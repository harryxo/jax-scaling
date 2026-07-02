"""Inference is two workloads wearing one API.

Prefill processes S tokens per forward pass — big matmuls, compute-bound.
Decode processes ONE token per pass but still touches every parameter and
the whole KV cache — memory-bound, with a hard floor:

    decode s/token >= (param_bytes + kv_cache_bytes) / hbm_bw

This file adds a KV cache to the model and measures both phases. The floor
prediction activates on chips we have specs for.
"""

import argparse
import os
import time
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp

import chips
from ledger import Ledger
from model import Config, count_params, forward, init, rmsnorm


def init_cache(cfg: Config, batch: int):
    shape = (cfg.n_layers, batch, cfg.n_heads, cfg.seq_len, cfg.head_dim)
    return {"k": jnp.zeros(shape), "v": jnp.zeros(shape)}


def cache_bytes(cfg: Config, batch: int) -> int:
    return 2 * cfg.n_layers * batch * cfg.n_heads * cfg.seq_len * cfg.head_dim * 4


@partial(jax.jit, donate_argnums=(1,), static_argnames="cfg")
def decode_step(params, cache, token, pos, cfg: Config):
    """One token for the whole batch: compute q/k/v for position `pos`,
    append k/v to the cache, attend over everything cached so far."""
    b = token.shape[0]
    h, hd = cfg.n_heads, cfg.head_dim
    x = params["embed"][token] + params["pos"][pos]          # (B, 1, D)

    for i, p in enumerate(params["blocks"]):
        xn = rmsnorm(x, p["ln1"])
        q = (xn @ p["wq"]).reshape(b, 1, h, hd).transpose(0, 2, 1, 3)
        k = (xn @ p["wk"]).reshape(b, 1, h, hd).transpose(0, 2, 1, 3)
        v = (xn @ p["wv"]).reshape(b, 1, h, hd).transpose(0, 2, 1, 3)
        cache["k"] = jax.lax.dynamic_update_slice(cache["k"], k[None], (i, 0, 0, pos, 0))
        cache["v"] = jax.lax.dynamic_update_slice(cache["v"], v[None], (i, 0, 0, pos, 0))
        valid = jnp.arange(cfg.seq_len) <= pos                # attend to <= pos
        scores = q @ cache["k"][i].transpose(0, 1, 3, 2) * hd ** -0.5
        scores = jnp.where(valid, scores, -jnp.inf)
        out = jax.nn.softmax(scores, -1) @ cache["v"][i]
        x = x + out.transpose(0, 2, 1, 3).reshape(b, 1, cfg.d_model) @ p["wo"]
        x = x + jax.nn.gelu(rmsnorm(x, p["ln2"]) @ p["w1"]) @ p["w2"]

    logits = rmsnorm(x, params["ln_f"]) @ params["embed"].T
    return jnp.argmax(logits[:, -1], -1, keepdims=True), cache


def bench(cfg: Config, params, batch: int, decode_tokens: int):
    prompt = jnp.zeros((batch, cfg.seq_len), jnp.int32)

    jax.block_until_ready(forward(params, prompt, cfg))      # compile
    t0 = time.perf_counter()
    jax.block_until_ready(forward(params, prompt, cfg))
    prefill_tps = batch * cfg.seq_len / (time.perf_counter() - t0)

    cache = init_cache(cfg, batch)
    token = jnp.zeros((batch, 1), jnp.int32)
    token, cache = decode_step(params, cache, token, 0, cfg)  # compile
    t0 = time.perf_counter()
    for pos in range(1, decode_tokens + 1):
        token, cache = decode_step(params, cache, token, pos, cfg)
    jax.block_until_ready(token)
    decode_tps = batch * decode_tokens / (time.perf_counter() - t0)
    return prefill_tps, decode_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = ap.parse_args()

    cfg = Config(vocab=96, d_model=args.d_model, n_heads=args.d_model // 32,
                 n_layers=args.n_layers, d_ff=4 * args.d_model, seq_len=args.seq_len)
    params = init(cfg, jax.random.key(0))
    param_bytes = count_params(cfg) * 4
    chip = chips.identify(jax.devices()[0])
    print(f"params={count_params(cfg):,} ({param_bytes / 1e6:.1f} MB f32), "
          f"device={jax.devices()[0].device_kind}")

    ledger = Ledger("inference")
    # Universal receipt: prefill beats decode per token, by a lot. It does
    # S matmul-batched tokens per weight-read; decode does 1.
    ledger.predict_range("prefill_over_decode_at_b1", 2, 100_000,
                         note="same weights read, SxB more tokens amortized")
    if chip:
        floor_s = (param_bytes + cache_bytes(cfg, 1)) / chip.hbm_bw
        ledger.predict_range("decode_tps_b1", round(0.2 / floor_s), round(1 / floor_s),
                             note=f"bandwidth floor on {chip.name}; band covers kernel overheads")

    for b in args.batches:
        pre, dec = bench(cfg, params, b, args.decode_tokens)
        print(f"  B={b:>3}  prefill {pre:>12,.0f} tok/s   decode {dec:>10,.0f} tok/s   "
              f"kv={cache_bytes(cfg, b) / 1e6:.1f} MB")
        if b == 1:
            ledger.measure("prefill_over_decode_at_b1", round(pre / dec, 1))
            if chip:
                ledger.measure("decode_tps_b1", round(dec))

    return ledger.scoreboard()


if __name__ == "__main__":
    import sys
    ok = main()
    if "IPython" not in sys.modules:
        sys.exit(0 if ok else 1)
