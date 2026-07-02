"""Train the mini-transformer under any parallelism strategy, and check the
predictions that matter:

  1. All strategies compute the same thing -> loss curves agree.
  2. Each strategy's collectives are the ones the theory says (HLO receipt).
  3. On a known chip: measured step time lands in the MFU band.

  train.py --strategy all --steps 30            # laptop, 8 fake devices
  train.py --strategy fsdp --steps 200 --batch 64   # Kaggle v5e-8
"""

import argparse
import csv
import os
import re
import time
import urllib.request
from collections import Counter
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding

import chips
import parallel
from ledger import Ledger, RECEIPTS
from model import Config, count_params, init, loss_fn, train_flops_per_token

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
COLLECTIVE_RE = re.compile(r"all-reduce|all-gather|reduce-scatter|collective-permute")
optimizer = optax.adamw(3e-4)


def load_data():
    path = Path(__file__).parent / "data" / "input.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print(f"downloading tinyshakespeare -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    return np.array([stoi[c] for c in text], np.int32), len(vocab)


def batches(data, batch, seq, steps, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(data) - seq - 1, batch)
        yield np.stack([data[i:i + seq + 1] for i in idx])


@partial(jax.jit, donate_argnums=(0, 1), static_argnames="cfg")
def train_step(params, opt_state, tokens, cfg):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, cfg)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


def run(strategy, cfg, data, args, mesh, ledger):
    params = parallel.shard(init(cfg, jax.random.key(0)), strategy, mesh)
    opt_state = jax.jit(optimizer.init)(params)  # inherits param sharding
    bspec = NamedSharding(mesh, parallel.batch_spec(strategy))

    # HLO receipt: which collectives did the annotations force?
    example = jax.device_put(next(batches(data, args.batch, cfg.seq_len, 1)), bspec)
    hlo = train_step.lower(params, opt_state, example, cfg).compile().as_text()
    coll = Counter(COLLECTIVE_RE.findall(hlo))
    print(f"\n[{strategy}] collectives in one compiled step: {dict(coll) or 'none'}")

    losses, step_ms = [], []
    for i, b in enumerate(batches(data, args.batch, cfg.seq_len, args.steps, seed=1)):
        tokens = jax.device_put(b, bspec)
        t0 = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state, tokens, cfg)
        loss = float(loss)  # blocks until the step is done
        if i > 2:  # skip compile + warmup
            step_ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(loss)
    med_ms = sorted(step_ms)[len(step_ms) // 2]

    flops_step = train_flops_per_token(cfg) * args.batch * cfg.seq_len
    chip = chips.identify(jax.devices()[0])
    line = f"[{strategy}] loss {losses[0]:.3f} -> {losses[-1]:.3f}, median step {med_ms:.1f} ms"
    if chip:
        mfu = flops_step / (med_ms / 1e3) / (len(jax.devices()) * chip.peak_flops)
        line += f", MFU {mfu:.1%} on {len(jax.devices())}x {chip.name}"
        ledger.measure(f"{strategy}/step_ms", round(med_ms, 2))
    print(line)

    with open(RECEIPTS / f"losses_{strategy}.csv", "w", newline="") as f:
        csv.writer(f).writerows(enumerate(losses))
    return losses, med_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all", choices=["dp", "fsdp", "tp", "all"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    data, vocab = load_data()
    cfg = Config(vocab=vocab, d_model=args.d_model, n_heads=args.d_model // 32,
                 n_layers=args.n_layers, d_ff=4 * args.d_model, seq_len=args.seq_len)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), (parallel.AXIS,))
    print(f"{len(devices)}x {devices[0].device_kind}, params={count_params(cfg):,}, "
          f"batch={args.batch}, seq={cfg.seq_len}")

    ledger = Ledger("train")
    chip = chips.identify(devices[0])
    if chip:
        # Step-time band from the napkin: assume MFU lands in [10%, 60%].
        flops_step = train_flops_per_token(cfg) * args.batch * cfg.seq_len
        ideal_ms = flops_step / (len(devices) * chip.peak_flops) * 1e3
        for s in (["dp", "fsdp", "tp"] if args.strategy == "all" else [args.strategy]):
            ledger.predict_range(f"{s}/step_ms", round(ideal_ms / 0.6, 2),
                                 round(ideal_ms / 0.1, 2), unit="ms",
                                 note=f"6ND napkin, MFU 10-60% on {chip.name}")

    strategies = ["dp", "fsdp", "tp"] if args.strategy == "all" else [args.strategy]
    finals = {}
    for s in strategies:
        losses, _ = run(s, cfg, data, args, mesh, ledger)
        finals[s] = losses[-1]

    if len(finals) > 1:
        vals = list(finals.values())
        spread = (max(vals) - min(vals)) / min(vals)
        ledger.predict_range("strategy_loss_spread", 0.0, 0.005,
                             note="same computation -> same loss curve")
        ledger.measure("strategy_loss_spread", round(spread, 6))
        print(f"\nfinal losses: { {k: round(v, 4) for k, v in finals.items()} } "
              f"(spread {spread:.4%})")

    return ledger.scoreboard()


if __name__ == "__main__":
    import sys
    ok = main()
    if "IPython" not in sys.modules:
        sys.exit(0 if ok else 1)
