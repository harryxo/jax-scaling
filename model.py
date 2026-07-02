"""A minimal decoder-only transformer in pure JAX, with the accounting done
inline: every parameter block is annotated with its param count and the
training FLOPs it incurs (2x forward matmul FLOPs for the backward pass on
activations, 2x for the backward on weights -> 6x per matmul param per token).

The receipt: count_params(cfg) — pure arithmetic from the annotations —
must equal the actual pytree. If the accounting and the code ever disagree,
the scoreboard catches it.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int = 96
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    seq_len: int = 256

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


def init(cfg: Config, key):
    ks = iter(jax.random.split(key, 2 + 6 * cfg.n_layers))
    dm, ff = cfg.d_model, cfg.d_ff
    scale = dm ** -0.5

    def dense(k, shape):
        return jax.random.normal(k, shape, jnp.float32) * scale

    params = {
        # embed: vocab*d params; tied output head adds 6*vocab*d train FLOPs/token
        "embed": jax.random.normal(next(ks), (cfg.vocab, dm), jnp.float32) * 0.02,
        # pos: seq*d params; lookup, ~0 FLOPs
        "pos": jax.random.normal(next(ks), (cfg.seq_len, dm), jnp.float32) * 0.02,
        "blocks": [],
        # ln_f: d params
        "ln_f": jnp.ones(dm),
    }
    for _ in range(cfg.n_layers):
        params["blocks"].append({
            # ln1, ln2: 2*d params
            "ln1": jnp.ones(dm),
            "ln2": jnp.ones(dm),
            # wq,wk,wv,wo: 4*d^2 params -> 6*4*d^2 train FLOPs/token
            "wq": dense(next(ks), (dm, dm)),
            "wk": dense(next(ks), (dm, dm)),
            "wv": dense(next(ks), (dm, dm)),
            "wo": dense(next(ks), (dm, dm)),
            # w1,w2: 2*d*ff params -> 6*2*d*ff train FLOPs/token
            "w1": dense(next(ks), (dm, ff)),
            "w2": dense(next(ks), (ff, dm)),
        })
    return params


def count_params(cfg: Config) -> int:
    """Sum of the annotations above — arithmetic only, no arrays."""
    dm, ff = cfg.d_model, cfg.d_ff
    per_layer = 4 * dm * dm + 2 * dm * ff + 2 * dm
    return cfg.vocab * dm + cfg.seq_len * dm + cfg.n_layers * per_layer + dm


def measure_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def train_flops_per_token(cfg: Config) -> float:
    """6 * (matmul params) + attention scores. Attention QK^T and AV are
    2*S*d fwd FLOPs/token each -> 12*S*d*L for training."""
    dm, ff, s = cfg.d_model, cfg.d_ff, cfg.seq_len
    matmul_params = cfg.n_layers * (4 * dm * dm + 2 * dm * ff) + cfg.vocab * dm
    return 6 * matmul_params + 12 * s * dm * cfg.n_layers


def rmsnorm(x, g):
    return x * g * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)


def attention(x, p, cfg: Config, mask):
    b, s, dm = x.shape
    h, hd = cfg.n_heads, cfg.head_dim
    q = (x @ p["wq"]).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
    k = (x @ p["wk"]).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
    v = (x @ p["wv"]).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) * hd ** -0.5 + mask[:s, :s]
    out = jax.nn.softmax(scores, -1) @ v
    return out.transpose(0, 2, 1, 3).reshape(b, s, dm) @ p["wo"]


def block(x, p, cfg: Config, mask):
    x = x + attention(rmsnorm(x, p["ln1"]), p, cfg, mask)
    return x + jax.nn.gelu(rmsnorm(x, p["ln2"]) @ p["w1"]) @ p["w2"]


@partial(jax.jit, static_argnames="cfg")
def forward(params, tokens, cfg: Config):
    s = tokens.shape[1]
    mask = jnp.where(jnp.tril(jnp.ones((cfg.seq_len, cfg.seq_len), bool)), 0.0, -jnp.inf)
    x = params["embed"][tokens] + params["pos"][:s]
    for p in params["blocks"]:
        x = block(x, p, cfg, mask)
    x = rmsnorm(x, params["ln_f"])
    return x @ params["embed"].T  # tied head


def loss_fn(params, tokens, cfg: Config):
    """Next-token cross-entropy over tokens[:, :-1] -> tokens[:, 1:]."""
    logits = forward(params, tokens[:, :-1], cfg)
    logp = jax.nn.log_softmax(logits, -1)
    targets = jax.nn.one_hot(tokens[:, 1:], cfg.vocab)
    return -jnp.mean(jnp.sum(logp * targets, -1))


if __name__ == "__main__":
    from ledger import Ledger

    cfg = Config()
    ledger = Ledger("model")
    ledger.predict("param_count", count_params(cfg),
                   note="sum of the inline annotations in init()")
    params = init(cfg, jax.random.key(0))
    ledger.measure("param_count", measure_params(params))

    toks = jax.random.randint(jax.random.key(1), (2, cfg.seq_len), 0, cfg.vocab)
    print(f"loss on random tokens: {loss_fn(params, toks, cfg):.3f} "
          f"(uniform baseline ln({cfg.vocab}) = {jnp.log(cfg.vocab):.3f})")

    import sys
    ok = ledger.scoreboard(tolerance=0.0)
    if "IPython" not in sys.modules:
        sys.exit(0 if ok else 1)
