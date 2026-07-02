"""The napkin, as a CLI. Everything here is arithmetic — no arrays, no
accelerators. Predictions from this file get checked against real runs by
train.py (step time, memory) and the scoreboard.

  calc.py                          # the repo's default mini model
  calc.py --d-model 4096 ...       # any config
  calc.py --params 7e9             # skip the config: cost tiers for an N-param model
"""

import argparse

import chips
from model import Config, count_params, train_flops_per_token

CHIPS = {"v5e": chips.V5E, "v6e": chips.V6E, "t4": chips.T4, "l4": chips.L4,
         "a100": chips.A100, "h100": chips.H100}

V5E_ONDEMAND_PER_CHIP_HR = 1.20  # us-central, TODO(verify): prices drift
SPOT_DISCOUNT = 0.5


def fmt(n):
    """Counts: K/M/B/T, scientific above that."""
    if abs(n) >= 1e15:
        return f"{n:.2e}"
    for div, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(n) >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:.0f}"


def fmt_bytes(n):
    for div, suffix in [(1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")]:
        if abs(n) >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:.0f}B"


def training_memory_bytes(n_params, batch, seq, d_model, n_layers):
    """Mixed precision: bf16 weights (2N) + f32 master/m/v (12N) + activations
    (~20 bytes per activation-site with checkpointing every block)."""
    states = 14 * n_params
    activations = 20 * batch * seq * d_model * n_layers
    return states, activations


def step_report(cfg: Config, batch, chip, mfu, n_chips):
    n = count_params(cfg)
    flops_step = train_flops_per_token(cfg) * batch * cfg.seq_len
    t_step = flops_step / (n_chips * chip.peak_flops * mfu)
    states, acts = training_memory_bytes(n, batch, cfg.seq_len, cfg.d_model, cfg.n_layers)

    print(f"config: {cfg}")
    print(f"params:            {fmt(n)}  ({count_params(cfg):,})")
    print(f"FLOPs/step:        {fmt(flops_step)} (batch={batch}, seq={cfg.seq_len})")
    print(f"memory:            {fmt_bytes(states)} states + {fmt_bytes(acts)} activations "
          f"vs {fmt_bytes(chip.hbm_bytes * n_chips)} HBM on {n_chips}x {chip.name}"
          f"{'  ** OOM without sharding **' if states + acts > chip.hbm_bytes else ''}")
    print(f"predicted step:    {t_step * 1e3:.1f} ms at {mfu:.0%} MFU on {n_chips}x {chip.name}")
    return t_step


def tier_table(n_params, chip):
    """What does it cost to touch an N-param model? Three tiers."""
    per_chip_flops = lambda mfu: chip.peak_flops * mfu
    rate = V5E_ONDEMAND_PER_CHIP_HR

    print(f"cost tiers for a {fmt(n_params)}-param model on {chip.name} "
          f"(on-demand ${rate}/chip-hr, spot ~{SPOT_DISCOUNT:.0%}):\n")
    tiers = [
        ("pretrain, Chinchilla 20N tokens", 6 * n_params * 20 * n_params, 0.40),
        ("fine-tune, 2B tokens", 6 * n_params * 2e9, 0.40),
        ("verify: 100 steps @ 1M tokens/step", 6 * n_params * 100 * 1e6, 0.35),
    ]

    for name, flops, mfu in tiers:
        hours = flops / per_chip_flops(mfu) / 3600
        print(f"  {name:<38} {fmt(flops)} FLOPs  "
              f"{hours:>9.0f} chip-hr  ${hours * rate:>9,.0f} on-demand  "
              f"${hours * rate * SPOT_DISCOUNT:>9,.0f} spot")
    print(f"\n  memory check: {fmt_bytes(14 * n_params)} of training state -> "
          f"{max(1, round(14 * n_params / chip.hbm_bytes))}+ chips (FSDP), "
          f"{fmt_bytes(chip.hbm_bytes)} HBM each")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, help="skip config; tier table for N params")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=0, help="default 4*d_model")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chip", choices=CHIPS, default="v5e")
    ap.add_argument("--n-chips", type=int, default=1)
    ap.add_argument("--mfu", type=float, default=0.40)
    args = ap.parse_args()

    if args.params:
        tier_table(args.params, CHIPS[args.chip])
    else:
        cfg = Config(vocab=args.vocab, d_model=args.d_model, n_heads=args.n_heads,
                     n_layers=args.n_layers, d_ff=args.d_ff or 4 * args.d_model,
                     seq_len=args.seq_len)
        step_report(cfg, args.batch, CHIPS[args.chip], args.mfu, args.n_chips)
