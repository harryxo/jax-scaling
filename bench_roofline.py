"""The roofline receipt: below a critical batch size, a matmul's wall-clock
time is flat — you pay for memory movement and the FLOPs ride along free.

For C[B,N] = A[B,K] @ W[K,N] with B << K,N, arithmetic intensity is
2BKN / (dtype_bytes * KN) = 2B/dtype_bytes ops/byte. The chip flips from
memory-bound to compute-bound where that equals its peak_flops/hbm_bw, so:

    predicted crossover B = chip.crossover_batch * dtype_bytes / 2

Runs anywhere; the prediction only exists on chips we have specs for.
"""

import csv
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ["MPLBACKEND"] = "Agg"

import jax
import jax.numpy as jnp

import chips
from ledger import Ledger, RECEIPTS

K = N = int(os.environ.get("ROOFLINE_KN", "4096"))
_DEFAULT_BATCHES = "1 2 4 8 16 32 64 128 256 512 1024 2048"
BATCHES = [int(b) for b in os.environ.get("ROOFLINE_BATCHES", _DEFAULT_BATCHES).split()]
DTYPE = jnp.bfloat16
DTYPE_BYTES = 2

def median_ms(fn, *args, reps=int(os.environ.get("ROOFLINE_REPS", "10"))):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1e3


def main():
    dev = jax.devices()[0]
    chip = chips.identify(dev)
    print(f"device: {dev.device_kind} ({dev.platform}), matmul K=N={K}, {DTYPE.__name__}")

    ledger = Ledger("roofline")
    if chip:
        pred = chip.crossover_batch * DTYPE_BYTES / 2
        # Band: +/-2x. The spec sheet gives peak numbers; real kernels are
        # imperfect, so we claim the order of magnitude, not the digit.
        ledger.predict_range("crossover_batch", round(pred / 2), round(pred * 2),
                             note=f"{chip.name}: {chip.crossover_batch:.0f} ops/byte")
        ledger.predict_range("peak_tflops", round(chip.peak_flops / 1e12 * 0.3),
                             round(chip.peak_flops / 1e12 * 1.1),
                             note="sweep should approach the roof")
        print(f"predicted crossover batch on {chip.name}: ~{pred:.0f}")
    else:
        print("unknown chip (no spec entry) — measuring only, no prediction")

    matmul = jax.jit(lambda a, w: a @ w)
    w = jnp.ones((K, N), DTYPE)

    rows = []
    for b in BATCHES:
        a = jnp.ones((b, K), DTYPE)
        ms = median_ms(matmul, a, w)
        tflops = 2 * b * K * N / (ms / 1e3) / 1e12
        rows.append((b, ms, tflops))
        print(f"  B={b:>5}  {ms:8.3f} ms   {tflops:7.2f} TFLOP/s")

    # Empirical crossover: intersect the two asymptotes. Flat region time
    # comes from the smallest batches; the compute slope from the largest
    # (linear regime, overhead amortized). No saturation required.
    t_flat_ms = sorted(ms for _, ms, _ in rows[:3])[1]
    slope = rows[-1][1] / rows[-1][0]              # ms per unit batch, large-B secant
    measured = round(t_flat_ms / slope)
    print(f"\nempirical crossover (flat {t_flat_ms:.3f} ms / slope {slope * 1e3:.3f} us per B): "
          f"B ~ {measured}")

    # Diagnose the flat region: is it memory-bound (the roofline story) or
    # launch-overhead-bound (the third regime the napkin ignores)?
    weight_bytes = K * N * DTYPE_BYTES
    implied_bw = weight_bytes / (t_flat_ms / 1e3)
    if chip:
        share = implied_bw / chip.hbm_bw
        regime = "memory-bound" if share > 0.3 else "LAUNCH-OVERHEAD-bound"
        print(f"flat region implies {implied_bw / 1e9:.0f} GB/s vs {chip.hbm_bw / 1e9:.0f} spec "
              f"({share:.0%}) -> {regime}")
        if share <= 0.3:
            print(f"  -> the crossover measurement is contaminated by dispatch overhead;"
                  f" rerun with ROOFLINE_KN={K * 4} to raise the memory floor above it")
        ledger.measure("crossover_batch", measured)
        ledger.measure("peak_tflops", round(max(t for _, _, t in rows), 1))

    RECEIPTS.mkdir(exist_ok=True)
    with open(RECEIPTS / "roofline.csv", "w", newline="") as f:
        csv.writer(f).writerows([("batch", "ms", "tflops"), *rows])

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o-")
        ax.axvline(measured, ls="--", c="gray")
        ax.set(xlabel="batch size", ylabel="matmul ms",
               title=f"{dev.device_kind}: flat = memory-bound, linear = compute-bound")
        fig.savefig(RECEIPTS / "roofline.png", dpi=120, bbox_inches="tight")
        print(f"plot -> {RECEIPTS / 'roofline.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")

    return ledger.scoreboard(tolerance=1.0) if chip else True


if __name__ == "__main__":
    import sys
    ok = main()
    if "IPython" not in sys.modules:
        sys.exit(0 if ok else 1)
