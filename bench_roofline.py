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

import jax
import jax.numpy as jnp

import chips
from ledger import Ledger, RECEIPTS

K = N = int(os.environ.get("ROOFLINE_KN", "4096"))
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DTYPE = jnp.bfloat16
DTYPE_BYTES = 2

def median_ms(fn, *args, reps=10):
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

    # Empirical crossover: smallest batch reaching half the peak throughput
    # seen in the sweep (throughput doubles with B while memory-bound, then
    # plateaus once compute-bound).
    peak = max(t for _, _, t in rows)
    measured = next(b for b, _, t in rows if t >= peak / 2)
    print(f"\nempirical crossover batch (first B at >=50% of sweep peak): {measured}")
    if chip:
        ledger.measure("crossover_batch", measured)

    RECEIPTS.mkdir(exist_ok=True)
    with open(RECEIPTS / "roofline.csv", "w", newline="") as f:
        csv.writer(f).writerows([("batch", "ms", "tflops"), *rows])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o-")
        ax.axvline(measured, ls="--", c="gray")
        ax.set(xlabel="batch size", ylabel="matmul ms",
               title=f"{dev.device_kind}: flat = memory-bound, linear = compute-bound")
        fig.savefig(RECEIPTS / "roofline.png", dpi=120, bbox_inches="tight")
        print(f"plot -> {RECEIPTS / 'roofline.png'}")
    except ImportError:
        pass

    return ledger.scoreboard(tolerance=1.0) if chip else True


if __name__ == "__main__":
    import sys
    ok = main()
    if "IPython" not in sys.modules:
        sys.exit(0 if ok else 1)
