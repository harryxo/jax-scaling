"""One matmul, four shardings: predict which collectives XLA inserts, then
read the compiled HLO to check. We never write communication code — the
sharding annotation alone determines it.

Runs identically on:
  - any laptop:   8 fake CPU devices (set automatically below)
  - Kaggle v5e-8: 8 real chips (delete nothing, change nothing)

The four cases:
  replicated       A(n,n) x B(n,n), both whole on every device -> no comm
  data_parallel    A row-sharded, B whole -> rows are independent -> no comm
  contracting      A col-sharded, B row-sharded -> every device holds a
                   PARTIAL SUM of the full output -> all-reduce, unavoidable
  weight_gather    B col-sharded, output forced replicated -> every device
                   computes a slice, then all-gather
"""

import os
import re
import time

# Must happen before importing jax. Harmless on a real TPU runtime (jax
# prefers the TPU backend over forced CPU devices).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import comms
from ledger import Ledger

# 2048 (16 MB arrays) is laptop-friendly. On real chips bump it so the
# collective is bandwidth-bound, not latency-bound: SCALING_N=8192 python ...
N = int(os.environ.get("SCALING_N", "2048"))
COLLECTIVE_RE = re.compile(r"all-reduce|all-gather|reduce-scatter|collective-permute")


def collectives_in_hlo(sharded_args, out_sharding):
    """Compile A @ B with the given shardings; return sorted collective op
    names found in the optimized HLO."""
    jitted = jax.jit(lambda a, b: a @ b, out_shardings=out_sharding)
    hlo = jitted.lower(*sharded_args).compile().as_text()
    return sorted(set(COLLECTIVE_RE.findall(hlo))) or ["none"]


def _median_time_s(fn, arg, reps=20):
    jax.block_until_ready(fn(arg))  # compile
    for _ in range(3):              # warm
        jax.block_until_ready(fn(arg))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def time_collectives(mesh, ledger, link):
    """Time the two collectives the matmul cases predicted, in isolation,
    at exactly the byte counts on the ledger. Fills the pending rows."""
    try:
        shard_map = jax.shard_map
    except AttributeError:  # older jax
        from jax.experimental.shard_map import shard_map

    repl = NamedSharding(mesh, P(None, None))
    full = jax.device_put(jnp.ones((N, N), jnp.float32), repl)
    rows = jax.device_put(jnp.ones((N, N), jnp.float32),
                          NamedSharding(mesh, P("x", None)))
    bytes_ = N * N * 4

    # Pure all-reduce: every device holds the full array ("partial sums"),
    # psum combines them. Same traffic as the contracting matmul case.
    allreduce = jax.jit(shard_map(lambda a: jax.lax.psum(a, "x"), mesh=mesh,
                                  in_specs=P(None, None), out_specs=P(None, None)))
    # Pure all-gather: row shards in, replicated out. The identity fn
    # compiles to exactly one resharding collective.
    allgather = jax.jit(lambda a: a, out_shardings=repl)

    for key, fn, arg, factor in [("contracting/comm_us", allreduce, full, 2),
                                 ("weight_gather/comm_us", allgather, rows, 1)]:
        # Safety receipt: confirm the isolated op really is the collective
        # we think we are timing, before trusting the number.
        found = sorted(set(COLLECTIVE_RE.findall(
            fn.lower(arg).compile().as_text()))) or ["none"]
        t = _median_time_s(fn, arg)
        ledger.measure(key, round(t * 1e6, 1))
        eff_gbs = factor * bytes_ * 7 / 8 / t / 1e9
        print(f"{key}: HLO={found}  measured {t * 1e6:.0f} us  "
              f"-> effective {eff_gbs:.1f} GB/s (spec sheet says {link.bandwidth / 1e9:.0f})")


def main():
    devices = jax.devices()
    assert len(devices) == 8, f"expected 8 devices, got {len(devices)}: {devices}"
    platform = devices[0].platform
    mesh = Mesh(np.array(devices), ("x",))
    repl = NamedSharding(mesh, P(None, None))

    print(f"platform={platform}, devices={len(devices)}, mesh={mesh.shape}")

    A = jnp.ones((N, N), jnp.float32)
    B = jnp.ones((N, N), jnp.float32)
    out_bytes = N * N * 4  # float32 output

    #                 A spec        B spec        out       expected collectives
    cases = {
        "replicated":    (P(None, None), P(None, None), repl, "none"),
        "data_parallel": (P("x", None),  P(None, None), None, "none"),
        "contracting":   (P(None, "x"),  P("x", None),  repl, "all-reduce"),
        "weight_gather": (P(None, None), P(None, "x"),  repl, "all-gather"),
    }

    ledger = Ledger("shard_matmul")
    # Comm cost is always predicted for real silicon — fake CPU devices have
    # no interconnect. Act 5b measures this on Kaggle's v5e-8.
    link = comms.V5E_ICI

    for name, (spec_a, spec_b, out, expected) in cases.items():
        # Prediction goes on the ledger BEFORE we compile anything.
        ledger.predict(f"{name}/collectives", expected,
                       note=f"A={spec_a} B={spec_b}")

        a = jax.device_put(A, NamedSharding(mesh, spec_a))
        b = jax.device_put(B, NamedSharding(mesh, spec_b))
        found = collectives_in_hlo((a, b), out)
        ledger.measure(f"{name}/collectives", ",".join(found))

        # What should the communication cost on real hardware?
        cost = {"all-reduce": comms.all_reduce_s,
                "all-gather": comms.all_gather_s}.get(expected)
        if cost:
            # Band: links are bidirectional (up to 2x the per-axis spec) and
            # the naive model is one direction of one axis (1x). Measured on
            # v5e-8: all-reduce 1.67x, all-gather 1.21x — both in band.
            hi_us = cost(out_bytes, 8, link) * 1e6
            lo_us = hi_us / 2
            cost_str = f"predicted comm on {link}: {lo_us:.0f}-{hi_us:.0f} us"
            ledger.predict_range(f"{name}/comm_us", round(lo_us, 1), round(hi_us, 1),
                                 unit="us", note="[2x bidirectional, 1x one-way] ICI band")
        else:
            cost_str = "predicted comm: zero"
        print(f"\n--- {name}: A={spec_a} B={spec_b} -> {found} | {cost_str}")

        if name == "data_parallel":  # show one layout, not four
            jax.debug.visualize_array_sharding(a)

    if platform == "tpu" or os.environ.get("SCALING_FORCE_TIMING"):
        print("\ntiming collectives in isolation...")
        time_collectives(mesh, ledger, link)
    else:
        print("\ntiming skipped: fake CPU devices have no interconnect — "
              "run this same file on Kaggle v5e-8 to fill the pending rows")

    ok = ledger.scoreboard()
    print("RESULT:", "all HLO receipts match predictions" if ok else "MISMATCH — see board")
    return ok


if __name__ == "__main__":
    import sys
    ok = main()
    if "IPython" not in sys.modules:  # don't spam tracebacks in notebooks
        sys.exit(0 if ok else 1)
