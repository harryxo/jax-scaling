"""The trace doesn't lie: attribute every profiled microsecond to compute,
memory movement, or communication.

Runs a few training steps under jax.profiler.trace, then parses the emitted
Perfetto/Chrome trace events into the three buckets the whole course reasons
about. Open the same trace in Perfetto (ui.perfetto.dev) for the visual pass;
this parser is the receipt that the buckets sum to something sensible.

  trace.py --strategy fsdp --steps 10
"""

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding

import parallel
from ledger import RECEIPTS
from model import Config, init
from train import batches, load_data, optimizer, train_step

TRACE_DIR = RECEIPTS / "trace"

COMM = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all")
MEMORY = ("copy", "transpose", "dynamic-update-slice", "dynamic-slice", "broadcast",
          "concatenate", "reshape", "slice", "pad", "bitcast")


def bucket(name: str) -> str:
    n = name.lower()
    if any(c in n for c in COMM):
        return "communication"
    if any(m in n for m in MEMORY):
        return "memory"
    return "compute"


def emit_trace(strategy: str, steps: int, batch: int):
    data, vocab = load_data()
    cfg = Config(vocab=vocab, d_model=128, n_heads=4, n_layers=2, d_ff=512, seq_len=128)
    mesh = Mesh(np.array(jax.devices()), (parallel.AXIS,))
    params = parallel.shard(init(cfg, jax.random.key(0)), strategy, mesh)
    opt_state = jax.jit(optimizer.init)(params)
    bspec = NamedSharding(mesh, parallel.batch_spec(strategy))

    it = batches(data, batch, cfg.seq_len, steps + 3)
    for b in list(it)[:3]:  # compile + warm OUTSIDE the trace
        params, opt_state, _ = train_step(params, opt_state, jax.device_put(b, bspec), cfg)

    with jax.profiler.trace(str(TRACE_DIR)):
        for b in batches(data, batch, cfg.seq_len, steps, seed=2):
            params, opt_state, loss = train_step(params, opt_state,
                                                 jax.device_put(b, bspec), cfg)
        jax.block_until_ready(loss)
    print(f"traced {steps} {strategy} steps -> {TRACE_DIR}")


def parse_latest_trace():
    files = sorted(TRACE_DIR.rglob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no *.trace.json.gz under {TRACE_DIR} — did the trace emit?")
    path = files[-1]
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]

    # Process-name metadata tells us which pids are device timelines.
    pid_names = {e["pid"]: e["args"]["name"] for e in events
                 if e.get("ph") == "M" and e.get("name") == "process_name"}
    device_pids = {pid for pid, name in pid_names.items()
                   if any(k in name.lower() for k in ("device", "tpu", "gpu", "xla", "stream"))}

    # In fallback mode (no device pids, e.g. CPU) drop host-framework spans —
    # they are bookkeeping, not XLA ops. Real device timelines don't have them.
    host_noise = ("thunk", "block_until", "try_to_block", "rendezvous", "$", "execute")

    totals, ops = Counter(), Counter()
    for e in events:
        if e.get("ph") == "X" and "dur" in e and (not device_pids or e["pid"] in device_pids):
            name = e.get("name", "?")
            if not device_pids and any(h in name.lower() for h in host_noise):
                continue
            totals[bucket(name)] += e["dur"]
            ops[name[:60]] += e["dur"]

    total = sum(totals.values())
    print(f"\ntrace: {path.name}  (device timelines: "
          f"{[pid_names[p] for p in device_pids] or 'none found — using all events'})")
    print(f"{'bucket':<15} {'ms':>10}  share")
    for b, us in totals.most_common():
        print(f"{b:<15} {us / 1e3:>10.2f}  {us / total:>6.1%}")
    print("\ntop ops by device time:")
    for name, us in ops.most_common(8):
        print(f"  {us / 1e3:>9.2f} ms  [{bucket(name):<13}] {name}")
    print(f"\nopen in ui.perfetto.dev: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fsdp", choices=["dp", "fsdp", "tp"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()

    if not args.parse_only:
        emit_trace(args.strategy, args.steps, args.batch)
    parse_latest_trace()
