# jax-scaling

Predict LLM training and inference performance from arithmetic, then verify
every prediction on hardware you can use for free. Companion repo to the
Surprisal video/course.

The protocol, everywhere: **predict → measure → scoreboard** (see `ledger.py`).

## Quickstart (any laptop, no accelerator needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -U pip jax rich optax matplotlib
make all            # every act's receipt, on this machine
make shard          # or one act at a time
```

## Act ↔ file map

| Act | File | Make target | Receipt |
|-----|------|-------------|---------|
| 0 Cold open | `ledger.py` | `ledger` | the scoreboard itself |
| 1 Rooflines | `chips.py`, `bench_roofline.py` | `roofline` | measured crossover vs spec-sheet band |
| 2 Count everything | `model.py`, `calc.py` | `model`, `calc` | analytic param count == pytree, exactly |
| 3 Price of communication | `comms.py`, `shard_matmul.py` | `shard` | predicted collectives vs compiled HLO + timed on real ICI |
| 4 Parallelize | `parallel.py`, `train.py` | `train` | 3 strategies, identical loss curves + per-strategy HLO collectives |
| 5 Reckoning | (same files, real chips) | — | the full scoreboard on TPU v5e |
| 6 Inference | `bench_inference.py` | `inference` | prefill/decode split + decode bandwidth floor |
| 7 Trace | `trace.py` | `trace` | every profiled microsecond bucketed: compute/memory/comm |

Verified so far: Act 3 receipts 6/6 green on Kaggle TPU v5e-8 (2026-07-01).
The first hardware-sized v5e-8 pass is recorded in
`docs/hardware-v5e8-20260702.md`: DP/FSDP training and decode are in band,
while roofline crossover remains out of band and TP training is explained by
communication-heavy traces.
Use `make hardware-v5e8` for the repeatable Kaggle pass.

## Hardware tiers

- **Tier 0** — any laptop: `XLA_FLAGS=--xla_force_host_platform_device_count=8`
  gives an 8-device mesh on CPU. Structure receipts (HLO, loss curves) run here.
- **Tier 1** — free: Colab (TPU v5e-1 / v6e-1, T4) and Kaggle (TPU v5e-8).
  Timing receipts run here.
- **Tier 2** — optional: GCP v5e spot for the 7B lab.

## Notebook troubleshooting (Colab / Kaggle)

- **Kaggle v5e-8 hardware validation:** after setup, run
  `make hardware-v5e8`. It continues through out-of-band receipts, writes a
  log, and tars `receipts/` under `/kaggle/working`.
- **`AttributeError` after `git pull`:** the kernel cached the old modules.
  Restart the session, or `import importlib, ledger, comms;
  importlib.reload(ledger); importlib.reload(comms)` before `%run`.
- **`os.fork()` RuntimeWarning** when using `!` shell commands after JAX has
  initialized: harmless for git/ls; restart the session if anything hangs.
- **Hugepages UserWarning / `SliceBuilder port 8471` error** on TPU startup:
  standard Kaggle/Colab noise, affects nothing.

## Source

The reasoning framework follows *How to Scale Your Model* — Austin, Douglas,
Frostig, Levskaya, et al., Google DeepMind, 2025.
https://jax-ml.github.io/scaling-book/

```bibtex
@misc{scalingbook,
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya,
            Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico
            and Choy, Peter and Ramasesh, Vinay and Webson, Albert and
            Pope, Reiner},
  title  = {How to Scale Your Model},
  year   = {2025},
  publisher = {Google DeepMind},
  howpublished = {\url{https://jax-ml.github.io/scaling-book/}}
}
```

All code, benchmarks, and figures in this repo are original.
