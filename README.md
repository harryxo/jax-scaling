# jax-scaling

Predict LLM training and inference performance from arithmetic, then verify
every prediction on hardware you can use for free. Companion repo to the
Surprisal video/course.

The protocol, everywhere: **predict → measure → scoreboard** (see `ledger.py`).

## Quickstart (any laptop, no accelerator needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -U pip jax rich
make shard          # one matmul, four shardings, HLO receipts
```

## Act ↔ file map

| Act | File | Receipt |
|-----|------|---------|
| 0 Cold open | `ledger.py` | the scoreboard itself |
| 1 Rooflines | `chips.py`, `bench_roofline.py` | measured crossover vs spec-sheet prediction |
| 2 Count everything | `model.py`, `calc.py` | calculator totals vs a real training run |
| 3 Price of communication | `comms.py`, `shard_matmul.py` | predicted collectives vs compiled HLO |
| 4 Parallelize | `parallel.py`, `train.py` | 3 strategies, identical loss curves |
| 5 Reckoning | (same files, real chips) | the full scoreboard on TPU v5e |
| 6 Inference | `bench_inference.py` | measured decode vs bandwidth floor |
| 7 Trace | `trace.py` | profiler buckets sum to step time |

## Hardware tiers

- **Tier 0** — any laptop: `XLA_FLAGS=--xla_force_host_platform_device_count=8`
  gives an 8-device mesh on CPU. Structure receipts (HLO, loss curves) run here.
- **Tier 1** — free: Colab (TPU v5e-1 / v6e-1, T4) and Kaggle (TPU v5e-8).
  Timing receipts run here.
- **Tier 2** — optional: GCP v5e spot for the 7B lab.

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
