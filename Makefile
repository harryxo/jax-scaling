PY := .venv/bin/python

.PHONY: all shard roofline model calc train inference trace ledger devices hardware-v5e8

# Run every act's receipt on this machine (8 fake devices where needed).
all: roofline model shard train inference trace ledger

roofline:           ## Act 1: matmul batch sweep vs spec-sheet crossover
	$(PY) bench_roofline.py

model:              ## Act 2: analytic param count vs actual pytree
	$(PY) model.py

calc:               ## Act 2: the napkin CLI (try: make calc ARGS="--params 7e9")
	$(PY) calc.py $(ARGS)

shard:              ## Act 3: one matmul, four shardings, HLO receipts
	$(PY) shard_matmul.py

train:              ## Act 4: three strategies, identical loss curves
	$(PY) train.py --strategy all --steps 30

inference:          ## Act 6: prefill vs decode, KV cache, bandwidth floor
	$(PY) bench_inference.py

trace:              ## Act 7: profile train steps, bucket every microsecond
	$(PY) trace.py

ledger:             ## Print every scoreboard
	$(PY) -c "from pathlib import Path; from ledger import Ledger; \
	[Ledger(p.stem).scoreboard() for p in sorted(Path('receipts').glob('*.json'))]"

devices:
	$(PY) -c "import os; os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=8'); import jax; print(jax.devices())"

hardware-v5e8:       ## Kaggle TPU v5e-8 pass: continue through out-of-band receipts
	bash scripts/kaggle_hardware_validation.sh
