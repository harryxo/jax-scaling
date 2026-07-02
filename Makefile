PY := .venv/bin/python

.PHONY: shard devices ledger

shard:
	$(PY) shard_matmul.py

devices:
	$(PY) -c "import os; os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=8'); import jax; print(jax.devices())"

ledger:
	$(PY) -c "from pathlib import Path; from ledger import Ledger; [Ledger(p.stem).scoreboard() for p in Path('receipts').glob('*.json')]"
