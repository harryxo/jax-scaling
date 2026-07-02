#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M)}"
OUT_DIR="${OUT_DIR:-/kaggle/working}"
if [[ ! -d "$OUT_DIR" ]]; then
  OUT_DIR="$ROOT"
fi
LOG="${LOG:-$OUT_DIR/jax-scaling-hw-validation-v5e8-$STAMP.log}"
TARBALL="${TARBALL:-$OUT_DIR/jax-scaling-receipts-v5e8-$STAMP.tgz}"

run() {
  local name="$1"
  shift
  echo "===== $name =====" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  echo "===== $name exit=$status =====" | tee -a "$LOG"
}

: > "$LOG"

run devices "$PY" -c "import jax; print(jax.devices())"

run roofline-extended env \
  ROOFLINE_BATCHES="${ROOFLINE_BATCHES:-1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192}" \
  "$PY" bench_roofline.py

run train-hardware "$PY" train.py \
  --strategy "${TRAIN_STRATEGY:-all}" \
  --steps "${TRAIN_STEPS:-100}" \
  --batch "${TRAIN_BATCH:-64}" \
  --d-model "${TRAIN_D_MODEL:-512}" \
  --n-layers "${TRAIN_LAYERS:-6}" \
  --seq-len "${TRAIN_SEQ_LEN:-256}"

run inference-hardware "$PY" bench_inference.py \
  --d-model "${INFER_D_MODEL:-512}" \
  --n-layers "${INFER_LAYERS:-6}" \
  --seq-len "${INFER_SEQ_LEN:-512}" \
  --decode-tokens "${INFER_DECODE_TOKENS:-128}" \
  --batches ${INFER_BATCHES:-1 8 32}

run trace-hardware "$PY" trace.py \
  --strategy "${TRACE_STRATEGY:-fsdp}" \
  --steps "${TRACE_STEPS:-20}" \
  --batch "${TRACE_BATCH:-64}"

run ledger "$PY" -c "from pathlib import Path; from ledger import Ledger; [Ledger(p.stem).scoreboard() for p in sorted(Path('receipts').glob('*.json'))]"

tar -czf "$TARBALL" receipts
echo "log -> $LOG"
echo "receipts -> $TARBALL"
