#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-supplementary_anonymous/configs/train_default.yaml}"
TRAIN_JSONL="${TRAIN_JSONL:-data/graphs/train.jsonl}"
VALID_JSONL="${VALID_JSONL:-data/graphs/valid.jsonl}"
OUT_DIR="${OUT_DIR:-runs/repocontraaudit}"

python -m repocontraaudit.cli.train \
  --train-data "${TRAIN_JSONL}" \
  --valid-data "${VALID_JSONL}" \
  --out "${OUT_DIR}" \
  --epochs 30 \
  --hidden-dim 256 \
  --graph-layers 3 \
  --chain-budget 6 \
  --beam-width 6

echo "Training template used config ${CONFIG}."

