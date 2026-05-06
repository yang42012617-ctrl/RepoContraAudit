#!/usr/bin/env bash
set -euo pipefail

BASELINE_JSONL="${BASELINE_JSONL:-results/baseline_predictions.jsonl}"
MODEL_JSONL="${MODEL_JSONL:-results/repocontraaudit_predictions.jsonl}"
METRIC="${METRIC:-f1}"

python supplementary_anonymous/significance/paired_bootstrap.py \
  --baseline "${BASELINE_JSONL}" \
  --model "${MODEL_JSONL}" \
  --metric "${METRIC}" \
  --iterations 10000 \
  --seed 13

