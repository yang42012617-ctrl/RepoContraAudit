#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-runs/repocontraaudit/model.pt}"
TEST_JSONL="${TEST_JSONL:-data/graphs/test.jsonl}"
PRED_JSONL="${PRED_JSONL:-results/repocontraaudit_predictions.jsonl}"

mkdir -p "$(dirname "${PRED_JSONL}")"

python -m repocontraaudit.cli.infer \
  --checkpoint "${CHECKPOINT}" \
  --data "${TEST_JSONL}" \
  > "${PRED_JSONL}"

python supplementary_anonymous/metrics/compute_detection.py \
  --input "${PRED_JSONL}" \
  --label-field gold_label \
  --prediction-field label \
  --score-field vulnerable_probability

