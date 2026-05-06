#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-supplementary_anonymous/configs/llm_baselines.yaml}"
SETTING="${SETTING:-llm_rag}"
TEST_JSONL="${TEST_JSONL:-data/graphs/test.jsonl}"
OUT_JSONL="${OUT_JSONL:-results/${SETTING}_predictions.jsonl}"

mkdir -p "$(dirname "${OUT_JSONL}")"

python -m repocontraaudit.llm.run_baseline \
  --config "${CONFIG}" \
  --setting "${SETTING}" \
  --data "${TEST_JSONL}" \
  --out "${OUT_JSONL}" \
  --deterministic

python supplementary_anonymous/metrics/compute_detection.py \
  --input "${OUT_JSONL}" \
  --label-field gold_label \
  --prediction-field label \
  --score-field confidence

