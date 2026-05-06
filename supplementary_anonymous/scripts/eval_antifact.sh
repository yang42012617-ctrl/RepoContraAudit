#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-runs/repocontraaudit/model.pt}"
TEST_JSONL="${TEST_JSONL:-data/graphs/test.jsonl}"
AF_JSONL="${AF_JSONL:-results/repocontraaudit_antifact.jsonl}"

mkdir -p "$(dirname "${AF_JSONL}")"

python -m repocontraaudit.cli.infer \
  --checkpoint "${CHECKPOINT}" \
  --data "${TEST_JSONL}" \
  --anti-fact \
  > "${AF_JSONL}"

python supplementary_anonymous/metrics/compute_antifact_metrics.py \
  --input "${AF_JSONL}"

