#!/usr/bin/env bash
set -euo pipefail

LOCALIZATION_JSONL="${LOCALIZATION_JSONL:-results/repocontraaudit_localization.jsonl}"

python supplementary_anonymous/metrics/compute_localization.py \
  --input "${LOCALIZATION_JSONL}" \
  --gold-field rationales \
  --score-field statement_marginals

