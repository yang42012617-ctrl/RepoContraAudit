#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-runs/repocontraaudit/model.pt}"
TEST_JSONL="${TEST_JSONL:-data/graphs/test.jsonl}"
PROFILE_OUT="${PROFILE_OUT:-results/runtime_profile.jsonl}"

mkdir -p "$(dirname "${PROFILE_OUT}")"

python -m repocontraaudit.profiling.profile_runtime \
  --checkpoint "${CHECKPOINT}" \
  --data "${TEST_JSONL}" \
  --profile-graph-construction \
  --profile-inference \
  --profile-antifact \
  --out "${PROFILE_OUT}"

echo "Runtime profiling template completed."

