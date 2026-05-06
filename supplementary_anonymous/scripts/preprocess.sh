#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-data/raw}"
WORK_ROOT="${WORK_ROOT:-data/work}"
MANIFEST_DIR="${MANIFEST_DIR:-supplementary_anonymous/splits}"

mkdir -p "${WORK_ROOT}/pre_fix" "${WORK_ROOT}/metadata"

python -m repocontraaudit.cli.preprocess \
  --raw-root "${RAW_ROOT}" \
  --out-root "${WORK_ROOT}/pre_fix" \
  --manifest-dir "${MANIFEST_DIR}" \
  --exclude-post-fix-code \
  --exclude-patch-hunks \
  --exclude-fix-commit-text \
  --filter-label-revealing-text

echo "Preprocessing template completed. Replace the module above with the dataset-specific loader when using private benchmark checkouts."

