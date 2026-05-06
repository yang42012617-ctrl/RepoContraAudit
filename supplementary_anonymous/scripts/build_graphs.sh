#!/usr/bin/env bash
set -euo pipefail

PRE_FIX_ROOT="${PRE_FIX_ROOT:-data/work/pre_fix}"
ALERT_ROOT="${ALERT_ROOT:-data/work/alerts}"
TRACE_ROOT="${TRACE_ROOT:-data/work/traces}"
OUT_ROOT="${OUT_ROOT:-data/graphs}"

mkdir -p "${OUT_ROOT}"

python -m repocontraaudit.cli.build_graphs \
  --input-root "${PRE_FIX_ROOT}" \
  --alerts-root "${ALERT_ROOT}" \
  --traces-root "${TRACE_ROOT}" \
  --out "${OUT_ROOT}/all.jsonl" \
  --schema typed_multimodal_evidence_graph \
  --missing-modalities-as-observation-mask

echo "Graph construction template completed."

