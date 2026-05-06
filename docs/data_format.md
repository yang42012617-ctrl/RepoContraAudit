# Data Format

RepoContraAudit consumes repository-level audit units in JSONL or JSON. Each row is a typed
multimodal evidence graph plus optional partial supervision.

## Required Fields

- `repo_id`: stable repository or audit-unit id.
- `label`: `1` for vulnerable, `0` for benign.
- `nodes`: evidence units.
- `edges`: typed relations between evidence units.

## Node Fields

- `id`: unique within the audit unit.
- `type`: one of `statement`, `function`, `file`, `variable`, `api`, `external_call`,
  `static_alert`, `trace_event`, `text_segment`, `repository`, or `other`.
- `modality`: one of `code`, `graph`, `alert`, `trace`, or `text`.
- `text`: code, alert message, trace event description, or textual context.
- `file`, `line`, `end_line`: optional grounding location.
- `observed`: whether this evidence was observed.
- `confidence`: extractor or alignment confidence.
- `attrs`: arbitrary extractor metadata.

## Edge Fields

- `src`, `dst`: node ids.
- `type`: one of `ast_containment`, `control_flow`, `data_flow`, `call`, `import`,
  `inheritance`, `alert_to_code`, `trace_to_code`, `text_to_code`,
  `repository_contains`, `file_contains`, `function_contains`, `next_statement`,
  `related`, or `other`.
- `confidence`: optional relation confidence.

## Optional Supervision

- `rationales`: statement node ids with observed localization labels.
- `positive_chains`: weakly observed positive evidence chains. Multiple chains are allowed.
- `patch_nodes`: pre-fix evidence nodes aligned to human changes. Do not include post-fix code.
- `observation_mask`: modality availability flags.

Missing rationales, chains, patches, traces, alerts, and text are normal. The training objective
applies each loss only when the corresponding supervision is available.

