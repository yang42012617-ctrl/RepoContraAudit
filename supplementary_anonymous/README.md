# RepoContraAudit Anonymous Supplementary Package

This anonymous supplementary package contains split manifests, prompt templates,
configuration files, metric scripts, significance-test scripts, profiling
templates, and command templates used to reproduce the evaluation protocol
described in the paper.

The package is anonymized for double-blind review. It contains no author names,
institution names, personal paths, private repository URLs, or non-anonymous
metadata.

The implementation and scripts are also mirrored in the anonymous review
repository:
https://anonymous.4open.science/r/RepoContraAudit-9EB3/

## Package Contents

- `configs/`: default training, evaluation, and LLM-baseline configurations.
- `splits/`: leakage-controlled split manifest templates for the four benchmark
  families used in the paper.
- `prompts/`: deterministic JSON-output prompt templates for repository-level
  LLM baselines.
- `scripts/`: command templates for preprocessing, graph construction, training,
  evaluation, LLM baselines, significance testing, and profiling.
- `metrics/`: standalone JSONL metric scripts for detection, localization,
  evidence-chain agreement, and anti-fact evaluation.
- `significance/`: paired-bootstrap significance testing.
- `profiling/`: runtime profiling command template.

## Expected Workflow

1. Prepare leakage-filtered repository audit units with pre-fix artifacts only.
2. Build typed multimodal evidence graphs.
3. Train RepoContraAudit with partially observed supervision.
4. Evaluate detection, localization, evidence-chain agreement, and anti-fact
   decision sensitivity.
5. Run paired-bootstrap significance tests against selected baselines.
6. Profile graph construction, inference, and anti-fact generation.

The command templates are intentionally parameterized. They do not contain local
absolute paths, account names, private URLs, or private dataset locations.

## Data and Leakage Policy

The manifests record the split policy and aggregate counts needed to reproduce
the protocol. They do not include private checkout paths or post-fix code. Model
inputs must contain only pre-fix source artifacts and permitted pre-fix auxiliary
evidence. Patch-derived information may be used only as offline supervision
targets after removing post-fix code, patch hunks, and vulnerability-revealing
commit text from model inputs.

## Anti-Fact Scope

Anti-fact perturbations are evidence-level decision-sensitivity tests with
lightweight plausibility checks. They are not executable repairs, formal proofs
of vulnerability removal, or semantic equivalence guarantees.

