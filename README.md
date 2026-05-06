# RepoContraAudit Open Reference

This repository is a compact open-source reference implementation of
**RepoContraAudit: Explainable Repository-Level Code Auditing via Multimodal
Anti-Fact Reasoning**.

It is intentionally not the full paper artifact. The goal is to expose the core
methodology in readable, runnable code:

- typed multimodal repository evidence graphs;
- hierarchical typed message passing across code, graph, alert, trace, and text evidence;
- reliability-aware cross-modal fusion with observation masks;
- partially supervised latent evidence-chain scoring and marginal localization;
- anti-fact evidence perturbation search with sparse, connected, grounded projections.

The reference version uses a lightweight hashing text encoder so it can run on a
normal laptop. The paper-scale setup can replace that encoder with a pretrained
code Transformer and plug in production static-analysis, trace, and dataset
reconstruction pipelines.

## Quick Start

```bash
python -m pip install -e .
python -m repocontraaudit.cli.train --train-data examples/toy_audit_units.jsonl --epochs 5 --out runs/toy
python -m repocontraaudit.cli.infer --checkpoint runs/toy/model.pt --data examples/toy_audit_units.jsonl --anti-fact
```

You can also build an evidence graph from a small source tree:

```bash
python -m repocontraaudit.cli.build_graph --repo examples/sample_repo --label 0 --out runs/sample_graph.jsonl
```

## What This Implements

The code maps directly to the method sections:

- `repocontraaudit.data`: JSONL audit-unit schema and typed graph objects.
- `repocontraaudit.builders`: lightweight repository-to-evidence-graph builder.
- `repocontraaudit.model`: statement encoder, typed graph encoder, hierarchical pooling, and cross-modal fusion.
- `repocontraaudit.chains`: evidence-chain scoring, beam extraction, and latent chain loss.
- `repocontraaudit.antifact`: differentiable mask proposal, projection, and plausibility checks.
- `repocontraaudit.training`: partial-supervision training loop.

See [docs/method_mapping.md](docs/method_mapping.md) and
[docs/data_format.md](docs/data_format.md) for details.

## Scope

This open version does **not** include private preprocessing manifests, benchmark
splits, LLM baseline infrastructure, auditor-study materials, or the full
large-scale training recipe. Anti-fact outputs are evidence-level
decision-sensitivity tests with lightweight plausibility checks. They are not
semantic repairs and do not prove vulnerability removal.

## Minimal Data Format

Each JSONL row is one repository-level audit unit:

```json
{
  "repo_id": "toy/reentrancy",
  "label": 1,
  "nodes": [
    {"id": "s1", "type": "statement", "modality": "code", "text": "balances[msg.sender] -= amount;", "file": "Vault.sol", "line": 8},
    {"id": "a1", "type": "static_alert", "modality": "alert", "text": "external call before state update"}
  ],
  "edges": [
    {"src": "a1", "dst": "s1", "type": "alert_to_code"}
  ],
  "rationales": ["s1"],
  "positive_chains": [["entry", "call", "a1", "s1"]]
}
```

## License

MIT.

