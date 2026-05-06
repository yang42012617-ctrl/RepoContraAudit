# Method Mapping

This repository is a reference implementation of the paper method, not a full reproduction artifact.

## Typed Multimodal Evidence Graph

- Paper: Sections 3.1 and 4.2.
- Code: `repocontraaudit.data`, `repocontraaudit.builders`.
- Open version: JSONL audit units and a lightweight regex source builder. Production users can
  replace the builder with precise parsers, CodeQL/Semgrep/Slither outputs, dynamic traces, and
  curated text sources.

## Hierarchical Cross-File Encoder

- Paper: Section 4.3.
- Code: `repocontraaudit.model.TypedGraphLayer`, `RepoContraAuditModel`.
- Open version: deterministic hash text embeddings, node-type embeddings, modality embeddings,
  three typed message-passing layers, statement localization head, and attention repository pooling.
  The paper-scale 12-layer code Transformer can replace `HashingTextEncoder`.

## Reliability-Aware Cross-Modal Fusion

- Paper: Section 4.4.
- Code: `repocontraaudit.model.ReliabilityAwareFusion`.
- Open version: per-modality neighbor evidence aggregation, observation masks, modality dropout,
  and soft gates over code, graph, alert, trace, and text evidence.

## Latent Evidence Chains

- Paper: Sections 3.2, 4.5, and 4.6.
- Code: `repocontraaudit.chains`.
- Open version: chain scores combine selected node scores and induced typed-edge scores. The loss
  marginalizes over available positive weak chains and contrasts corrupted chains. Beam search
  extracts connected evidence structures under a size budget.

## Anti-Fact Evidence Perturbation

- Paper: Sections 3.3, 4.7, 4.8, and Appendix C.
- Code: `repocontraaudit.antifact`.
- Open version: continuous masks over chain nodes plus one-hop neighbors are optimized to reduce
  vulnerable confidence while remaining sparse and connected. The thresholded mask is projected to
  grounded evidence units and checked for simple plausibility properties.

Anti-fact outputs are evidence-level decision-sensitivity explanations. They are not executable
patches, semantic repairs, or formal vulnerability-removal proofs.

## Deliberately Omitted from the Open Reference

- Private benchmark reconstruction manifests and leakage-filtered splits.
- Full static analyzer, tracer, and LLM baseline infrastructure.
- Human auditor-study materials.
- A100-scale training recipe and pretrained code Transformer checkpoints.
- Claims of semantic repair correctness.

