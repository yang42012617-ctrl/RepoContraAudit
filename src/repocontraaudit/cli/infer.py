"""Run repository-level inference and evidence extraction."""

from __future__ import annotations

import argparse
import json

import torch

from repocontraaudit.antifact import AntiFactSearcher
from repocontraaudit.chains import beam_search_chains, statement_marginals_from_chains
from repocontraaudit.data import load_audit_units
from repocontraaudit.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer RepoContraAudit predictions and explanations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--anti-fact", action="store_true", help="Run anti-fact perturbation search.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence chains to print.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, args.device)
    units = load_audit_units(args.data)
    searcher = AntiFactSearcher()

    for unit in units:
        with torch.no_grad():
            output = model(unit)
            probs = torch.softmax(output.logits, dim=-1)
            chains = beam_search_chains(model, unit, output)
            marginals = statement_marginals_from_chains(unit, chains)
        record = {
            "repo_id": unit.repo_id,
            "label": int(probs.argmax().item()),
            "confidence": float(probs.max().item()),
            "vulnerable_probability": float(probs[1].item()) if probs.numel() > 1 else None,
            "evidence_chains": [chain.to_dict() for chain in chains[: args.top_k]],
            "statement_marginals": marginals,
        }
        if args.anti_fact:
            result = searcher.search(model, unit, chains[0] if chains else None)
            record["anti_fact"] = result.to_dict()
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

