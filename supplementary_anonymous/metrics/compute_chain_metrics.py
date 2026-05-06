#!/usr/bin/env python3
"""Compute weak evidence-chain and induced-edge agreement metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chain_nodes(chain) -> list[str]:
    if isinstance(chain, dict):
        return [str(item) for item in chain.get("nodes", [])]
    return [str(item) for item in chain]


def edge_set(chain: list[str]) -> set[tuple[str, str]]:
    return set(zip(chain, chain[1:]))


def prf(gold: set, pred: set) -> tuple[float, float, float]:
    tp = len(gold & pred)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold-field", default="positive_chains")
    parser.add_argument("--prediction-field", default="evidence_chains")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    totals = {"chain_precision": 0.0, "chain_recall": 0.0, "chain_f1": 0.0, "edge_f1": 0.0}
    count = 0
    for row in rows:
        if args.gold_field not in row or args.prediction_field not in row:
            continue
        gold_nodes = set()
        pred_nodes = set()
        gold_edges = set()
        pred_edges = set()
        for chain in row[args.gold_field]:
            nodes = chain_nodes(chain)
            gold_nodes.update(nodes)
            gold_edges.update(edge_set(nodes))
        for chain in row[args.prediction_field]:
            nodes = chain_nodes(chain)
            pred_nodes.update(nodes)
            pred_edges.update(edge_set(nodes))
        precision, recall, f1 = prf(gold_nodes, pred_nodes)
        _edge_precision, _edge_recall, edge_f1 = prf(gold_edges, pred_edges)
        totals["chain_precision"] += precision
        totals["chain_recall"] += recall
        totals["chain_f1"] += f1
        totals["edge_f1"] += edge_f1
        count += 1

    metrics = {key: value / max(1, count) for key, value in totals.items()}
    metrics["n"] = count
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

