#!/usr/bin/env python3
"""Paired bootstrap significance testing for JSONL prediction files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_label(value) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"vulnerable", "vul", "1", "true", "yes"}:
            return 1
        if text in {"benign", "safe", "0", "false", "no"}:
            return 0
    return int(value)


def metric(labels: list[int], preds: list[int], name: str) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    if name == "accuracy":
        return (tp + tn) / max(1, len(labels))
    if name == "precision":
        return precision
    if name == "recall":
        return recall
    if name == "f1":
        return f1
    raise ValueError(f"unsupported metric: {name}")


def align_rows(
    baseline_rows: list[dict],
    model_rows: list[dict],
    id_field: str,
) -> tuple[list[dict], list[dict]]:
    baseline_by_id = {str(row.get(id_field, idx)): row for idx, row in enumerate(baseline_rows)}
    model_by_id = {str(row.get(id_field, idx)): row for idx, row in enumerate(model_rows)}
    common = sorted(set(baseline_by_id) & set(model_by_id))
    return [baseline_by_id[key] for key in common], [model_by_id[key] for key in common]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", default="f1", choices=["accuracy", "precision", "recall", "f1"])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--id-field", default="repo_id")
    parser.add_argument("--label-field", default="gold_label")
    parser.add_argument("--prediction-field", default="label")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    baseline_rows, model_rows = align_rows(read_jsonl(args.baseline), read_jsonl(args.model), args.id_field)
    labels = [as_label(row[args.label_field]) for row in model_rows]
    baseline_preds = [as_label(row[args.prediction_field]) for row in baseline_rows]
    model_preds = [as_label(row[args.prediction_field]) for row in model_rows]
    observed = metric(labels, model_preds, args.metric) - metric(labels, baseline_preds, args.metric)

    diffs = []
    n = len(labels)
    for _ in range(args.iterations):
        sample = [rng.randrange(n) for _item in range(n)]
        sample_labels = [labels[idx] for idx in sample]
        sample_baseline = [baseline_preds[idx] for idx in sample]
        sample_model = [model_preds[idx] for idx in sample]
        diffs.append(
            metric(sample_labels, sample_model, args.metric)
            - metric(sample_labels, sample_baseline, args.metric)
        )
    diffs.sort()
    lower = diffs[int(0.025 * len(diffs))]
    upper = diffs[int(0.975 * len(diffs))]
    p_value = sum(1 for diff in diffs if diff <= 0.0) / max(1, len(diffs))
    print(json.dumps({
        "n": n,
        "metric": args.metric,
        "observed_delta": observed,
        "ci95": [lower, upper],
        "one_sided_p_model_not_better": p_value,
        "iterations": args.iterations,
        "seed": args.seed
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

