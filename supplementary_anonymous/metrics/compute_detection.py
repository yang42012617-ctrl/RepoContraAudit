#!/usr/bin/env python3
"""Compute detection metrics from JSONL predictions."""

from __future__ import annotations

import argparse
import json
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


def binary_metrics(labels: list[int], preds: list[int], scores: list[float] | None = None) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {
        "n": len(labels),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn)
    }
    if scores is not None:
        out["auroc"] = auroc(labels, scores)
    return out


def auroc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j + 1) / 2.0
        rank_sum += avg_rank * sum(label for _score, label in pairs[idx:j + 1])
        idx = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--label-field", default="gold_label")
    parser.add_argument("--prediction-field", default="label")
    parser.add_argument("--score-field", default="vulnerable_probability")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    labels = [as_label(row[args.label_field]) for row in rows if args.label_field in row]
    preds = [as_label(row[args.prediction_field]) for row in rows if args.label_field in row]
    scores = [
        float(row[args.score_field])
        for row in rows
        if args.label_field in row and args.score_field in row and row[args.score_field] is not None
    ]
    score_list = scores if len(scores) == len(labels) else None
    print(json.dumps(binary_metrics(labels, preds, score_list), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

