#!/usr/bin/env python3
"""Compute statement, function, and file localization metrics from JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def precision_recall_f1(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    tp = len(gold & pred)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def ranked_keys(score_map: dict[str, float]) -> list[str]:
    return [key for key, _value in sorted(score_map.items(), key=lambda item: item[1], reverse=True)]


def prefix_projection(statement_ids: list[str], level: str) -> list[str]:
    projected = []
    for item in statement_ids:
        if level == "file":
            projected.append(item.split(":", 2)[1] if item.startswith("stmt:") and ":" in item else item)
        elif level == "function":
            parts = item.split(":")
            projected.append(":".join(parts[:3]) if len(parts) >= 3 else item)
        else:
            projected.append(item)
    return projected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold-field", default="rationales")
    parser.add_argument("--score-field", default="statement_marginals")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    totals = {
        "statement_precision": 0.0,
        "statement_recall": 0.0,
        "statement_f1": 0.0,
        "statement_recall_at_5": 0.0,
        "function_recall_at_3": 0.0,
        "file_recall_at_3": 0.0
    }
    count = 0
    for row in rows:
        if args.gold_field not in row or args.score_field not in row:
            continue
        gold = set(row[args.gold_field])
        scores = {str(key): float(value) for key, value in row[args.score_field].items()}
        ranked = ranked_keys(scores)
        pred = {key for key, value in scores.items() if value >= args.threshold}
        if not pred:
            pred = set(ranked[:5])
        p, r, f1 = precision_recall_f1(gold, pred)
        totals["statement_precision"] += p
        totals["statement_recall"] += r
        totals["statement_f1"] += f1
        totals["statement_recall_at_5"] += recall_at_k(gold, ranked, 5)
        totals["function_recall_at_3"] += recall_at_k(
            set(prefix_projection(list(gold), "function")),
            prefix_projection(ranked, "function"),
            3
        )
        totals["file_recall_at_3"] += recall_at_k(
            set(prefix_projection(list(gold), "file")),
            prefix_projection(ranked, "file"),
            3
        )
        count += 1

    metrics = {key: value / max(1, count) for key, value in totals.items()}
    metrics["n"] = count
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

