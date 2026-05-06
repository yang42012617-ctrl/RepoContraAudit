#!/usr/bin/env python3
"""Compute anti-fact decision-sensitivity and plausibility metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--field", default="anti_fact")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if args.field in row and row[args.field]]
    totals = {
        "flip_rate": 0.0,
        "confidence_drop": 0.0,
        "intervention_size": 0.0,
        "connectivity": 0.0,
        "overall_plausibility": 0.0,
        "projection_failure_rate": 0.0
    }
    for row in rows:
        af = row[args.field]
        intervention = af.get("intervention", {})
        checks = intervention.get("checks", {})
        selected_nodes = intervention.get("selected_nodes", [])
        totals["flip_rate"] += 1.0 if af.get("flipped") else 0.0
        totals["confidence_drop"] += float(af.get("confidence_drop", 0.0))
        totals["intervention_size"] += len(selected_nodes) + len(intervention.get("selected_edges", []))
        totals["connectivity"] += 1.0 if checks.get("connected") else 0.0
        check_values = [1.0 if value else 0.0 for value in checks.values()]
        totals["overall_plausibility"] += sum(check_values) / max(1, len(check_values))
        totals["projection_failure_rate"] += 1.0 if intervention.get("projection_failed") else 0.0

    metrics = {key: value / max(1, len(rows)) for key, value in totals.items()}
    metrics["n"] = len(rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

