"""Build JSONL evidence graphs from source repositories."""

from __future__ import annotations

import argparse

from repocontraaudit.builders import SourceRepositoryBuilder, load_sidecar_json
from repocontraaudit.data import write_audit_units


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a typed evidence graph from a source tree.")
    parser.add_argument("--repo", required=True, help="Path to the source repository.")
    parser.add_argument("--label", type=int, default=0, help="Repository-level vulnerability label.")
    parser.add_argument("--repo-id", default=None, help="Optional audit-unit id.")
    parser.add_argument("--alerts", default=None, help="Optional normalized alerts JSON.")
    parser.add_argument("--traces", default=None, help="Optional normalized traces JSON.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    builder = SourceRepositoryBuilder()
    unit = builder.build(
        args.repo,
        label=args.label,
        repo_id=args.repo_id,
        alerts=load_sidecar_json(args.alerts),
        traces=load_sidecar_json(args.traces),
    )
    write_audit_units([unit], args.out)
    print(f"wrote 1 audit unit with {len(unit.nodes)} nodes and {len(unit.edges)} edges to {args.out}")


if __name__ == "__main__":
    main()

