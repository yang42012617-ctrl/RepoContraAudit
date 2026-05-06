"""Lightweight repository-to-evidence-graph construction.

The paper-scale implementation can plug in precise parsers, static analyzers,
and trace collectors. This reference builder deliberately uses conservative
regex heuristics so the whole method can be run without external tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable

from repocontraaudit.data import AuditUnit, EvidenceEdge, EvidenceNode
from repocontraaudit.text import compact_text

DEFAULT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".java",
    ".rs",
    ".sol",
}

FUNC_PATTERNS = [
    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*(?:public|private|protected|static|inline|virtual|external|internal|\w+\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?\s*$"),
]
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
COMMENT_RE = re.compile(r"^\s*(//|#|\*)\s?(.*)")
IMPORT_RE = re.compile(r"^\s*(?:import|from|#include)\b(.*)")
KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "require",
    "assert",
    "emit",
    "sizeof",
    "function",
}


@dataclass
class SourceRepositoryBuilder:
    """Construct a typed evidence graph from a source tree."""

    suffixes: set[str] = field(default_factory=lambda: set(DEFAULT_SUFFIXES))
    max_files: int = 256
    max_file_bytes: int = 512_000

    def build(
        self,
        repo_path: str | Path,
        label: int = 0,
        repo_id: str | None = None,
        alerts: Iterable[dict] | None = None,
        traces: Iterable[dict] | None = None,
    ) -> AuditUnit:
        repo_path = Path(repo_path)
        repo_id = repo_id or repo_path.name
        nodes: list[EvidenceNode] = [
            EvidenceNode(
                id="repo",
                type="repository",
                modality="graph",
                text=f"repository {repo_id}",
                file=str(repo_path),
            )
        ]
        edges: list[EvidenceEdge] = []

        statement_by_location: dict[tuple[str, int], str] = {}
        function_by_name: dict[str, str] = {}
        source_files = list(self._iter_source_files(repo_path))

        for file_index, source_file in enumerate(source_files):
            rel = source_file.relative_to(repo_path).as_posix()
            file_id = f"file:{rel}"
            nodes.append(EvidenceNode(id=file_id, type="file", modality="graph", text=rel, file=rel))
            edges.append(EvidenceEdge("repo", file_id, "repository_contains"))

            current_function_id: str | None = None
            previous_statement_id: str | None = None
            lines = source_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_number, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                import_match = IMPORT_RE.match(line)
                if import_match:
                    import_id = f"import:{rel}:{line_number}"
                    nodes.append(
                        EvidenceNode(
                            id=import_id,
                            type="api",
                            modality="graph",
                            text=compact_text("import", import_match.group(1)),
                            file=rel,
                            line=line_number,
                        )
                    )
                    edges.append(EvidenceEdge(file_id, import_id, "import"))

                function_name = _match_function_name(line)
                if function_name:
                    current_function_id = f"fn:{rel}:{function_name}:{line_number}"
                    function_by_name.setdefault(function_name, current_function_id)
                    nodes.append(
                        EvidenceNode(
                            id=current_function_id,
                            type="function",
                            modality="graph",
                            text=function_name,
                            file=rel,
                            line=line_number,
                        )
                    )
                    edges.append(EvidenceEdge(file_id, current_function_id, "file_contains"))
                    previous_statement_id = None

                comment_match = COMMENT_RE.match(line)
                if comment_match:
                    comment_id = f"text:{rel}:{line_number}"
                    nodes.append(
                        EvidenceNode(
                            id=comment_id,
                            type="text_segment",
                            modality="text",
                            text=comment_match.group(2),
                            file=rel,
                            line=line_number,
                        )
                    )
                    edges.append(EvidenceEdge(comment_id, current_function_id or file_id, "text_to_code"))
                    continue

                statement_id = f"stmt:{rel}:{line_number}"
                statement_by_location[(rel, line_number)] = statement_id
                nodes.append(
                    EvidenceNode(
                        id=statement_id,
                        type="statement",
                        modality="code",
                        text=stripped,
                        file=rel,
                        line=line_number,
                    )
                )
                edges.append(
                    EvidenceEdge(current_function_id or file_id, statement_id, "function_contains")
                )
                if previous_statement_id:
                    edges.append(EvidenceEdge(previous_statement_id, statement_id, "control_flow"))
                previous_statement_id = statement_id

                for call_name in _iter_call_names(stripped):
                    api_id = f"api:{call_name}"
                    if not any(node.id == api_id for node in nodes):
                        nodes.append(EvidenceNode(id=api_id, type="api", modality="graph", text=call_name))
                    edge_type = "call" if call_name in function_by_name else "related"
                    edges.append(EvidenceEdge(statement_id, api_id, edge_type))

            if file_index + 1 >= self.max_files:
                break

        self._attach_alerts(nodes, edges, statement_by_location, alerts or [])
        self._attach_traces(nodes, edges, statement_by_location, traces or [])
        return AuditUnit(repo_id=repo_id, label=label, nodes=nodes, edges=edges)

    def _iter_source_files(self, repo_path: Path) -> Iterable[Path]:
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            if any(part.startswith(".") for part in path.relative_to(repo_path).parts):
                continue
            if path.stat().st_size > self.max_file_bytes:
                continue
            yield path

    def _attach_alerts(
        self,
        nodes: list[EvidenceNode],
        edges: list[EvidenceEdge],
        statement_by_location: dict[tuple[str, int], str],
        alerts: Iterable[dict],
    ) -> None:
        for idx, alert in enumerate(alerts):
            file_name = str(alert.get("file") or alert.get("path") or "")
            line = int(alert.get("line") or alert.get("start_line") or 0)
            alert_id = str(alert.get("id") or f"alert:{idx}")
            nodes.append(
                EvidenceNode(
                    id=alert_id,
                    type="static_alert",
                    modality="alert",
                    text=compact_text(
                        alert.get("tool"),
                        alert.get("rule_id"),
                        alert.get("severity"),
                        alert.get("message") or alert.get("message_template"),
                    ),
                    file=file_name or None,
                    line=line or None,
                    confidence=float(alert.get("confidence", 1.0)),
                    attrs=dict(alert),
                )
            )
            target = statement_by_location.get((file_name, line))
            if target:
                edges.append(EvidenceEdge(alert_id, target, "alert_to_code"))

    def _attach_traces(
        self,
        nodes: list[EvidenceNode],
        edges: list[EvidenceEdge],
        statement_by_location: dict[tuple[str, int], str],
        traces: Iterable[dict],
    ) -> None:
        previous_trace_id: str | None = None
        for idx, event in enumerate(traces):
            file_name = str(event.get("file") or event.get("path") or "")
            line = int(event.get("line") or 0)
            trace_id = str(event.get("id") or f"trace:{idx}")
            nodes.append(
                EvidenceNode(
                    id=trace_id,
                    type="trace_event",
                    modality="trace",
                    text=compact_text(event.get("event_type"), event.get("function"), event.get("message")),
                    file=file_name or None,
                    line=line or None,
                    confidence=float(event.get("confidence", 1.0)),
                    attrs=dict(event),
                )
            )
            target = statement_by_location.get((file_name, line))
            if target:
                edges.append(EvidenceEdge(trace_id, target, "trace_to_code"))
            if previous_trace_id:
                edges.append(EvidenceEdge(previous_trace_id, trace_id, "control_flow"))
            previous_trace_id = trace_id


def load_sidecar_json(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return raw.get("items") or raw.get("alerts") or raw.get("traces") or []


def _match_function_name(line: str) -> str | None:
    for pattern in FUNC_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _iter_call_names(line: str) -> Iterable[str]:
    for match in CALL_RE.finditer(line):
        name = match.group(1)
        if name not in KEYWORDS:
            yield name

