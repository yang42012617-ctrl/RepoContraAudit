"""Data structures and JSONL IO for repository evidence graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from repocontraaudit.schema import (
    EDGE_TYPES,
    MODALITIES,
    NODE_TYPES,
    canonical_edge_type,
    canonical_modality,
    canonical_node_type,
)


@dataclass(slots=True)
class EvidenceNode:
    """One typed evidence unit in a repository graph."""

    id: str
    type: str
    modality: str
    text: str = ""
    file: str | None = None
    line: int | None = None
    end_line: int | None = None
    observed: bool = True
    confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidenceNode":
        node_type = canonical_node_type(raw.get("type"))
        modality = canonical_modality(raw.get("modality"), node_type)
        attrs = dict(raw.get("attrs") or raw.get("attributes") or {})
        return cls(
            id=str(raw["id"]),
            type=node_type,
            modality=modality,
            text=str(raw.get("text") or raw.get("message") or ""),
            file=raw.get("file") or raw.get("path"),
            line=_optional_int(raw.get("line")),
            end_line=_optional_int(raw.get("end_line") or raw.get("line_end")),
            observed=bool(raw.get("observed", True)),
            confidence=float(raw.get("confidence", 1.0)),
            attrs=attrs,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "modality": self.modality,
            "text": self.text,
            "observed": self.observed,
            "confidence": self.confidence,
        }
        if self.file is not None:
            data["file"] = self.file
        if self.line is not None:
            data["line"] = self.line
        if self.end_line is not None:
            data["end_line"] = self.end_line
        if self.attrs:
            data["attrs"] = self.attrs
        return data

    @property
    def grounded(self) -> bool:
        return self.file is not None or self.type in {"static_alert", "trace_event", "text_segment"}

    @property
    def source_label(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        if self.file:
            return self.file
        return self.id


@dataclass(slots=True)
class EvidenceEdge:
    """One typed relation between evidence units."""

    src: str
    dst: str
    type: str
    confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidenceEdge":
        return cls(
            src=str(raw.get("src") or raw.get("source")),
            dst=str(raw.get("dst") or raw.get("target")),
            type=canonical_edge_type(raw.get("type") or raw.get("relation")),
            confidence=float(raw.get("confidence", 1.0)),
            attrs=dict(raw.get("attrs") or raw.get("attributes") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "confidence": self.confidence,
        }
        if self.attrs:
            data["attrs"] = self.attrs
        return data


@dataclass
class AuditUnit:
    """A repository-level training or inference example."""

    repo_id: str
    label: int
    nodes: list[EvidenceNode]
    edges: list[EvidenceEdge]
    rationales: list[str] = field(default_factory=list)
    positive_chains: list[list[str]] = field(default_factory=list)
    patch_nodes: list[str] = field(default_factory=list)
    observation_mask: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = int(self.label)
        self.node_index = {node.id: idx for idx, node in enumerate(self.nodes)}
        self.nodes_by_id = {node.id: node for node in self.nodes}
        self.edges = [edge for edge in self.edges if edge.src in self.node_index and edge.dst in self.node_index]
        if not self.observation_mask:
            self.observation_mask = {
                modality: any(node.modality == modality and node.observed for node in self.nodes)
                for modality in MODALITIES
            }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AuditUnit":
        raw_chains = raw.get("positive_chains") or raw.get("chains") or []
        positive_chains = [_normalize_chain(chain) for chain in raw_chains]
        return cls(
            repo_id=str(raw.get("repo_id") or raw.get("id") or raw.get("name")),
            label=int(raw.get("label", 0)),
            nodes=[EvidenceNode.from_dict(item) for item in raw.get("nodes", [])],
            edges=[EvidenceEdge.from_dict(item) for item in raw.get("edges", [])],
            rationales=[str(item) for item in raw.get("rationales", [])],
            positive_chains=positive_chains,
            patch_nodes=[str(item) for item in raw.get("patch_nodes", [])],
            observation_mask={str(k): bool(v) for k, v in (raw.get("observation_mask") or {}).items()},
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "repo_id": self.repo_id,
            "label": self.label,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if self.rationales:
            data["rationales"] = self.rationales
        if self.positive_chains:
            data["positive_chains"] = self.positive_chains
        if self.patch_nodes:
            data["patch_nodes"] = self.patch_nodes
        if self.observation_mask:
            data["observation_mask"] = self.observation_mask
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @property
    def statement_ids(self) -> list[str]:
        return [node.id for node in self.nodes if node.type == "statement"]

    @property
    def statement_indices(self) -> list[int]:
        return [self.node_index[node_id] for node_id in self.statement_ids]

    def neighbors(self, node_id: str, undirected: bool = True) -> set[str]:
        out: set[str] = set()
        for edge in self.edges:
            if edge.src == node_id:
                out.add(edge.dst)
            if undirected and edge.dst == node_id:
                out.add(edge.src)
        return out

    def edge_lookup(self) -> dict[tuple[str, str], list[EvidenceEdge]]:
        lookup: dict[tuple[str, str], list[EvidenceEdge]] = {}
        for edge in self.edges:
            lookup.setdefault((edge.src, edge.dst), []).append(edge)
        return lookup

    def type_id(self, node: EvidenceNode) -> int:
        return NODE_TYPES.index(node.type) if node.type in NODE_TYPES else NODE_TYPES.index("other")

    def modality_id(self, node: EvidenceNode) -> int:
        return MODALITIES.index(node.modality) if node.modality in MODALITIES else 0

    def edge_type_id(self, edge: EvidenceEdge) -> int:
        return EDGE_TYPES.index(edge.type) if edge.type in EDGE_TYPES else EDGE_TYPES.index("other")


def load_audit_units(path: str | Path) -> list[AuditUnit]:
    """Load JSONL or JSON audit units."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        rows = raw if isinstance(raw, list) else raw.get("audit_units", [raw])
    return [AuditUnit.from_dict(row) for row in rows]


def write_audit_units(units: Iterable[AuditUnit], path: str | Path) -> None:
    """Write audit units as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalize_chain(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        raw = raw.get("nodes") or raw.get("node_ids") or []
    return [str(item) for item in raw]

