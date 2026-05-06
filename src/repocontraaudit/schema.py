"""Shared schema constants for typed multimodal evidence graphs."""

from __future__ import annotations

MODALITIES = ("code", "graph", "alert", "trace", "text")

NODE_TYPES = (
    "statement",
    "function",
    "file",
    "variable",
    "api",
    "external_call",
    "static_alert",
    "trace_event",
    "text_segment",
    "repository",
    "other",
)

EDGE_TYPES = (
    "ast_containment",
    "control_flow",
    "data_flow",
    "call",
    "import",
    "inheritance",
    "alert_to_code",
    "trace_to_code",
    "text_to_code",
    "repository_contains",
    "file_contains",
    "function_contains",
    "next_statement",
    "related",
    "other",
)

TYPE_ALIASES = {
    "alert": "static_alert",
    "trace": "trace_event",
    "text": "text_segment",
    "contains": "ast_containment",
    "containment": "ast_containment",
    "cfg": "control_flow",
    "dfg": "data_flow",
    "alert-code": "alert_to_code",
    "trace-code": "trace_to_code",
    "text-code": "text_to_code",
}


def canonical_node_type(value: str | None) -> str:
    """Normalize user data into the reference node-type vocabulary."""

    if not value:
        return "other"
    value = str(value).strip().lower()
    value = TYPE_ALIASES.get(value, value)
    return value if value in NODE_TYPES else "other"


def canonical_edge_type(value: str | None) -> str:
    """Normalize user data into the reference edge-type vocabulary."""

    if not value:
        return "other"
    value = str(value).strip().lower()
    value = TYPE_ALIASES.get(value, value)
    return value if value in EDGE_TYPES else "other"


def canonical_modality(value: str | None, node_type: str | None = None) -> str:
    """Infer a modality from the explicit modality or node type."""

    if value:
        value = str(value).strip().lower()
        if value in MODALITIES:
            return value
    node_type = canonical_node_type(node_type)
    if node_type == "static_alert":
        return "alert"
    if node_type == "trace_event":
        return "trace"
    if node_type == "text_segment":
        return "text"
    if node_type in {"function", "file", "variable", "api", "repository"}:
        return "graph"
    return "code"

