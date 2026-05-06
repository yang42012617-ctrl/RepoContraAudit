"""Neural components for the RepoContraAudit reference implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from repocontraaudit.data import AuditUnit
from repocontraaudit.schema import EDGE_TYPES, MODALITIES, NODE_TYPES
from repocontraaudit.text import stable_hash, tokenize


@dataclass
class RepoContraAuditConfig:
    """Model hyperparameters for the open reference implementation."""

    hidden_dim: int = 128
    vocab_size: int = 8192
    graph_layers: int = 3
    dropout: float = 0.10
    modality_dropout: float = 0.10
    chain_budget: int = 6
    beam_width: int = 6
    num_classes: int = 2

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "RepoContraAuditConfig":
        return cls(**{key: value for key, value in raw.items() if key in cls.__annotations__})


class ModelOutput(NamedTuple):
    logits: Tensor
    statement_logits: Tensor
    statement_indices: list[int]
    node_embeddings: Tensor
    repo_embedding: Tensor
    modality_gates: Tensor


class HashingTextEncoder(nn.Module):
    """A tiny deterministic token encoder.

    Paper-scale runs should replace this with a pretrained code Transformer. The
    reference encoder keeps the repository small and makes CI smoke tests cheap.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, texts: list[str], device: torch.device) -> Tensor:
        vectors: list[Tensor] = []
        for text in texts:
            token_ids = [stable_hash(tok.lower(), self.vocab_size - 1) + 1 for tok in tokenize(text)]
            if not token_ids:
                token_ids = [0]
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            vectors.append(self.embedding(ids).mean(dim=0))
        return torch.stack(vectors, dim=0)


class TypedGraphLayer(nn.Module):
    """Relation-aware message passing matching the paper's typed aggregation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in EDGE_TYPES
        )
        self.attention = nn.ModuleList(
            nn.Linear(hidden_dim * 2 + 1, 1, bias=False) for _ in EDGE_TYPES
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_type_ids: Tensor,
        edge_confidence: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return self.norm(h)

        num_nodes = h.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        messages = torch.zeros_like(h)
        logits = torch.empty(edge_index.size(1), device=h.device)
        transformed: list[Tensor] = []

        for edge_pos in range(edge_index.size(1)):
            relation_id = int(edge_type_ids[edge_pos].item())
            msg = self.relation_linears[relation_id](h[src[edge_pos]])
            transformed.append(msg)
            att_input = torch.cat([h[dst[edge_pos]], msg, edge_confidence[edge_pos].view(1)])
            logits[edge_pos] = self.attention[relation_id](att_input).squeeze(0)

        for node_idx in range(num_nodes):
            incoming = (dst == node_idx).nonzero(as_tuple=False).flatten()
            if incoming.numel() == 0:
                continue
            weights = F.softmax(logits[incoming], dim=0)
            for local_pos, edge_pos in enumerate(incoming.tolist()):
                messages[node_idx] = messages[node_idx] + weights[local_pos] * transformed[edge_pos]

        updated = self.update(self.dropout(messages), h)
        return self.norm(h + self.dropout(updated))


class ReliabilityAwareFusion(nn.Module):
    """Reliability-aware cross-modal fusion with observation masks."""

    def __init__(self, hidden_dim: int, dropout: float, modality_dropout: float) -> None:
        super().__init__()
        self.value_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in MODALITIES)
        self.gates = nn.ModuleList(nn.Linear(hidden_dim * 2 + 1, 1) for _ in MODALITIES)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.modality_dropout = modality_dropout

    def forward(
        self,
        h: Tensor,
        node_modality_ids: Tensor,
        neighbors: list[list[int]],
        observation_mask: Tensor,
        training: bool,
    ) -> tuple[Tensor, Tensor]:
        fused_rows: list[Tensor] = []
        gate_rows: list[Tensor] = []
        for node_idx in range(h.size(0)):
            neighbor_ids = neighbors[node_idx] or [node_idx]
            evidence_vectors: list[Tensor] = []
            gate_logits: list[Tensor] = []
            for modality_id, _modality in enumerate(MODALITIES):
                candidates = [
                    idx for idx in neighbor_ids if int(node_modality_ids[idx].item()) == modality_id
                ]
                observed = observation_mask[modality_id].clone()
                if training and self.modality_dropout > 0 and bool(observed.item()):
                    drop = torch.rand((), device=h.device) < self.modality_dropout
                    observed = observed & (~drop)
                if candidates:
                    evidence = h[torch.tensor(candidates, dtype=torch.long, device=h.device)].mean(dim=0)
                    evidence = self.value_layers[modality_id](evidence)
                else:
                    evidence = torch.zeros_like(h[node_idx])
                evidence_vectors.append(evidence)
                gate_input = torch.cat([h[node_idx], evidence, observed.float().view(1)])
                logit = self.gates[modality_id](gate_input).squeeze(0)
                if not bool(observed.item()) or not candidates:
                    logit = logit - 1e4
                gate_logits.append(logit)
            logits = torch.stack(gate_logits)
            if torch.all(logits < -999):
                logits = torch.zeros_like(logits)
            gates = F.softmax(logits, dim=0)
            evidence_sum = torch.stack(evidence_vectors, dim=0).mul(gates[:, None]).sum(dim=0)
            fused_rows.append(self.norm(h[node_idx] + self.fuse(torch.cat([h[node_idx], evidence_sum]))))
            gate_rows.append(gates)
        return torch.stack(fused_rows, dim=0), torch.stack(gate_rows, dim=0)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor, indices: list[int] | None = None) -> Tensor:
        if indices:
            idx = torch.tensor(indices, dtype=torch.long, device=h.device)
            values = h[idx]
        else:
            values = h
        weights = F.softmax(self.score(values).squeeze(-1), dim=0)
        return (weights[:, None] * values).sum(dim=0)


class RepoContraAuditModel(nn.Module):
    """Compact implementation of RepoContraAudit's shared evidence graph model."""

    def __init__(self, config: RepoContraAuditConfig | None = None) -> None:
        super().__init__()
        self.config = config or RepoContraAuditConfig()
        hdim = self.config.hidden_dim
        self.text_encoder = HashingTextEncoder(self.config.vocab_size, hdim)
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hdim)
        self.modality_embedding = nn.Embedding(len(MODALITIES), hdim)
        self.numeric_projection = nn.Linear(3, hdim)
        self.input_norm = nn.LayerNorm(hdim)
        self.graph_layers = nn.ModuleList(
            TypedGraphLayer(hdim, self.config.dropout) for _ in range(self.config.graph_layers)
        )
        self.fusion = ReliabilityAwareFusion(
            hdim, self.config.dropout, self.config.modality_dropout
        )
        self.repo_pool = AttentionPool(hdim)
        self.classifier = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hdim, self.config.num_classes),
        )
        self.localization_head = nn.Linear(hdim, 1)
        self.chain_node_head = nn.Linear(hdim, 1)
        self.edge_type_embedding = nn.Embedding(len(EDGE_TYPES), hdim)
        self.chain_edge_head = nn.Sequential(
            nn.Linear(hdim * 3, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1),
        )

    def forward(self, unit: AuditUnit, node_suppression: Tensor | None = None) -> ModelOutput:
        device = next(self.parameters()).device
        node_type_ids, modality_ids, numeric_features = self._node_tensors(unit, device)
        edge_index, edge_type_ids, edge_confidence = self._edge_tensors(unit, device)
        texts = [_node_text(node) for node in unit.nodes]

        h = (
            self.text_encoder(texts, device)
            + self.node_type_embedding(node_type_ids)
            + self.modality_embedding(modality_ids)
            + self.numeric_projection(numeric_features)
        )
        h = self.input_norm(h)
        if node_suppression is not None:
            h = h * (1.0 - node_suppression.to(device).float().view(-1, 1).clamp(0, 1))

        for layer in self.graph_layers:
            h = layer(h, edge_index, edge_type_ids, edge_confidence)

        neighbors = _neighbors_for_fusion(unit)
        observation = torch.tensor(
            [unit.observation_mask.get(modality, False) for modality in MODALITIES],
            dtype=torch.bool,
            device=device,
        )
        h, gates = self.fusion(h, modality_ids, neighbors, observation, self.training)

        statement_indices = unit.statement_indices
        statement_logits = (
            self.localization_head(h[torch.tensor(statement_indices, dtype=torch.long, device=device)])
            .squeeze(-1)
            if statement_indices
            else torch.empty(0, device=device)
        )
        task_indices = [
            idx
            for idx, node in enumerate(unit.nodes)
            if node.type in {"statement", "static_alert", "trace_event", "text_segment"}
        ]
        repo_embedding = self.repo_pool(h, task_indices or None)
        logits = self.classifier(repo_embedding)
        return ModelOutput(logits, statement_logits, statement_indices, h, repo_embedding, gates)

    def score_chain(self, unit: AuditUnit, node_embeddings: Tensor, chain_nodes: list[str]) -> Tensor:
        """Score a connected evidence chain using node and induced-edge scores."""

        device = node_embeddings.device
        valid_ids = [node_id for node_id in chain_nodes if node_id in unit.node_index]
        if not valid_ids:
            return torch.tensor(0.0, device=device)
        indices = torch.tensor([unit.node_index[node_id] for node_id in valid_ids], device=device)
        score = self.chain_node_head(node_embeddings[indices]).sum()
        selected = set(valid_ids)
        for edge in unit.edges:
            if edge.src in selected and edge.dst in selected:
                src = node_embeddings[unit.node_index[edge.src]]
                dst = node_embeddings[unit.node_index[edge.dst]]
                edge_type = self.edge_type_embedding(
                    torch.tensor(unit.edge_type_id(edge), dtype=torch.long, device=device)
                )
                score = score + self.chain_edge_head(torch.cat([src, dst, edge_type])).squeeze(0)
        return score

    def _node_tensors(self, unit: AuditUnit, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        node_type_ids = torch.tensor([unit.type_id(node) for node in unit.nodes], dtype=torch.long, device=device)
        modality_ids = torch.tensor(
            [unit.modality_id(node) for node in unit.nodes], dtype=torch.long, device=device
        )
        line_scale = max([node.line or 0 for node in unit.nodes] + [1])
        numeric = torch.tensor(
            [
                [
                    float(node.observed),
                    float(node.confidence),
                    float(node.line or 0) / float(line_scale),
                ]
                for node in unit.nodes
            ],
            dtype=torch.float32,
            device=device,
        )
        return node_type_ids, modality_ids, numeric

    def _edge_tensors(self, unit: AuditUnit, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if not unit.edges:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
            )
        src = [unit.node_index[edge.src] for edge in unit.edges]
        dst = [unit.node_index[edge.dst] for edge in unit.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_type_ids = torch.tensor(
            [unit.edge_type_id(edge) for edge in unit.edges], dtype=torch.long, device=device
        )
        edge_confidence = torch.tensor(
            [edge.confidence for edge in unit.edges], dtype=torch.float32, device=device
        )
        return edge_index, edge_type_ids, edge_confidence


def _node_text(node) -> str:
    fields = [
        node.type,
        node.modality,
        node.text,
        node.file or "",
        node.attrs.get("rule_id", ""),
        node.attrs.get("severity", ""),
        node.attrs.get("tool", ""),
    ]
    return " ".join(str(field) for field in fields if field)


def _neighbors_for_fusion(unit: AuditUnit) -> list[list[int]]:
    neighbors: list[set[int]] = [set([idx]) for idx in range(len(unit.nodes))]
    for edge in unit.edges:
        src = unit.node_index[edge.src]
        dst = unit.node_index[edge.dst]
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return [sorted(values) for values in neighbors]
