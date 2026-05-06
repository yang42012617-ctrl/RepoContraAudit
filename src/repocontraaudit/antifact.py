"""Anti-fact evidence perturbation search.

Anti-fact interventions in this repository are evidence-level perturbations:
they suppress selected graph evidence and test model-level decision sensitivity.
They are not semantic vulnerability repairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast

import torch
from torch import Tensor
import torch.nn.functional as F

from repocontraaudit.chains import EvidenceChain, beam_search_chains, is_connected
from repocontraaudit.data import AuditUnit
from repocontraaudit.model import RepoContraAuditModel


@dataclass
class AntiFactIntervention:
    selected_nodes: list[dict]
    selected_edges: list[dict]
    checks: dict[str, bool]
    projection_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "selected_nodes": self.selected_nodes,
            "selected_edges": self.selected_edges,
            "checks": self.checks,
            "projection_failed": self.projection_failed,
        }


@dataclass
class AntiFactResult:
    before_label: int
    after_label: int
    before_confidence: float
    after_confidence: float
    confidence_drop: float
    flipped: bool
    intervention: AntiFactIntervention

    def to_dict(self) -> dict:
        return {
            "before_label": self.before_label,
            "after_label": self.after_label,
            "before_confidence": self.before_confidence,
            "after_confidence": self.after_confidence,
            "confidence_drop": self.confidence_drop,
            "flipped": self.flipped,
            "intervention": self.intervention.to_dict(),
        }


class AntiFactSearcher:
    """Differentiable mask proposal followed by conservative projection."""

    def __init__(
        self,
        steps: int = 40,
        lr: float = 5e-2,
        node_threshold: float = 0.50,
        edge_threshold: float = 0.40,
        lambda_sparse: float = 0.05,
        lambda_conn: float = 0.20,
        lambda_plaus: float = 0.10,
    ) -> None:
        self.steps = steps
        self.lr = lr
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.lambda_sparse = lambda_sparse
        self.lambda_conn = lambda_conn
        self.lambda_plaus = lambda_plaus

    def search(
        self,
        model: RepoContraAuditModel,
        unit: AuditUnit,
        chain: EvidenceChain | None = None,
    ) -> AntiFactResult:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            before = model(unit)
            before_probs = torch.softmax(before.logits, dim=-1)
            before_label = int(before_probs.argmax().item())
            before_confidence = float(before_probs[before_label].item())
            if chain is None:
                chains = beam_search_chains(model, unit, before)
                chain = chains[0] if chains else EvidenceChain(tuple(unit.statement_ids[:1]), 0.0)

        candidate_ids = self._candidate_ids(unit, list(chain.node_ids))
        candidate_indices = [unit.node_index[node_id] for node_id in candidate_ids]
        if not candidate_indices:
            intervention = AntiFactIntervention([], [], {"grounded": False}, projection_failed=True)
            return AntiFactResult(
                before_label,
                before_label,
                before_confidence,
                before_confidence,
                0.0,
                False,
                intervention,
            )

        init = torch.full((len(candidate_indices),), -1.0, device=device)
        chain_set = set(chain.node_ids)
        for pos, node_id in enumerate(candidate_ids):
            if node_id in chain_set:
                init[pos] = 1.5
        mask_logits = torch.nn.Parameter(init)
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)

        benign = torch.tensor([0], dtype=torch.long, device=device)
        for _step in range(self.steps):
            optimizer.zero_grad()
            candidate_mask = torch.sigmoid(mask_logits)
            full_mask = torch.zeros(len(unit.nodes), dtype=torch.float32, device=device)
            full_mask[torch.tensor(candidate_indices, dtype=torch.long, device=device)] = candidate_mask
            masked = model(unit, node_suppression=full_mask)
            flip_loss = F.cross_entropy(masked.logits.view(1, -1), benign)
            sparse = candidate_mask.mean()
            conn = self._connectivity_surrogate(unit, candidate_ids, candidate_mask)
            plaus = self._plausibility_surrogate(unit, candidate_ids, candidate_mask)
            loss = flip_loss + self.lambda_sparse * sparse + self.lambda_conn * conn + self.lambda_plaus * plaus
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_logits], 1.0)
            optimizer.step()

        with torch.no_grad():
            final_scores = torch.sigmoid(mask_logits)
            selected_ids = [
                node_id
                for node_id, score in zip(candidate_ids, final_scores.tolist())
                if score >= self.node_threshold
            ]
            if not selected_ids and candidate_ids:
                best_pos = int(torch.argmax(final_scores).item())
                selected_ids = [candidate_ids[best_pos]]
            intervention = self._project(unit, selected_ids)
            hard_mask = torch.zeros(len(unit.nodes), dtype=torch.float32, device=device)
            for node_id in selected_ids:
                hard_mask[unit.node_index[node_id]] = 1.0
            after = model(unit, node_suppression=hard_mask)
            after_probs = torch.softmax(after.logits, dim=-1)
            after_label = int(after_probs.argmax().item())
            after_confidence = float(after_probs[before_label].item())

        if was_training:
            model.train()
        return AntiFactResult(
            before_label=before_label,
            after_label=after_label,
            before_confidence=before_confidence,
            after_confidence=after_confidence,
            confidence_drop=before_confidence - after_confidence,
            flipped=before_label != after_label,
            intervention=intervention,
        )

    def _candidate_ids(self, unit: AuditUnit, chain_ids: list[str]) -> list[str]:
        candidates = set(node_id for node_id in chain_ids if node_id in unit.node_index)
        for node_id in list(candidates):
            candidates.update(unit.neighbors(node_id, undirected=True))
        return sorted(candidates, key=lambda node_id: unit.node_index[node_id])

    def _connectivity_surrogate(self, unit: AuditUnit, candidate_ids: list[str], mask: Tensor) -> Tensor:
        selected_mass = mask.sum()
        if selected_mass <= 1:
            return torch.zeros((), device=mask.device)
        index = {node_id: idx for idx, node_id in enumerate(candidate_ids)}
        edge_mass = torch.zeros((), device=mask.device)
        for edge in unit.edges:
            if edge.src in index and edge.dst in index:
                edge_mass = edge_mass + mask[index[edge.src]] * mask[index[edge.dst]]
        return torch.relu((selected_mass - 1.0) - edge_mass) / (selected_mass + 1e-6)

    def _plausibility_surrogate(self, unit: AuditUnit, candidate_ids: list[str], mask: Tensor) -> Tensor:
        penalties = []
        for node_id in candidate_ids:
            node = unit.nodes_by_id[node_id]
            grounded = float(node.grounded)
            penalties.append(1.0 - grounded)
        penalty_tensor = torch.tensor(penalties, dtype=torch.float32, device=mask.device)
        return (penalty_tensor * mask).sum() / (mask.sum() + 1e-6)

    def _project(self, unit: AuditUnit, selected_ids: list[str]) -> AntiFactIntervention:
        grounded_nodes = [node_id for node_id in selected_ids if unit.nodes_by_id[node_id].grounded]
        projection_failed = bool(selected_ids) and len(grounded_nodes) < len(selected_ids)
        selected_set = set(grounded_nodes)
        selected_nodes = [
            {
                "id": node.id,
                "type": node.type,
                "modality": node.modality,
                "file": node.file,
                "line": node.line,
                "text": node.text,
            }
            for node_id in grounded_nodes
            for node in [unit.nodes_by_id[node_id]]
        ]
        selected_edges = [
            edge.to_dict()
            for edge in unit.edges
            if edge.src in selected_set and edge.dst in selected_set
        ]
        checks = {
            "grounded": len(grounded_nodes) == len(selected_ids),
            "connected": is_connected(unit, grounded_nodes) if grounded_nodes else False,
            "has_concrete_artifact": bool(selected_nodes),
            "edge_endpoints_grounded": all(
                edge["src"] in selected_set and edge["dst"] in selected_set for edge in selected_edges
            ),
            "syntax_parse_available": self._syntax_check(unit, grounded_nodes),
        }
        return AntiFactIntervention(selected_nodes, selected_edges, checks, projection_failed)

    def _syntax_check(self, unit: AuditUnit, selected_ids: list[str]) -> bool:
        """Parse Python files when they are locally available; otherwise report true.

        The anti-fact output is not an edited program, so this is a lightweight
        grounding check rather than a semantic-repair proof.
        """

        files = {unit.nodes_by_id[node_id].file for node_id in selected_ids}
        for file_name in files:
            if not file_name or not file_name.endswith(".py"):
                continue
            path = Path(file_name)
            if not path.exists():
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                return False
        return True
