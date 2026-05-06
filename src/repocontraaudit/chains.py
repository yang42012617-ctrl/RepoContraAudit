"""Evidence-chain extraction, marginalization, and losses."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
import math
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from repocontraaudit.data import AuditUnit
from repocontraaudit.model import ModelOutput, RepoContraAuditModel


@dataclass(frozen=True)
class EvidenceChain:
    node_ids: tuple[str, ...]
    score: float

    def to_dict(self) -> dict:
        return {"nodes": list(self.node_ids), "score": self.score}


def latent_chain_loss(
    model: RepoContraAuditModel,
    unit: AuditUnit,
    output: ModelOutput,
    max_negatives: int = 8,
) -> Tensor:
    """Partially supervised contrastive chain objective.

    It implements `-log sum exp(pos) / (sum exp(pos) + sum exp(neg))`.
    If no weak chain supervision is available, the returned zero tensor keeps
    the partial-observation objective well defined.
    """

    device = output.node_embeddings.device
    positives = [
        [node_id for node_id in chain if node_id in unit.node_index]
        for chain in unit.positive_chains
    ]
    positives = [chain for chain in positives if chain]
    if not positives:
        return torch.zeros((), device=device)

    negatives = corrupt_chains(unit, positives, max_negatives=max_negatives)
    pos_scores = torch.stack(
        [model.score_chain(unit, output.node_embeddings, chain) for chain in positives]
    )
    if negatives:
        neg_scores = torch.stack(
            [model.score_chain(unit, output.node_embeddings, chain) for chain in negatives]
        )
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
    else:
        all_scores = pos_scores
    return -(torch.logsumexp(pos_scores, dim=0) - torch.logsumexp(all_scores, dim=0))


def localization_loss(unit: AuditUnit, output: ModelOutput) -> Tensor:
    """Statement-level rationale BCE applied only when rationales are observed."""

    device = output.logits.device
    if not unit.rationales or output.statement_logits.numel() == 0:
        return torch.zeros((), device=device)
    rationale_set = set(unit.rationales)
    labels = torch.tensor(
        [1.0 if unit.nodes[idx].id in rationale_set else 0.0 for idx in output.statement_indices],
        dtype=torch.float32,
        device=device,
    )
    return F.binary_cross_entropy_with_logits(output.statement_logits, labels)


def classification_loss(unit: AuditUnit, output: ModelOutput) -> Tensor:
    label = torch.tensor([unit.label], dtype=torch.long, device=output.logits.device)
    return F.cross_entropy(output.logits.view(1, -1), label)


def statement_marginals_from_chains(unit: AuditUnit, chains: list[EvidenceChain]) -> dict[str, float]:
    """Approximate statement marginals over retained beam-search chains."""

    if not chains:
        return {node_id: 0.0 for node_id in unit.statement_ids}
    scores = torch.tensor([chain.score for chain in chains], dtype=torch.float32)
    weights = torch.softmax(scores, dim=0).tolist()
    marginals = {node_id: 0.0 for node_id in unit.statement_ids}
    for chain, weight in zip(chains, weights):
        selected = set(chain.node_ids)
        for node_id in marginals:
            if node_id in selected:
                marginals[node_id] += float(weight)
    return marginals


@torch.no_grad()
def beam_search_chains(
    model: RepoContraAuditModel,
    unit: AuditUnit,
    output: ModelOutput,
    budget: int | None = None,
    beam_width: int | None = None,
) -> list[EvidenceChain]:
    """Extract high-scoring connected evidence chains."""

    budget = budget or model.config.chain_budget
    beam_width = beam_width or model.config.beam_width
    node_scores = model.chain_node_head(output.node_embeddings).squeeze(-1)

    seed_indices = _seed_indices(unit, output, node_scores, beam_width)
    beams: list[tuple[tuple[int, ...], Tensor]] = [
        ((idx,), model.score_chain(unit, output.node_embeddings, [unit.nodes[idx].id]))
        for idx in seed_indices
    ]

    finished: list[tuple[tuple[int, ...], Tensor]] = list(beams)
    for _depth in range(1, budget):
        expanded: list[tuple[tuple[int, ...], Tensor]] = []
        for chain_indices, _score in beams:
            chain_set = set(chain_indices)
            frontier = set()
            for idx in chain_indices:
                for neighbor_id in unit.neighbors(unit.nodes[idx].id, undirected=True):
                    frontier.add(unit.node_index[neighbor_id])
            for next_idx in frontier - chain_set:
                new_indices = tuple(sorted((*chain_indices, next_idx)))
                node_ids = [unit.nodes[idx].id for idx in new_indices]
                expanded.append((new_indices, model.score_chain(unit, output.node_embeddings, node_ids)))
        if not expanded:
            break
        unique: dict[tuple[int, ...], Tensor] = {}
        for indices, score in expanded:
            if indices not in unique or float(score) > float(unique[indices]):
                unique[indices] = score
        beams = nlargest(beam_width, unique.items(), key=lambda item: float(item[1]))
        finished.extend(beams)

    dedup: dict[tuple[str, ...], float] = {}
    for indices, score in finished:
        node_ids = tuple(unit.nodes[idx].id for idx in indices)
        dedup[node_ids] = max(float(score), dedup.get(node_ids, -math.inf))
    return [
        EvidenceChain(node_ids=node_ids, score=score)
        for node_ids, score in nlargest(beam_width, dedup.items(), key=lambda item: item[1])
    ]


def corrupt_chains(
    unit: AuditUnit,
    positives: list[list[str]],
    max_negatives: int = 8,
    seed: int = 13,
) -> list[list[str]]:
    """Create negative chains by type-compatible and locality-breaking corruption."""

    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for node in unit.nodes:
        by_type.setdefault(node.type, []).append(node.id)

    negatives: list[list[str]] = []
    for chain in positives:
        if len(negatives) >= max_negatives:
            break
        if not chain:
            continue
        for replace_pos, node_id in enumerate(chain):
            node = unit.nodes_by_id.get(node_id)
            if not node:
                continue
            candidates = [candidate for candidate in by_type.get(node.type, []) if candidate not in chain]
            if not candidates:
                continue
            corrupted = list(chain)
            corrupted[replace_pos] = rng.choice(candidates)
            if not is_connected(unit, corrupted):
                negatives.append(corrupted)
                break

    all_ids = [node.id for node in unit.nodes]
    while len(negatives) < max_negatives and all_ids:
        size = min(max(2, len(positives[0])), len(all_ids))
        sampled = rng.sample(all_ids, size)
        if not is_connected(unit, sampled):
            negatives.append(sampled)
        else:
            break
    return negatives[:max_negatives]


def is_connected(unit: AuditUnit, node_ids: list[str] | tuple[str, ...]) -> bool:
    selected = set(node_ids)
    if len(selected) <= 1:
        return True
    start = next(iter(selected))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in unit.neighbors(current, undirected=True):
            if neighbor in selected and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == selected


def _seed_indices(unit: AuditUnit, output: ModelOutput, node_scores: Tensor, beam_width: int) -> list[int]:
    statement_pairs = list(zip(output.statement_indices, output.statement_logits.tolist()))
    statement_pairs.sort(key=lambda item: item[1], reverse=True)
    seeds = [idx for idx, _score in statement_pairs[:beam_width]]
    alert_trace = [
        idx
        for idx, node in enumerate(unit.nodes)
        if node.type in {"static_alert", "trace_event"} and idx not in seeds
    ]
    alert_trace.sort(key=lambda idx: float(node_scores[idx]), reverse=True)
    seeds.extend(alert_trace[: max(0, beam_width - len(seeds))])
    if not seeds:
        scores = node_scores.detach().cpu().tolist()
        seeds = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:beam_width]
    return seeds

