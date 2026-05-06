"""Training utilities for partial-supervision RepoContraAudit runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random

import torch
from torch.optim import AdamW

from repocontraaudit.chains import classification_loss, latent_chain_loss, localization_loss
from repocontraaudit.data import AuditUnit
from repocontraaudit.model import RepoContraAuditConfig, RepoContraAuditModel


@dataclass
class LossWeights:
    loc: float = 1.0
    chain: float = 0.5
    patch: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 13
    patience: int = 5
    loss_weights: LossWeights = field(default_factory=LossWeights)


def train_model(
    train_units: list[AuditUnit],
    valid_units: list[AuditUnit] | None = None,
    model_config: RepoContraAuditConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str | None = None,
) -> tuple[RepoContraAuditModel, dict]:
    """Train the compact reference model."""

    train_config = train_config or TrainConfig()
    set_seed(train_config.seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RepoContraAuditModel(model_config or RepoContraAuditConfig()).to(torch_device)
    optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    valid_units = valid_units or []
    history: list[dict] = []
    best_score = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        random.shuffle(train_units)
        total_loss = 0.0
        parts = {"cls": 0.0, "loc": 0.0, "chain": 0.0, "patch": 0.0}
        for unit in train_units:
            optimizer.zero_grad()
            output = model(unit)
            cls = classification_loss(unit, output)
            loc = localization_loss(unit, output)
            chain = latent_chain_loss(model, unit, output)
            patch = patch_alignment_loss(model, unit, output)
            loss = (
                cls
                + train_config.loss_weights.loc * loc
                + train_config.loss_weights.chain * chain
                + train_config.loss_weights.patch * patch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            parts["cls"] += float(cls.detach().cpu())
            parts["loc"] += float(loc.detach().cpu())
            parts["chain"] += float(chain.detach().cpu())
            parts["patch"] += float(patch.detach().cpu())

        train_metrics = evaluate(model, train_units)
        valid_metrics = evaluate(model, valid_units) if valid_units else train_metrics
        record = {
            "epoch": epoch,
            "loss": total_loss / max(1, len(train_units)),
            "loss_parts": {key: value / max(1, len(train_units)) for key, value in parts.items()},
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(record)

        score = valid_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_macro_f1": best_score}


@torch.no_grad()
def evaluate(model: RepoContraAuditModel, units: list[AuditUnit]) -> dict[str, float]:
    if not units:
        return {"accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
    was_training = model.training
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    for unit in units:
        output = model(unit)
        preds.append(int(output.logits.argmax().item()))
        labels.append(unit.label)
    if was_training:
        model.train()
    return classification_metrics(labels, preds)


def classification_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    correct = sum(int(y == p) for y, p in zip(labels, preds))
    accuracy = correct / max(1, len(labels))
    per_class_f1 = []
    precisions = []
    recalls = []
    for cls in sorted(set(labels) | set(preds) | {0, 1}):
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        per_class_f1.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(per_class_f1) / len(per_class_f1),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
    }


def patch_alignment_loss(model: RepoContraAuditModel, unit: AuditUnit, output) -> torch.Tensor:
    """Optional patch-derived auxiliary loss over pre-fix changed evidence nodes."""

    device = output.logits.device
    patch_ids = [node_id for node_id in unit.patch_nodes if node_id in unit.node_index]
    if not patch_ids:
        return torch.zeros((), device=device)
    target = torch.zeros(len(unit.nodes), dtype=torch.float32, device=device)
    target[[unit.node_index[node_id] for node_id in patch_ids]] = 1.0
    logits = model.chain_node_head(output.node_embeddings).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def save_checkpoint(model: RepoContraAuditModel, path: str | Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(payload["metadata"], indent=2), encoding="utf-8"
    )


def load_checkpoint(path: str | Path, device: str | None = None) -> RepoContraAuditModel:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(Path(path), map_location=torch_device)
    model = RepoContraAuditModel(RepoContraAuditConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.to(torch_device)
    model.eval()
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
