"""Train the open RepoContraAudit reference model."""

from __future__ import annotations

import argparse
from pathlib import Path

from repocontraaudit.data import load_audit_units
from repocontraaudit.model import RepoContraAuditConfig
from repocontraaudit.training import LossWeights, TrainConfig, save_checkpoint, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RepoContraAudit on JSONL audit units.")
    parser.add_argument("--train-data", required=True, help="Training JSONL/JSON path.")
    parser.add_argument("--valid-data", default=None, help="Optional validation JSONL/JSON path.")
    parser.add_argument("--out", default="runs/repocontraaudit", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--chain-budget", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--loc-weight", type=float, default=1.0)
    parser.add_argument("--chain-weight", type=float, default=0.5)
    parser.add_argument("--patch-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_units = load_audit_units(args.train_data)
    valid_units = load_audit_units(args.valid_data) if args.valid_data else None
    model_config = RepoContraAuditConfig(
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        chain_budget=args.chain_budget,
        beam_width=args.beam_width,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        loss_weights=LossWeights(args.loc_weight, args.chain_weight, args.patch_weight),
    )
    model, metadata = train_model(train_units, valid_units, model_config, train_config, args.device)
    out_dir = Path(args.out)
    save_checkpoint(model, out_dir / "model.pt", metadata)
    final = metadata["history"][-1] if metadata["history"] else {}
    print(f"saved checkpoint to {out_dir / 'model.pt'}")
    print(f"final metrics: {final.get('valid', final.get('train', {}))}")


if __name__ == "__main__":
    main()

