from pathlib import Path
import unittest

import torch

from repocontraaudit.antifact import AntiFactSearcher
from repocontraaudit.chains import beam_search_chains, latent_chain_loss, localization_loss
from repocontraaudit.data import load_audit_units
from repocontraaudit.model import RepoContraAuditConfig, RepoContraAuditModel


ROOT = Path(__file__).resolve().parents[1]


class ReferencePipelineTest(unittest.TestCase):
    def test_forward_chain_and_antifact(self):
        units = load_audit_units(ROOT / "examples" / "toy_audit_units.jsonl")
        model = RepoContraAuditModel(
            RepoContraAuditConfig(hidden_dim=32, graph_layers=1, chain_budget=4, beam_width=3)
        )
        unit = units[0]
        output = model(unit)

        self.assertEqual(output.logits.shape[-1], 2)
        self.assertEqual(output.statement_logits.numel(), len(unit.statement_ids))
        self.assertGreaterEqual(float(localization_loss(unit, output).detach()), 0.0)
        self.assertGreaterEqual(float(latent_chain_loss(model, unit, output).detach()), 0.0)

        chains = beam_search_chains(model, unit, output, budget=4, beam_width=3)
        self.assertTrue(chains)
        result = AntiFactSearcher(steps=2).search(model, unit, chains[0])
        self.assertIn("grounded", result.intervention.checks)

    def test_model_can_train_one_step(self):
        units = load_audit_units(ROOT / "examples" / "toy_audit_units.jsonl")
        model = RepoContraAuditModel(RepoContraAuditConfig(hidden_dim=32, graph_layers=1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model(units[0])
        loss = latent_chain_loss(model, units[0], output) + localization_loss(units[0], output)
        loss = loss + torch.nn.functional.cross_entropy(output.logits.view(1, -1), torch.tensor([1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
