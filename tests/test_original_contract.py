import ast
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from runtime import (
    ROOT, build_model, load_checkpoint, parse_options, run_epoch, save_checkpoint,
)


def options():
    return parse_options(["--device", "cpu", "--output-n", "25", "--d-model", "4", "--log-every", "0"])


def definition_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class KnownFiveOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = None

    def forward(self, full):
        self.seen = full.detach().clone()
        template = torch.ones_like(full)
        template[:, :10] *= 2
        return tuple(template * self.scale * stage for stage in range(1, 6))


class OriginalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_released_model_and_active_layers_are_verbatim(self):
        manifest = json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))
        nodes = ast.parse((ROOT/"model/model_h36m_long.py").read_text(encoding="utf-8")).body
        node = next(n for n in nodes if isinstance(n, ast.ClassDef))
        self.assertEqual(definition_hash(node), manifest["model_class_ast_sha256"])
        self.assertIn("gt_spatial = gt.reshape(bs, self.in_features // self.node_n, self.node_n, -1)",
                      (ROOT/"model/model_h36m_long.py").read_text(encoding="utf-8"))
        base_nodes = {n.name: n for n in ast.parse((ROOT/"model/BaseModel.py").read_text(encoding="utf-8")).body
                      if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertEqual(set(base_nodes), set(manifest["base_definition_ast_sha256"]))
        for name, expected in manifest["base_definition_ast_sha256"].items():
            self.assertEqual(definition_hash(base_nodes[name]), expected, name)

    def test_original_preprocessing_definitions_are_unchanged(self):
        manifest = json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))
        for relative, definitions in manifest["original_preprocessing_ast_sha256"].items():
            nodes = {n.name: n for n in ast.parse((ROOT/relative).read_text(encoding="utf-8")).body
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for name, expected in definitions.items():
                self.assertEqual(definition_hash(nodes[name]), expected, f"{relative}:{name}")

    def test_full_sequence_forward_shapes(self):
        for horizon in (25, 50):
            with self.subTest(horizon=horizon):
                opt = parse_options(["--device", "cpu", "--output-n", str(horizon), "--d-model", "4"])
                model = build_model(opt).eval()
                with torch.no_grad():
                    outputs = model(torch.randn(2, 10+horizon, 66))
                self.assertEqual(len(outputs), 5)
                for output in outputs:
                    self.assertEqual(tuple(output.shape), (2, 10+horizon, 66))
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_future_ground_truth_still_has_a_gradient_path(self):
        torch.manual_seed(213)
        model = build_model(options()).eval()
        full = torch.randn(2, 35, 66, requires_grad=True)
        model(full)[0][:, 10:].square().mean().backward()
        self.assertIsNotNone(full.grad)
        self.assertGreater(full.grad[:, 10:].abs().sum().item(), 0)

    def test_training_passes_full_window_and_preserves_original_loss_and_metric(self):
        opt = options()
        model = KnownFiveOutputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        result = run_epoch(model, [torch.zeros(2, 35, 96)], opt, optimizer)
        self.assertEqual(tuple(model.seen.shape), (2, 35, 66))
        # Every stage predicts XYZ=2 in history and XYZ=1 in future, scaled
        # by stage index. Original loss includes all 35 frames and all heads.
        expected = math.sqrt(3) * (2*10 + 25) / 35 * 15
        self.assertAlmostEqual(result["objective"], expected, places=4)
        # 22 predicted joints + 6 equivalent copied joints, 4 GT-filled joints.
        self.assertAlmostEqual(result["loss"], math.sqrt(3)*28/32, places=5)

    def test_original_unused_graph_biases_remain_unused(self):
        model = build_model(options()).train()
        model(torch.randn(2, 35, 66))[0].square().mean().backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [f"gcn_blocks.{i}.bias" for i in range(7)])

    def test_checkpoint_roundtrip_restores_weights_optimizer_and_rng(self):
        opt = options()
        model = build_model(opt)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        metrics = run_epoch(model, [torch.randn(2, 35, 96)], opt, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"last.pt"
            record = {"train": metrics, "val": metrics}
            save_checkpoint(path, model, optimizer, 1, metrics["loss"], record, opt)
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restored = build_model(opt)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=opt.lr)
            checkpoint = load_checkpoint(path, restored, opt, restored_optimizer)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(restored_optimizer.state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)

    def test_corrected_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixed.pt"
            torch.save({"model_format": "geobiasnet-history-xyz-v1"}, path)
            with self.assertRaisesRegex(ValueError, "another implementation"):
                load_checkpoint(path, build_model(options()), options())


if __name__ == "__main__":
    unittest.main()
