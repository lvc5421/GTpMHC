"""Numerical wiring and strict checkpoint tests using untrained random weights.

The release tests are self-contained. To additionally compare the exact original
network, set GTPMHC_LEGACY_DIR to its source directory. No weights or features are
downloaded, and these tests establish no biological prediction accuracy.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "torch_geometric"))
if HAS_RUNTIME:
    import torch
    from gtpmhc import GTpMHC, inference_model_kwargs, load_gtpmhc_state


def graph_inputs(lengths, seed=31, mature_positions=True):
    generator = torch.Generator().manual_seed(seed)

    def features(length):
        identities = torch.randint(0, 20, (length,), generator=generator)
        raw = torch.cat((torch.nn.functional.one_hot(identities, 21).float(),
                         torch.rand(length, 12, generator=generator)), dim=1)
        return raw, torch.randn(length, 1280, generator=generator)

    def edges(sizes):
        result, offset = [], 0
        for size in sizes:
            nodes = torch.arange(size) + offset
            contact = torch.tensor([[offset, offset + size - 1], [offset + size - 1, offset]])
            sequential = torch.stack((torch.cat((nodes, nodes[:-1], nodes[1:])),
                                      torch.cat((nodes, nodes[1:], nodes[:-1]))))
            result.append(torch.cat((sequential, contact), dim=1))
            offset += size
        return torch.cat(result, dim=1)

    peptide_raw, peptide_esm = features(sum(lengths))
    hla_raw, hla_esm = features(34 * len(lengths))
    peptide_edges, hla_edges = edges(lengths), edges([34] * len(lengths))
    positions = ([7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
                  84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]
                 if mature_positions else list(range(34)))
    return {
        "x_pep": peptide_raw, "x_pretrain_pep": peptide_esm,
        "peptide_edge_index": peptide_edges,
        "x_p": hla_raw, "x_pretrain_p": hla_esm, "residue_edge_index": hla_edges,
        "residue_edge_weight": torch.rand(hla_edges.shape[1], generator=generator),
        "peptide_edge_weight": torch.rand(peptide_edges.shape[1], generator=generator),
        "prot_node_pos": torch.tensor(positions).reshape(-1, 1).repeat(len(lengths), 1),
        "pep_batch": torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)),
        "prot_batch": torch.repeat_interleave(torch.arange(len(lengths)), 34),
    }


@unittest.skipUnless(HAS_RUNTIME, "Model tests require PyTorch and PyG.")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2026)
        self.kwargs = {"hidden_channels": 16, "K": [2, 3], "heads": 2, "max_pep_len": 25}
        self.model = GTpMHC(**self.kwargs)

    def test_direct_nn_module_without_training_heads(self):
        self.assertEqual(GTpMHC.__bases__, (torch.nn.Module,))
        self.assertFalse(self.model.training)
        self.assertFalse(hasattr(self.model, "masked_span_loss"))
        self.assertFalse(hasattr(self.model, "cls_out_branch1"))
        self.assertFalse(hasattr(self.model, "transpmhc_pair_projection"))
        with self.assertRaises(TypeError):
            GTpMHC(**self.kwargs, window_padding=2)

    def test_single_logits_for_short_medium_long_and_mixed_batch(self):
        with torch.inference_mode():
            for lengths in ([9], [17], [25], [9, 17, 25]):
                with self.subTest(lengths=lengths):
                    output = self.model(**graph_inputs(lengths))
                    self.assertEqual(tuple(output.shape), (len(lengths), 1))
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_compact_checkpoint_roundtrip_is_exact_and_deterministic(self):
        restored = GTpMHC(**self.kwargs)
        report = load_gtpmhc_state(restored, {"state_dict": self.model.state_dict()})
        self.assertEqual(report["ignored_legacy_keys"], [])
        inputs = graph_inputs([9, 17, 25])
        with torch.inference_mode():
            expected = self.model(**inputs)
            self.assertTrue(torch.equal(expected, restored(**inputs)))
            self.assertTrue(torch.equal(expected, restored(**inputs)))

    def test_dataparallel_prefix_and_known_unused_legacy_tensor(self):
        state = {"module." + key: value for key, value in self.model.state_dict().items()}
        state["module.mae_pep_mask_token"] = torch.zeros(1, 16)
        report = load_gtpmhc_state(self.model, {"model_state_dict": state})
        self.assertEqual(report["ignored_legacy_keys"], ["mae_pep_mask_token"])

    def test_missing_retained_tensor_rejected_before_loading(self):
        state = dict(self.model.state_dict())
        key = "graph_sequence_projection.6.weight"
        del state[key]
        before = self.model.graph_sequence_projection[6].weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "Missing retained keys.*" + key):
            load_gtpmhc_state(self.model, state)
        self.assertTrue(torch.equal(before, self.model.graph_sequence_projection[6].weight))

    def test_unknown_keys_including_lookalike_auxiliary_and_wrong_layout_rejected(self):
        for key in ("unknown.weight", "mae_pep_decoder_typo.weight", "_hla_first_layout_version"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unexpected checkpoint keys"):
                load_gtpmhc_state(self.model, {**self.model.state_dict(), key: torch.zeros(1)})

    def test_wrong_shapes_for_retained_or_unused_tensors_are_rejected(self):
        for key, value in (("pep_convs.0.att", torch.zeros(1, 3, 16)),
                           ("pep_sinusoidal_position.encoding", torch.zeros(1, 15, 16)),
                           ("mae_pep_mask_token", torch.zeros(16))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "shape/type mismatch"):
                load_gtpmhc_state(self.model, {**self.model.state_dict(), key: value})

    def test_invalid_wrappers_and_mixed_prefixes_are_rejected(self):
        for checkpoint in ({}, {"state_dict": None}, [], {1: torch.zeros(1)}):
            with self.subTest(checkpoint=type(checkpoint).__name__), self.assertRaises(ValueError):
                load_gtpmhc_state(self.model, checkpoint)
        state = dict(self.model.state_dict())
        state["module.mae_pep_mask_token"] = torch.zeros(1, 16)
        with self.assertRaisesRegex(ValueError, "mixes prefixed"):
            load_gtpmhc_state(self.model, state)

    def test_explicit_legacy_configuration_filter_has_no_unknown_key_fallback(self):
        kwargs = {**self.kwargs, "dropout": 0.1, "transformer_dropout": 0.1,
                  "device": "cuda:0", "window_padding": 2, "total_layer": 2,
                  "use_raw_feature_fusion": True}
        self.assertEqual(inference_model_kwargs(kwargs), self.kwargs)
        for changes in ({"unknown_flag": 0}, {"use_raw_feature_fusion": False}, {"total_layer": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inference_model_kwargs({**kwargs, **changes})

    def test_training_mode_and_excessive_sequence_length_are_rejected(self):
        self.model.train()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            self.model(**graph_inputs([9]))
        self.model.eval()
        with self.assertRaisesRegex(ValueError, "exceeds max_pep_len"):
            self.model(**graph_inputs([26]))


@unittest.skipUnless(HAS_RUNTIME and bool(os.environ.get("GTPMHC_LEGACY_DIR")),
                     "Optional original-model comparison needs GTPMHC_LEGACY_DIR.")
class OriginalModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        directory = Path(os.environ["GTPMHC_LEGACY_DIR"]).resolve()
        source = directory / "pep_model_d4transformer_mae_merged_transpmhc_fulltoken_multidataset.py"
        if not source.is_file():
            raise ValueError("GTPMHC_LEGACY_DIR must contain the original peptide-first fulltoken model.")
        sys.path.insert(0, str(directory))
        try:
            spec = importlib.util.spec_from_file_location("_gtpmhc_original_comparison", source)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
        finally:
            sys.path.remove(str(directory))
        cls.Reference = reference.pMHC_classifier

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def compare(self, kwargs, batches, mature_positions=True):
        torch.manual_seed(44)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            original = self.Reference(**kwargs, device="cpu").eval()
        compact = GTpMHC(**kwargs)
        original_state = original.state_dict()
        report = load_gtpmhc_state(compact, original_state)
        self.assertEqual(len(report["ignored_legacy_keys"]), 76)
        self.assertEqual(set(original_state) - set(compact.state_dict()), set(report["ignored_legacy_keys"]))
        self.assertLess(sum(parameter.numel() for parameter in compact.parameters()),
                        sum(parameter.numel() for parameter in original.parameters()))
        with torch.inference_mode():
            for lengths in batches:
                with self.subTest(kwargs=kwargs, lengths=lengths):
                    inputs = graph_inputs(lengths, mature_positions=mature_positions)
                    expected = original(**inputs)[0]
                    actual = compact(**inputs)
                    # Same CPU operations and fixed slots: require exact equality,
                    # rather than merely comparing classifications or ranks.
                    self.assertTrue(torch.equal(expected, actual),
                                    f"max |logit difference| = {(expected - actual).abs().max().item()}")

    def test_original_logits_match_for_9_17_25mers_and_mixed_batch(self):
        self.compare({"hidden_channels": 16, "K": [2, 3], "heads": 2, "max_pep_len": 25},
                     ([9], [17], [25], [9, 17, 25]))

    def test_original_default_width_and_two_transformer_layers_match(self):
        self.compare({"hidden_channels": 128, "K": [5, 10], "heads": 4,
                      "max_pep_len": 25, "transformer_layers": 2}, ([9, 17, 25],),
                     mature_positions=False)


if __name__ == "__main__":
    unittest.main()
