"""Sequence numbering and numeric feature contracts, without ESM downloads."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import features

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


class NumberingTests(unittest.TestCase):
    def test_import_and_sequence_validation_require_no_ml_packages(self):
        command = [sys.executable, "-I", "-S", "-c",
                   "import runpy; m=runpy.run_path(" + repr(str(ROOT / "features.py")) + "); "
                   "assert m['normalize_sequence'](' aaA\\nDIVNFL ') == 'AAADIVNFL'"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(features.normalize_sequence(">one\naaadi\nvnfl", "Peptide"), "AAADIVNFL")
        for invalid in ("", "HLA-A*02:01", "AA*", ">one\nAAA\n>two\nCCC", ">empty", None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                features.normalize_sequence(invalid)

    def test_hla_i_precursor_and_mature_map_to_same_34_sites(self):
        full = features.REFERENCE_SEQUENCES["I"]
        for sequence, offset in ((full, 24), (full[24:], 0), (full[25:], -1)):
            with self.subTest(offset=offset):
                result = features.map_hla_sequence(sequence)
                self.assertEqual(result["sequence_offset"], offset)
                self.assertEqual(result["indices"], [offset + p - 1 for p in features.PSEUDO_POS_MATURE])
                self.assertEqual(result["pseudo_sequence"], "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY")

    def test_hla_mapping_rejects_ambiguous_indel_unknown_and_wrong_offset(self):
        full = features.REFERENCE_SEQUENCES["I"]
        for sequence in (full + full, full[:100] + "A" + full[100:],
                         full[:30] + "X" + full[31:], "A" * 350):
            with self.subTest(length=len(sequence)), self.assertRaises(ValueError):
                features.map_hla_sequence(sequence)
        for kwargs in ({"sequence_offset": 0}, {"sequence_offset": True},
                       {"pseudo_sequence": "A" * 34}, {"max_hla_position": 170}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                features.map_hla_sequence(full, **kwargs)

    def test_hla_dr_maps_two_chains_and_keeps_the_15_plus_19_order(self):
        dra, drb = (features.REFERENCE_SEQUENCES[key] for key in ("DRA", "DRB1"))
        for alpha, beta, offsets in ((dra, drb, (25, 29)), (dra[25:], drb[29:], (0, 0))):
            result = features.map_hla_ii_sequences(alpha, beta, max_hla_position=33)
            self.assertEqual((result["dra_mapping"]["sequence_offset"],
                              result["drb1_mapping"]["sequence_offset"]), offsets)
            expected = ("".join(dra[25 + p - 1] for p in features.DRA_PSEUDO_POS_MATURE)
                        + "".join(drb[29 + p - 1] for p in features.DRB1_PSEUDO_POS_MATURE))
            self.assertEqual(result["pseudo_sequence"], expected)
            self.assertEqual(len(result["dra_mapping"]["indices"]), 15)
            self.assertEqual(len(result["drb1_mapping"]["indices"]), 19)
        with self.assertRaises(ValueError):
            features.map_hla_ii_sequences(drb, dra)
        with self.assertRaises(ValueError):
            features.map_hla_ii_sequences(dra, drb, sequence_offset=0)


@unittest.skipIf(torch is None, "PyTorch and NumPy are required for numeric feature checks")
class NumericFeatureTests(unittest.TestCase):
    def test_raw33_matches_frozen_training_feature_matrix_exactly(self):
        # SHA256 of original preprocessing's float32 [21,33] matrix in this exact
        # alphabet order. This catches table values, normalization and ordering.
        matrix = features.seq_feature(features.RESIDUES)
        self.assertEqual(matrix.shape, (21, 33))
        self.assertEqual(hashlib.sha256(matrix.tobytes()).hexdigest(),
                         "e4bb927be46dc2b8c8b93f57ab76b44084d174f8bbfd574605968d89e85291d4")
        np.testing.assert_array_equal(matrix[:, :21], np.eye(21, dtype=np.float32))
        np.testing.assert_array_equal(matrix[-1, 21:26], np.zeros(5))
        np.testing.assert_array_equal(matrix[-1, 26:], np.full(7, 0.5))

    def test_contacts_keep_max_weights_neighbor_edges_and_last_self_loops(self):
        contacts = torch.eye(4) * 0.9
        contacts[0, 3], contacts[3, 0] = 0.8, 0.6
        edges, weights = features.contact_map(contacts)
        expected = [[i, j] for i in range(4) for j in range(4) if i != j] + [[i, i] for i in range(4)]
        self.assertEqual(edges.t().tolist(), expected)
        for (source, target), weight in zip(expected, weights.tolist()):
            self.assertAlmostEqual(weight, 1. if source == target else
                                   0.8 if {source, target} == {0, 3} else 0.5)
        self.assertEqual(features.contact_map(torch.empty(0, 0))[0].shape, (2, 0))
        self.assertEqual(features.contact_map(torch.zeros(1, 1))[1].tolist(), [1.])
        # Thresholding precedes symmetrization in training: an invalid/weak
        # reverse direction must not erase an eligible forward contact.
        contacts[3, 0] = float("nan")
        _, actual_weights = features.contact_map(contacts)
        torch.testing.assert_close(actual_weights, weights)

    def stub_esm(self, dim=4):
        class ESM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.fragments = []

            def converter(self, data):
                sequence = data[0][1]
                self.fragments.append(sequence)
                return None, None, torch.tensor([[0] + [ord(aa) for aa in sequence] + [0]])

            def forward(self, tokens, repr_layers, return_contacts):
                length = tokens.shape[1] - 2
                positions = torch.arange(tokens.shape[1], dtype=torch.float32)[None, :]
                value = tokens.float() / 100 + positions
                representations = {layer: (value + layer).unsqueeze(-1).repeat(1, 1, dim)
                                   for layer in repr_layers}
                local = torch.arange(length, dtype=torch.float32)
                contacts = ((local[:, None] + local[None, :]) / (2 * length)).unsqueeze(0)
                return {"representations": representations, "contacts": contacts,
                        "logits": value.unsqueeze(-1).repeat(1, 1, 3)}
        return ESM()

    def test_short_esm_excludes_special_tokens_and_supports_layer_aggregation(self):
        model = self.stub_esm()
        sequence = "AAADIVNFL"
        for approach, multiplier, bias in (("last", 1, 3), ("sum", 3, 6), ("mean", 1, 2)):
            embedding, contacts = features.esm_extract(model, model.converter, sequence,
                                                        layer=3, approach=approach, dim=4)
            expected = (torch.tensor([ord(aa) for aa in sequence]) / 100
                        + torch.arange(1, 10)) * multiplier + bias
            torch.testing.assert_close(embedding[:, 0], expected)
            self.assertEqual(embedding.shape, (9, 4))
            self.assertEqual(contacts.shape, (9, 9))
            self.assertFalse(embedding.requires_grad)

    def test_long_esm_uses_600_stride_200_windows_and_overlap_averages(self):
        model = self.stub_esm()
        sequence = (features.RESIDUES * 53)[:1100]
        embedding, contacts = features.esm_extract(model, model.converter, sequence, layer=3, dim=4)
        self.assertEqual(model.fragments, [sequence[start:start + 600] for start in (0, 200, 400, 500)])
        expected = ord(sequence[500]) / 100 + (501 + 301 + 101 + 1) / 4 + 3
        self.assertAlmostEqual(embedding[500, 0].item(), expected, places=4)
        self.assertAlmostEqual(contacts[500, 500].item(), (500 + 300 + 100 + 0) / (4 * 600))
        self.assertEqual(contacts[0, 1099].item(), 0.)
        self.assertEqual(embedding.shape, (1100, 4))


@unittest.skipIf(torch is None, "PyTorch is required for live-feature assembly checks")
class ProviderTests(unittest.TestCase):
    def provider(self, **settings):
        return features.GTpMHCFeatures(settings, ROOT, "cpu")

    def synthetic(self, sequence, kind):
        length = len(sequence)
        item = {"seq": sequence, "num_nodes": length, "node_pos": torch.arange(length).reshape(-1, 1),
                "edge_weight": torch.ones(length), "edge_index": torch.arange(length).repeat(2, 1),
                "seq_feat": torch.arange(length, dtype=torch.float32)[:, None].repeat(1, 33),
                "token_representation": torch.arange(length, dtype=torch.float32)[:, None].repeat(1, 1280)}
        if kind == "peptide":
            item["x_pep"] = item.pop("seq_feat")
            item["x_pretrain_pep"] = item.pop("token_representation")
            item["peptide_edge_index"] = item.pop("edge_index")
        return item

    def test_hla_i_is_extracted_at_full_length_before_slicing(self):
        provider = self.provider()
        provider._sequence_features = Mock(side_effect=self.synthetic)
        full = features.REFERENCE_SEQUENCES["I"]
        _, item, source = provider.protein_feature(full)
        selected = torch.tensor([24 + p - 1 for p in features.PSEUDO_POS_MATURE])
        torch.testing.assert_close(item["seq_feat"][:, 0], selected.float())
        torch.testing.assert_close(item["token_representation"][:, 0], selected.float())
        self.assertEqual(item["pseudo_pos_mature"].reshape(-1).tolist(), list(features.PSEUDO_POS_MATURE))
        self.assertEqual(item["edge_index"].tolist(), [list(range(34))] * 2)
        provider._sequence_features.assert_called_once_with(full, "chain")
        self.assertEqual(source, "live_esm")
        self.assertEqual(provider.protein_feature(full)[2], "memory_cache")
        self.assertEqual(provider._sequence_features.call_count, 1)

    def test_hla_dr_uses_separate_full_chains_and_reuses_dra_cache(self):
        provider = self.provider(mhc_class="II")
        provider._sequence_features = Mock(side_effect=self.synthetic)
        dra, drb = (features.REFERENCE_SEQUENCES[key] for key in ("DRA", "DRB1"))
        _, item, _ = provider.protein_feature(drb, dra_sequence=dra)
        self.assertEqual([call.args[0] for call in provider._sequence_features.call_args_list], [dra, drb])
        self.assertEqual(item["node_pos"].reshape(-1).tolist(), list(range(34)))
        self.assertNotIn("pseudo_pos_mature", item)
        self.assertNotIn("full_node_pos", item)
        torch.testing.assert_close(item["seq_feat"][:15, 0],
                                   torch.tensor([25 + p - 1 for p in features.DRA_PSEUDO_POS_MATURE]).float())
        torch.testing.assert_close(item["seq_feat"][15:, 0],
                                   torch.tensor([29 + p - 1 for p in features.DRB1_PSEUDO_POS_MATURE]).float())
        for source, target in item["edge_index"].t().tolist():
            self.assertEqual(source < 15, target < 15)
        provider.protein_feature(drb + "A", dra_sequence=dra)
        self.assertEqual(provider._sequence_features.call_count, 3)

    def test_cache_is_bounded_lru_and_can_be_disabled(self):
        provider = self.provider(max_cached_sequences=2)
        provider._sequence_features = Mock(side_effect=self.synthetic)
        for sequence in ("AAA", "CCC", "AAA", "DDD"):
            provider.peptide_feature(sequence)
        self.assertEqual(list(provider._peptide_cache), ["AAA", "DDD"])
        self.assertEqual(provider._sequence_features.call_count, 3)
        provider = self.provider(max_cached_sequences=0)
        provider._sequence_features = Mock(side_effect=self.synthetic)
        provider.peptide_feature("AAA")
        provider.peptide_feature("AAA")
        self.assertEqual(len(provider._peptide_cache), 0)
        self.assertEqual(provider._sequence_features.call_count, 2)

    def test_invalid_sequence_and_missing_second_chain_do_not_trigger_esm(self):
        provider = self.provider()
        provider._sequence_features = Mock()
        for sequence in ("A" * 34, "HLA-A*02:01", "A" * 350):
            with self.assertRaises(ValueError):
                provider.protein_feature(sequence)
        provider._sequence_features.assert_not_called()
        with self.assertRaises(ValueError):
            self.provider(mhc_class="II").protein_feature(features.REFERENCE_SEQUENCES["DRB1"])

    def test_missing_or_empty_local_weights_are_rejected_without_importing_esm(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = features.GTpMHCFeatures({}, directory, "cpu")
            with patch.object(features.importlib, "import_module") as imports:
                with self.assertRaisesRegex(RuntimeError, "esm2_t33_650M_UR50D.pt"):
                    provider.prepare()
            imports.assert_not_called()
            (Path(directory) / "assets").mkdir()
            (Path(directory) / "assets/esm2_t33_650M_UR50D.pt").touch()
            with self.assertRaises(RuntimeError):
                provider.prepare()

    def test_loader_reads_both_local_checkpoints_with_core_loader_only(self):
        with tempfile.TemporaryDirectory() as directory:
            main, contacts = Path(directory) / "main.pt", Path(directory) / "contacts.pt"
            torch.save({"main": 1}, main)
            torch.save({"head": 2}, contacts)
            model = Mock()
            model.eval.return_value = model
            model.to.return_value = model
            alphabet = Mock()
            core = Mock(return_value=(model, alphabet))
            esm = SimpleNamespace(pretrained=SimpleNamespace(load_model_and_alphabet_core=core))
            provider = features.GTpMHCFeatures({"esm_checkpoint": str(main),
                                               "esm_contact_checkpoint": str(contacts)}, directory, "cpu")
            with patch.dict(sys.modules, {"esm": esm}):
                self.assertIs(provider.prepare(), provider)
                provider.prepare()
            core.assert_called_once_with("esm2_t33_650M_UR50D", {"main": 1}, {"head": 2})
            alphabet.get_batch_converter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
