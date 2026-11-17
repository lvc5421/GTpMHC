"""Input validation and actual compact-model engine integration; no ESM downloads."""

import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import sys
import re
import threading
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app


class InputTests(unittest.TestCase):
    def request(self, mhc_class="I", **changes):
        payload = {"protein": "A" * 180, "peptide": "AAADIVNFL"}
        if mhc_class == "II":
            payload["dra_sequence"] = "A" * 200
        payload.update(changes)
        return app.normalize_request(payload, 25, mhc_class)

    def test_fasta_and_whitespace_normalize(self):
        item = self.request(protein=">HLA\n" + "a" * 180 + "\n", peptide=" aaa divnfl ")
        self.assertEqual(item["protein"], "A" * 180)
        self.assertEqual(item["peptide"], "AAADIVNFL")

    def test_threshold_boundaries_and_rejections(self):
        for value in (0, 0.5, 1):
            self.assertEqual(self.request(threshold=value)["threshold"], value)
        for value in (True, "0.5", None, -1, 2, float("inf"), float("nan"), 10 ** 1000):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                self.request(threshold=value)

    def test_sequence_boundaries_and_invalid_characters(self):
        for changes in ({"peptide": "A" * 26}, {"peptide": ""}, {"peptide": "ABC"},
                        {"protein": "A" * 34}, {"protein": ">one\nAAAA\n>two\nAAAA"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.request(**changes)

    def test_class_ii_needs_dra_and_preserves_zero_offsets(self):
        item = self.request("II", sequence_offset=0, dra_sequence_offset=0)
        self.assertEqual(item["dra_sequence_offset"], 0)
        self.assertEqual(item["sequence_offset"], 0)
        for changes in ({"dra_sequence": None}, {"dra_sequence": "A" * 34}):
            with self.assertRaises(ValueError):
                self.request("II", **changes)
        with self.assertRaises(ValueError):
            self.request(dra_sequence="A" * 200)

    def test_server_files_and_model_selection_cannot_be_request_fields(self):
        for key in ("model_checkpoint", "protein_fasta", "mhc_class", "model_kwargs"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                app.normalize_request({"protein": "A" * 180, "peptide": "AAADIVNFL",
                                       key: "arbitrary"}, 25, "I")

    def test_offsets_and_pseudo_length(self):
        for value in (False, 1.5, -2, 61):
            with self.assertRaises(ValueError):
                self.request(sequence_offset=value)
        self.assertEqual(self.request(pseudo_sequence="A" * 34)["pseudo_sequence"], "A" * 34)
        with self.assertRaises(ValueError):
            self.request(pseudo_sequence="A" * 33)

    def test_missing_weights_report_unavailable_without_ml(self):
        engine = app.GTpMHCEngine()
        self.assertEqual(engine.status()["model_name"], "GTpMHC")
        self.assertFalse(engine.ready)
        with self.assertRaises(RuntimeError):
            engine.predict({})


class HlaDrFastaTests(unittest.TestCase):
    DRA = "A" * 200
    DRB1 = "C" * 180
    PEPTIDE = "DAGQIILQEAVPVKRGD"

    def fasta(self):
        return ">DRA\n" + self.DRA + "\n>DRB1\n" + self.DRB1 + "\n"

    def request(self, text=None, **options):
        return app.normalize_request({"protein": self.fasta() if text is None else text,
                                      "peptide": self.PEPTIDE, **options}, 25, "II")

    def test_example_has_both_full_chains(self):
        text = (ROOT / "examples/hla_dr_example.fasta").read_text(encoding="utf-8-sig")
        pair = app.parse_hla_dr_fasta(text)
        self.assertEqual(len(pair["dra_sequence"]), 254)
        self.assertEqual(len(pair["protein"]), 266)
        from features import map_hla_ii_sequences
        mapping = map_hla_ii_sequences(pair["dra_sequence"], pair["protein"])
        self.assertEqual(len(mapping["pseudo_sequence"]), 34)

    def test_record_order_case_descriptions_and_bom_do_not_change_chain_roles(self):
        text = "\ufeff\n>drb1 beta chain\n" + self.DRB1.lower() + "\n\n>dra alpha chain\n" + self.DRA.lower()
        self.assertEqual(app.parse_hla_dr_fasta(text), {"protein": self.DRB1, "dra_sequence": self.DRA})

    def test_combined_request_equals_old_input_and_does_not_mutate_payload(self):
        payload = {"protein": self.fasta(), "peptide": self.PEPTIDE,
                   "sequence_offset": 0, "dra_sequence_offset": 0, "threshold": 0.0}
        original = dict(payload)
        old = {**payload, "protein": self.DRB1, "dra_sequence": self.DRA}
        self.assertEqual(app.normalize_request(payload, 25, "II"), app.normalize_request(old, 25, "II"))
        self.assertEqual(payload, original)

    def test_missing_extra_duplicate_empty_or_unlabeled_chains_are_rejected(self):
        invalid = ("", self.DRA, ">DRA\n" + self.DRA, ">DRB1\n" + self.DRB1,
                   ">alpha\n" + self.DRA + "\n>beta\n" + self.DRB1,
                   self.fasta() + ">DRA\n" + self.DRA,
                   self.fasta() + ">OTHER\nAAA", ">DRA\n\n>DRB1\n" + self.DRB1,
                   self.DRA + "\n" + self.fasta(), ">\n" + self.DRA,
                   ">DRA>DRB1\n" + self.DRA, self.fasta().replace(self.DRA, "A" * 199 + "*"))
        for text in invalid:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                app.parse_hla_dr_fasta(text)

    def test_combined_file_and_separate_dra_cannot_be_mixed(self):
        for dra in (None, "", self.DRA):
            with self.subTest(dra=dra), self.assertRaisesRegex(ValueError, "双链 FASTA 已包含 DRA"):
                self.request(dra_sequence=dra)

    def test_each_chain_length_is_checked_independently(self):
        for chain in ("DRA", "DRB1"):
            for length in (99, 2001):
                pair = {"DRA": self.DRA, "DRB1": self.DRB1, chain: "A" * length}
                text = ">DRA\n" + pair["DRA"] + "\n>DRB1\n" + pair["DRB1"]
                with self.subTest(chain=chain, length=length), self.assertRaises(ValueError):
                    self.request(text)

    def test_combined_raw_size_allows_two_chains_but_is_bounded(self):
        text = ">DRA\n" + "a " * 1800 + "\n>DRB1\n" + "c " * 1800
        request = self.request(text)
        self.assertEqual(len(request["dra_sequence"]), 1800)
        self.assertEqual(len(request["protein"]), 1800)
        with self.assertRaisesRegex(ValueError, "输入过长"):
            self.request(self.fasta() + " " * 10000)

    def test_class_i_rejects_multi_record_fasta_and_preserves_single_record(self):
        with self.assertRaises(ValueError):
            app.normalize_request({"protein": self.fasta(), "peptide": self.PEPTIDE}, 25, "I")
        result = app.normalize_request({"protein": ">arbitrary description\n" + self.DRB1,
                                        "peptide": self.PEPTIDE}, 25, "I")
        self.assertEqual(result["protein"], self.DRB1)

    def cli_engine(self):
        class ParsedInputEngine:
            instances = []

            def __init__(self, config_path, mhc_class):
                self.mhc_class = mhc_class
                self.max_pep_len = 25
                self.settings = {}
                self.loaded = False
                self.instances.append(self)

            def load(self):
                self.loaded = True

            def predict(self, payload):
                return {"parsed_input": app.normalize_request(payload, 25, self.mhc_class)}

        return ParsedInputEngine

    def test_cli_reads_one_file_for_class_ii(self):
        engine = self.cli_engine()
        stdout = io.StringIO()
        with patch.object(app, "GTpMHCEngine", engine), redirect_stdout(stdout):
            code = app.main(["predict", "--mhc-class", "II", "--protein-fasta",
                             str(ROOT / "examples/hla_dr_example.fasta"),
                             "--peptide", self.PEPTIDE, "--threshold", "0.5"])
        self.assertEqual(code, 0)
        parsed = json.loads(stdout.getvalue())["parsed_input"]
        self.assertEqual(len(parsed["dra_sequence"]), 254)
        self.assertEqual(len(parsed["protein"]), 266)
        self.assertEqual(parsed["peptide"], self.PEPTIDE)
        self.assertTrue(engine.instances[0].loaded)

    def test_cli_preserves_class_i_and_legacy_separate_class_ii_files(self):
        with tempfile.TemporaryDirectory() as directory:
            protein = Path(directory) / "protein.fasta"
            dra = Path(directory) / "dra.fasta"
            protein.write_text(">DRB1\n" + self.DRB1, encoding="utf-8-sig")
            dra.write_text(">DRA\n" + self.DRA, encoding="utf-8")
            for mhc_class in ("I", "II"):
                engine = self.cli_engine()
                stdout = io.StringIO()
                args = ["predict", "--mhc-class", mhc_class, "--protein-fasta", str(protein),
                        "--peptide", self.PEPTIDE]
                if mhc_class == "II":
                    args += ["--dra-fasta", str(dra)]
                with patch.object(app, "GTpMHCEngine", engine), redirect_stdout(stdout):
                    code = app.main(args)
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout.getvalue())["parsed_input"]["protein"], self.DRB1)

    def test_bad_cli_fasta_is_rejected_before_loading_weights(self):
        engine = self.cli_engine()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_dra.fasta"
            path.write_text(">DRB1\n" + self.DRB1, encoding="utf-8")
            with patch.object(app, "GTpMHCEngine", engine), redirect_stderr(stderr):
                code = app.main(["predict", "--mhc-class", "II", "--protein-fasta", str(path),
                                 "--peptide", self.PEPTIDE])
        self.assertEqual(code, 1)
        self.assertFalse(engine.instances[0].loaded)
        self.assertIn(">DRA", json.loads(stderr.getvalue())["detail"])


ML = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"))


@unittest.skipUnless(ML, "PyTorch/PyG required for actual model integration")
class EngineTests(unittest.TestCase):
    def setUp(self):
        import torch
        from gtpmhc import GTpMHC
        self.torch = torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.kwargs = {"hidden_channels": 8, "K": [2], "heads": 2, "max_pep_len": 25,
                       "pep_evo_channels": 8, "prot_evo_channels": 8}
        torch.manual_seed(71)
        model = GTpMHC(**self.kwargs).eval()
        torch.save(model.state_dict(), self.directory / "model.pth")
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        config.update(device="cpu", threshold=0.0)
        for profile in config["models"].values():
            profile.update(checkpoint="model.pth", kwargs=self.kwargs)
        self.config = self.directory / "config.json"
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def test_checkpoint_check_does_not_import_esm_or_extract_features(self):
        engine = app.GTpMHCEngine(self.config, base_dir=self.directory)
        with patch("features.GTpMHCFeatures", side_effect=AssertionError("ESM must not load")):
            report = engine.check()
        self.assertTrue(report["compatible"])
        self.assertEqual(report["model_name"], "GTpMHC")
        self.assertFalse(engine.ready)

    def test_direct_tensor_engine_matches_model_and_chain_positions(self):
        torch = self.torch

        def graph(length):
            nodes = torch.arange(length)
            return torch.stack((nodes, nodes)), torch.ones(length)

        torch.manual_seed(94)
        p_edge, p_weight = graph(9)
        h_edge, h_weight = graph(34)
        peptide = {"x_pep": torch.randn(9, 33), "x_pretrain_pep": torch.randn(9, 8),
                   "peptide_edge_index": p_edge, "edge_weight": p_weight}
        protein = {"seq": "A" * 34, "seq_feat": torch.randn(34, 33),
                   "token_representation": torch.randn(34, 8), "edge_index": h_edge,
                   "edge_weight": h_weight, "node_pos": torch.arange(34),
                   "hla_mapping": {"test": "synthetic only"}}

        class SyntheticFeatures:
            def __init__(self, *args, **kwargs):
                pass

            def prepare(self):
                pass

            def protein_feature(self, full, **options):
                return full, protein, "test_synthetic"

            def peptide_feature(self, full):
                return peptide, "test_synthetic"

        for mhc_class in ("I", "II"):
            with self.subTest(mhc_class=mhc_class):
                if mhc_class == "I":
                    protein["pseudo_pos_mature"] = torch.arange(1, 35)
                else:
                    protein.pop("pseudo_pos_mature", None)
                engine = app.GTpMHCEngine(self.config, mhc_class, self.directory)
                with patch("features.GTpMHCFeatures", SyntheticFeatures):
                    engine.load()
                positions = protein.get("pseudo_pos_mature", protein["node_pos"])
                with torch.inference_mode():
                    expected = torch.sigmoid(engine.model(
                        x_pep=peptide["x_pep"], x_pretrain_pep=peptide["x_pretrain_pep"],
                        peptide_edge_index=p_edge, peptide_edge_weight=p_weight,
                        x_p=protein["seq_feat"], x_pretrain_p=protein["token_representation"],
                        residue_edge_index=h_edge, residue_edge_weight=h_weight,
                        prot_node_pos=positions.reshape(-1, 1))).item()
                payload = {"protein": "A" * 180, "peptide": "AAADIVNFL"}
                if mhc_class == "II":
                    payload["dra_sequence"] = "A" * 200
                result = engine.predict(payload)
                self.assertAlmostEqual(result["score"], expected, places=7)
                self.assertTrue(result["is_binder"])
                self.assertEqual(result["threshold"], 0.0)
                self.assertEqual(result["model_name"], "GTpMHC")
                self.assertEqual(result["mhc_class"], mhc_class)
                if mhc_class == "II":
                    combined = {key: value for key, value in payload.items() if key != "dra_sequence"}
                    combined["protein"] = ">DRA\n" + payload["dra_sequence"] + "\n>DRB1\n" + payload["protein"]
                    result_combined = engine.predict(combined)
                    self.assertEqual(result_combined["score"], result["score"])
                    self.assertEqual(result_combined["protein"], payload["protein"])
                    self.assertEqual(result_combined["dra_sequence"], payload["dra_sequence"])


APP = app


class FixtureEngine:
    """Only echoes validated JSON to check routing and error translation."""

    def __init__(self):
        self.ready = True
        self.mhc_class = "I"
        self.calls = []
        self.failure = None

    def status(self):
        return {"ready": self.ready, "model_name": "GTpMHC", "mhc_class": self.mhc_class,
                "max_pep_len": 25, "threshold": 0.5, "device": "transport-test-only",
                "artifacts": [], "warnings": [], "message": "HTTP fixture",
                "feature_mode": "sequence", "input_mode": "sequence_only"}

    def predict(self, payload):
        self.calls.append(payload)
        if self.failure is not None:
            raise self.failure
        if not self.ready:
            raise RuntimeError("HTTP fixture unavailable")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        if not payload.get("protein") or not payload.get("peptide"):
            raise ValueError("Protein and peptide are required")
        return {"transport_fixture": True, "received": payload}


class HttpContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = FixtureEngine()
        self.server = APP.create_server(self.engine, host="127.0.0.1", port=0,
                                        index_path=ROOT / "index.html")
        self.worker = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)

    def request(self, method, path, body=None, content_type=None, headers=None):
        request_headers = dict(headers or {})
        if content_type is not None:
            request_headers["Content-Type"] = content_type
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, payload, path="/api/predict"):
        return self.request("POST", path, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8")

    def assert_detail(self, status, response):
        actual, _, body = response
        self.assertEqual(actual, status)
        self.assertTrue(json.loads(body)["detail"])

    def test_single_html_is_served_at_both_public_page_paths(self):
        expected = (ROOT / "index.html").read_bytes()
        for path in ("/", "/index.html"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith("text/html"))
                self.assertEqual(body, expected)
                self.assertEqual(int(headers["Content-Length"]), len(expected))
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_inline_script_and_styles_are_authorized_by_exact_csp_hashes(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        csp = headers["Content-Security-Policy"]
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        html = body.decode("utf-8")
        for tag in ("script", "style"):
            blocks = re.findall(r"<" + tag + r"\b[^>]*>([\s\S]*?)</" + tag + ">", html)
            self.assertEqual(len(blocks), 1)
            digest = base64.b64encode(hashlib.sha256(blocks[0].encode("utf-8")).digest()).decode("ascii")
            self.assertIn("'sha256-" + digest + "'", csp)

    def test_status_reports_the_engine_without_inference(self):
        status, headers, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), self.engine.status())
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.engine.calls, [])

    def test_prediction_transports_sequences_and_zero_offsets_without_rewriting(self):
        payload = {"protein": ">protein\nACDEFGHIKLMNPQRSTVWY", "peptide": "GILGFVFTL",
                   "threshold": 0, "sequence_offset": 0, "protein_kind": "auto"}
        status, headers, body = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(json.loads(body), {"transport_fixture": True, "received": payload})
        self.assertEqual(self.engine.calls, [payload])

    def test_class_II_payload_keeps_both_chains_and_independent_offsets(self):
        self.engine.mhc_class = "II"
        payload = {"protein": "DRB1-sequence-fixture", "dra_sequence": "DRA-sequence-fixture",
                   "peptide": "PKYVKQNTLKLAT", "sequence_offset": 29,
                   "dra_sequence_offset": 0, "pseudo_sequence": "A" * 34}
        status, _, body = self.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["received"], payload)

    def test_class_ii_combined_fasta_is_normalized_through_http(self):
        self.engine.predict = lambda payload: {"parsed_input": APP.normalize_request(payload, 25, "II")}
        text = (ROOT / "examples/hla_dr_example.fasta").read_text(encoding="utf-8-sig")
        payload = {"protein": text, "peptide": "DAGQIILQEAVPVKRGD",
                   "sequence_offset": 29, "dra_sequence_offset": 25}
        status, _, body = self.post(payload)
        self.assertEqual(status, 200)
        parsed = json.loads(body)["parsed_input"]
        self.assertEqual(len(parsed["protein"]), 266)
        self.assertEqual(len(parsed["dra_sequence"]), 254)
        self.assertEqual(parsed["sequence_offset"], 29)
        self.assertEqual(parsed["dra_sequence_offset"], 25)
        self.assert_detail(422, self.post({**payload, "dra_sequence": parsed["dra_sequence"]}))

    def test_query_strings_cannot_select_files_models_or_replace_json_input(self):
        status, _, body = self.request("GET", "/?file=../../config.json")
        self.assertEqual(status, 200)
        self.assertEqual(body, (ROOT / "index.html").read_bytes())
        status, _, body = self.request("GET", "/api/status?config=private.json&mhc_class=II")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), self.engine.status())
        payload = {"protein": "sequence-fixture", "peptide": "ACDE"}
        status, _, body = self.post(payload, "/api/predict?model_path=private.pth&threshold=0")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["received"], payload)

    def test_invalid_json_unicode_and_nonfinite_literals_are_400(self):
        for body in (b"{", b"\xff", b'{"threshold":NaN}', b'{"threshold":Infinity}', b'{"threshold":-Infinity}'):
            with self.subTest(body=body):
                self.assert_detail(400, self.request("POST", "/api/predict", body, "application/json"))
        self.assertEqual(self.engine.calls, [])

    def test_nonobject_and_incomplete_payloads_are_422(self):
        for payload in (None, [], "sequence", 12, {}, {"protein": "sequence-fixture"}):
            with self.subTest(payload=payload):
                self.assert_detail(422, self.post(payload))

    def test_engine_validation_error_is_422(self):
        self.engine.failure = ValueError("Fixture validation error")
        self.assert_detail(422, self.post({"protein": "sequence", "peptide": "ACDE"}))

    def test_model_unavailable_is_503(self):
        self.engine.ready = False
        response = self.post({"protein": "sequence", "peptide": "ACDE"})
        self.assert_detail(503, response)
        self.assertIn("unavailable", json.loads(response[2])["detail"])

    def test_unexpected_engine_error_is_500_without_internal_exception_text(self):
        self.engine.failure = OSError("private/internal/weight-file.pth")
        response = self.post({"protein": "sequence", "peptide": "ACDE"})
        self.assert_detail(500, response)
        self.assertNotIn("private/internal", response[2].decode("utf-8"))

    def test_media_type_is_required(self):
        for content_type in (None, "text/plain", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=content_type):
                self.assert_detail(415, self.request("POST", "/api/predict", b"{}", content_type))
        self.assertEqual(self.engine.calls, [])

    def test_oversized_or_empty_bodies_are_rejected_before_inference(self):
        for body in (b"", b" " * (APP.MAX_BODY_BYTES + 1)):
            with self.subTest(length=len(body)):
                self.assert_detail(413, self.request("POST", "/api/predict", body, "application/json"))
        self.assertEqual(self.engine.calls, [])

    def test_malformed_content_length_is_400(self):
        self.assert_detail(400, self.request("POST", "/api/predict", b"{}", "application/json",
                                             headers={"Content-Length": "invalid"}))
        self.assertEqual(self.engine.calls, [])

    def test_other_routes_cannot_expose_release_files_or_traverse_directories(self):
        for path in ("/config.json", "/app.py", "/features.py", "/model.py", "/README.md",
                     "/assets/model.pth", "/models/last_model.pth", "/api/proteins",
                     "/../app.py", "/%2e%2e/app.py", "/static/../app.py", "/static/style.css"):
            with self.subTest(path=path):
                self.assert_detail(404, self.request("GET", path))
        self.assertEqual(self.engine.calls, [])

    def test_unknown_post_route_never_calls_the_engine(self):
        for path in ("/api/status", "/config.json", "/api/predict/extra"):
            with self.subTest(path=path):
                self.assert_detail(404, self.request("POST", path, b"{}", "application/json"))
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
