"""GTpMHC sequence prediction, checkpoint checks and HTTP service."""

import argparse
import base64
from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlsplit

from features import normalize_sequence


BASE_DIR = Path(__file__).resolve().parent
MAX_BODY_BYTES = 16 * 1024
LOGGER = logging.getLogger("GTpMHC")


def validate_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Threshold must be a number between 0 and 1.")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Threshold must be a finite number between 0 and 1.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1.")
    return float(value)


def parse_hla_dr_fasta(text, max_protein_length=2000):
    """Read one DRA and one DRB1 record, identified by header rather than order."""
    if not isinstance(text, str) or len(text) > max_protein_length * 4 + 2000:
        raise ValueError("HLA-DR dual-chain FASTA is missing or too long.")
    records = {}
    chain = None
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip().split()
            chain = header[0].upper() if header else ""
            if chain not in ("DRA", "DRB1"):
                raise ValueError("Class II FASTA headers must be >DRA and >DRB1; an optional space and description may follow the name.")
            if chain in records:
                raise ValueError("Duplicate record in Class II FASTA: " + chain)
            records[chain] = []
        elif chain is None:
            raise ValueError("A >DRA or >DRB1 header is required before FASTA sequence lines.")
        else:
            records[chain].append(line)
    if set(records) != {"DRA", "DRB1"}:
        raise ValueError("Class II FASTA must contain exactly one >DRA record and one >DRB1 record.")
    sequences = {name: normalize_sequence("\n".join(lines), name) for name, lines in records.items()}
    for name, sequence in sequences.items():
        if not 100 <= len(sequence) <= max_protein_length:
            raise ValueError("{} length must be between 100 and {}.".format(name, max_protein_length))
    return {"protein": sequences["DRB1"], "dra_sequence": sequences["DRA"]}


def normalize_request(payload, max_pep_len, mhc_class, max_protein_length=2000):
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object.")
    allowed = {"protein", "peptide", "threshold", "protein_kind", "sequence_offset",
               "pseudo_sequence", "dra_sequence", "dra_sequence_offset"}
    if set(payload) - allowed:
        raise ValueError("Unsupported fields: " + ", ".join(sorted(set(payload) - allowed)))
    if mhc_class == "II" and isinstance(payload.get("protein"), str):
        protein = payload["protein"]
        header_count = sum(line.lstrip().startswith(">") for line in protein.lstrip("\ufeff").splitlines())
        if header_count > 1 or (header_count and "dra_sequence" not in payload):
            if "dra_sequence" in payload:
                raise ValueError("Dual-chain FASTA already includes DRA; do not also provide dra_sequence or --dra-* arguments.")
            payload = {**payload, **parse_hla_dr_fasta(protein, max_protein_length)}
    result = {"threshold": validate_threshold(payload.get("threshold", 0.5))}
    for field in ("protein", "peptide"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) > max_protein_length * 2 + 1000:
            raise ValueError(field + " is missing or too long.")
        result[field] = normalize_sequence(value, field)
    if not 1 <= len(result["peptide"]) <= max_pep_len:
        raise ValueError("Peptide length must be between 1 and {}.".format(max_pep_len))
    if not (170 if mhc_class == "I" else 100) <= len(result["protein"]) <= max_protein_length:
        raise ValueError("Please provide the complete corresponding HLA protein chain (maximum length: {}).".format(max_protein_length))
    result["protein_kind"] = payload.get("protein_kind", "auto")
    if result["protein_kind"] not in ("auto", "full_sequence"):
        raise ValueError("protein_kind must be 'auto' or 'full_sequence'.")
    for field in ("sequence_offset", "dra_sequence_offset"):
        value = payload.get(field)
        if value is not None and (type(value) is not int or not -1 <= value <= 60):
            raise ValueError(field + " must be an integer between -1 and 60.")
        result[field] = value
    pseudo = payload.get("pseudo_sequence")
    result["pseudo_sequence"] = normalize_sequence(pseudo, "pseudo_sequence") if pseudo is not None else None
    if result["pseudo_sequence"] is not None and len(result["pseudo_sequence"]) != 34:
        raise ValueError("pseudo_sequence must contain 34 residues.")
    if mhc_class == "II":
        dra = payload.get("dra_sequence")
        if not isinstance(dra, str) or len(dra) > max_protein_length * 2 + 1000:
            raise ValueError("Class II requires a dual-chain FASTA containing >DRA and >DRB1, or a separate complete dra_sequence.")
        result["dra_sequence"] = normalize_sequence(dra, "DRA")
        if not 100 <= len(result["dra_sequence"]) <= max_protein_length:
            raise ValueError("DRA length is out of the allowed range.")
    elif "dra_sequence" in payload or "dra_sequence_offset" in payload:
        raise ValueError("DRA parameters are only supported for Class II HLA-DR.")
    return result


class GTpMHCEngine:
    """One trained model and bounded sequence caches per process."""

    def __init__(self, config_path=None, mhc_class="I", base_dir=BASE_DIR):
        self.base_dir = Path(base_dir)
        self.mhc_class = mhc_class
        if mhc_class not in ("I", "II"):
            raise ValueError("mhc_class must be 'I' or 'II'.")
        path = Path(config_path) if config_path else self.base_dir / "config.json"
        try:
            self.settings = json.loads(path.read_text(encoding="utf-8-sig"))
            profile = self.settings["models"][mhc_class]
            self.kwargs = profile["kwargs"]
            self.settings["model_checkpoint"] = profile["checkpoint"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Failed to read GTpMHC configuration and specified MHC class: {}".format(exc)) from exc
        self.settings["mhc_class"] = mhc_class
        self.settings["threshold"] = validate_threshold(self.settings.get("threshold", 0.5))
        for key in ("model_checkpoint", "esm_checkpoint", "esm_contact_checkpoint"):
            if not isinstance(self.settings.get(key), str) or not self.settings[key].strip():
                raise ValueError("Missing path in configuration: " + key)
        if not isinstance(self.kwargs, dict):
            raise ValueError("models.<class>.kwargs must be a model parameter object.")
        self.max_pep_len = self.kwargs.get("max_pep_len", 25)
        if type(self.max_pep_len) is not int or self.max_pep_len < 1:
            raise ValueError("max_pep_len must be a positive integer.")
        for key, default in (("max_protein_length", 2000), ("max_cached_sequences", 256)):
            if type(self.settings.get(key, default)) is not int or self.settings.get(key, default) < 1:
                raise ValueError(key + " must be a positive integer.")
        self.model = self.features = None
        self.ready = False
        self.load_error = None
        self.device = self.settings.get("device", "auto")
        self._lock = threading.Lock()

    def path(self, key):
        path = Path(self.settings[key]).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    def artifacts(self):
        result = []
        for key in ("model_checkpoint", "esm_checkpoint", "esm_contact_checkpoint"):
            path = self.path(key)
            result.append({"name": key, "path": self.settings[key], "exists": path.is_file(),
                           "size_bytes": path.stat().st_size if path.is_file() else 0, "required": True})
        return result

    def status(self):
        artifacts = self.artifacts()
        missing = [item["path"] for item in artifacts if not item["size_bytes"]]
        warnings = (["Missing local checkpoint(s): " + ", ".join(missing)] if missing else [])
        warnings.append("Scores are not probability-calibrated; thresholds should be determined by independent validation.")
        return {"model_name": "GTpMHC", "mhc_class": self.mhc_class, "ready": self.ready,
                "input_mode": "sequence_only", "feature_mode": "sequence",
                "max_pep_len": self.max_pep_len, "threshold": self.settings["threshold"],
                "device": str(self.device), "artifacts": artifacts, "warnings": warnings,
                "message": "Model loaded and ready for prediction." if self.ready else self.load_error or "Waiting to load local model."}

    def _load_model(self):
        if not self.path("model_checkpoint").is_file() or not self.path("model_checkpoint").stat().st_size:
            raise RuntimeError("Please place the trained local model checkpoint at: " + str(self.path("model_checkpoint")))
        import torch
        from gtpmhc import GTpMHC, load_gtpmhc_state

        selected = self.settings.get("device", "auto")
        selected = ("cuda:0" if torch.cuda.is_available() else "cpu") if selected == "auto" else selected
        device = torch.device(selected)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this environment; please set device to 'cpu'.")
        model = GTpMHC(**self.kwargs)
        # Administrator-managed local artifacts; HTTP requests cannot supply files.
        state = torch.load(self.path("model_checkpoint"), map_location="cpu", weights_only=False)
        report = load_gtpmhc_state(model, state)
        self.model = model.to(device).eval()
        self.device = device
        return report

    def check(self):
        """Check model weights without loading ESM or predicting a binding score."""
        with self._lock:
            report = self._load_model()
            return {"compatible": True, "model_name": "GTpMHC", "mhc_class": self.mhc_class,
                    "scope": "model checkpoint only", "checkpoint": str(self.path("model_checkpoint")),
                    "parameter_count": sum(p.numel() for p in self.model.parameters()),
                    "checkpoint_report": report}

    def load(self):
        with self._lock:
            if self.ready:
                return
            try:
                self._load_model()
                from features import GTpMHCFeatures
                self.features = GTpMHCFeatures(self.settings, self.base_dir, str(self.device),
                                              max_hla_position=self.kwargs.get("max_hla_position", 256))
                self.features.prepare()
                self.ready = True
                self.load_error = None
            except Exception as exc:
                self.model = self.features = None
                self.ready = False
                self.load_error = str(exc)
                raise RuntimeError(self.load_error) from exc

    def predict(self, payload):
        if not self.ready:
            raise RuntimeError(self.load_error or "Model is not ready yet; please check /api/status.")
        if isinstance(payload, dict):
            payload = {"threshold": self.settings["threshold"], **payload}
        request = normalize_request(payload, self.max_pep_len, self.mhc_class,
                                    self.settings.get("max_protein_length", 2000))
        with self._lock:
            return self._predict(request)

    def _predict(self, request):
        import torch

        start = time.monotonic()
        options = {key: request[key] for key in ("sequence_offset", "pseudo_sequence")}
        if self.mhc_class == "II":
            options.update(dra_sequence=request["dra_sequence"], dra_sequence_offset=request["dra_sequence_offset"])
        _, protein, protein_source = self.features.protein_feature(
            request["protein"], kind=request["protein_kind"], **options)
        peptide, peptide_source = self.features.peptide_feature(request["peptide"])
        if len(protein["seq"]) != 34:
            raise RuntimeError("HLA features must contain 34 positions.")
        positions = next((protein[key] for key in ("pseudo_pos_mature", "full_node_pos", "node_pos")
                          if protein.get(key) is not None), torch.arange(34))
        tensors = {"x_pep": peptide["x_pep"], "x_pretrain_pep": peptide["x_pretrain_pep"],
                   "x_p": protein["seq_feat"], "x_pretrain_p": protein["token_representation"],
                   "peptide_edge_index": peptide["peptide_edge_index"], "residue_edge_index": protein["edge_index"],
                   "peptide_edge_weight": peptide["edge_weight"], "residue_edge_weight": protein["edge_weight"],
                   "prot_node_pos": torch.as_tensor(positions).reshape(-1, 1)}
        for key, value in tensors.items():
            dtype = torch.long if key.endswith("edge_index") or key == "prot_node_pos" else torch.float32
            tensors[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        with torch.inference_mode():
            logit = self.model(**tensors)
            if logit.numel() != 1:
                raise RuntimeError("Model output count does not match the number of input pairs.")
            score = float(torch.sigmoid(logit).reshape(-1)[0].cpu())
        if not math.isfinite(score):
            raise RuntimeError("Model returned a non-finite score; please check the checkpoints and input.")
        binder = score >= request["threshold"]
        result = {"model_name": "GTpMHC", "mhc_class": self.mhc_class,
                  "protein": request["protein"], "peptide": request["peptide"],
                  "score": score, "threshold": request["threshold"], "is_binder": binder,
                  "prediction": "Predicted binder" if binder else "Predicted non-binder",
                  "feature_sources": {"protein": protein_source, "peptide": peptide_source},
                  "protein_mapping": protein.get("hla_mapping", {}),
                  "elapsed_seconds": round(time.monotonic() - start, 3)}
        if self.mhc_class == "II":
            result["dra_sequence"] = request["dra_sequence"]
        return result


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def make_handler(engine, index_path=None):
    index_path = Path(index_path) if index_path else BASE_DIR / "index.html"
    page = index_path.read_bytes() if index_path.is_file() else None
    hashes = {}
    for tag in ("style", "script"):
        blocks = re.findall(rb"<" + tag.encode() + rb"\b[^>]*>(.*?)</" + tag.encode() + rb">", page or b"", re.S)
        hashes[tag] = " ".join("'sha256-" + base64.b64encode(hashlib.sha256(
            block.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).digest()).decode() + "'" for block in blocks)
    csp = ("default-src 'none'; style-src " + (hashes["style"] or "'none'") +
           "; script-src " + (hashes["script"] or "'none'") +
           "; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")

    class GTpMHCHandler(BaseHTTPRequestHandler):
        server_version = "GTpMHC"

        def send_payload(self, status, data, content_type="application/json; charset=utf-8"):
            body = json_bytes(data) if content_type.startswith("application/json") else data
            self.send_response(status)
            for key, value in (("Content-Type", content_type), ("Content-Length", str(len(body))),
                               ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                               ("Content-Security-Policy", csp)):
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                self.send_payload(200, engine.status())
            elif path in ("/", "/index.html") and page is not None:
                self.send_payload(200, page, "text/html; charset=utf-8")
            else:
                self.send_payload(404, {"detail": "Endpoint or page not found."})

        def discard_small_body(self):
            # Unread sent bytes can reset a Windows socket and hide the error response.
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if 0 < size <= MAX_BODY_BYTES:
                    self.rfile.read(size)
            except ValueError:
                pass

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                self.discard_small_body()
                self.send_payload(404, {"detail": "Endpoint not found."})
                return
            if self.headers.get_content_type() != "application/json":
                self.discard_small_body()
                self.send_payload(415, {"detail": "Content-Type must be application/json."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    self.send_payload(413, {"detail": "Request body is empty or exceeds 16KB."})
                    return
                payload = json.loads(self.rfile.read(size).decode("utf-8"),
                                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (ValueError, UnicodeError) as exc:
                self.send_payload(400, {"detail": "Invalid JSON: " + str(exc)})
                return
            try:
                self.send_payload(200, engine.predict(payload))
            except (ValueError, KeyError) as exc:
                self.send_payload(422, {"detail": str(exc)})
            except RuntimeError as exc:
                self.send_payload(503, {"detail": str(exc)})
            except Exception:
                LOGGER.exception("GTpMHC prediction failed")
                self.send_payload(500, {"detail": "Prediction failed; please check the server logs."})

        def log_message(self, fmt, *args):
            LOGGER.info("%s - %s", self.client_address[0], fmt % args)

    return GTpMHCHandler


def create_server(engine, host="127.0.0.1", port=8000, index_path=None):
    server = ThreadingHTTPServer((host, port), make_handler(engine, index_path))
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "predict", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="Configuration file; relative checkpoint paths are resolved against the project directory.")
        command.add_argument("--mhc-class", choices=("I", "II"), default="I")
    serve = commands.choices["serve"]
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    predict = commands.choices["predict"]
    protein = predict.add_mutually_exclusive_group(required=True)
    protein.add_argument("--protein", help="Complete HLA-I sequence, or Class II dual-chain FASTA text containing >DRA and >DRB1.")
    protein.add_argument("--protein-fasta", type=Path,
                         help="Class I: single-chain FASTA; Class II: a single file containing both >DRA and >DRB1 records.")
    dra = predict.add_mutually_exclusive_group()
    dra.add_argument("--dra-sequence", help="Legacy dual-field input compatibility; omit when using dual-chain FASTA.")
    dra.add_argument("--dra-fasta", type=Path, help="Legacy dual-file input compatibility; omit when using dual-chain FASTA.")
    predict.add_argument("--peptide", required=True)
    predict.add_argument("--threshold", type=float)
    predict.add_argument("--sequence-offset", type=int)
    predict.add_argument("--dra-sequence-offset", type=int)
    predict.add_argument("--pseudo-sequence")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        engine = GTpMHCEngine(args.config, args.mhc_class)
        if args.command == "check":
            with redirect_stdout(sys.stderr):
                report = engine.check()
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "predict":
            payload = {"protein": args.protein_fasta.read_text(encoding="utf-8-sig") if args.protein_fasta else args.protein,
                       "peptide": args.peptide}
            for key in ("threshold", "sequence_offset", "dra_sequence_offset", "pseudo_sequence"):
                if getattr(args, key) is not None:
                    payload[key] = getattr(args, key)
            if args.dra_sequence is not None or args.dra_fasta is not None:
                payload["dra_sequence"] = args.dra_fasta.read_text(encoding="utf-8-sig") if args.dra_fasta else args.dra_sequence
            # Fail on malformed input before loading large ESM/model checkpoints.
            normalize_request(payload, engine.max_pep_len, args.mhc_class,
                              engine.settings.get("max_protein_length", 2000))
            with redirect_stdout(sys.stderr):
                engine.load()
                result = engine.predict(payload)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        try:
            engine.load()
        except RuntimeError as exc:
            LOGGER.warning("Model unavailable: %s", exc)
        server = create_server(engine, args.host, args.port)
        print("GTpMHC: http://{}:{}".format(args.host, server.server_address[1]), flush=True)
        try:
            server.serve_forever(poll_interval=0.3)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (ValueError, RuntimeError, OSError, TypeError, ImportError) as exc:
        print(json.dumps({"detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
