# GTpMHC

Peptide binding to major histocompatibility complex (MHC) molecules is a prerequisite for antigen presentation and subsequent T-cell recognition. Accurate prediction of peptide binding to MHC molecules is important for studying antigen presentation and designing vaccines. Existing methods capture sequence features and peptide–MHC interactions, but how individual MHC residues contribute jointly to peptide recognition remains difficult to model. Learning residue-group patterns from MHC sequence context and relating these patterns to different peptides may improve the prediction of binding specificity across alleles.

Here, we present GTpMHC, a graph–Transformer framework trained and evaluated separately on MHC class I and class II datasets. GTpMHC extracts contextualized residue representations at pseudo-sequence positions from protein language model encodings of full MHC sequences. Transformer encoders incorporate peptide–MHC sequence context, and graph clustering organizes MHC residues into groups for interaction modeling with peptides. The resulting residue representations are combined through joint self-attention to predict binding.

 The GTpMHC online prediction webserver and experimental datasets are available at http://bliulab.net/GTpMHC/document

## Files

| File | Purpose |
|---|---|
| `gtpmhc.py` | Standalone `GTpMHC` model and audited checkpoint loader; no project-specific parent model |
| `features.py` | Raw33/ESM features, contact graph, HLA numbering and bounded in-memory caches |
| `app.py` | Configuration, input validation, prediction engine, HTTP server and CLI |
| `index.html` | Complete web interface with inline CSS and JavaScript; no CDN |
| `config.json` | Class I / II checkpoint paths and inference architecture parameters |

Training, MAE reconstruction, auxiliary prediction heads, SupCon, offline preprocessing, precomputed feature dictionaries, dataset loaders and unused inherited implementations are omitted. The original peptide-first model order is retained. This release does not use the separate HLA-first model variant.

## Install

Use Python 3.8–3.11; Python 3.10 is a practical choice for the pinned original deployment dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An existing environment that runs the original webserver can also be used. For GPU deployment, install a PyTorch build compatible with your CUDA environment. Extra `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv` packages are not required by this edge-index inference path.

## Local weights and model settings

Create a `weights` directory and place these trusted local artifacts in it:

```text
weights/
  gtpmhc_i.pth                                  # trained class I checkpoint
  gtpmhc_ii.pth                                 # trained HLA-DR checkpoint
  esm2_t33_650M_UR50D.pt
  esm2_t33_650M_UR50D-contact-regression.pt
```

Only the GTpMHC checkpoint for the selected class is needed. Both ESM files are required for sequence prediction. Use existing ESM files from your server or obtain the official checkpoints from the [ESM project](https://github.com/facebookresearch/esm).

Edit `config.json` to match the actual training architecture. Its class I example uses the reconstructed TranspMHC settings discussed for the original webserver; the class II values are a template and must be checked against the actual class II run. In particular, `max_pep_len`, hidden width, GAT heads and cluster sizes must match the checkpoint. The template is not evidence of a trained class II model.

Paths inside `config.json` are relative to this repository directory, even when another configuration file is selected. Absolute paths are also accepted. Historical training paths and training-only options are not required.

The checkpoint loader accepts native compact GTpMHC checkpoints and the original peptide-first fulltoken checkpoints. Only explicitly identified, unused training/auxiliary parameters may be discarded. Missing required parameters, unknown extra parameters and mismatched tensor shapes are errors. HLA-first checkpoints are not compatible with this release. Removing auxiliary parameters does not require retraining because the binding computation is preserved; numerical comparison details are below.

When migrating an existing training configuration, `inference_model_kwargs(old_config["model_kwargs"])` from `gtpmhc.py` selects the supported inference parameters and rejects unknown options. Put the resulting dictionary in `models.I.kwargs` or `models.II.kwargs`; the original long module name is no longer needed.

Check model weights and architecture without loading ESM:

```bash
python app.py check --mhc-class I
python app.py check --mhc-class II
```

## Web interface

```bash
python app.py serve --mhc-class I --host 127.0.0.1 --port 8000
# HLA-DR service:
python app.py serve --mhc-class II --host 127.0.0.1 --port 8001
```

Open the displayed address. If weights are missing, the page and `/api/status` explain what is required; the service does not fabricate predictions. Restart after replacing weights or changing configuration.

For a remote web deployment, run this application behind the server's reverse proxy and set the listening address as appropriate. The included HTTP server is intended as the application backend and does not provide TLS or authentication.

## Command line

Class I, using a full protein sequence and peptide:

```bash
python app.py predict --mhc-class I \
  --protein-fasta /path/to/hla.fasta \
  --peptide AAADIVNFL --threshold 0.5
```

Class II HLA-DR requires both chains:

```bash
python app.py predict --mhc-class II \
  --protein-fasta /path/to/drb1.fasta \
  --dra-fasta /path/to/dra.fasta \
  --peptide DAGQIILQEAVPVKRGD --threshold 0.5
```

Direct sequence arguments are `--protein "$PROTEIN"` and `--dra-sequence "$DRA"`. In class II mode, `--protein` is DRB1. The 17-residue example requires a checkpoint trained with `max_pep_len >= 17`; changing the setting cannot increase a trained checkpoint's capacity. Optional `--sequence-offset`, `--dra-sequence-offset` and `--pseudo-sequence` check mature-chain mapping. The CLI prints result JSON to stdout and diagnostics to stderr.

Use `--config /path/to/config.json` after any subcommand to select another configuration.

## API

`GET /api/status` returns readiness, the selected class and local artifact status.

`POST /api/predict` accepts `Content-Type: application/json`:

```json
{
  "protein": "FULL_HLA_OR_DRB1_SEQUENCE",
  "peptide": "AAADIVNFL",
  "threshold": 0.5
}
```

For HLA-DR, also provide `dra_sequence`. The server class is selected when starting the process; requests cannot change the loaded model or provide server file paths. Results include `model_name: "GTpMHC"`, `score`, `threshold`, `is_binder`, `prediction`, mapping details and feature sources.

A score at or above the threshold is classified as binding. Scores are not IC50 values or calibrated binding probabilities; select the decision threshold using independent validation data.

## Features

Features are generated from the complete input chains before selecting 34 HLA positions. Class I selects 34 mature-chain positions; HLA-DR selects 15 DRA and 19 DRB1 positions. The two DR chains have separate contact graphs. Raw33 channel order, ESM layer 33 embeddings and contact graph construction match the original webserver. The contact-regression checkpoint supplies the ESM contact prediction head.

The fixed public numbering references and their sources are embedded in `features.py`. Internal insertions/deletions or ambiguous mapping are rejected. Mapping is not an allele identification service. Long chains use the original 600-residue windows with stride 200, so contact edges across separate windows are not available. Feature caches remain in memory and are bounded by `max_cached_sequences`.

## Tests and GitHub upload

```bash
python -m unittest discover -s tests
node tests/test_ui.js
```

Some numerical tests require PyTorch/PyG. ESM feature tests use deterministic stubs where stated; they do not establish prediction accuracy. Two optional comparisons against the original model can be enabled with `GTPMHC_LEGACY_DIR=/path/to/pmhc_webserver_test` when running the Python suite. Ordinary release tests do not depend on that original directory.

Upload this directory as the repository root. `.gitignore` excludes local weights, generated caches, logs and archives. No pretrained model, private server path, training data or generated result is included. This task only prepares repository files; it does not create or push a remote GitHub repository. No project license has been assigned automatically.

## Validation (2026-11-16)

- Full Python suite: 51 tests passed, including the two original-model comparisons; frontend suite: 18 tests passed. HTTP tests used a fixture engine and actual local loopback requests.
- Original versus compact model: final CPU logits were exactly equal for 9-, 17-, and 25-residue peptides, a mixed-length batch, and a width-128 model with two Transformer layers. Both mature HLA positions and the class II 0–33 position convention were exercised. These comparisons used synthetic graphs and random reference weights.
- The loader requires every retained tensor. It may discard only the 76 individually identified unused legacy keys, with shape checks even for those discarded tensors; it never uses `strict=False`.
- Raw33 channels matched the original 21-residue feature matrix exactly; 12 contact graph cases matched edges, ordering and weights. Short and 1100-residue window extraction matched original helper outputs using deterministic ESM stubs.
- Engine tests exercised a real compact checkpoint and direct graph tensors for both classes; checkpoint checks did not load ESM. UI/API, FASTA validation, class II fields and missing-weight behavior were checked.
- Runtime Python files were reduced from 12 to 3. The original 42 text/source files were checked with SHA-256 and were not modified.

The numerical test environment was existing Python 3.13.5, PyTorch 2.14.0+cpu and PyG 2.8.0.post1; no environment was installed or changed. Python sources also passed Python 3.8 grammar checks, and CLI help/missing-weight diagnostics worked without site packages. The pinned original deployment environment was not reinstalled for this task. Real trained checkpoints, real ESM extraction, CUDA equivalence, biological prediction accuracy and speed improvements have not been validated here.
