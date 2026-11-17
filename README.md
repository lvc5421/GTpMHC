# GTpMHC

Peptide binding to major histocompatibility complex (MHC) molecules is a prerequisite for antigen presentation and subsequent T-cell recognition. Accurate prediction of peptide binding to MHC molecules is important for studying antigen presentation and designing vaccines. Existing methods capture sequence features and peptide–MHC interactions, but how individual MHC residues contribute jointly to peptide recognition remains difficult to model. Learning residue-group patterns from MHC sequence context and relating these patterns to different peptides may improve the prediction of binding specificity across alleles.

Here, we present GTpMHC, a graph–Transformer framework trained and evaluated separately on MHC class I and class II datasets. GTpMHC extracts contextualized residue representations at pseudo-sequence positions from protein language model encodings of full MHC sequences. Transformer encoders incorporate peptide–MHC sequence context, and graph clustering organizes MHC residues into groups for interaction modeling with peptides. The resulting residue representations are combined through joint self-attention to predict binding.

 The GTpMHC online prediction webserver and experimental datasets are available at http://bliulab.net/GTpMHC/document


<img width="8000" height="4500" alt="pmhc0911" src="https://github.com/user-attachments/assets/52d68a7c-5fdb-4dae-8127-bb06ff74c94e" />

Figure. 1 The framework of GTpMHC. (A) Feature Extraction and Graph Construction, (B) Interaction Modeling and Prediction, (C) Downstream Experiments

## 1. Install

```bash
conda env create -f GTpMHC.yml
```

### 2. Download Pre-trained Models

Create a `weights` directory and download the pre-trained GTpMHC checkpoints along with the ESM-2 backbone weights:

```bash
mkdir -p weights

# GTpMHC class I
curl -fL --retry 3 \
  -o weights/model_I.pth \
  https://github.com/lvc5421/GTpMHC/releases/download/v1.0.0/model_I.pth

# GTpMHC class II
curl -fL --retry 3 \
  -o weights/model_II.pth \
  https://github.com/lvc5421/GTpMHC/releases/download/v1.0.0/model_II.pth

# ESM-2 sequence model
curl -fL --retry 3 \
  -o weights/esm2_t33_650M_UR50D.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

# ESM-2 contact prediction head
curl -fL --retry 3 \
  -o weights/esm2_t33_650M_UR50D-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

### 3. Usage

**MHC Class I** (requires the full-length MHC heavy chain sequence and a peptide):

```bash
python app.py predict --mhc-class I \
  --config config.json \
  --protein-fasta examples/MHC_I.fasta \
  --peptide AAADIVNFL --threshold 0.5
```

**MHC Class II** (requires both $\alpha$ and $\beta$ chain sequences in the FASTA file, along with a peptide):

```bash
python app.py predict --mhc-class II \
  --config config.json \
  --protein-fasta examples/MHC_II.fasta \
  --peptide DAGQIILQEAVPVKRGD --threshold 0.5
```
