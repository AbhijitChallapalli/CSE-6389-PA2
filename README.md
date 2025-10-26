# GNN for Alzheimer’s Disease (AD) Classification using Brain Connectivity

## Overview

This repository implements a Graph Convolutional Network (GCN) to perform **binary classification** of subjects as **Alzheimer’s Disease (AD)** or **Cognitively Normal (CN)** using **functional connectivity (FC)** and **structural connectivity (SC)** matrices. The emphasis of this project is on **methodology, analysis, and experimental design** rather than raw accuracy.

- **SC (structural connectivity):** used as the **graph topology** (adjacency). We apply log1p compression, optional unit scaling, add self-loops, and perform **GCN symmetric normalization** \(\hat A = D^{-1/2}(A + I)D^{-1/2}\).
- **FC (functional connectivity):** used as **node features**. We optionally apply Fisher z-transform and then **standardize** either globally per subject or row-wise per node.
- **Model:** a compact, regularized **two-layer GCN** with a small MLP head and **global mean pooling**.
- **Evaluation:** Stratified **k-fold** cross-validation (default 5-fold), reporting **Accuracy**, **Balanced Accuracy**, **F1**, and **ROC-AUC**. We also plot a confusion matrix per fold.

> This README follows the structure and tone of your previous project’s documentation and report, adapted to the GNN setting.

## Dataset

- **Subjects:** 20 total (10 AD, 10 CN).
- **Per subject files:** two 150×150 matrices in text format:
  - `FunctionalConnectivity.txt` (FC)
  - `StructuralConnectivity.txt` (SC)
- **Directory layout (example):**

```
project_root/
├── AD1/
│   ├── FunctionalConnectivity.txt
│   └── StructuralConnectivity.txt
├── AD2/
│   ├── FunctionalConnectivity.txt
│   └── StructuralConnectivity.txt
├── ...
├── CN1/
│   ├── FunctionalConnectivity.txt
│   └── StructuralConnectivity.txt
└── CN10/
    ├── FunctionalConnectivity.txt
    └── StructuralConnectivity.txt
```

## Preprocessing & Graph Construction

### Structural Connectivity (SC) → Adjacency

1. Symmetrize and zero the diagonal (if needed by data source).
2. `log1p` to compress heavy-tailed tract count distributions.
3. Optional unit scaling to [0, 1].
4. Add self-loops and compute **GCN normalization:** \(\hat A = D^{-1/2}(A + I)D^{-1/2}\).

### Functional Connectivity (FC) → Node Features

1. Optionally apply **Fisher z-transform** (arctanh of clipped correlations).
2. **Standardize** either per subject (global) or **row-wise** per node.
3. Use the resulting 150-d feature vector at each node (the row corresponding to that node).

## Model

We use a lightweight, regularized **GCN** suitable for small-N neuroimaging datasets:

- Input node features: 150-d (from FC)
- Two GCN layers (message passing with pre-normalized \(\hat A\))
- Global mean pooling
- MLP head (hidden → hidden → logits), with dropout
  This design reflects common practice in connectomics: **use SC as stable topology** and inject FC as features.

## Training & Cross-Validation

- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW (defaults provided)
- **Regularization:** Dropout and gradient clipping
- **Early stopping:** patience based on validation ROC-AUC
- **CV:** **StratifiedKFold** (default 5 folds) to maintain class balance in every fold
- **Metrics:** Accuracy, Balanced Accuracy, F1, ROC-AUC
- **Artifacts:** Confusion matrix heatmap per fold saved as `cm_fold_{k}.png`

## Installation

```bash
git clone https://github.com/AbhijitChallapalli/CSE-6389-PA2.git
cd CSE-6389-PA2
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)

# .venv\Scripts\activate  (Windows PowerShell)
pip install -r requirements.txt
```

## Usage

1. Place your subject folders (`AD*`, `CN*`) under the project root.
2. Run the script:

```bash
python train.py
```

Key arguments (examples):

- `k_folds=5` – number of folds in StratifiedKFold
- `epochs=200` – max epochs (early stopping usually stops earlier)
- `batch_size=2`
- `hidden=128`, `dropout=0.5`, `lr=1e-3`, `weight_decay=1e-6`

### Outputs

- Per-fold console logs including training loss every N epochs
- Confusion matrix images: `cm_fold_1.png`, `cm_fold_2.png`, ...
- Mean metrics over folds printed at the end

## Ablation Study (suggested)

- **FC role:** FC as adjacency (|z|, sparsified) versus FC as features (current baseline)
- **Normalization choices:** with/without Fisher z-transform; global vs row-wise standardization
- **Hidden size / depth:** vary `hidden` and number of GCN layers (1–3)

## Reproducibility

- Fixed seeds for NumPy and PyTorch via `set_seed(...)`
- Deterministic settings for cuDNN (as compatible)

## Repository Structure (example)

```
project_root/
├── AD*/ CN*/                 # subject folders with .txt matrices
├── train.py                  # training + evaluation entry point
├── SimpleGCN.py              # Sample GCN code provided
├── requirements.txt
├── Project_report.pdf        # Document for the project
├── cm_fold_*.png             # generated confusion matrices
└── README.md
```

## Related Work and Rationale

- **GCN normalization:** \(\hat A = D^{-1/2}(A + I)D^{-1/2}\) has become the standard for stable message passing.
- **SC as topology, FC as features:** SC provides a relatively stable anatomical backbone across subjects, while FC contributes condition-dependent connectivity patterns. This split is supported by connectomics literature and helps avoid issues with signed FC edges in vanilla GCNs.
- **Compact backbones:** With only 20 subjects, small models with strong regularization reduce overfitting risk while remaining interpretable.

## References

- Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.
- General connectomics practice for SC/FC modelling and graph normalization in neuroimaging literature.

## License

MIT . See `LICENSE`.
