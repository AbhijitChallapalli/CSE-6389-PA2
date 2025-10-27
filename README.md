# GCN for Alzheimer’s Disease (AD) Classification using Brain Connectivity

## Overview

This repository implements a Graph Convolutional Network (GCN) to perform **binary classification** of subjects as **Alzheimer’s Disease (AD)** or **Cognitively Normal (CN)** using **functional connectivity (FC)** and **structural connectivity (SC)** matrices. The emphasis of this project is on **methodology, analysis, and experimental design** rather than raw accuracy.

- **SC (structural connectivity):** used as the **graph topology** (adjacency). We apply log1p compression, optional unit scaling, add self-loops, and perform **GCN symmetric normalization** $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$.
- **FC (functional connectivity):** used as **node features**. We optionally apply Fisher z-transform and then **standardize** either globally per subject or row-wise per node.
- **Model:** a compact, regularized **two-layer GCN** with a small MLP head and **global mean pooling**.
- **Evaluation:** Stratified **k-fold** cross-validation (default 5-fold), reporting **Accuracy**, **Balanced Accuracy**, **F1**, and **ROC-AUC**. We also plot a confusion matrix per fold.

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

## Literature-Review for the model

I reviewed recent works combining **FC & SC with GNNs** and distilled the following takeaways for a small dataset (N=20):

- **Joint-GCN (DTI + rs-fMRI, joint graph):** builds separate SC and FC graphs and adds **learnable inter-network links** between matching ROIs; shared encoders and fusion typically outperform single-modality GCNs. _Takeaway:_ **light cross-modal coupling helps**, but adds complexity.
- **MMTGCN (Mutual Multi-Scale Triplet GCN):** multi-scale graphs and triplet interactions capture higher-order relationships; shows robust gains but is heavier to implement. _Takeaway:_ **multi-scale and modality-aware branches** can improve robustness, but may overfit with very small N.
- **GCNNs for AD spectrum:** two-layer GCNs with **GCN normalization** and **graph-level pooling** are strong **baselines** on connectome data. _Takeaway:_ a **compact two-layer GCN** is a solid starting point.
- **SC ↔ FC relation with GCN encoders:** SC provides a **stable anatomical backbone** that constrains FC; using SC as the message-passing graph and injecting FC as features is principled. _Takeaway:_ **use SC for topology** and **FC as features**.

### What I implemented

Given the dataset size, I adopted the **simplest defensible** design: **single-branch GCN over SC** with **FC as node features**. A dual-branch fusion or a cross-modal gate is left as **future work/ablation**, consistent with Joint-GCN’s motivation.

## Preprocessing & Graph Construction

### Structural Connectivity (SC) → Adjacency

1. Symmetrize and zero the diagonal.
2. `log1p` to compress heavy-tailed tract count distributions.
3. Optional unit scaling to [0, 1].
4. Add self-loops and compute **GCN normalization:** \(\hat A = D^{-1/2}(A + I)D^{-1/2}\).

### Functional Connectivity (FC) → Node Features

1. Optionally apply **Fisher z-transform** (arctanh of clipped correlations).
2. **Standardize** either per subject (global) or **row-wise** per node.
3. Use the resulting 150-d feature vector at each node (the row corresponding to that node).

## Model Architecture

I developed a **two-layer GCN** with batch normalization, dropout, and a small MLP head.

> I began with a dual-branch idea inspired by Joint-GCN (an SC branch and an FC graph branch with late fusion). In early trials and based on the small sample size (N=20), I simplified it to a **single-branch design** to reduce parameters and stabilize training. SC serves as the message-passing topology (after GCN normalization), while FC provides node features. I retained **batch normalization** and **dropout** for regularization and used **global mean pooling** to obtain graph-level embeddings. This compact setup echoes prior connectome GCN baselines and aligns with literature that treats SC as a stable anatomical scaffold and FC as a functional signal injected at the nodes.

**Final architecture (per subject):**

- **Input:** SC-derived features, FC-derived features Each 150 x 150.
- Linear(150 → 64) → ReLU → Dropout(0.5)
- GCN layer (64 → 64) with pre-normalized \(\hat A\) → ReLU → Dropout(0.5)
- GCN layer (64 → 64) → **Global mean pool** over nodes → 64-d vector
- MLP head: Linear(64 → 64) → ReLU → Dropout(0.5) → Linear(64 → 2)

This balances parameter efficiency and representational power for small-N connectomics.

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
```

```bash
cd CSE-6389-PA2
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
```

```
# .venv\Scripts\activate (Windows PowerShell)

pip install -r requirements.txt

```

## Usage

1. Place your subject folders (`AD*`, `CN*`) under the project root.
2. Run the script:

```bash
python train.py
```

Key arguments:

- `k_folds=5` – number of folds in StratifiedKFold
- `epochs=200` – max epochs (early stopping usually stops earlier)
- `batch_size=2`
- `hidden=128`, `dropout=0.5`, `lr=1e-3`, `weight_decay=1e-6`

### Outputs

- Per-fold console logs including training loss every N epochs
- Confusion matrix images: `cm_fold_1.png`, `cm_fold_2.png`, ...
- Mean metrics over folds printed at the end

## Ablation Study

- **FC role:** FC as adjacency (|z|, sparsified) versus FC as features (current baseline)
- **Normalization choices:** with/without Fisher z-transform; global vs row-wise standardization
- **Hidden size / depth:** vary `hidden` and number of GCN layers (1–3)

## Reproducibility

- Fixed seeds for NumPy and PyTorch via `set_seed(...)`
- Deterministic settings for cuDNN (as compatible)

## Repository Structure

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

- **GCN normalization:** $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$ has become the standard for stable message passing.
- **SC as topology, FC as features:** SC provides a relatively stable anatomical backbone across subjects, while FC contributes condition-dependent connectivity patterns. This split is supported by connectomics literature and helps avoid issues with signed FC edges in vanilla GCNs.
- **Compact backbones:** With only 20 subjects, small models with strong regularization reduce overfitting risk while remaining interpretable.

## References

1. Kipf, T. N., & Welling, M. (2017, April). _Semi-supervised classification with graph convolutional networks_. International Conference on Learning Representations (ICLR). [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907) ([arXiv][1])

2. Li, Y., Wei, Q., Adeli, E., Pohl, K. M., & Zhao, Q. (2022). _Joint graph convolution for analyzing brain structural and functional connectome_. In **Medical Image Computing and Computer Assisted Intervention – MICCAI 2022** (Lecture Notes in Computer Science). Springer. [https://doi.org/10.1007/978-3-031-16431-6_22](https://doi.org/10.1007/978-3-031-16431-6_22) ([MICCAI Conferences][2])

3. Yao, D., Liu, M., Lin, C., Li, H., Zhang, J., & Shen, D. (2021). A mutual multi-scale triplet graph convolutional network for classification of brain disorders using functional or structural connectivity. _IEEE Transactions on Medical Imaging, 40_(4), 1279–1289. [https://doi.org/10.1109/TMI.2021.3051604](https://doi.org/10.1109/TMI.2021.3051604) ([PMC][3])

4. Li, Y., Shafipour, R., Mateos, G., & Zhang, Z. (2019, November). _Mapping brain structural connectivities to functional networks via graph encoder–decoder with interpretable latent embeddings_. In **2019 IEEE Global Conference on Signal and Information Processing (GlobalSIP)**. [https://www.hajim.rochester.edu/ece/sites/gmateos/pubs/brain/SCFC_GLOBALSIP19.pdf](https://www.hajim.rochester.edu/ece/sites/gmateos/pubs/brain/SCFC_GLOBALSIP19.pdf) ([hajim.rochester.edu][5])

5. Liu, J., Ma, G., Jiang, F., Lu, C.-T., Yu, P. S., & Ragin, A. B. (2019). Community-preserving graph convolutions for structural and functional joint embedding of brain networks. _arXiv Preprint_. [https://arxiv.org/abs/1911.03583](https://arxiv.org/abs/1911.03583) ([arXiv][6])
