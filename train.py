import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns

def gcn_normalize_sc(SC: np.ndarray,*,log1p: bool = True,scale_to_unit: bool = True,add_self_loops: bool = True,eps: float = 1e-8,) -> torch.FloatTensor:
    """
    Structural connectivity → GCN-normalized adjacency:
      sym + zeroDiag → (log1p) → (scale to [0,1]) → D^-1/2 (A+I) D^-1/2
    Returns torch.FloatTensor [N, N].
    """
    if log1p:
        SC = np.log1p(np.maximum(SC, 0.0))
    
    if scale_to_unit and SC.max() > 0:
        SC = SC / (SC.max() + eps)

    if add_self_loops:
        SC = SC + np.eye(SC.shape[0], dtype=SC.dtype)

    d = SC.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    A_hat = (d_inv_sqrt[:, None]) * SC * (d_inv_sqrt[None, :])
    return torch.tensor(A_hat, dtype=torch.float32)

def normalize_fc_features(FC: np.ndarray,*,fisher_z: bool = False,standardize: str = "zscore",rowwise: bool = False,eps: float = 1e-8,) -> torch.FloatTensor:
    """
    Functional connectivity → node feature matrix (NOT an adjacency).
      (Fisher-z) → (standardize globally or per-row)
    Returns torch.FloatTensor [N, N] where row i is node i's feature vector.
    """
    if fisher_z:
        r = np.clip(FC, -0.999999, 0.999999)
        FC = np.arctanh(r)

    if standardize == "zscore":
        if rowwise:
            mu = FC.mean(axis=1, keepdims=True)
            sd = FC.std(axis=1, keepdims=True)
            FC = (FC - mu) / (sd +eps) 
        else:
            mu, sd = FC.mean(), FC.std()
            FC = (FC - mu) / (sd + eps)

    elif standardize == "minmax":
        if rowwise:
            x_min = FC.min(axis=1, keepdims=True)
            x_max = FC.max(axis=1, keepdims=True)
            FC = (FC - x_min) / (x_max - x_min + eps)
        else:
            x_min, x_max = FC.min(), FC.max()
            FC = (FC - x_min) / (x_max - x_min + eps)

    return torch.tensor(FC.astype(np.float32), dtype=torch.float32)

def read_data(root_dir,*,fc_rowwise=True):
    """
    Reads the root_dir for AD and CN folders and 
    """
    out = []
    for files in os.listdir(root_dir):
        
        #We don't want other files in the directory to be read as data
        if not (files.startswith("AD") or files.startswith("CN")):
            continue

        #Subject directories    
        subject_dir = os.path.join(root_dir, files)
        if not os.path.isdir(subject_dir):
            continue
        
        #labeled AD as 1 and CN as 0
        label = 1 if files.startswith("AD") else 0


        functional_path=os.path.join(subject_dir,"FunctionalConnectivity.txt")
        structural_path=os.path.join(subject_dir,"StructuralConnectivity.txt")
        FC = np.loadtxt(functional_path)
        SC = np.loadtxt(structural_path)

        A_sc = gcn_normalize_sc(SC)
        X_fc = normalize_fc_features(FC, rowwise=fc_rowwise)

        out.append({"id": files, "label": label, "FC": FC, "SC": SC,"A_sc":A_sc,"X_fc":X_fc})

    return out



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X):
        """
        A: [B,N,N] pre-normalized adjacency (your A_sc)
        X: [B,N,C] node features
        """
        H = A @ X                    
        H = self.linear(H)            # [B,N,out]
        B, N, C = H.shape
        H = H.reshape(B*N, C)
        H = self.bn(H)
        H = H.view(B, N, C)
        H = torch.relu(self.dropout(H))
        return H

class GCNClassifier(nn.Module):
    def __init__(self, in_feats=150, hidden=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.in_lin = nn.Linear(in_feats, hidden)
        self.gcn1 = GCNLayer(hidden, hidden, dropout=dropout)
        self.gcn2 = GCNLayer(hidden, hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, A, X):
        """
        A: [B,N,N]; X: [B,N,F]
        """
        if X.dim() == 2:  
            A = A.unsqueeze(0)
            X = X.unsqueeze(0)
        H = torch.relu(self.in_lin(X))      # [B,N,H]
        H = self.gcn1(A, H)                 # [B,N,H]
        H = self.gcn2(A, H)                 # [B,N,H]
        G = H.mean(dim=1)                   # [B,H]
        logits = self.head(G)               # [B,num_classes]
        return logits



def set_seed(seed=7321):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for A, X, y in loader:
        A, X, y = A.to(device), X.to(device), y.to(device)
        logits = model(A, X)
        p = torch.softmax(logits, dim=1)[:, 1]
        yhat = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        preds.extend(yhat.cpu().numpy().tolist())
        probs.extend(p.cpu().numpy().tolist())

    acc  = accuracy_score(ys, preds)
    bacc = balanced_accuracy_score(ys, preds)
    f1   = f1_score(ys, preds)
    try:
        auc = roc_auc_score(ys, probs) if len(set(ys)) == 2 else 0.5
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(ys, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {"acc": acc, "bacc": bacc, "f1": f1, "auc": auc, "cm": cm,"sensitivity":sensitivity,"specificity":specificity}

def tensors_from_subjects(subjects):
    A_sc = torch.stack([s["A_sc"] for s in subjects])  # [S,N,N]
    X_fc = torch.stack([s["X_fc"] for s in subjects])  # [S,N,N]
    y_all = torch.tensor([int(s["label"]) for s in subjects], dtype=torch.long)
    return A_sc, X_fc, y_all



def run_stratified_cv_allfolds(subjects, model_cls, *,k_folds=5, batch_size=4, epochs=200,
                               hidden=64, dropout=0.5, lr=1e-3, weight_decay=1e-4,
                               patience=20, seed=123):
    """
    Trains and evaluates inside the StratifiedKFold loop. No separate train_one_fold helper.
    - subjects: list from read_data(), each with A_sc, X_fc, label
    - model_cls: your GCNClassifier class
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stack all samples once
    A_sc, X_fc, y_all = tensors_from_subjects(subjects)
    S, N, F = X_fc.shape  # samples, nodes, features

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_stats = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(S), y_all.numpy()), 1):
        print(f"\n=== Fold {fold}/{k_folds} ===")

        # Split tensors by indices
        A_train, X_train, y_train = A_sc[train_indices], X_fc[train_indices], y_all[train_indices]
        A_val, X_val, y_val = A_sc[val_indices], X_fc[val_indices], y_all[val_indices]

        # DataLoaders (default [N,N] into [B,N,N])
        train_loader = DataLoader(TensorDataset(A_train, X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(A_val, X_val, y_val), batch_size=batch_size, shuffle=False)

        # model per fold
        model = model_cls(in_feats=F, hidden=hidden, num_classes=2, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.CrossEntropyLoss()

        # early stopping on AUROC
        best_auc, best_state, no_improve = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            running_loss = 0.0                       
            num_samples = 0

            for A, X, y in train_loader:
                A, X, y = A.to(device), X.to(device), y.to(device)
                opt.zero_grad()
                logits = model(A, X)
                loss = crit(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

                running_loss += loss.item() * y.size(0)
                num_samples += y.size(0)

            # Validate each epoch
            val_metrics = evaluate(model, val_loader, device)
            auc = val_metrics["auc"]

            # print loss every 5 epochs or on last epoch
            if ((ep + 1) % 5 == 0) or (ep == epochs - 1):
                avg_loss = running_loss / max(1, num_samples)
                print(f"[Fold {fold}] Epoch {ep+1:3d}/{epochs}  "
                      f"train_loss={avg_loss:.4f}  "
                      f"val_auc={auc:.3f}  val_acc={val_metrics['acc']:.3f}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # eval on validation fold
        m = evaluate(model, val_loader, device)
        fold_stats.append(m)
        print(f"Acc={m['acc']:.3f}  BalAcc={m['bacc']:.3f}  F1={m['f1']:.3f} "f"AUC={m['auc']:.3f}  sensitivity={m['sensitivity']:.3f}  Specificity={m['specificity']:.3f}")
        print(f"Confusion Matrix (array):\n{m['cm']}")

        # confusion-matrix heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="mako", cbar=False, ax=ax,xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — Fold {fold}")
        plt.tight_layout()
        plt.savefig(f"cm_fold_{fold}.png", dpi=200)
        plt.show()

    # Summarize across folds 
    keys = ["acc", "bacc", "f1", "auc"]
    means = {k: float(np.mean([m[k] for m in fold_stats])) for k in keys}

    print(means)
    print("\n=== Mean across folds ===")
    for k, mu in means.items():
        print(f"{k:6s}: {mu:.3f}")

    return fold_stats, means


subjects = read_data(root_dir="./", fc_rowwise=False)
fold_stats, summary = run_stratified_cv_allfolds(
    subjects,
    model_cls=GCNClassifier,   
    k_folds=5,
    batch_size=2,
    epochs=200,
    hidden=128,
    dropout=0.5,
    lr=1e-3,
    weight_decay=1e-6,
    patience=20,
    seed=1
)
