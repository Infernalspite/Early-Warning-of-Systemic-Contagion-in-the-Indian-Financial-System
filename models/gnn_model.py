"""
models/gnn_model.py
===================
GraphSAGE-based GNN for systemic risk early warning.

Each DATE is represented as a GRAPH:
  - Nodes  : 20 Indian banks
  - Edges  : correlation > 0.4 in rolling 30-day window
  - Node features : [5d_return, 10d_volatility, 30d_beta_to_nifty] per bank

Graph-level pooling -> MLP -> binary classification (high_stress_next_30d)

This tests whether TOPOLOGY adds signal beyond aggregated tabular features.

Run: python models/gnn_model.py
Requires: pip install torch torch_geometric
"""

import os, sys, json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler

# ── PyTorch & PyG Import Guard ───────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError:
    print("PyTorch not installed.  Run: pip install torch")
    sys.exit(0)

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn   import SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("WARNING: torch_geometric not installed.")
    print("         Run: pip install torch_geometric")
    print("         Using simplified manual GNN with pure PyTorch as fallback.")

os.makedirs("models", exist_ok=True)

print("="*60)
print("GNN MODEL (GraphSAGE) -- INDIAN RISK ENGINE")
print("="*60)

# ================================================================
# LOAD DATA
# ================================================================
print("\n[1] Loading data...")

returns  = pd.read_csv("data/processed/bank_returns_nse.csv", index_col=0, parse_dates=True)
returns.index = pd.to_datetime(returns.index, dayfirst=True)
returns  = returns.sort_index()

features = pd.read_csv("data/processed/features_india.csv",  index_col=0, parse_dates=True)
features.index = pd.to_datetime(features.index, dayfirst=True)
features = features.sort_index()

bank_names = list(returns.columns)
N_BANKS    = len(bank_names)
SEQ_LEN    = 30   # rolling window for graph construction

print(f"   Banks: {N_BANKS}  |  Returns shape: {returns.shape}")
print(f"   Features shape: {features.shape}")

# ================================================================
# BUILD PER-NODE FEATURES PER DATE
# ================================================================
print("\n[2] Building node features and graphs...")

def get_node_features(idx, returns_df, features_df, seq_len=30):
    """
    For date at position idx, build node feature matrix (N_banks x 3):
      col 0: 5-day mean return per bank
      col 1: 10-day rolling std (vol) per bank
      col 2: 30-day beta of bank vs Nifty Bank (proxy: vs mean of all banks)
    """
    if idx < seq_len:
        return None
    window = returns_df.iloc[idx-seq_len:idx]
    node_feats = np.zeros((len(returns_df.columns), 3), dtype=np.float32)
    mkt_ret = window.mean(axis=1)  # equal-weight market proxy

    for j, col in enumerate(returns_df.columns):
        bank_ret = window[col].fillna(0).values
        node_feats[j, 0] = bank_ret[-5:].mean()         # 5d mean return
        node_feats[j, 1] = bank_ret.std() + 1e-8        # 30d vol
        # Beta vs market proxy
        cov = np.cov(bank_ret, mkt_ret.values)[0, 1]
        var = np.var(mkt_ret.values) + 1e-8
        node_feats[j, 2] = cov / var
    return node_feats

def get_edge_index(idx, returns_df, threshold=0.4, seq_len=30):
    """Build edge_index tensor from rolling correlation graph."""
    if idx < seq_len:
        return np.zeros((2, 0), dtype=np.int64)
    window = returns_df.iloc[idx-seq_len:idx].fillna(0)
    corr   = window.corr().fillna(0).values
    src, dst = [], []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if corr[i, j] > threshold:
                src += [i, j]
                dst += [j, i]
    return np.array([src, dst], dtype=np.int64)

# Common index (intersection of returns and features)
common_idx = returns.index.intersection(features.index)
target_col = "high_stress_next_30d"
if target_col not in features.columns:
    print(f"ERROR: '{target_col}' not in features. Run crisis_labels.py first.")
    sys.exit(1)

labels_series = features[target_col].reindex(common_idx).fillna(0).astype(int)
returns_aligned = returns.reindex(common_idx)

graph_list   = []
label_list   = []
date_list    = []

for pos, dt in enumerate(common_idx):
    idx_in_returns = returns.index.get_loc(dt)
    if not isinstance(idx_in_returns, int):
        idx_in_returns = idx_in_returns if isinstance(idx_in_returns, int) else int(idx_in_returns)
    if pos < SEQ_LEN:
        continue

    node_x = get_node_features(pos, returns_aligned, features, SEQ_LEN)
    if node_x is None:
        continue

    ei = get_edge_index(pos, returns_aligned, threshold=0.4, seq_len=SEQ_LEN)
    label = int(labels_series.iloc[pos])

    graph_list.append((node_x, ei))
    label_list.append(label)
    date_list.append(dt)

print(f"   Graphs built: {len(graph_list)} (each = one trading day)")

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================
split_date  = pd.Timestamp("2022-01-01")
train_mask  = [d < split_date  for d in date_list]
test_mask   = [d >= split_date for d in date_list]

print(f"   Train graphs: {sum(train_mask)}  |  Test graphs: {sum(test_mask)}")
pos_train = sum(label_list[i] for i, m in enumerate(train_mask) if m)
print(f"   Train positives: {pos_train} ({pos_train/sum(train_mask)*100:.1f}%)")

# ================================================================
# BUILD TORCH GEOMETRIC DATA OBJECTS  (or manual tensors if no PyG)
# ================================================================

def build_pyg_dataset(mask):
    data_list = []
    for i, (nf, ei) in enumerate(graph_list):
        if not mask[i]:
            continue
        x = torch.tensor(nf, dtype=torch.float)
        if ei.shape[1] > 0:
            edge_index = torch.tensor(ei, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        y = torch.tensor([label_list[i]], dtype=torch.float)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list

# ================================================================
# MODEL DEFINITIONS
# ================================================================

if PYG_AVAILABLE:
    class GraphRiskNet(nn.Module):
        def __init__(self, in_channels=3, hidden=64, out_classes=1):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.fc1   = nn.Linear(hidden, 32)
            self.fc2   = nn.Linear(32, out_classes)
            self.drop  = nn.Dropout(0.3)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = self.drop(x)
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)   # graph-level pooling
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            return torch.sigmoid(self.fc2(x)).squeeze(-1)

else:
    # Fallback: simplified manual message passing (no PyG dependency)
    class GraphRiskNet(nn.Module):
        """
        Manual GNN without torch_geometric.
        Simple mean-aggregation over neighbours then MLP.
        """
        def __init__(self, in_channels=3, hidden=64, out_classes=1):
            super().__init__()
            self.lin1 = nn.Linear(in_channels * 2, hidden)
            self.lin2 = nn.Linear(hidden, 32)
            self.lin3 = nn.Linear(32, out_classes)
            self.drop = nn.Dropout(0.3)

        def forward_graph(self, x, edge_index):
            # x: (N, F), edge_index: (2, E)
            N = x.shape[0]
            if edge_index.shape[1] > 0:
                agg = torch.zeros_like(x)
                cnt = torch.zeros(N, 1, dtype=x.dtype)
                for src, dst in zip(edge_index[0], edge_index[1]):
                    agg[dst] += x[src]
                    cnt[dst] += 1
                cnt = cnt.clamp(min=1)
                agg = agg / cnt
            else:
                agg = torch.zeros_like(x)
            combined = torch.cat([x, agg], dim=-1)   # (N, 2F)
            return combined.mean(0, keepdim=True)    # (1, 2F) graph-level

        def forward(self, x, edge_index, batch=None):
            graph_emb = self.forward_graph(x, edge_index)
            h = F.relu(self.lin1(graph_emb))
            h = self.drop(h)
            h = F.relu(self.lin2(h))
            return torch.sigmoid(self.lin3(h)).squeeze()

# ================================================================
# TRAINING
# ================================================================
print("\n[3] Training GraphSAGE model...")

EPOCHS     = 40
BATCH_SIZE = 32
LR         = 0.001

pos_w = (sum(train_mask) - pos_train) / max(pos_train, 1)
pos_weight = torch.tensor([pos_w], dtype=torch.float)
criterion  = nn.BCELoss()  # we apply pos_weight manually via sample weights

model  = GraphRiskNet(in_channels=3, hidden=64)
optim  = Adam(model.parameters(), lr=LR)

# Convert to simple lists of tensors for training loop (works with/without PyG)
def make_batch_tensors(indices):
    xs, eis, ys = [], [], []
    for i in indices:
        nf, ei = graph_list[i]
        xs.append(torch.tensor(nf, dtype=torch.float))
        eis.append(torch.tensor(ei, dtype=torch.long) if ei.shape[1] > 0
                   else torch.zeros((2,0), dtype=torch.long))
        ys.append(torch.tensor(float(label_list[i])))
    return xs, eis, ys

train_indices = [i for i, m in enumerate(train_mask) if m]
test_indices  = [i for i, m in enumerate(test_mask)  if m]

np.random.seed(42)
model.train()

for epoch in range(1, EPOCHS + 1):
    np.random.shuffle(train_indices)
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(train_indices), BATCH_SIZE):
        batch_idx = train_indices[start:start+BATCH_SIZE]
        xs, eis, ys = make_batch_tensors(batch_idx)

        optim.zero_grad()
        batch_loss = torch.tensor(0.0, requires_grad=True)

        for x_i, ei_i, y_i in zip(xs, eis, ys):
            if PYG_AVAILABLE:
                batch_vec = torch.zeros(x_i.shape[0], dtype=torch.long)
                pred = model(x_i, ei_i, batch_vec)
            else:
                pred = model(x_i, ei_i)
            pred = pred.clamp(1e-6, 1-1e-6)
            w    = pos_weight.item() if y_i.item() == 1 else 1.0
            loss = w * F.binary_cross_entropy(pred.unsqueeze(0), y_i.unsqueeze(0))
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        optim.step()
        total_loss += batch_loss.item()
        n_batches  += 1

    if epoch % 8 == 0 or epoch == 1:
        print(f"   Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/max(n_batches,1):.4f}")

# ================================================================
# EVALUATION
# ================================================================
print("\n[4] Evaluating on test set...")

model.eval()
all_probs, all_preds, all_true = [], [], []

with torch.no_grad():
    for i in test_indices:
        nf, ei = graph_list[i]
        x_t  = torch.tensor(nf, dtype=torch.float)
        ei_t = torch.tensor(ei, dtype=torch.long) if ei.shape[1] > 0 \
               else torch.zeros((2,0), dtype=torch.long)
        if PYG_AVAILABLE:
            bv   = torch.zeros(x_t.shape[0], dtype=torch.long)
            prob = model(x_t, ei_t, bv).item()
        else:
            prob = model(x_t, ei_t).item()
        pred = 1 if prob >= 0.5 else 0
        all_probs.append(prob)
        all_preds.append(pred)
        all_true.append(label_list[i])

acc  = accuracy_score(all_true, all_preds)
prec = precision_score(all_true, all_preds, zero_division=0)
rec  = recall_score(all_true, all_preds, zero_division=0)
f1   = f1_score(all_true, all_preds, zero_division=0)
try:
    auc = roc_auc_score(all_true, all_probs)
except Exception:
    auc = float("nan")

print("\n--- GNN EVALUATION RESULTS ---")
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1 Score  : {f1:.4f}")
print(f"   ROC-AUC   : {auc:.4f}")

print("\n" + classification_report(all_true, all_preds,
      target_names=["Normal","High Stress"], zero_division=0))

# ================================================================
# SAVE
# ================================================================
torch.save(model.state_dict(), "models/gnn_model.pt")

metrics = {
    "model"    : "GNN (GraphSAGE)" if PYG_AVAILABLE else "GNN (Manual SAGE)",
    "accuracy" : round(acc,  4),
    "precision": round(prec, 4),
    "recall"   : round(rec,  4),
    "f1"       : round(f1,   4),
    "roc_auc"  : round(auc,  4) if not np.isnan(auc) else None,
    "pyg_used" : PYG_AVAILABLE
}
with open("models/gnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nModel saved : models/gnn_model.pt")
print("Metrics saved: models/gnn_metrics.json")
print("\nGNN training complete.")
