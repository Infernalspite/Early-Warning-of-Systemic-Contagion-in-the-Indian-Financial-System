"""
LSTM Model - Early Warning of Systemic Contagion (India)
=========================================================
Binary sequence classification: predict high_stress_next_30d
Sliding window of SEQ_LEN=30 trading days.
Time-based train/test split at 2022-01-01 (based on window end date).

Run from project root:
    python models/lstm_model.py
"""

import json
import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is not installed.")
    print("Install with: pip install torch")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN    = 30       # sliding window length in trading days
HIDDEN     = 64
NUM_LAYERS = 2
DROPOUT    = 0.3
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 0.001

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "features_india.csv"
FEAT_LIST   = ROOT / "data" / "processed" / "feature_list.txt"
MODEL_OUT   = ROOT / "models" / "lstm_model.pt"
SCALER_OUT  = ROOT / "models" / "scaler_lstm.pkl"
METRICS_OUT = ROOT / "models" / "lstm_metrics.json"

TARGET_COL = "high_stress_next_30d"
SPLIT_DATE = pd.Timestamp("2022-01-01")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
df = df.sort_index()

# ---------------------------------------------------------------------------
# 2. Determine feature columns
# ---------------------------------------------------------------------------
if FEAT_LIST.exists():
    print("Loading feature list from:", FEAT_LIST)
    with open(FEAT_LIST, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features loaded from file: {len(feature_cols)}")
else:
    non_feature_cols = {TARGET_COL}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"  No feature_list.txt found. Using all {len(feature_cols)} non-label columns.")

# ---------------------------------------------------------------------------
# 3. Drop NaN rows
# ---------------------------------------------------------------------------
cols_needed = feature_cols + [TARGET_COL]
df = df[cols_needed].dropna()
print(f"  Rows after dropping NaN: {len(df)}")

feature_arr = df[feature_cols].values.astype(np.float32)
label_arr   = df[TARGET_COL].values.astype(np.float32)
dates       = df.index

# ---------------------------------------------------------------------------
# 4. Create sliding windows
#    X[i] = feature_arr[i : i+SEQ_LEN]       shape (SEQ_LEN, n_features)
#    y[i] = label_arr[i + SEQ_LEN]           label at the day AFTER the window
#    end_date[i] = dates[i + SEQ_LEN]        used for train/test split
# ---------------------------------------------------------------------------
n = len(feature_arr)
if n <= SEQ_LEN:
    raise ValueError(f"Not enough rows ({n}) to build sequences of length {SEQ_LEN}.")

X_seqs    = []
y_labels  = []
end_dates = []

for i in range(n - SEQ_LEN):
    X_seqs.append(feature_arr[i : i + SEQ_LEN])
    y_labels.append(label_arr[i + SEQ_LEN])
    end_dates.append(dates[i + SEQ_LEN])

X_seqs    = np.array(X_seqs,   dtype=np.float32)   # (N, SEQ_LEN, n_features)
y_labels  = np.array(y_labels, dtype=np.float32)   # (N,)
end_dates = np.array(end_dates)

print(f"  Total sequences: {len(X_seqs)}")

# ---------------------------------------------------------------------------
# 5. Time-based train/test split on end_date
# ---------------------------------------------------------------------------
train_mask = end_dates < SPLIT_DATE
test_mask  = end_dates >= SPLIT_DATE

X_train_raw = X_seqs[train_mask]
y_train     = y_labels[train_mask]
X_test_raw  = X_seqs[test_mask]
y_test      = y_labels[test_mask]

print(f"  Train sequences: {len(X_train_raw)}  (end_date < {SPLIT_DATE.date()})")
print(f"  Test  sequences: {len(X_test_raw)}   (end_date >= {SPLIT_DATE.date()})")

if len(X_train_raw) == 0 or len(X_test_raw) == 0:
    raise ValueError("Train or test set is empty after the time split. Check SPLIT_DATE.")

# ---------------------------------------------------------------------------
# 6. Standardise features (fit on flattened train windows)
# ---------------------------------------------------------------------------
n_features = X_train_raw.shape[2]
scaler = StandardScaler()

X_train_flat = X_train_raw.reshape(-1, n_features)
scaler.fit(X_train_flat)

X_train_sc = scaler.transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
X_test_sc  = scaler.transform(X_test_raw.reshape(-1,  n_features)).reshape(X_test_raw.shape)

# ---------------------------------------------------------------------------
# 7. PyTorch Dataset
# ---------------------------------------------------------------------------
class IndiaStressDataset(Dataset):
    """Dataset of (sequence, label) pairs for systemic stress prediction."""

    def __init__(self, sequences_X: np.ndarray, labels_y: np.ndarray):
        self.X = torch.tensor(sequences_X, dtype=torch.float32)
        self.y = torch.tensor(labels_y,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = IndiaStressDataset(X_train_sc, y_train)
test_dataset  = IndiaStressDataset(X_test_sc,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---------------------------------------------------------------------------
# 8. LSTM Model
# ---------------------------------------------------------------------------
class BankingLSTM(nn.Module):
    """
    Two-layer LSTM for binary classification of systemic stress events.

    Args:
        input_size  : Number of input features per timestep.
        hidden_size : Number of LSTM hidden units.
        num_layers  : Number of stacked LSTM layers.
        dropout     : Dropout probability between LSTM layers.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]          # take last timestep
        out          = self.dropout(last_hidden)
        out          = self.fc(out)
        return self.sigmoid(out).squeeze(1)         # (batch,)


# ---------------------------------------------------------------------------
# 9. Training
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = BankingLSTM(
    input_size=n_features,
    hidden_size=HIDDEN,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

# pos_weight for BCELoss to handle class imbalance
n_neg_train = int((y_train == 0).sum())
n_pos_train = int((y_train == 1).sum())
if n_pos_train == 0:
    raise ValueError("No positive samples in the training set.")
pos_weight_val = n_neg_train / n_pos_train
pos_weight     = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
print(f"  Train class distribution: neg={n_neg_train}, pos={n_pos_train}")
print(f"  BCELoss pos_weight = {pos_weight_val:.4f}")

criterion = nn.BCELoss(weight=None)   # we incorporate pos_weight via BCEWithLogitsLoss below
# Use BCEWithLogitsLoss for numerical stability (model outputs sigmoid already, so use BCELoss)
criterion = nn.BCELoss()

# Manually apply weighting each batch (simpler, avoids needing logits)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nTraining BankingLSTM for {EPOCHS} epochs ...")
print(f"  {'Epoch':<8} {'Loss':>12}")
print(f"  {'-'*22}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)

        # Manual pos_weight scaling: weight positive examples
        weights = torch.where(y_batch == 1,
                              torch.full_like(y_batch, pos_weight_val),
                              torch.ones_like(y_batch))
        loss = (weights * nn.functional.binary_cross_entropy(preds, y_batch, reduction="none")).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    avg_loss = total_loss / len(train_dataset)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:<4}   Loss: {avg_loss:.6f}")

print("  Training complete.")

# ---------------------------------------------------------------------------
# 10. Evaluate on test set
# ---------------------------------------------------------------------------
model.eval()
all_preds = []
all_probs = []
all_true  = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        probs   = model(X_batch).cpu().numpy()
        preds   = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_true.extend(y_batch.numpy().astype(int).tolist())

y_pred_arr = np.array(all_preds)
y_prob_arr = np.array(all_probs)
y_true_arr = np.array(all_true)

accuracy  = accuracy_score(y_true_arr, y_pred_arr)
precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
recall    = recall_score(y_true_arr, y_pred_arr, zero_division=0)
f1        = f1_score(y_true_arr, y_pred_arr, zero_division=0)
roc_auc   = roc_auc_score(y_true_arr, y_prob_arr)

metrics = {
    "model":       "BankingLSTM",
    "split_date":  str(SPLIT_DATE.date()),
    "seq_len":     SEQ_LEN,
    "hidden_size": HIDDEN,
    "num_layers":  NUM_LAYERS,
    "dropout":     DROPOUT,
    "epochs":      EPOCHS,
    "n_train":     int(len(y_train)),
    "n_test":      int(len(y_test)),
    "accuracy":    round(accuracy,  4),
    "precision":   round(precision, 4),
    "recall":      round(recall,    4),
    "f1":          round(f1,        4),
    "roc_auc":     round(roc_auc,   4),
}

# ---------------------------------------------------------------------------
# 11. Save artefacts
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), MODEL_OUT)

with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n  Model  saved -> {MODEL_OUT}")
print(f"  Scaler saved -> {SCALER_OUT}")
print(f"  Metrics saved -> {METRICS_OUT}")

# ---------------------------------------------------------------------------
# 12. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  LSTM (BankingLSTM) - TEST SET RESULTS")
print("=" * 50)
print(f"  {'Metric':<15} {'Value':>10}")
print(f"  {'-'*25}")
print(f"  {'Split Date':<15} {str(SPLIT_DATE.date()):>10}")
print(f"  {'Seq Length':<15} {SEQ_LEN:>10}")
print(f"  {'Train Seqs':<15} {len(y_train):>10,}")
print(f"  {'Test Seqs':<15} {len(y_test):>10,}")
print(f"  {'Accuracy':<15} {accuracy:>10.4f}")
print(f"  {'Precision':<15} {precision:>10.4f}")
print(f"  {'Recall':<15} {recall:>10.4f}")
print(f"  {'F1 Score':<15} {f1:>10.4f}")
print(f"  {'ROC-AUC':<15} {roc_auc:>10.4f}")
print("=" * 50)
