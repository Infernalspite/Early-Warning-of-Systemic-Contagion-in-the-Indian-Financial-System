"""
src/models/lstm_model.py
========================
PyTorch LSTM Deep Learning Architecture with Dual Configuration Support:
- Macro-only Configuration: macro indicators only
- Macro + Network Configuration: macro + flattened network centrality & econometric features
"""

import os
import pathlib
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class SystemicLSTM(nn.Module):
    """
    Bidirectional / Multi-layer LSTM for systemic crisis prediction.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        h = F.relu(self.fc1(self.dropout(last_hidden)))
        return self.fc2(h).squeeze(-1)  # Logits for BCEWithLogitsLoss

class TimeSeriesDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        X_clean = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num((y_seq > 0).astype(np.float32), nan=0.0)
        self.X = torch.tensor(X_clean, dtype=torch.float32)
        self.y = torch.tensor(y_clean, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X_data, y_data, seq_len=30):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_data)):
        X_seq.append(X_data[i-seq_len:i])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(X_train, y_train, X_test, y_test, config_mode="macro_plus_network", seq_len=30, epochs=30, batch_size=32, lr=0.001):
    device = torch.device("cpu")  # CPU execution for guaranteed numerical stability
    print(f"Training LSTM ({config_mode}) on device: {device}...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train))
    X_test_scaled = scaler.transform(np.nan_to_num(X_test))

    X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, y_train, seq_len)
    X_te_seq, y_te_seq = create_sequences(X_test_scaled, y_test, seq_len)

    train_ds = TimeSeriesDataset(X_tr_seq, y_tr_seq)
    test_ds = TimeSeriesDataset(X_te_seq, y_te_seq)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = SystemicLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3).to(device)

    pos_weight = float((y_tr_seq == 0).sum() / max((y_tr_seq == 1).sum(), 1))
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {total_loss / max(len(train_ds),1):.4f}")

    # Inference
    model.eval()
    test_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            p = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(p)

    return model, np.array(test_probs), y_te_seq
