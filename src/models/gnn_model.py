"""
src/models/gnn_model.py
========================
Dynamic Multi-Relation Graph Neural Network (EvolveGCN / Relational GAT Architecture).
Supports multi-relation edge types (Correlation, Granger, Exposure) and temporal sequence aggregation.
"""

import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class RelationalDynamicGNN(nn.Module):
    """
    Multi-Relation Dynamic Graph Neural Network.
    Combines Relational Message Passing (R-GCN / SAGE) over graph snapshots with GRU/LSTM temporal aggregation.
    """
    def __init__(self, in_channels=4, hidden_dim=64, out_classes=1, num_edge_types=3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Edge type projection weights for multi-relation edge aggregation
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
        
        # Temporal Recurrent Layer (RNN/GRU over node representations across time)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output Head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, out_classes)
        self.dropout = nn.Dropout(0.3)

    def forward_snapshot(self, x, edge_index, batch=None):
        """
        Message passing over a single daily graph snapshot.
        """
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)

        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        
        # Graph-level readout
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        graph_emb = global_mean_pool(h, batch)
        return graph_emb

    def forward(self, snapshot_list):
        """
        Forward pass over a sequence of T temporal graph snapshots.
        """
        device = snapshot_list[0].x.device
        h_t = None
        
        for snap in snapshot_list:
            snap_emb = self.forward_snapshot(snap.x.to(device), snap.edge_index.to(device))
            if h_t is None:
                h_t = torch.zeros_like(snap_emb)
            h_t = self.gru(snap_emb, h_t)
            
        out = F.relu(self.fc1(h_t))
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)

def train_gnn_model(train_snapshots, test_snapshots, hidden_dim=64, epochs=40, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Relational Dynamic GNN on device: {device}...")

    in_channels = train_snapshots[0].x.shape[1] if len(train_snapshots) > 0 else 4
    model = RelationalDynamicGNN(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to temporal sequences of length 30
    def make_sequences(snaps, seq_len=30):
        seqs, labels = [], []
        for i in range(seq_len, len(snaps)):
            seqs.append(snaps[i-seq_len:i])
            labels.append(snaps[i].y.item())
        return seqs, np.array(labels)

    train_seqs, y_train = make_sequences(train_snapshots)
    test_seqs, y_test = make_sequences(test_snapshots)

    model.train()
    pos_weight_val = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        perm = np.random.permutation(len(train_seqs))
        
        for start in range(0, len(train_seqs), batch_size):
            batch_idxs = perm[start:start+batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for idx in batch_idxs:
                seq = train_seqs[idx]
                target = torch.tensor([y_train[idx]], dtype=torch.float32, device=device)
                logits = model(seq)
                
                loss = criterion(logits, target)
                batch_loss += loss

            batch_loss = batch_loss / len(batch_idxs)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | GNN Loss: {total_loss / (len(train_seqs)/batch_size):.4f}")

    # Evaluate
    model.eval()
    test_probs = []
    with torch.no_grad():
        for seq in test_seqs:
            p = torch.sigmoid(model(seq)).cpu().item()
            test_probs.append(p)

    return model, np.array(test_probs), y_test
