"""
src/graphs/dataset.py
======================
PyTorch Geometric Dataset wrapper and temporal walk-forward fold constructor.
"""

import pathlib
import torch
import pandas as pd
from torch_geometric.data import Dataset

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class DynamicGraphDataset(Dataset):
    """
    Dataset wrapper for dynamic financial network snapshots.
    """
    def __init__(self, snapshots):
        super().__init__()
        self.snapshots = snapshots

    def len(self):
        return len(self.snapshots)

    def get(self, idx):
        return self.snapshots[idx]

def get_temporal_splits(snapshots, split_date="2022-01-01"):
    """
    Time-respecting train/test split.
    """
    train_snaps = []
    test_snaps = []
    
    for snap in snapshots:
        if hasattr(snap, "date") and snap.date >= split_date:
            test_snaps.append(snap)
        else:
            train_snaps.append(snap)
            
    return DynamicGraphDataset(train_snaps), DynamicGraphDataset(test_snaps)
