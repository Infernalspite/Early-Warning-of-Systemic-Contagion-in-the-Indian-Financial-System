"""
scripts/run_ablations.py
========================
Runs all ablation experiments:
1. Topology Ablation (LSTM macro-only vs macro+network)
2. Edge-Type Ablation (Correlation-only vs +Granger vs +Exposure)

Skips ablations whose output CSVs already exist.
"""

import sys
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

TABLES_DIR = ROOT_DIR / "results" / "tables"

def run_all_ablations():
    print("=" * 60)
    print("RUNNING ALL ABLATION EXPERIMENTS FOR PAPER DEFENSE")
    print("=" * 60)

    # 1. Topology ablation — skip if already done
    topo_path = TABLES_DIR / "topology_ablation.csv"
    if topo_path.exists():
        print(f"\n1. Topology Ablation already done -> {topo_path}")
        import pandas as pd
        top_df = pd.read_csv(topo_path)
    else:
        print("\n1. Running Topology Value Ablation...")
        from src.ablation.topology_ablation import run_topology_ablation
        top_df = run_topology_ablation()

    # 2. Edge-type ablation
    edge_path = TABLES_DIR / "edge_type_ablation.csv"
    if edge_path.exists():
        print(f"\n2. Edge-Type Ablation already done -> {edge_path}")
        import pandas as pd
        edge_df = pd.read_csv(edge_path)
    else:
        print("\n2. Running Edge-Type Layer Ablation...")
        from src.ablation.edge_ablation import run_edge_type_ablation
        edge_df = run_edge_type_ablation()

    from scripts.run_pipeline import copy_to_downloads
    copy_to_downloads()
    print("\nAll Ablation Experiments Completed Successfully!")

if __name__ == "__main__":
    run_all_ablations()
