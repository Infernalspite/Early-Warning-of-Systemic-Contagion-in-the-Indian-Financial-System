#!/usr/bin/env python3
"""
build_and_deploy.py
===================
Comprehensive automation script to:
1. Download latest market data
2. Label crisis periods
3. Engineer features
4. Train all models (LR, RF, XGBoost, LSTM, GNN)
5. Evaluate models
6. Export dashboard data
7. Deploy to Vercel

Run this ONCE to fully deploy the system on Vercel.

Usage:
    python build_and_deploy.py [--vercel-token YOUR_TOKEN] [--no-deploy]
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime

print("\n" + "="*70)
print("🚀 EARLY WARNING SYSTEM - FULL BUILD & DEPLOYMENT PIPELINE")
print("="*70)

# ========================================================================
# CONFIGURATION
# ========================================================================

VERCEL_TOKEN = os.environ.get("VERCEL_TOKEN") or os.environ.get("VERCEL_AUTH_TOKEN")
VERCEL_ORG = os.environ.get("VERCEL_ORG_ID")
VERCEL_PROJECT = "early-warning-systemic-contagion"
DEPLOY_ENABLED = "--no-deploy" not in sys.argv
AUTO_CONFIRM = "--yes" in sys.argv

PIPELINE_STEPS = [
    ("Download Data", "python download_data.py"),
    ("Label Crisis Periods", "python crisis_labels.py"),
    ("Engineer Features", "python feature_engineering.py"),
    ("Train Logistic Regression", "python logistic_regression_model.py"),
    ("Train Random Forest", "python train_model.py"),
    ("Train XGBoost", "python xgboost_model.py"),
    ("Train LSTM", "python lstm_model.py"),
    ("Train GNN", "python gnn_model.py"),
    ("Evaluate All Models", "python evaluate_all_models.py"),
]

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def log_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def log_step(num, total, step_name):
    print(f"\n[{num}/{total}] 🔨 {step_name}")
    print("-" * 70)

def run_command(cmd, shell=True):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_file_exists(path):
    """Check if a file exists."""
    return os.path.isfile(path)

def create_github_action():
    """Create GitHub Actions workflow for continuous deployment."""
    workflow_dir = ".github/workflows"
    os.makedirs(workflow_dir, exist_ok=True)
    
    workflow_content = """name: Deploy Early Warning System

on:
  schedule:
    - cron: '0 9 * * 1-5'  # Daily at 9 AM IST (Monday-Friday)
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install pandas numpy scikit-learn xgboost yfinance matplotlib seaborn
          pip install torch statsmodels networkx
          pip install torch-geometric
      
      - name: Run full pipeline
        run: python build_and_deploy.py --yes --vercel-token ${{ secrets.VERCEL_TOKEN }}
      
      - name: Commit updates
        run: |
          git config user.name "Systemic Risk Bot"
          git config user.email "bot@systemic-risk.dev"
          git add data/ models/ web/ outputs/
          git commit -m "⚙️ Auto-update: models retrained + data refreshed on $(date +%Y-%m-%d)" || echo "Nothing to commit"
          git push || echo "Push skipped"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
"""
    
    workflow_path = os.path.join(workflow_dir, "deploy-pipeline.yml")
    os.makedirs(workflow_dir, exist_ok=True)
    with open(workflow_path, "w") as f:
        f.write(workflow_content)
    print(f"✅ Created GitHub Actions workflow: {workflow_path}")
    return workflow_path

# ========================================================================
# STEP 1: VERIFY DEPENDENCIES
# ========================================================================

log_section("STEP 0: CHECKING DEPENDENCIES")

required_packages = [
    "pandas", "numpy", "scikit-learn", "xgboost", "yfinance",
    "matplotlib", "seaborn", "statsmodels", "networkx"
]

print("Checking Python packages...")
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} (missing)")
        missing_packages.append(pkg)

if missing_packages:
    print(f"\n⚠️  Installing missing packages: {', '.join(missing_packages)}")
    run_command(f"pip install {' '.join(missing_packages)}")

# ========================================================================
# STEP 2: RUN PIPELINE
# ========================================================================

log_section("STEP 1: RUNNING FULL ML PIPELINE")

failed_steps = []
for idx, (step_name, command) in enumerate(PIPELINE_STEPS, 1):
    log_step(idx, len(PIPELINE_STEPS), step_name)
    
    if run_command(command):
        print(f"✅ {step_name} completed successfully")
    else:
        print(f"❌ {step_name} failed")
        failed_steps.append(step_name)

if failed_steps:
    print(f"\n⚠️  {len(failed_steps)} steps failed:")
    for step in failed_steps:
        print(f"  - {step}")
else:
    print(f"\n✅ All {len(PIPELINE_STEPS)} pipeline steps completed successfully!")

# ========================================================================
# STEP 3: VERIFY MODEL FILES
# ========================================================================

log_section("STEP 2: VERIFYING MODEL FILES")

required_models = [
    "models/logistic_regression.pkl",
    "models/scaler_lr.pkl",
    "models/random_forest_india.pkl",
    "models/random_forest_binary.pkl",
    "models/xgboost.pkl",
    "models/lstm_model.pt",
    "web/data/dashboard_data.json",
]

print("Checking model outputs...")
missing_models = []
for model_path in required_models:
    if check_file_exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"  ✅ {model_path:<40} ({size_kb:>8.1f} KB)")
    else:
        print(f"  ❌ {model_path:<40} (missing)")
        missing_models.append(model_path)

if missing_models:
    print(f"\n⚠️  {len(missing_models)} model files missing. Some models may not have trained successfully.")

# ========================================================================
# STEP 4: PREPARE DEPLOYMENT PACKAGE
# ========================================================================

log_section("STEP 3: PREPARING DEPLOYMENT PACKAGE")

print("Organizing files for Vercel deployment...")

# Ensure API directory has models
os.makedirs("api/models", exist_ok=True)
for model_file in ["logistic_regression.pkl", "scaler_lr.pkl", "random_forest_binary.pkl", "xgboost.pkl"]:
    src = f"models/{model_file}"
    dst = f"api/models/{model_file}"
    if check_file_exists(src) and not check_file_exists(dst):
        import shutil
        shutil.copy(src, dst)
        print(f"  ✅ Copied {model_file} to api/models/")

# Ensure data directory has processed features
os.makedirs("api/data/processed", exist_ok=True)
if check_file_exists("data/processed/features_india.csv"):
    import shutil
    shutil.copy("data/processed/features_india.csv", "api/data/processed/features_india.csv")
    print(f"  ✅ Copied features_india.csv to api/data/")
if check_file_exists("data/processed/feature_list.txt"):
    import shutil
    shutil.copy("data/processed/feature_list.txt", "api/data/processed/feature_list.txt")
    print(f"  ✅ Copied feature_list.txt to api/data/")

print("✅ Deployment package ready")

# ========================================================================
# STEP 5: CREATE GITHUB ACTIONS WORKFLOW
# ========================================================================

log_section("STEP 4: SETTING UP CI/CD")

create_github_action()

# ========================================================================
# STEP 6: DEPLOY TO VERCEL
# ========================================================================

if DEPLOY_ENABLED and VERCEL_TOKEN:
    log_section("STEP 5: DEPLOYING TO VERCEL")
    
    print(f"🌐 Deploying to Vercel...")
    print(f"   Project: {VERCEL_PROJECT}")
    
    # Install Vercel CLI if not present
    print("\n   Installing Vercel CLI...")
    run_command("npm install -g vercel")
    
    # Deploy
    print("\n   Deploying...")
    deploy_cmd = f"vercel deploy --prod --token {VERCEL_TOKEN}"
    if VERCEL_ORG:
        deploy_cmd += f" --scope {VERCEL_ORG}"
    
    if run_command(deploy_cmd):
        print(f"✅ Deployment to Vercel completed successfully")
    else:
        print(f"⚠️  Vercel deployment may have issues. Check manually:")
        print(f"    1. Install Vercel CLI: npm install -g vercel")
        print(f"    2. Run: vercel deploy --prod")
else:
    if not DEPLOY_ENABLED:
        print("⏭️  Skipping Vercel deployment (--no-deploy flag set)")
    if not VERCEL_TOKEN:
        print("⏭️  Skipping Vercel deployment (VERCEL_TOKEN not set)")
        print("\n   To enable auto-deployment, set:")
        print("      export VERCEL_TOKEN=<your_vercel_token>")

# ========================================================================
# FINAL SUMMARY
# ========================================================================

log_section("BUILD COMPLETE")

print("📊 Pipeline Summary:")
print(f"   ✅ Steps completed: {len(PIPELINE_STEPS) - len(failed_steps)}/{len(PIPELINE_STEPS)}")
print(f"   ✅ Models trained: {sum(1 for m in required_models if check_file_exists(m))}/{len(required_models)}")
print(f"   ⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n🚀 Next steps:")
print("   1. Push changes to GitHub:")
print("      git add -A")
print('      git commit -m "🔄 Full pipeline run - $(date +%Y-%m-%d)"')
print("      git push origin main")
print()
print("   2. Test the live API:")
print("      curl https://<your-deployment>.vercel.app/api/live_score")
print()
print("   3. Open the dashboard:")
print("      https://early-warning-of-systemic-contagion-weld.vercel.app/")
print()
print("   4. Set up scheduled runs via GitHub Actions (.github/workflows/deploy-pipeline.yml)")

print("\n" + "="*70)
print("✨ DEPLOYMENT PIPELINE COMPLETE ✨")
print("="*70 + "\n")
