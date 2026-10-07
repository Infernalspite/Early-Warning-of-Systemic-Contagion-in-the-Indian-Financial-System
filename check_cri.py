import json, re
with open("web/index.html", "r", encoding="utf-8") as f:
    text = f.read()
m = re.search(r'<script id="app-data"[^>]*>(.*?)</script>', text, re.DOTALL)
data = json.loads(m.group(1))

cri = data["cri_series"]
vals = [x["v"] for x in cri]
print(f"CRI series: {len(cri)} points")
print(f"Min: {min(vals):.2f}  Max: {max(vals):.2f}  Std: {(sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5:.4f}")
print(f"First 5: {vals[:5]}")
print(f"Last 5:  {vals[-5:]}")
print(f"Sample from middle: {vals[2500:2505]}")
print()
# Check uniqueness
unique_vals = len(set(round(v,2) for v in vals))
print(f"Unique rounded values: {unique_vals}")

import csv
with open("data/processed/features.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"\nFeatures CSV rows: {len(rows)}")
cols = list(rows[0].keys())
print(f"Columns: {cols}")
# Check raw stress values
stress_col = "continuous_stress_score"
if stress_col in cols:
    stress_vals = [float(r[stress_col]) if r[stress_col] else 0 for r in rows]
    unique_stress = len(set(round(v,4) for v in stress_vals))
    print(f"continuous_stress_score: min={min(stress_vals):.4f} max={max(stress_vals):.4f} unique={unique_stress}")
