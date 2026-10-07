import json, re
with open("web/index.html", "r", encoding="utf-8") as f:
    text = f.read()
m = re.search(r'<script id="app-data"[^>]*>(.*?)</script>', text, re.DOTALL)
data = json.loads(m.group(1))

cri = data["cri_series"]
print("=== CRI by year (mean/max) ===")
from collections import defaultdict
by_year = defaultdict(list)
for x in cri:
    yr = x["d"][:4]
    by_year[yr].append(x["v"])
for yr in sorted(by_year.keys()):
    vals = by_year[yr]
    print(f"{yr}: mean={sum(vals)/len(vals):.1f}  max={max(vals):.1f}  pts={len(vals)}")
