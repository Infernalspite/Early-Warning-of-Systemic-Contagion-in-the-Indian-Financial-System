"""
app.py — Full-stack web server for the Early Warning Systemic Contagion dashboard.

Serves:
  GET /           → index.html (the dashboard with all embedded charts)
  GET /api/live_score → live yfinance pull + LR, RF, XGBoost re-scoring (JSON)
  GET /health     → health check for Render

Run locally:
  python app.py
"""
import os
import sys

from flask import Flask, send_file, jsonify, request

# Allow importing api.live_score from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.live_score import compute_payload

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/api/live_score", methods=["GET", "OPTIONS"])
def live_score():
    if request.method == "OPTIONS":
        resp = app.make_response("")
        resp.status_code = 204
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    payload = compute_payload()
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
