import os, sys, json, time, hashlib, secrets
from datetime import datetime, timedelta
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------- USERS ----------------
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Admin User", "role": "admin",
    }
}

# ---------------- CACHE ----------------
_cache = {}
CACHE_TTL = 600

def cache_get(key):
    e = _cache.get(key)
    if e and time.time() - e["ts"] < CACHE_TTL:
        return e["data"]
    return None

def cache_set(key, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ---------------- AUTH ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- DATA ----------------
def load_stock(ticker, days=None):
    try:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df.tail(days) if days else df
    except Exception as e:
        print("LOAD ERROR:", e)
        return None

def available_tickers():
    if not os.path.exists(DATA_DIR):
        return []
    return [f.replace(".csv","") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "StockSeer Backend Running ✅"

# ---------------- LOGIN ----------------
@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    user = USERS.get(data.get("username"))

    if not user or user["password_hash"] != hash_pw(data.get("password")):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user"] = {"username": data["username"]}
    return jsonify({"success": True})

# ---------------- STOCKS ----------------
@app.route("/api/stocks")
@login_required
def stocks():
    tickers = available_tickers()
    return jsonify({"stocks": tickers})

# ---------------- FORECAST ----------------
@app.route("/api/forecast", methods=["POST"])
@login_required
def forecast():
    try:
        body = request.get_json()
        ticker = body.get("ticker", "AAPL")

        df = load_stock(ticker, 100)
        if df is None:
            return jsonify({"error": "Data not found"}), 404

        # SAFE IMPORT (IMPORTANT FIX)
        try:
            from models.forecaster import EnsembleForecaster
        except Exception as e:
            return jsonify({
                "error": "Model import failed",
                "details": str(e)
            }), 500

        model = EnsembleForecaster(horizon=30)
        model.fit(df)

        pred = model.predict()

        return jsonify({
            "ticker": ticker,
            "forecast": pred.to_dict(orient="records")
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Render FIX
    app.run(host="0.0.0.0", port=port)
