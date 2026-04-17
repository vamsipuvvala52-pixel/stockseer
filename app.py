"""
app.py — StockSeer AI Flask Backend
Endpoints:
  POST /api/auth/login          Login
  POST /api/auth/register       Register
  POST /api/auth/logout         Logout
  GET  /api/stocks              All stocks overview
  GET  /api/stocks/<ticker>     Single stock OHLCV
  POST /api/forecast            Run ML forecast
  GET  /api/signals             Trading signals
  GET  /api/portfolio           Portfolio stats
  GET  /api/market              Market context
  GET  /api/news/<ticker>       Simulated news
"""
import os, sys, json, time, hashlib, secrets
from datetime import datetime, timedelta
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, redirect, url_for

app = Flask(__name__)
# Use env var SECRET_KEY on Render; fall back to a fixed dev key locally
app.secret_key = os.environ.get("SECRET_KEY", "stockseer-dev-secret-change-in-prod")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── In-memory user store (replace with DB in production) ──────────────────────
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Admin User", "role": "admin",
        "created": "2024-01-01",
    },
    "demo": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Demo User", "role": "user",
        "created": "2024-01-01",
    },
}

# ── Forecast cache ─────────────────────────────────────────────────────────────
_cache    = {}
_models   = {}
CACHE_TTL = 600   # 10 min

def cache_get(key):
    e = _cache.get(key)
    if e and time.time() - e["ts"] < CACHE_TTL:
        return e["data"]
    return None

def cache_set(key, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ── Auth helpers ───────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Data loaders ───────────────────────────────────────────────────────────────
def load_stock(ticker, days=None):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.tail(days) if days else df

def available_tickers():
    return sorted([f.replace(".csv","") for f in os.listdir(DATA_DIR)
                   if f.endswith(".csv") and f != "all_stocks.csv"])

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("index.html", user=session["user"])

@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

# ── Auth API ───────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = data.get("username","").strip().lower()
    password = data.get("password","")
    user = USERS.get(username)
    if not user or user["password_hash"] != hash_pw(password):
        return jsonify({"error": "Invalid credentials"}), 401
    session.permanent = True
    session["user"] = {"username": username, "name": user["name"], "role": user["role"]}
    return jsonify({"success": True, "user": session["user"]})

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    username = data.get("username","").strip().lower()
    password = data.get("password","")
    name     = data.get("name","").strip()
    if not username or not password or not name:
        return jsonify({"error": "All fields required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be ≥ 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be ≥ 6 characters"}), 400
    if username in USERS:
        return jsonify({"error": "Username already taken"}), 409
    USERS[username] = {
        "password_hash": hash_pw(password),
        "name": name, "role": "user",
        "created": str(datetime.today().date()),
    }
    session.permanent = True
    session["user"] = {"username": username, "name": name, "role": "user"}
    return jsonify({"success": True, "user": session["user"]})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/me")
def api_me():
    if "user" not in session:
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "user": session["user"]})

# ── Stocks API ─────────────────────────────────────────────────────────────────
@app.route("/api/stocks")
@login_required
def api_stocks():
    ck = "stocks_overview"
    cached = cache_get(ck)
    if cached: return jsonify(cached)

    STOCK_INFO = {
        "AAPL":{"name":"Apple Inc.","sector":"Technology"},
        "MSFT":{"name":"Microsoft Corp.","sector":"Technology"},
        "GOOGL":{"name":"Alphabet Inc.","sector":"Technology"},
        "TSLA":{"name":"Tesla Inc.","sector":"EV"},
        "NVDA":{"name":"NVIDIA Corp.","sector":"Chips"},
        "AMZN":{"name":"Amazon.com","sector":"Retail"},
        "META":{"name":"Meta Platforms","sector":"Social"},
        "NFLX":{"name":"Netflix Inc.","sector":"Media"},
    }
    result = []
    for t in available_tickers():
        df = load_stock(t, days=5)
        if df is None: continue
        last  = round(float(df["Close"].iloc[-1]), 2)
        prev  = round(float(df["Close"].iloc[-2]), 2) if len(df)>1 else last
        chg   = round((last-prev)/prev*100, 2)
        vol   = int(df["Volume"].iloc[-1])
        info  = STOCK_INFO.get(t, {"name":t,"sector":"—"})
        rsi   = round(float(df["RSI"].dropna().iloc[-1]), 1) if "RSI" in df else 50.0
        result.append({
            "ticker": t, "name": info["name"], "sector": info["sector"],
            "price": last, "change_pct": chg, "volume": vol, "rsi": rsi,
            "trend": "up" if chg >= 0 else "down",
        })
    out = {"stocks": result, "count": len(result),
           "updated": datetime.now().strftime("%H:%M:%S")}
    cache_set(ck, out)
    return jsonify(out)

@app.route("/api/stocks/<ticker>")
@login_required
def api_stock(ticker):
    days = int(request.args.get("days", 252))
    df = load_stock(ticker.upper(), days=days)
    if df is None:
        return jsonify({"error": f"{ticker} not found"}), 404
    # Return only numeric columns, replace NaN with None
    numeric = df.select_dtypes(include=[np.number])
    records = numeric.replace({np.nan: None}).reset_index().to_dict(orient="records")
    return jsonify({
        "ticker": ticker.upper(), "rows": len(df),
        "columns": list(numeric.columns),
        "data": records
    })

# ── Forecast API ───────────────────────────────────────────────────────────────
@app.route("/api/forecast", methods=["POST"])
@login_required
def api_forecast():
    body    = request.get_json(force=True)
    ticker  = body.get("ticker","AAPL").upper()
    model   = body.get("model","ensemble").lower()
    horizon = int(body.get("horizon", 30))
    days    = int(body.get("training_days", 504))

    ck = f"{ticker}_{model}_{horizon}_{days}"
    cached = cache_get(ck)
    if cached: return jsonify({**cached, "cached": True})

    df = load_stock(ticker, days=days)
    if df is None:
        return jsonify({"error": f"{ticker} not found"}), 404

    start = time.time()
    try:
        from models.forecaster import EnsembleForecaster, ProphetModel, ARIMAModel, MLForecaster, trading_signal

        if model == "prophet":
            m = ProphetModel(n_changepoints=20, interval_width=0.80)
            m.fit(df.index, df["Close"].values)
            future = pd.bdate_range(df.index[-1]+pd.Timedelta(days=1), periods=horizon)
            pred   = m.predict(future)
            pred["date"] = pred["ds"].astype(str)
            records = pred[["date","yhat","lower","upper"]].to_dict(orient="records")
            sig  = trading_signal(df, pred.rename(columns={"ds":"date"}))
            result = {"ticker":ticker,"model":"prophet","horizon":horizon,
                      "forecast":records,"signal":sig,
                      "metrics":{"note":"Prophet only"},
                      "elapsed_s":round(time.time()-start,2)}

        elif model == "arima":
            m    = ARIMAModel(max_p=4, max_q=2)
            m.fit(df["Close"].values)
            pred = m.predict(steps=horizon)
            records = pred.to_dict(orient="records")
            last_pred = pred.tail(1).rename(columns={"yhat":"yhat"})
            sig  = trading_signal(df, pd.DataFrame({"yhat":[float(pred["yhat"].iloc[-1])]}))
            result = {"ticker":ticker,"model":"arima","horizon":horizon,
                      "order":f"({m.p_},1,{m.q_})","forecast":records,
                      "signal":sig,"metrics":{"note":"ARIMA only"},
                      "elapsed_s":round(time.time()-start,2)}

        else:  # ensemble (default)
            ens  = EnsembleForecaster(horizon=horizon)
            ens.fit(df)
            pred = ens.predict()
            pred["date"] = pred["date"].astype(str)
            records = pred.to_dict(orient="records")
            sig  = trading_signal(df, pred)
            result = {"ticker":ticker,"model":"ensemble","horizon":horizon,
                      "weights":ens.weights,"metrics":ens.metrics_,
                      "forecast":records,"signal":sig,
                      "feature_importance":ens.rf.feature_importance(),
                      "elapsed_s":round(time.time()-start,2)}
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}),500

    cache_set(ck, result)
    return jsonify({**result,"cached":False})

# ── Signals API ────────────────────────────────────────────────────────────────
@app.route("/api/signals")
@login_required
def api_signals():
    ck = "signals_all"
    cached = cache_get(ck)
    if cached: return jsonify(cached)

    from models.forecaster import EnsembleForecaster, trading_signal
    results = []
    for t in available_tickers():
        df = load_stock(t, days=252)
        if df is None: continue
        try:
            ens  = EnsembleForecaster(horizon=20)
            ens.fit(df)
            pred = ens.predict()
            sig  = trading_signal(df, pred)
            sig["ticker"] = t
            sig["price"]  = round(float(df["Close"].iloc[-1]),2)
            sig["model_weights"] = ens.weights
            results.append(sig)
        except:
            pass

    results.sort(key=lambda x: x["confidence"], reverse=True)
    out = {"signals":results,"generated":datetime.now().strftime("%Y-%m-%d %H:%M")}
    cache_set(ck, out)
    return jsonify(out)

# ── Portfolio API ──────────────────────────────────────────────────────────────
@app.route("/api/portfolio")
@login_required
def api_portfolio():
    ck = "portfolio"
    cached = cache_get(ck)
    if cached: return jsonify(cached)

    tickers = available_tickers()
    prices  = {}
    for t in tickers:
        df = load_stock(t, days=504)
        if df is not None:
            prices[t] = df["Close"]
    prices_df = pd.DataFrame(prices).dropna()

    rets   = prices_df.pct_change().dropna()
    mu     = rets.mean() * 252
    cov    = rets.cov() * 252
    n      = len(tickers)
    w_ew   = np.ones(n)/n
    w_mv   = w_ew.copy()

    from scipy.optimize import minimize
    def neg_sharpe(w):
        r = float(w @ mu.values); v = float(np.sqrt(w@cov.values@w))
        return -(r-0.05)/(v+1e-10)
    bounds = [(0.02,0.45)]*n
    cons   = [{"type":"eq","fun":lambda w:sum(w)-1}]
    res    = minimize(neg_sharpe, w_ew, method="SLSQP",
                      bounds=bounds, constraints=cons,
                      options={"maxiter":500,"ftol":1e-9})
    if res.success:
        w_mv = np.maximum(res.x,0); w_mv/=w_mv.sum()

    def pstats(w):
        r=float(w@mu.values); v=float(np.sqrt(w@cov.values@w))
        return round(r*100,2),round(v*100,2),round((r-0.05)/(v+1e-10),3)

    ret_ew,vol_ew,sh_ew = pstats(w_ew)
    ret_mv,vol_mv,sh_mv = pstats(w_mv)

    out = {
        "equal_weight": {
            "weights":{t:round(float(w),4) for t,w in zip(tickers,w_ew)},
            "return_%":ret_ew,"vol_%":vol_ew,"sharpe":sh_ew
        },
        "max_sharpe": {
            "weights":{t:round(float(w),4) for t,w in zip(tickers,w_mv)},
            "return_%":ret_mv,"vol_%":vol_mv,"sharpe":sh_mv
        },
        "correlation": rets.corr().round(3).to_dict(),
        "ann_returns":  {t:round(float(mu[t]*100),2) for t in tickers},
        "ann_vols":     {t:round(float(np.sqrt(cov.loc[t,t])*100),2) for t in tickers},
    }
    cache_set(ck, out)
    return jsonify(out)

# ── Market API ─────────────────────────────────────────────────────────────────
@app.route("/api/market")
@login_required
def api_market():
    np.random.seed(int(datetime.today().strftime("%j")))
    n = 30
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    sp500 = (4700*np.exp(np.cumsum(0.0003+np.random.randn(n)*0.01))).round(2)
    vix   = np.clip(20+np.cumsum(np.random.randn(n)*0.4),12,45).round(2)
    return jsonify({
        "sp500":  {"labels":[str(d.date()) for d in dates],"values":sp500.tolist()},
        "vix":    {"labels":[str(d.date()) for d in dates],"values":vix.tolist()},
        "summary":{
            "sp500_last":  round(float(sp500[-1]),2),
            "sp500_chg_%": round(float((sp500[-1]-sp500[-2])/sp500[-2]*100),2),
            "vix_last":    round(float(vix[-1]),2),
            "fear_greed":  int(np.clip(100-vix[-1]*2,10,95)),
        }
    })

# ── News API (simulated) ───────────────────────────────────────────────────────
@app.route("/api/news/<ticker>")
@login_required
def api_news(ticker):
    NEWS = {
        "AAPL": [
            {"title":"Apple Vision Pro demand exceeds analyst estimates",
             "sentiment":"positive","time":"2h ago","source":"Bloomberg"},
            {"title":"iPhone 16 supply chain expansion reported in Asia",
             "sentiment":"positive","time":"5h ago","source":"Reuters"},
            {"title":"Apple exploring AI chip partnership with TSMC",
             "sentiment":"positive","time":"1d ago","source":"WSJ"},
        ],
        "NVDA": [
            {"title":"NVIDIA H100 backlog extends to 12 months",
             "sentiment":"positive","time":"1h ago","source":"Bloomberg"},
            {"title":"Data center revenue hits new quarterly record",
             "sentiment":"positive","time":"4h ago","source":"CNBC"},
        ],
        "TSLA": [
            {"title":"Tesla price cuts impact Q3 margins",
             "sentiment":"negative","time":"3h ago","source":"Reuters"},
            {"title":"Cybertruck production ramp slower than expected",
             "sentiment":"negative","time":"6h ago","source":"Bloomberg"},
            {"title":"Full Self-Driving v12 rollout praised by beta users",
             "sentiment":"positive","time":"8h ago","source":"Electrek"},
        ],
        "MSFT": [
            {"title":"Azure cloud growth accelerates on AI demand",
             "sentiment":"positive","time":"2h ago","source":"CNBC"},
            {"title":"Microsoft Copilot adoption surges in enterprise",
             "sentiment":"positive","time":"1d ago","source":"Bloomberg"},
        ],
    }
    items = NEWS.get(ticker.upper(), [
        {"title":f"Analysts maintain BUY rating on {ticker}",
         "sentiment":"positive","time":"3h ago","source":"MarketWatch"},
        {"title":f"{ticker} reports inline earnings, guidance raised",
         "sentiment":"positive","time":"1d ago","source":"Barron's"},
    ])
    return jsonify({"ticker": ticker.upper(), "news": items})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "tickers": available_tickers()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  StockSeer AI — Flask Server")
    print(f"  http://localhost:{port}")
    print("  Login: admin / admin123  |  demo / demo123")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
