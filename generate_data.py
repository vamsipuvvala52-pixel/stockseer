"""Generate realistic OHLCV stock datasets with technical indicators."""
import numpy as np
import pandas as pd
import os, sys

np.random.seed(42)

STOCKS = {
    "AAPL":  {"start": 148.0, "drift": 0.00045, "vol": 0.016, "name": "Apple Inc.",      "sector": "Technology"},
    "MSFT":  {"start": 310.0, "drift": 0.00050, "vol": 0.014, "name": "Microsoft Corp.", "sector": "Technology"},
    "GOOGL": {"start": 135.0, "drift": 0.00040, "vol": 0.018, "name": "Alphabet Inc.",   "sector": "Technology"},
    "TSLA":  {"start": 200.0, "drift": 0.00025, "vol": 0.035, "name": "Tesla Inc.",      "sector": "EV"},
    "NVDA":  {"start": 380.0, "drift": 0.00095, "vol": 0.030, "name": "NVIDIA Corp.",    "sector": "Chips"},
    "AMZN":  {"start": 165.0, "drift": 0.00042, "vol": 0.017, "name": "Amazon.com",      "sector": "Retail"},
    "META":  {"start": 255.0, "drift": 0.00058, "vol": 0.022, "name": "Meta Platforms",  "sector": "Social"},
    "NFLX":  {"start": 378.0, "drift": 0.00028, "vol": 0.025, "name": "Netflix Inc.",    "sector": "Media"},
}

def garch_vol(n):
    sigma2 = np.zeros(n); sigma2[0] = 0.0001
    eps = np.random.randn(n)
    for t in range(1, n):
        sigma2[t] = 0.000005 + 0.08*(eps[t-1]**2)*sigma2[t-1] + 0.91*sigma2[t-1]
    return np.sqrt(sigma2), eps

def make_ohlcv(ticker, cfg, start="2021-01-04", end="2024-12-31"):
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    sigma, eps = garch_vol(n)
    t = np.arange(n)
    season = 0.005*np.sin(2*np.pi*t/5) + 0.008*np.sin(2*np.pi*t/252)
    events = np.zeros(n)
    idxs = np.random.choice(range(30, n-5), 12, replace=False)
    for i in idxs:
        events[i] = np.random.choice([-1,1]) * np.random.uniform(0.04, 0.13)
    ret   = cfg["drift"] + (sigma * cfg["vol"] / sigma.mean()) * eps + season + events
    close = np.exp(np.log(cfg["start"]) + np.cumsum(ret))
    dr = close * np.random.uniform(0.005, 0.025, n)
    op = close * (1 + np.random.randn(n)*0.003)
    hi = np.maximum(close, op) + dr * np.random.uniform(0.3, 0.7, n)
    lo = np.minimum(close, op) - dr * np.random.uniform(0.3, 0.7, n)
    vol = np.random.lognormal(np.log(5e7), 0.6, n).astype(int)
    for i in idxs:
        if i < n: vol[i] = int(vol[i] * np.random.uniform(2.5, 5.0))
    df = pd.DataFrame({
        "Date": dates, "Open": op.round(2), "High": hi.round(2),
        "Low": lo.round(2), "Close": close.round(2),
        "Volume": vol, "Ticker": ticker, "Name": cfg["name"], "Sector": cfg["sector"]
    }).set_index("Date")
    return df

def add_ta(df):
    c = df["Close"]
    df["MA5"]   = c.rolling(5).mean().round(3)
    df["MA20"]  = c.rolling(20).mean().round(3)
    df["MA50"]  = c.rolling(50).mean().round(3)
    df["MA200"] = c.rolling(200).mean().round(3)
    df["EMA12"] = c.ewm(span=12, adjust=False).mean().round(3)
    df["EMA26"] = c.ewm(span=26, adjust=False).mean().round(3)
    df["MACD"]  = (df["EMA12"] - df["EMA26"]).round(4)
    df["Signal"]= df["MACD"].ewm(span=9, adjust=False).mean().round(4)
    df["Hist"]  = (df["MACD"] - df["Signal"]).round(4)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]   = (100 - 100/(1 + gain/loss.replace(0, np.nan))).round(2)
    s20 = c.rolling(20).std()
    df["BB_upper"] = (df["MA20"] + 2*s20).round(3)
    df["BB_lower"] = (df["MA20"] - 2*s20).round(3)
    df["BB_width"] = ((df["BB_upper"] - df["BB_lower"]) / df["MA20"]).round(4)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - c.shift()).abs()
    lc = (df["Low"]  - c.shift()).abs()
    df["ATR"]   = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().round(3)
    lo14 = df["Low"].rolling(14).min(); hi14 = df["High"].rolling(14).max()
    df["Stoch"] = (100*(c-lo14)/(hi14-lo14)).round(2)
    df["OBV"]   = (np.sign(c.diff()).fillna(0)*df["Volume"]).cumsum().astype(int)
    df["Ret1d"] = c.pct_change(1).round(6)
    df["Ret5d"] = c.pct_change(5).round(6)
    df["Ret20d"]= c.pct_change(20).round(6)
    df["LogRet"]= np.log(c/c.shift(1)).round(6)
    df["Vol20"] = df["LogRet"].rolling(20).std().round(6)
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"Lag{lag}"] = c.shift(lag).round(3)
    return df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    print("Generating datasets...")
    all_dfs = []
    for ticker, cfg in STOCKS.items():
        df = make_ohlcv(ticker, cfg)
        df = add_ta(df)
        path = os.path.join(data_dir, f"{ticker}.csv")
        df.to_csv(path)
        all_dfs.append(df)
        print(f"  {ticker}: {len(df)} rows, {len(df.columns)} features → {path}")
    combined = pd.concat(all_dfs)
    combined.to_csv(os.path.join(data_dir, "all_stocks.csv"))
    print(f"\nAll done. Combined: {len(combined)} rows.")
