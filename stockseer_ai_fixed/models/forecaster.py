"""
models/forecaster.py
ML forecasting engine:
  - Prophet-style decomposable model
  - ARIMA (pure NumPy)
  - Random Forest (sklearn)
  - Ensemble (weighted average)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
#  Prophet-style model
# ════════════════════════════════════════════════════════════
class ProphetModel:
    def __init__(self, n_changepoints=20, interval_width=0.80,
                 yearly_order=8, weekly_order=3):
        self.n_cp = n_changepoints
        self.ci   = interval_width
        self.yo   = yearly_order
        self.wo   = weekly_order
        self.fitted = False

    def _fourier(self, t, period, order):
        cols = []
        for k in range(1, order+1):
            cols += [np.sin(2*np.pi*k*t/period),
                     np.cos(2*np.pi*k*t/period)]
        return np.column_stack(cols)

    def _trend(self, t, k, m, deltas, s_j):
        a = np.array([(t >= sj).astype(float) for sj in s_j]).T
        return (k + a @ deltas) * t + (m + a @ (-s_j * deltas))

    def fit(self, dates, y):
        dates = pd.to_datetime(dates)
        self.t0_    = dates[0]
        self.tscale_= max((dates[-1]-dates[0]).days, 1)
        t = np.array((dates - self.t0_).days) / self.tscale_
        self.y_mu_  = y.mean(); self.y_sd_ = y.std() + 1e-8
        ys = (y - self.y_mu_) / self.y_sd_
        # Changepoints on first 80% of time
        self.cps_ = t[np.linspace(0, int(0.8*len(t))-1,
                                   self.n_cp, dtype=int)]
        t_days = t * self.tscale_
        X_seas = np.hstack([self._fourier(t_days, 365.25, self.yo),
                             self._fourier(t_days, 7,      self.wo)])
        ns, nc = X_seas.shape[1], self.n_cp

        def loss(p):
            k, m = p[0], p[1]
            dlts = p[2:2+nc]
            beta = p[2+nc:]
            tr  = self._trend(t, k, m, dlts, self.cps_)
            se  = X_seas @ beta
            r   = ys - tr - se
            return 0.5*np.sum(r**2) + 0.05*np.sum(np.abs(dlts)) + 0.01*np.sum(beta**2)

        x0 = np.zeros(2+nc+ns)
        x0[0] = (ys[-1]-ys[0])/(t[-1]-t[0]+1e-9); x0[1] = ys[0]
        res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter":400})
        p = res.x
        self.k_    = p[0]; self.m_    = p[1]
        self.dlts_ = p[2:2+nc]; self.beta_ = p[2+nc:]
        yhat = self._trend(t,self.k_,self.m_,self.dlts_,self.cps_) + X_seas@self.beta_
        self.sigma_ = float(np.std(ys - yhat))
        self.fitted = True
        return self

    def predict(self, future_dates):
        from scipy.stats import norm
        fd = pd.to_datetime(future_dates)
        t  = np.array((fd - self.t0_).days) / self.tscale_
        td = t * self.tscale_
        X  = np.hstack([self._fourier(td,365.25,self.yo),
                         self._fourier(td,7,     self.wo)])
        tr   = self._trend(t,self.k_,self.m_,self.dlts_,self.cps_)
        yhat_s = tr + X @ self.beta_
        yhat   = yhat_s * self.y_sd_ + self.y_mu_
        z      = norm.ppf((1+self.ci)/2)
        unc    = z * self.sigma_ * self.y_sd_ * np.sqrt(1 + np.arange(len(t))*0.008)
        return pd.DataFrame({
            "ds":    fd, "yhat": np.round(yhat,3),
            "lower": np.round(yhat-unc,3),
            "upper": np.round(yhat+unc,3),
        })


# ════════════════════════════════════════════════════════════
#  ARIMA (pure NumPy, auto-order)
# ════════════════════════════════════════════════════════════
class ARIMAModel:
    def __init__(self, max_p=4, max_q=2):
        self.max_p = max_p; self.max_q = max_q

    def _yule_walker(self, y, p):
        n  = len(y); ym = y - y.mean()
        r  = np.array([np.dot(ym[k:],ym[:n-k])/n for k in range(p+1)])
        R  = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
        try:
            phi = np.linalg.solve(R + np.eye(p)*1e-8, r[1:])
        except:
            phi = np.zeros(p)
        return phi, max(r[0] - phi@r[1:], 1e-10)

    def _aic(self, y, p, q):
        phi = self._yule_walker(y, p)[0] if p else np.array([])
        eps = np.zeros(len(y))
        for t in range(max(p,q), len(y)):
            ar = sum(phi[k]*y[t-1-k] for k in range(p)) if p else 0
            eps[t] = y[t] - ar
        n   = len(eps); sse = np.sum(eps**2)
        return n*np.log(sse/n+1e-10) + 2*(p+q)

    def fit(self, series):
        y = np.array(series, dtype=float)
        # Difference once
        self.y0_  = y[-1]
        yd = np.diff(y); self.yd_ = yd
        # Select p, q by AIC
        best, bp, bq = np.inf, 1, 0
        for p in range(self.max_p+1):
            for q in range(self.max_q+1):
                if p==0 and q==0: continue
                try:
                    ic = self._aic(yd, p, q)
                    if ic < best: best, bp, bq = ic, p, q
                except: pass
        self.p_ = bp; self.q_ = bq
        self.phi_, self.sigma2_ = self._yule_walker(yd, bp) if bp else (np.array([]),np.var(yd))
        self.mu_  = float(yd.mean())
        # Residuals
        eps = np.zeros(len(yd))
        for t in range(max(bp,bq), len(yd)):
            ar = sum(self.phi_[k]*yd[t-1-k] for k in range(bp)) if bp else 0
            eps[t] = yd[t] - self.mu_ - ar
        self.eps_ = eps
        return self

    def predict(self, steps=30):
        yd = self.yd_.copy(); eps = self.eps_.copy()
        fcs = []
        for h in range(steps):
            ar = sum(self.phi_[k]*yd[-(k+1)] for k in range(min(self.p_,len(yd))))
            f  = self.mu_ + ar
            fcs.append(f); yd = np.append(yd, f); eps = np.append(eps, 0)
        # Integrate back
        preds = np.cumsum([self.y0_] + list(fcs))[1:]
        ci    = 1.96*np.sqrt(self.sigma2_)*np.sqrt(1+np.arange(steps)*0.3)
        return pd.DataFrame({
            "step":  range(1,steps+1),
            "yhat":  np.round(preds,3),
            "lower": np.round(preds-ci,3),
            "upper": np.round(preds+ci,3),
        })


# ════════════════════════════════════════════════════════════
#  Random Forest / GBM (sklearn)
# ════════════════════════════════════════════════════════════
class MLForecaster:
    FEATURES = ["MA5","MA20","MA50","RSI","MACD","Signal","ATR",
                "BB_width","Stoch","Ret1d","Ret5d","Vol20",
                "Lag1","Lag2","Lag3","Lag5"]

    def __init__(self, model_type="rf", seq=20):
        self.mtype = model_type; self.seq = seq
        self.scaler = StandardScaler()
        if model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                                min_samples_leaf=5, n_jobs=-1, random_state=42)
        else:
            self.model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                    learning_rate=0.05, random_state=42)

    def _build_xy(self, df):
        feats = [f for f in self.FEATURES if f in df.columns]
        X_raw = df[feats].values; y = df["Close"].values
        X_s   = self.scaler.fit_transform(X_raw)
        # Add rolling window features
        Xs, ys = [], []
        for i in range(self.seq, len(X_s)):
            row = X_s[i].tolist() + list(X_s[i-self.seq:i, 0])  # close lags
            Xs.append(row); ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, df):
        df = df.dropna(subset=self.FEATURES+["Close"])
        X, y = self._build_xy(df)
        self.model.fit(X, y)
        self.last_df_  = df
        self.last_X_   = X[-1:].copy()
        self.last_y_   = float(df["Close"].iloc[-1])
        self.fitted    = True
        # In-sample error
        yhat = self.model.predict(X)
        self.sigma_ = float(np.std(y - yhat))
        return self

    def predict(self, steps=30):
        feats = [f for f in self.FEATURES if f in self.last_df_.columns]
        n_named = len(feats)
        preds = []
        X = self.last_X_.copy()
        for h in range(steps):
            yhat = float(self.model.predict(X)[0])
            preds.append(yhat)
            new_row = X[0].copy()
            # Shift the rolling window lags left by one and insert the new scaled close
            new_close_scaled = (yhat - self.scaler.mean_[0]) / self.scaler.scale_[0]
            new_row[n_named+1:] = new_row[n_named:-1]   # shift window right→left
            new_row[n_named]    = new_close_scaled        # newest lag = current prediction
            X = new_row.reshape(1, -1)
        unc = self.sigma_ * np.sqrt(1 + np.arange(steps)*0.02)
        return pd.DataFrame({
            "step":  range(1,steps+1),
            "yhat":  np.round(preds,3),
            "lower": np.round(np.array(preds)-1.96*unc,3),
            "upper": np.round(np.array(preds)+1.96*unc,3),
        })

    def feature_importance(self):
        feats = [f for f in self.FEATURES if f in self.last_df_.columns]
        base  = feats + [f"lag_{i}" for i in range(self.seq)]
        imp   = self.model.feature_importances_[:len(base)]
        return sorted(zip(base, imp), key=lambda x:-x[1])[:10]


# ════════════════════════════════════════════════════════════
#  Ensemble
# ════════════════════════════════════════════════════════════
class EnsembleForecaster:
    def __init__(self, horizon=30):
        self.horizon  = horizon
        self.prophet  = ProphetModel(n_changepoints=20, interval_width=0.80)
        self.arima    = ARIMAModel(max_p=4, max_q=2)
        self.rf       = MLForecaster(model_type="rf")
        self.gbm      = MLForecaster(model_type="gbm")
        self.weights  = {"prophet":0.30,"arima":0.20,"rf":0.25,"gbm":0.25}
        self.metrics_ = {}

    def fit(self, df):
        df = df.copy().dropna(subset=["Close"])
        close = df["Close"].values
        dates = df.index if hasattr(df.index, "day") else pd.to_datetime(df.index)

        # Train / val split (85/15)
        split = int(len(df)*0.85)
        train_df  = df.iloc[:split]; val_df = df.iloc[split:]
        train_c   = close[:split];   val_c  = close[split:]

        print(f"    Prophet... ", end="", flush=True)
        self.prophet.fit(dates[:split], train_c)
        p_pred = self.prophet.predict(dates[split:])["yhat"].values[:len(val_c)]
        print("done")

        print(f"    ARIMA...   ", end="", flush=True)
        self.arima.fit(train_c)
        a_pred = self.arima.predict(len(val_c))["yhat"].values
        print("done")

        print(f"    RandomForest... ", end="", flush=True)
        self.rf.fit(train_df)
        r_pred = self.rf.predict(len(val_c))["yhat"].values
        print("done")

        print(f"    GBM...     ", end="", flush=True)
        self.gbm.fit(train_df)
        g_pred = self.gbm.predict(len(val_c))["yhat"].values
        print("done")

        # Compute MAPE per model → inverse-MAPE weights
        def mape(true, pred):
            n = min(len(true), len(pred))
            return np.mean(np.abs(true[:n]-pred[:n])/(np.abs(true[:n])+1e-8))*100

        mapes = {
            "prophet": mape(val_c, p_pred),
            "arima":   mape(val_c, a_pred),
            "rf":      mape(val_c, r_pred),
            "gbm":     mape(val_c, g_pred),
        }
        inv = {k: 1/(v+1e-6) for k,v in mapes.items()}
        total = sum(inv.values())
        self.weights = {k: round(v/total,4) for k,v in inv.items()}

        # Metrics on val
        ens_pred = sum(self.weights[k]*p for k,p in zip(
            ["prophet","arima","rf","gbm"],
            [p_pred, a_pred, r_pred, g_pred]
        ))
        n = min(len(val_c), len(ens_pred))
        y, yh = val_c[:n], ens_pred[:n]
        self.metrics_ = {
            "MAE":    round(float(np.mean(np.abs(y-yh))), 3),
            "RMSE":   round(float(np.sqrt(np.mean((y-yh)**2))), 3),
            "MAPE_%": round(float(np.mean(np.abs(y-yh)/(np.abs(y)+1e-8))*100), 3),
            "R2":     round(float(1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2+1e-8)), 4),
        }

        # Refit all on full data
        self.prophet.fit(dates, close)
        self.arima.fit(close)
        self.rf.fit(df)
        self.gbm.fit(df)
        self.dates_ = dates; self.df_ = df
        return self

    def predict(self, steps=None):
        h = steps or self.horizon
        future = pd.bdate_range(self.dates_[-1]+pd.Timedelta(days=1), periods=h)
        p = self.prophet.predict(future)["yhat"].values
        a = self.arima.predict(h)["yhat"].values
        r = self.rf.predict(h)["yhat"].values
        g = self.gbm.predict(h)["yhat"].values
        yhat = (self.weights["prophet"]*p + self.weights["arima"]*a +
                self.weights["rf"]*r      + self.weights["gbm"]*g)
        # Uncertainty from each model's spread
        last = float(self.df_["Close"].iloc[-1])
        unc  = np.abs(yhat - last) * 0.12 + last*0.01 + np.arange(h)*last*0.0005
        return pd.DataFrame({
            "date":  future, "yhat":  np.round(yhat,3),
            "lower": np.round(yhat-1.96*unc,3),
            "upper": np.round(yhat+1.96*unc,3),
            "prophet": np.round(p,3), "arima":   np.round(a,3),
            "rf":      np.round(r,3), "gbm":     np.round(g,3),
        })


# ════════════════════════════════════════════════════════════
#  Trading signals
# ════════════════════════════════════════════════════════════
def trading_signal(df, forecast_df):
    last   = float(df["Close"].iloc[-1])
    target = float(forecast_df["yhat"].iloc[-1])
    upside = (target - last) / last * 100
    rsi    = float(df["RSI"].dropna().iloc[-1]) if "RSI" in df else 50
    macd   = float(df["MACD"].dropna().iloc[-1]) if "MACD" in df else 0
    ma20   = float(df["MA20"].dropna().iloc[-1]) if "MA20" in df else last
    ma50   = float(df["MA50"].dropna().iloc[-1]) if "MA50" in df else last

    score  = 0
    if last > ma20: score += 1
    if last > ma50: score += 1
    if macd > 0:    score += 1
    if rsi < 40:    score += 1
    if rsi > 70:    score -= 1
    if upside > 3:  score += 1
    if upside < -3: score -= 1

    if   score >= 3: signal, color = "STRONG BUY",  "#00f5a0"
    elif score == 2: signal, color = "BUY",          "#4ade80"
    elif score == 1: signal, color = "WEAK BUY",     "#86efac"
    elif score == 0: signal, color = "HOLD",         "#60a5fa"
    elif score ==-1: signal, color = "WEAK SELL",    "#fca5a5"
    elif score ==-2: signal, color = "SELL",         "#f87171"
    else:            signal, color = "STRONG SELL",  "#ef4444"

    confidence = min(95, 50 + abs(score)*9)
    return {
        "signal": signal, "color": color, "score": score,
        "confidence": confidence, "upside_pct": round(upside,2),
        "target": round(target,2), "rsi": round(rsi,1),
        "above_ma20": last>ma20, "above_ma50": last>ma50,
    }
