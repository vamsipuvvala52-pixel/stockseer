# StockSeer AI — ML Stock Forecasting Platform

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets
python generate_data.py

# 3. Run the server
python app.py
# → Open http://localhost:5000
```

## Login Credentials
| Username | Password  | Role  |
|----------|-----------|-------|
| admin    | admin123  | Admin |
| demo     | demo123   | User  |

Or register a new account from the login page.

## Features
- **Login / Register** — session-based auth, SHA-256 password hashing
- **Dashboard** — real-time prices, KPIs, main chart with CI bands
- **Forecast** — Ensemble (Prophet + ARIMA + Random Forest + GBM)
- **Portfolio** — Mean-variance optimization, correlation matrix
- **Signals** — AI trading signals with confidence scores
- **Ticker bar** — Live scrolling prices

## ML Models
| Model        | Type        | Description                        |
|--------------|-------------|------------------------------------|
| Prophet      | Statistical | Trend + Fourier seasonality + CI   |
| ARIMA        | Statistical | Auto-order, integrated diff        |
| RandomForest | ML (sklearn)| 200 trees, 20+ technical features  |
| GBM          | ML (sklearn)| Gradient boosting regressor        |
| **Ensemble** | Combined    | Inverse-MAPE learned weights       |

## Project Structure
```
stockseer/
├── app.py              # Flask backend (auth + all API routes)
├── generate_data.py    # Dataset generator
├── models/
│   └── forecaster.py   # ProphetModel, ARIMAModel, MLForecaster, EnsembleForecaster
├── templates/
│   ├── login.html      # Login / Register page
│   └── index.html      # Main dashboard
├── data/               # Generated CSV datasets (8 stocks × 1042 rows)
├── requirements.txt
└── README.md
```

## API Endpoints
| Method | Endpoint               | Auth | Description           |
|--------|------------------------|------|-----------------------|
| POST   | /api/auth/login        | No   | Login                 |
| POST   | /api/auth/register     | No   | Register new user     |
| POST   | /api/auth/logout       | Yes  | Logout                |
| GET    | /api/stocks            | Yes  | All stocks overview   |
| GET    | /api/stocks/<ticker>   | Yes  | OHLCV data            |
| POST   | /api/forecast          | Yes  | Run ML forecast       |
| GET    | /api/signals           | Yes  | Trading signals       |
| GET    | /api/portfolio         | Yes  | Portfolio optimizer   |
| GET    | /api/market            | Yes  | Market index data     |
| GET    | /api/news/<ticker>     | Yes  | News & sentiment      |
