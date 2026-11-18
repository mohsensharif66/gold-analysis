# GoldTrend Pro — Backend Foundation

Backend-first desktop application for Iranian 18K gold market analysis.

## Requirements
- Windows 10/11
- Python 3.11+
- PyCharm

## Setup (PyCharm Terminal / PowerShell)
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.main init-db
python -m app.main collect
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Commands
```powershell
python -m app.main init-db
python -m app.main collect
python -m app.main build-candles
python -m app.main indicators
python -m app.main pipeline
python -m app.main audit
```

The first implementation is intentionally conservative:
- Raw observations are append-only and idempotent by source/symbol/timestamp.
- All timestamps are stored as UTC ISO-8601 strings.
- TGJU timestamps are interpreted as Asia/Tehran unless explicitly timezone-aware.
- Prices are stored in the target currency/unit: IRR source prices are converted to toman using 0.1.
- Missing or malformed provider fields are logged, not silently fabricated.
- XAUS is the primary XAU/USD spot source; if it times out or fails, the collector attempts Yahoo Finance (`XAUUSD=X`) as a fallback.
- HTTP GET requests use bounded retries and separate connect/read timeouts. A fallback quote is stored under source `YFINANCE`, so provenance remains visible.

The `audit` command reports raw observations by source, candle date ranges/sample counts, and the number of non-null RSI/EMA200 values. Four-hour candles are generated only when a sufficiently populated intraday observation cluster exists; daily observations are never re-labelled as 4-hour candles.

## Important source caveats
Public endpoints and unofficial packages can change without notice. XAUS and TGJU payload parsers are defensive but must be verified against live responses in your environment. Yahoo Finance ticker coverage and quote freshness can also vary by region/network; the fallback is best-effort and is not a guarantee of a live quote. `yfinance-ir` is an unofficial dependency; its ticker symbols and historical coverage may change. Do not treat stale quotes as current prices.

## Project layout
```text
GoldTrend_Pro/
├── app/
│   ├── collectors/
│   │   ├── base.py
│   │   ├── tgju.py
│   │   ├── xaus.py
│   │   └── tgju_history.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── db/
│   │   ├── database.py
│   │   └── schema.sql
│   ├── engines/
│   │   ├── candles.py
│   │   └── indicators.py
│   ├── services/
│   │   └── collection_service.py
│   └── main.py
├── data/
├── logs/
├── tests/
├── .env.example
├── .gitignore
└── requirements.txt
```


## XAU/USD historical data (v4)

The XAU/USD daily backfill now uses the documented XAUS endpoint:
`https://xaus.com/api/v1/history`. XAUS documents up to 1,825 daily closes
(about five years). The endpoint supplies date, close, high and low; it does
not supply open or volume. The application stores the daily close as the raw
observation price and preserves the full returned point in `raw_json`. The
current quote continues to come from `/api/v1/spot`.

The previous Yahoo Finance symbol `XAUUSD=X` returned HTTP 404 in the tested
environment and is no longer used for historical XAU/USD backfill. A 5-year
series is not a substitute for the project's longer-term 10-year research
requirement; a separately verified source is still needed for the older
period. XAUS data is indicative market data and should be cross-checked before
being used for financial decisions.
