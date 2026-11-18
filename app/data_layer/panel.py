"""Daily OHLC panel rebuilt from the raw observations' provider payloads.

Why this exists: the candle builder only sees `price` (a close), but the raw
payload of the history sources carries real open/high/low values:
  * YFINANCE_IR (gold 18k, USD): open/high/low/close in rial -> x0.1 = toman
  * XAUS_HISTORY (XAU/USD): close/high/low (open is null)
No value here is estimated: missing fields stay NaN and are flagged.

Calendar: Iranian symbols are keyed by their Asia/Tehran calendar date, XAU/USD
by its UTC date (as published). Alignment across symbols is as-of (never
looks forward) - see `aligned_panel`.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TEHRAN = ZoneInfo("Asia/Tehran")
SYMBOLS = ("GOLD18_IRR", "USD_IRR", "XAU_USD")
HISTORY_SOURCE = {"GOLD18_IRR": "YFINANCE_IR", "USD_IRR": "YFINANCE_IR", "XAU_USD": "XAUS_HISTORY"}
LIVE_SOURCE = {"GOLD18_IRR": "TGJU", "USD_IRR": "TGJU", "XAU_USD": "XAUS"}


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) and f > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def _fetch(db, symbol: str, source: str):
    with db.connect() as conn:
        return conn.execute(
            """SELECT r.ts_utc, r.price, r.raw_json, r.fetched_at_utc
               FROM raw_observations r JOIN sources s ON s.id=r.source_id
               WHERE r.symbol_code=? AND s.name=? ORDER BY r.ts_utc""",
            (symbol, source),
        ).fetchall()


def _local_date(symbol: str, ts_utc: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if symbol == "XAU_USD":
        return pd.Timestamp(ts.date())
    return pd.Timestamp(ts.tz_convert(TEHRAN).date())


def _history_bars(db, symbol: str) -> pd.DataFrame:
    recs = {}
    for row in _fetch(db, symbol, HISTORY_SOURCE[symbol]):
        close = _num(row["price"])
        if np.isnan(close):
            continue
        o = h = l = np.nan
        try:
            raw = json.loads(row["raw_json"] or "{}")
            rc = _num(raw.get("close"))
            scale = close / rc if rc == rc else 1.0   # provider unit -> stored unit
            o, h, l = (_num(raw.get(k)) * scale for k in ("open", "high", "low"))
        except Exception:
            pass
        d = _local_date(symbol, row["ts_utc"])
        recs[d] = dict(open=o, high=h, low=l, close=close, provisional=False,
                       source=HISTORY_SOURCE[symbol])
    return pd.DataFrame.from_dict(recs, orient="index").sort_index() if recs else pd.DataFrame()


def _live_bar(db, symbol: str):
    """Latest live tick as a provisional bar (never overwrites finished history)."""
    rows = _fetch(db, symbol, LIVE_SOURCE[symbol])
    if not rows:
        return None
    day = _local_date(symbol, rows[-1]["ts_utc"])
    same = [r for r in rows if _local_date(symbol, r["ts_utc"]) == day]
    prices = [float(r["price"]) for r in same]
    hi, lo = max(prices), min(prices)
    try:  # TGJU also publishes the day's range (rial -> toman)
        rec = json.loads(same[-1]["raw_json"] or "{}").get("record", {})
        h = _num(str(rec.get("h", "")).replace(",", "")) * 0.1
        l = _num(str(rec.get("l", "")).replace(",", "")) * 0.1
        if h == h and l == l:
            hi, lo = max(hi, h), min(lo, l)
    except Exception:
        pass
    return day, dict(open=np.nan, high=hi, low=lo, close=prices[-1], provisional=True,
                     source=LIVE_SOURCE[symbol])


def load_ohlc(db, symbol: str, include_provisional: bool = True) -> pd.DataFrame:
    """Daily bars indexed by local date. Columns: open high low close provisional
    source repaired. OHLC inconsistencies are repaired by envelope and flagged."""
    df = _history_bars(db, symbol)
    if include_provisional:
        live = _live_bar(db, symbol)
        if live is not None:
            d, bar = live
            if df.empty or d > df.index.max():
                df.loc[d] = bar
            elif d == df.index.max() and False:
                pass
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "provisional", "source", "repaired"])
    df = df.sort_index()
    df["provisional"] = df["provisional"].astype(bool)
    cols = df[["open", "high", "low", "close"]]
    env_hi = cols.max(axis=1, skipna=True)
    env_lo = cols.min(axis=1, skipna=True)
    bad_hi = df["high"].notna() & (df["high"] < env_hi - 1e-9)
    bad_lo = df["low"].notna() & (df["low"] > env_lo + 1e-9)
    df["repaired"] = bad_hi | bad_lo
    df.loc[bad_hi, "high"] = env_hi[bad_hi]
    df.loc[bad_lo, "low"] = env_lo[bad_lo]
    return df


def has_range(df: pd.DataFrame, min_share: float = 0.8) -> bool:
    """True when high/low are real (present and not just a copy of close)."""
    if df.empty or df["high"].isna().all():
        return False
    ok = (df["high"] > df["low"]).mean()
    return bool(ok >= min_share)


def complete_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["provisional"]]


def aligned_panel(db, xau_lag_days: int = 1, max_stale_days: int = 5):
    """Gold-calendar panel with USD and XAU aligned as-of (no look-ahead).

    * Master calendar = complete gold bars (the forecast target).
    * USD: last value at/before the date (same Tehran clock as gold).
    * XAU/USD: value of date - xau_lag_days (its UTC-day close happens after the
      Tehran close of the same date, so using it would leak the future).
    Values older than `max_stale_days` are NaN rather than silently carried.
    Returns (panel, info).
    """
    gold = complete_bars(load_ohlc(db, "GOLD18_IRR", include_provisional=False))
    usd = load_ohlc(db, "USD_IRR", include_provisional=False)["close"]
    xau = load_ohlc(db, "XAU_USD", include_provisional=False)["close"]
    panel = gold[["open", "high", "low", "close"]].rename(columns=lambda c: f"gold_{c}").copy()

    def asof(series, shift_days):
        s = series.dropna().sort_index()
        idx = panel.index - pd.Timedelta(days=shift_days)
        vals = s.reindex(s.index.union(idx)).ffill().reindex(idx).to_numpy()
        age = pd.Series(s.index, index=s.index).reindex(s.index.union(idx)).ffill().reindex(idx)
        age_days = (idx - pd.DatetimeIndex(age.to_numpy())).days
        vals = np.where(age_days <= max_stale_days, vals, np.nan)
        return pd.Series(vals, index=panel.index)

    panel["usd"] = asof(usd, 0)
    panel["xau"] = asof(xau, xau_lag_days)
    info = dict(rows=int(len(panel)), first=str(panel.index.min().date()) if len(panel) else None,
                last=str(panel.index.max().date()) if len(panel) else None,
                xau_lag_days=xau_lag_days,
                usd_missing=int(panel["usd"].isna().sum()), xau_missing=int(panel["xau"].isna().sum()))
    return panel, info


def current_quotes(db, stale_minutes: int = 180) -> dict:
    """Latest live quote per symbol with provenance and staleness (never presented
    as live when older than the threshold)."""
    now = datetime.now(tz=ZoneInfo("UTC"))
    out = {}
    for sym in SYMBOLS:
        rows = _fetch(db, sym, LIVE_SOURCE[sym])
        if not rows:
            out[sym] = dict(price=None, source=LIVE_SOURCE[sym], stale=True, reason="no live data")
            continue
        r = rows[-1]
        ts = pd.Timestamp(r["ts_utc"]).tz_convert("UTC")
        age = (now - ts.to_pydatetime()).total_seconds() / 60
        out[sym] = dict(price=float(r["price"]), source=LIVE_SOURCE[sym], ts_utc=r["ts_utc"],
                        fetched_at_utc=r["fetched_at_utc"], age_minutes=round(age, 1),
                        stale=bool(age > stale_minutes))
    return out
