"""Data quality & validation. Findings are written to `quality_flags`
(flag types prefixed q_) and are never silently 'fixed' in the history."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.data_layer.panel import SYMBOLS, load_ohlc, has_range, current_quotes

OUTLIER_Z = 7.0
MAX_GAP_DAYS = 7
STALE_BAR_DAYS = 4


def check_symbol(db, symbol: str) -> dict:
    df = load_ohlc(db, symbol, include_provisional=False)
    flags = []
    if df.empty:
        return dict(symbol=symbol, bars=0, flags=[dict(type="q_empty", severity="critical", date=None, detail="no data")])
    c = df["close"]
    r = np.log(c).diff().dropna()
    med = r.rolling(120, min_periods=40).median()
    mad = (r - med).abs().rolling(120, min_periods=40).median() * 1.4826
    z = ((r - med) / mad.replace(0, np.nan)).dropna()
    for d, v in z[z.abs() > OUTLIER_Z].items():
        flags.append(dict(type="q_outlier", severity="warning", date=str(d.date()),
                          detail=f"robust z={v:.1f}, return={r[d]*100:.2f}%"))
    gaps = pd.Series(df.index, index=df.index).diff().dt.days
    for d, g in gaps[gaps > MAX_GAP_DAYS].items():
        flags.append(dict(type="q_gap", severity="info", date=str(d.date()), detail=f"{int(g)} calendar days since previous bar"))
    flat = (r == 0).astype(int)
    run = flat.groupby((flat != flat.shift()).cumsum()).transform("sum") * flat
    for d in run[run >= 4].index[::4]:
        flags.append(dict(type="q_flat_run", severity="warning", date=str(d.date()), detail="≥4 identical consecutive closes"))
    if df["repaired"].any():
        flags.append(dict(type="q_ohlc_repaired", severity="info", date=None,
                          detail=f"{int(df['repaired'].sum())} bars had high/low inside open/close; envelope applied"))
    if not has_range(df):
        flags.append(dict(type="q_no_range", severity="warning", date=None, detail="high/low unavailable or degenerate"))
    last = df.index.max()
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - last).days
    if age > STALE_BAR_DAYS:
        flags.append(dict(type="q_stale_history", severity="warning", date=str(last.date()),
                          detail=f"last completed bar is {age} days old"))
    return dict(symbol=symbol, bars=int(len(df)), first=str(df.index.min().date()), last=str(last.date()),
                age_days=int(age), has_ohlc=has_range(df), repaired=int(df["repaired"].sum()), flags=flags)


def source_discrepancy(db) -> list[dict]:
    """Compare the live quote with the latest history close of the same calendar day."""
    out = []
    for sym in SYMBOLS:
        full = load_ohlc(db, sym, include_provisional=True)
        hist = load_ohlc(db, sym, include_provisional=False)
        q = current_quotes(db).get(sym, {})
        if hist.empty or q.get("price") is None:
            continue
        diff = (q["price"] / hist["close"].iloc[-1] - 1) * 100
        out.append(dict(symbol=sym, live=q["price"], last_history_close=float(hist["close"].iloc[-1]),
                        last_history_date=str(hist.index[-1].date()), diff_pct=float(diff),
                        severity="warning" if abs(diff) > 3 else "info",
                        note="اختلاف بین قیمت لحظه‌ای و آخرین کلوز تاریخچه؛ ثبت می‌شود و جایگزین یکدیگر نمی‌شوند"))
    return out


def run_quality(db, persist: bool = True) -> dict:
    per = [check_symbol(db, s) for s in SYMBOLS]
    disc = source_discrepancy(db)
    if persist:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        rows = []
        for p in per:
            for f in p["flags"]:
                rows.append((p["symbol"], "1d", f["date"] or "", f["type"], f["severity"], f["detail"], now))
        with db.connect() as conn:
            conn.execute("DELETE FROM quality_flags WHERE flag_type LIKE 'q\\_%' ESCAPE '\\'")
            conn.executemany("INSERT INTO quality_flags(symbol_code,timeframe,open_time_utc,flag_type,severity,detail,created_at)"
                             " VALUES(?,?,?,?,?,?,?)", rows)
    crit = sum(1 for p in per for f in p["flags"] if f["severity"] == "critical")
    warn = sum(1 for p in per for f in p["flags"] if f["severity"] == "warning")
    status = "critical" if crit else ("warning" if warn else "ok")
    return dict(symbols=per, discrepancies=disc, status=status, critical=crit, warnings=warn)
