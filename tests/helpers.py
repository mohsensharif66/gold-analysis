"""Synthetic fixtures. They exist ONLY inside tests to exercise the engines; the
application never generates data."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.db.database import Database


def make_db(tmp_path):
    db = Database(tmp_path / "t.db"); db.initialize(); return db


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def put_yf(db, symbol, day: pd.Timestamp, o, h, l, c, rial=True):
    """YFINANCE_IR style row: stored price = close in toman, payload in rial (x10)."""
    k = 10 if rial else 1
    ts = iso(datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(hours=3, minutes=30))  # Tehran midnight
    raw = json.dumps({"open": o * k, "high": h * k, "low": l * k, "close": c * k})
    db.insert_observation(db.source_id("YFINANCE_IR"), symbol, ts, float(c), raw, ts)


def put_xaus_hist(db, day: pd.Timestamp, h, l, c):
    ts = iso(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
    db.insert_observation(db.source_id("XAUS_HISTORY"), "XAU_USD", ts, float(c),
                          json.dumps({"high": h, "low": l, "close": c, "open": None}), ts)


def random_walk_ohlc(n=600, seed=0, drift=0.0, vol=0.01, start=100.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    c = start * np.exp(np.cumsum(r))
    o = np.r_[start, c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)
