import numpy as np
import pandas as pd

from app.analytics import features as ft, patterns as pt, volatility as vol
from tests.helpers import random_walk_ohlc


def test_theoretical_price_formula():
    assert abs(ft.theoretical_gold(2000, 50000) - 2000 * 50000 * 0.75 / 31.1034768) < 1e-6


def test_sqrt_time_baseline_and_calibrated_coverage():
    df = random_walk_ohlc(3000, seed=0, vol=0.01)
    sig = vol.ewma_vol(vol.log_returns(df["close"]))
    cal = vol.calibrate(df["close"], sig)
    assert cal[5]["valid"] and 0.72 < cal[5]["heldout_coverage"] < 0.88
    it = vol.interval(100.0, 0.01, 4, {})                       # uncalibrated fallback = baseline
    assert it["basis"].startswith("sqrt-time") and abs(it["sigma_h"] - 0.02) < 1e-12


def test_pattern_matches_have_full_future_and_no_overlap_with_query():
    c = random_walk_ohlc(900, seed=7)["close"]
    r = pt.find_similar(c, window=20)
    assert r["valid"] and 1 <= r["k"] <= 10
    last = len(c) - 1
    ends = [c.index.get_loc(pd.Timestamp(m["end_date"])) for m in r["matches"]]
    assert all(e <= last - 20 for e in ends)                       # no overlap, 10d future known
    assert all(abs(a - b) >= 10 for i, a in enumerate(ends) for b in ends[i + 1:])
    assert all(set(m["future_returns"]) == {1, 3, 5, 7, 10} for m in r["matches"])
    assert all("base_rate_up_pct" in s for s in r["summary"].values())


def test_pattern_engine_finds_planted_repeat():
    rng = np.random.default_rng(1); n = 800
    r = rng.normal(0, 0.004, n); shape = np.r_[np.linspace(0, 0.03, 10), np.linspace(0.03, -0.02, 10)] / 1
    for start in (100, 400, n - 20):
        r[start:start + 20] = np.diff(np.r_[0, shape])
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2018-01-01", periods=n))
    res = pt.find_similar(c, window=20)
    ends = [c.index.get_loc(pd.Timestamp(m["end_date"])) for m in res["matches"][:3]]
    assert any(abs(e - 419) <= 3 or abs(e - 119) <= 3 for e in ends)
