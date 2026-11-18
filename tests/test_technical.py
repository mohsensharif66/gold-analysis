import numpy as np
import pandas as pd

from app.analytics import technical as ta
from tests.helpers import random_walk_ohlc


def trending(n=400, drift=0.004):
    return random_walk_ohlc(n, seed=1, drift=drift, vol=0.004)


def test_rsi_extremes_and_range():
    c = pd.Series(np.arange(1.0, 120.0))
    assert ta.rsi(c).iloc[-1] > 99
    assert ta.rsi(c[::-1].reset_index(drop=True)).iloc[-1] < 1
    r = ta.rsi(random_walk_ohlc(300)["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_indicators_are_causal():
    df = random_walk_ohlc(500, seed=3)
    full = ta.analyze(df); cut = ta.analyze(df.iloc[:-60])
    a = full["indicators"].iloc[:-60].fillna(-1).to_numpy(); b = cut["indicators"].fillna(-1).to_numpy()
    assert np.allclose(a, b)
    assert np.allclose(full["score_series"].iloc[:-60].fillna(-1), cut["score_series"].fillna(-1))


def test_ichimoku_cloud_is_shifted_forward_and_chikou_not_future():
    df = random_walk_ohlc(200, seed=4)
    ind = ta.compute_indicators(df)
    assert ind["span_a"].first_valid_index() == df.index[26 + 25]       # kijun(26 bars) + 26 shift
    t = 150
    assert ind["chikou_diff"].iloc[t] == df["close"].iloc[t] - df["close"].iloc[t - 26]


def test_score_bullish_in_uptrend_bearish_in_downtrend_and_bounded():
    up = ta.analyze(trending(400, 0.004))["explain"]["score"]
    dn = ta.analyze(trending(400, -0.004))["explain"]["score"]
    assert up > 60 and dn < 40 and 0 <= dn <= up <= 100


def test_close_only_data_drops_ohlc_components_and_renormalises():
    df = random_walk_ohlc(400, seed=5)[["close"]]
    r = ta.analyze(df)
    e = r["explain"]
    assert {"ADX", "Ichimoku", "Volume"} <= set(e["dropped"])
    assert abs(sum(c["effective_weight"] for c in e["components"] if c["valid"]) - 1) < 1e-3   # weights are rounded to 4 d.p. in the output
    assert r["validity"]["ADX/DI"]["valid"] is False and r["validity"]["Volume"]["valid"] is False


def test_divergence_is_only_known_after_pivot_confirmation():
    price = pd.Series([5, 4, 3, 2, 3, 4, 5, 6, 5, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)  # lows at 3 and 10
    ind = pd.Series([5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float)     # higher low on indicator
    ev = ta.find_divergences(price, ind, "X", order=3, min_gap=3, max_gap=30)
    bull = [e for e in ev if e["type"] == "bullish"]
    assert bull and all(e["confirm_idx"] == e["pivot_idx"] + 3 for e in bull)


def test_adx_high_in_strong_trend():
    r = ta.analyze(trending(300, 0.01))["snapshot"]
    assert r["adx"] > 25 and r["di_plus"] > r["di_minus"]
