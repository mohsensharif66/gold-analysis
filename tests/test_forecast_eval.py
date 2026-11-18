import numpy as np
import pandas as pd

from app.analytics import evaluation as ev, features as ft, forecast as fc, volatility as vol
from tests.helpers import random_walk_ohlc


def build_ctx(n=1200, seed=0, drift=0.0):
    df = random_walk_ohlc(n, seed=seed, drift=drift, vol=0.01)
    r = np.log(df["close"]).diff()
    X = pd.DataFrame(np.random.default_rng(seed).normal(size=(n, len(ft.FULL))), index=df.index, columns=ft.FULL)
    return df, fc.Ctx(np.log(df["close"].to_numpy()), r.to_numpy(), X)


def test_models_never_use_labels_beyond_fit_time():
    """Changing prices after fit time T must not change the forecast made at T."""
    df, ctx1 = build_ctx(900, seed=1)
    logc2 = ctx1.logc.copy(); T = 700
    logc2[T + 1:] += np.random.default_rng(9).normal(0, 0.5, len(logc2) - T - 1).cumsum()
    r2 = np.r_[np.nan, np.diff(logc2)]
    ctx2 = fc.Ctx(logc2, r2, ctx1.F)
    for model_cls in (fc.Ridge_, fc.ElasticNet_, fc.ARDirect, fc.ARRecursive, fc.ExpSmoothing, fc.RidgeX):
        for h in (1, 5, 10):
            a, b = model_cls(), model_cls()
            a.fit(ctx1, h, T); b.fit(ctx2, h, T)
            pa, pb = a.predict(ctx1, np.array([T])), b.predict(ctx2, np.array([T]))
            assert np.allclose(pa, pb, equal_nan=True), (model_cls.__name__, h)


def test_walk_forward_returns_nan_before_init_and_aligned_length():
    df, ctx = build_ctx(900, seed=2)
    models = [fc.RandomWalk(), fc.Ridge_()]
    P = fc.walk_forward(ctx, models, horizons=(3,), init=500, live=True)
    assert len(P[3]["ridge"]) == len(df) and np.isnan(P[3]["ridge"][:500]).all() and np.isfinite(P[3]["ridge"][-1])


def test_dm_test_direction():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 500); good = base * 0.5
    stat, p = ev.dm_test(good, base, 1)
    assert stat < 0 and p < 0.01


def test_causal_interval_coverage_close_to_nominal_on_iid():
    rng = np.random.default_rng(0); n = 3000; sig = np.full(n, 0.01)
    y = rng.normal(0, 0.01, n); pred = np.zeros(n)
    lo, hi = ev.causal_interval(pred, y, sig, 1)
    ok = np.isfinite(lo)
    cov = np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))
    assert 0.76 < cov < 0.84


def test_ensemble_stays_off_for_noise_models():
    rng = np.random.default_rng(1); n = 1500
    y = rng.normal(0, 0.01, n)
    P = {"rw": np.zeros(n), "m1": rng.normal(0, 0.01, n), "m2": rng.normal(0, 0.01, n)}
    pred, W, active = ev.ensemble_predictions(P, y, 1)
    assert not active.any() and np.allclose(pred, 0)


def test_ensemble_turns_on_for_a_genuinely_informative_model():
    rng = np.random.default_rng(2); n = 1500
    signal = rng.normal(0, 0.01, n); y = signal + rng.normal(0, 0.005, n)
    P = {"rw": np.zeros(n), "good": signal, "noise": rng.normal(0, 0.01, n)}
    pred, W, active = ev.ensemble_predictions(P, y, 1)
    assert active[300:].mean() > 0.95
    assert W["good"].iloc[400:].mean() > 0.9 and np.corrcoef(pred[400:], y[400:])[0, 1] > 0.8


def test_backtest_reports_rw_baseline_ratio_of_one():
    df, ctx = build_ctx(1100, seed=3)
    models = [fc.RandomWalk(), fc.MovingAverage()]
    P = fc.walk_forward(ctx, models, horizons=(1,), init=500, live=False)
    v = vol.vol_table(df)
    rep = ev.run_backtest(df.index, df["close"].to_numpy(), P, {1: fc.realized(ctx, 1)}, v["ewma"].to_numpy(),
                          np.array(["sideways"] * len(df), dtype=object), {"rw": "rw", "ma": "ma", "ensemble": "e"})
    assert abs(rep[1]["models"]["rw"]["rmse_ratio"] - 1) < 1e-9
    assert rep[1]["models"]["ma"]["dir_acc"] is None or 0 <= rep[1]["models"]["ma"]["dir_acc"] <= 1
