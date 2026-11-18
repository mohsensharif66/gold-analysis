"""Backtest metrics, causal interval calibration, regime breakdown and the
performance-weighted ensemble (which stays OFF unless it really beats the RW)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

W_KNOWN = 250       # trailing known-error window for ensemble weights
CAL_WIN = 750       # trailing window for interval calibration
MIN_W = 60


def known_shift(a: np.ndarray, h: int) -> np.ndarray:
    """Value 'known at t': outcome of the forecast made at t-h."""
    out = np.full_like(a, np.nan, dtype=float)
    out[h:] = a[:-h]
    return out


def dm_test(e_model: np.ndarray, e_base: np.ndarray, h: int):
    """Diebold-Mariano on squared errors with Newey-West variance (lag h-1).
    Negative stat => model has lower loss than the base."""
    d = (e_model ** 2 - e_base ** 2); d = d[np.isfinite(d)]
    n = len(d)
    if n < 50: return None, None
    dm = d - d.mean()
    var = (dm @ dm) / n
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dm[k:] @ dm[:-k]) / n
    if var <= 0: return None, None
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def causal_interval(pred, y, sig, h, lo_q=0.10, hi_q=0.90):
    """Interval for each t from standardized errors of forecasts already resolved at t.
    Returns (lower, upper) in return space."""
    z = (y - pred) / (sig * np.sqrt(h))
    zk = known_shift(z, h)
    zs = pd.Series(zk)
    ql = zs.rolling(CAL_WIN, min_periods=MIN_W).quantile(lo_q).to_numpy()
    qh = zs.rolling(CAL_WIN, min_periods=MIN_W).quantile(hi_q).to_numpy()
    s = sig * np.sqrt(h)
    return pred + ql * s, pred + qh * s


def ensemble_predictions(P: dict, y: np.ndarray, h: int, smooth: float = 0.8, margin: float = 0.995):
    """Causal ensemble. A model joins only if its trailing known MSE beats the RW's
    trailing MSE (x margin). Inverse-MSE weights, smoothed to limit weight churn.
    Returns (pred, weights_df, active_flag_series)."""
    names = [n for n in P if n != "rw"]
    N = len(y)
    mse = {}
    for n in names + ["rw"]:
        e2 = (y - (P[n] if n != "rw" else np.zeros(N))) ** 2
        mse[n] = pd.Series(known_shift(e2, h)).rolling(W_KNOWN, min_periods=MIN_W).mean().to_numpy()
    W = np.zeros((N, len(names)))
    prev = np.zeros(len(names))
    pred = np.zeros(N); active = np.zeros(N, bool)
    for t in range(N):
        raw = np.zeros(len(names))
        for j, n in enumerate(names):
            if np.isfinite(mse[n][t]) and np.isfinite(mse["rw"][t]) and np.isfinite(P[n][t]) \
                    and mse[n][t] < margin * mse["rw"][t]:
                raw[j] = 1.0 / mse[n][t]
        if raw.sum() > 0:
            raw /= raw.sum()
            cur = smooth * prev + (1 - smooth) * raw
            cur = cur * (raw > 0) if (cur * (raw > 0)).sum() > 0 else raw
            cur /= cur.sum()
            prev = cur
            ok = np.isfinite([P[n][t] for n in names])
            w = cur * ok
            if w.sum() > 0:
                w /= w.sum()
                pred[t] = float(sum(w[j] * P[n][t] for j, n in enumerate(names) if ok[j]))
                W[t] = w; active[t] = True
        else:
            prev = prev * smooth
    return pred, pd.DataFrame(W, columns=names), active


def metrics_for(pred, y, sig, h, close, trend=None, span=None, rw_pred=None):
    ok = np.isfinite(pred) & np.isfinite(y)
    if span is not None: ok &= span
    if ok.sum() < 30:
        return dict(n=int(ok.sum()), valid=False)
    e = (y - pred)[ok]; e0 = (y - (0 if rw_pred is None else rw_pred))[ok]
    mae, mae0 = np.abs(e).mean(), np.abs(e0).mean()
    px = close[ok] * np.exp(pred[ok]); ax = close[ok] * np.exp(y[ok])
    smape = float(np.mean(2 * np.abs(px - ax) / (np.abs(px) + np.abs(ax))) * 100)
    nz = (y[ok] != 0) & (pred[ok] != 0)          # a zero forecast makes no directional call
    diracc = float(np.mean(np.sign(pred[ok][nz]) == np.sign(y[ok][nz]))) if nz.sum() >= 30 else None
    up_rate = float(np.mean(y[ok][nz] > 0)) if nz.sum() >= 30 else None
    n_dir = int(nz.sum())
    dmstat, dmp = dm_test(e, e0, h)
    lo, hi = causal_interval(pred, y, sig, h)
    cov_ok = ok & np.isfinite(lo) & np.isfinite(hi)
    cov = float(np.mean((y[cov_ok] >= lo[cov_ok]) & (y[cov_ok] <= hi[cov_ok]))) if cov_ok.sum() > 30 else None
    res = dict(valid=True, n=int(ok.sum()), mae=float(mae), rmse=float(np.sqrt((e ** 2).mean())),
               rmse_rw=float(np.sqrt((e0 ** 2).mean())), mase_vs_rw=float(mae / mae0),
               rmse_ratio=float(np.sqrt((e ** 2).mean()) / np.sqrt((e0 ** 2).mean())),
               smape_price=smape, dir_acc=diracc, base_up_rate=up_rate, n_dir=n_dir,
               interval_80_coverage=cov, dm_stat=dmstat, dm_p=dmp,
               span_start=None, span_end=None)
    if trend is not None:
        by = {}
        for reg in ("bullish_trend", "bearish_trend", "sideways"):
            m2 = ok & (trend == reg)
            if m2.sum() >= 40:
                ee, e00 = (y - pred)[m2], (y - (0 if rw_pred is None else rw_pred))[m2]
                dd = (y[m2] != 0) & (pred[m2] != 0)
                by[reg] = dict(n=int(m2.sum()), rmse_ratio=float(np.sqrt((ee ** 2).mean()) / np.sqrt((e00 ** 2).mean())),
                               dir_acc=float(np.mean(np.sign(pred[m2][dd]) == np.sign(y[m2][dd]))) if dd.sum() >= 30 else None)
        res["by_regime"] = by
    return res


def run_backtest(index, close, P_all: dict, Y_all: dict, sigma, trend, labels: dict) -> dict:
    """P_all[h][name] preds; Y_all[h] realized. Returns a JSON-able report."""
    report = {}
    for h, P in P_all.items():
        y = Y_all[h]
        rw = P["rw"]
        ens, W, act = ensemble_predictions(P, y, h)
        all_models = dict(P); all_models["ensemble"] = np.where(np.isfinite(ens), ens, np.nan)
        rows = {}
        for name, p in all_models.items():
            valid_span = np.isfinite(p)
            if name == "ensemble":
                valid_span = np.ones(len(p), bool) & np.isfinite(P["rw"])
                # evaluate the ensemble only where at least the warm-up has passed
                warm = pd.Series(known_shift((y - 0) ** 2, h)).rolling(W_KNOWN, min_periods=MIN_W).count().to_numpy() >= MIN_W
                valid_span &= warm
            rows[name] = metrics_for(p, y, sigma, h, close, trend, span=valid_span, rw_pred=np.zeros(len(p)))
            rows[name]["label"] = labels.get(name, name)
            if rows[name].get("valid"):
                sp = np.where(valid_span & np.isfinite(y) & np.isfinite(p))[0]
                rows[name]["span_start"] = str(index[sp[0]].date()); rows[name]["span_end"] = str(index[sp[-1]].date())
        e = rows["ensemble"]
        enabled = bool(e.get("valid") and e["rmse_ratio"] < 0.99 and e["dm_p"] is not None and e["dm_p"] < 0.10)
        best = min((n for n in rows if n not in ("rw", "ensemble") and rows[n].get("valid")),
                   key=lambda n: rows[n]["rmse_ratio"], default=None)
        report[h] = dict(models=rows, ensemble_enabled=enabled, best_single=best,
                         active_share=float(act[np.isfinite(y)].mean()) if np.isfinite(y).any() else 0.0)
        report[h]["_ens_pred"] = ens; report[h]["_weights"] = W; report[h]["_active"] = act
    return report
