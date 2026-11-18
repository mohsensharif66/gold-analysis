"""Gold(IRR) vs USD(free market) vs XAU/USD. Correlation != causation; series are
aligned on dates where each market really printed a bar (no carried values)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.analytics.features import THEORETICAL_K, theoretical_gold
from app.data_layer.panel import load_ohlc, current_quotes


def common_returns(db) -> pd.DataFrame:
    g = load_ohlc(db, "GOLD18_IRR", False)["close"]
    u = load_ohlc(db, "USD_IRR", False)["close"]
    x = load_ohlc(db, "XAU_USD", False)["close"]
    d = pd.concat({"gold": g, "usd": u, "xau": x}, axis=1).dropna()
    r = np.log(d).diff().dropna()
    r["gap_days"] = pd.Series(d.index, index=d.index).diff().dt.days.reindex(r.index)
    return r.rename_axis("date")


def correlations(r: pd.DataFrame, windows=(30, 60, 90)) -> dict:
    out = {"n_common_days": int(len(r)), "full": {}, "rolling": {}}
    for a, b in (("gold", "usd"), ("gold", "xau"), ("usd", "xau")):
        pr = stats.pearsonr(r[a], r[b]); sp = stats.spearmanr(r[a], r[b])
        out["full"][f"{a}~{b}"] = dict(pearson=float(pr[0]), pearson_p=float(pr[1]),
                                       spearman=float(sp[0]), spearman_p=float(sp[1]))
        out["rolling"][f"{a}~{b}"] = {str(w): (None if len(r) < w else float(r[a].tail(w).corr(r[b].tail(w))))
                                      for w in windows}
    return out


def lag_analysis(r: pd.DataFrame, max_lag: int = 5) -> dict:
    """corr(gold_t, other_{t-k}); k>0 means the other market moves first."""
    res = {}
    for other in ("usd", "xau"):
        rows = []
        for k in range(0, max_lag + 1):
            x = r["gold"]; y = r[other].shift(k)
            m = x.notna() & y.notna()
            c = stats.pearsonr(x[m], y[m])
            rows.append(dict(lag=k, corr=float(c[0]), p=float(c[1]), n=int(m.sum())))
        best = max(rows, key=lambda t: abs(t["corr"]))
        res[other] = dict(by_lag=rows, strongest_lag=best["lag"],
                          note="تأخیر مثبت = بازار دیگر زودتر حرکت کرده؛ علیت اثبات نمی‌شود")
    return res


def comovement(r: pd.DataFrame, window: int = 250) -> dict:
    d = r.tail(window).dropna()
    if len(d) < 60:
        return dict(valid=False)
    X = np.column_stack([np.ones(len(d)), d["usd"], d["xau"]]); y = d["gold"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta; ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    k = 20
    implied = float(beta[1] * d["usd"].tail(k).sum() + beta[2] * d["xau"].tail(k).sum())
    actual = float(d["gold"].tail(k).sum())
    return dict(valid=True, window=len(d), beta_usd=float(beta[1]), beta_xau=float(beta[2]), alpha=float(beta[0]),
                r2=float(1 - ss_res / ss_tot), implied_ret_20=implied, actual_ret_20=actual,
                divergence_20=actual - implied)


def premium_series(db, max_stale_days: int = 3) -> pd.DataFrame:
    """Premium on the gold calendar; USD / XAU are taken as-of (last value at or
    before the date, at most `max_stale_days` old) - never from the future."""
    g = load_ohlc(db, "GOLD18_IRR", False)["close"]
    u = load_ohlc(db, "USD_IRR", False)["close"].dropna()
    x = load_ohlc(db, "XAU_USD", False)["close"].dropna()

    def asof(s):
        both = s.reindex(s.index.union(g.index)).ffill()
        src = pd.Series(s.index, index=s.index).reindex(s.index.union(g.index)).ffill()
        age = (pd.Series(g.index, index=g.index) - pd.to_datetime(src.reindex(g.index))).dt.days
        return both.reindex(g.index).where(age <= max_stale_days)

    d = pd.DataFrame({"market": g, "usd": asof(u), "xau": asof(x)}).dropna()
    d["theoretical"] = theoretical_gold(d["xau"], d["usd"])
    d["premium_pct"] = (d["market"] / d["theoretical"] - 1) * 100
    d["z60"] = (d["premium_pct"] - d["premium_pct"].rolling(60, min_periods=40).mean()) / \
        d["premium_pct"].rolling(60, min_periods=40).std()
    return d


def live_premium(db) -> dict:
    q = current_quotes(db)
    g, u, x = (q[s].get("price") for s in ("GOLD18_IRR", "USD_IRR", "XAU_USD"))
    stale = [s for s in q if q[s].get("stale")]
    if None in (g, u, x):
        return dict(valid=False, reason="missing live quote")
    theo = float(theoretical_gold(x, u))
    return dict(valid=not stale, stale_symbols=stale, market=g, usd=u, xau=x, theoretical=theo,
                premium_pct=float((g / theo - 1) * 100),
                formula="XAU/USD × USD/IRR(toman) × 0.75 ÷ 31.1034768")


def analyze(db) -> dict:
    r = common_returns(db)
    prem = premium_series(db)
    last = prem.iloc[-1]
    hist = prem["premium_pct"]
    return dict(correlations=correlations(r), lags=lag_analysis(r), comovement=comovement(r),
                premium=dict(last_date=str(prem.index[-1].date()), last_close_based=float(last["premium_pct"]),
                             z60=None if pd.isna(last["z60"]) else float(last["z60"]),
                             mean=float(hist.mean()), std=float(hist.std()),
                             p05=float(hist.quantile(.05)), p95=float(hist.quantile(.95)),
                             series_n=int(len(prem))),
                live=live_premium(db), premium_df=prem)
