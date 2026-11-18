"""Statistical volatility engine. sqrt(h) scaling is only the baseline; the
actual prediction interval comes from empirical standardized-return quantiles
(calibrated on past data, checked on a held-out later period)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 7, 10)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean()
    return np.sqrt(var)


def parkinson_vol(high: pd.Series, low: pd.Series, win: int = 20) -> pd.Series:
    x = (np.log(high / low) ** 2) / (4 * np.log(2))
    return np.sqrt(x.rolling(win, min_periods=win).mean())


def vol_table(df: pd.DataFrame) -> pd.DataFrame:
    r = log_returns(df["close"])
    out = pd.DataFrame({"ret": r, "std20": r.rolling(20, min_periods=20).std(),
                        "std60": r.rolling(60, min_periods=60).std(), "ewma": ewma_vol(r)})
    if {"high", "low"} <= set(df.columns) and df["high"].notna().any():
        out["parkinson20"] = parkinson_vol(df["high"], df["low"])
        out["atr_pct"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean() / df["close"]
    return out


def forward_returns(close: pd.Series, h: int) -> pd.Series:
    """Log return from t to t+h, NaN where the future is unknown."""
    return np.log(close.shift(-h) / close)


def calibrate(close: pd.Series, sigma: pd.Series, levels=(0.10, 0.90), split: float = 0.7,
              horizons=HORIZONS, regimes: pd.Series | None = None):
    """Empirical quantiles of z = r_h / (sigma_t*sqrt(h)). Reports held-out coverage
    (quantiles from the first `split` of history, evaluated on the rest)."""
    res = {}
    for h in horizons:
        z = (forward_returns(close, h) / (sigma * np.sqrt(h))).replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 150:
            res[h] = dict(valid=False, n=int(len(z)))
            continue
        k = int(len(z) * split)
        ql, qh = np.quantile(z.iloc[:k], levels)
        test = z.iloc[k:]
        cov = float(((test >= ql) & (test <= qh)).mean())
        ql_f, qh_f = np.quantile(z, levels)
        item = dict(valid=True, n=int(len(z)), q_lo=float(ql_f), q_hi=float(qh_f),
                    z_std=float(z.std()), nominal=levels[1] - levels[0],
                    heldout_coverage=cov, normal_assumption_z=1.2816)
        if regimes is not None:
            by = {}
            for name, grp in z.groupby(regimes.reindex(z.index)):
                if len(grp) >= 120:
                    by[str(name)] = dict(q_lo=float(np.quantile(grp, levels[0])),
                                         q_hi=float(np.quantile(grp, levels[1])), n=int(len(grp)))
            item["by_regime"] = by
        res[h] = item
    return res


def interval(price: float, sigma_t: float, h: int, calib: dict, regime: str | None = None):
    c = calib.get(h, {})
    if not c.get("valid"):
        z_lo, z_hi, basis = -1.2816, 1.2816, "sqrt-time baseline (uncalibrated)"
    else:
        q = c.get("by_regime", {}).get(regime) if regime else None
        z_lo, z_hi = (q["q_lo"], q["q_hi"]) if q else (c["q_lo"], c["q_hi"])
        basis = "empirical calibrated" + (f" ({regime})" if q else "")
    s = sigma_t * np.sqrt(h)
    return dict(lower=float(price * np.exp(z_lo * s)), upper=float(price * np.exp(z_hi * s)),
                sigma_h=float(s), basis=basis)
