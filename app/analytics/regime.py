"""Market regime detection (rule based v1; HMM can plug in via `Detector`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_FA = {"bullish_trend": "روند صعودی", "bearish_trend": "روند نزولی", "sideways": "بدون روند / خنثی"}
VOL_FA = {"high": "نوسان بالا", "normal": "نوسان عادی", "low": "نوسان پایین"}


def classify(ind: pd.DataFrame, ewma: pd.Series, adx_thr: float = 22.0) -> pd.DataFrame:
    c = ind["close"]
    up = (c > ind["ema_50"]) & (ind["ema_20"] > ind["ema_50"])
    dn = (c < ind["ema_50"]) & (ind["ema_20"] < ind["ema_50"])
    if ind["adx"].notna().any():
        strong = ind["adx"] >= adx_thr
        up, dn = up & strong & (ind["di_plus"] > ind["di_minus"]), dn & strong & (ind["di_plus"] < ind["di_minus"])
    mom = c.pct_change(20)
    up, dn = up & (mom > 0), dn & (mom < 0)
    trend = pd.Series(np.where(up, "bullish_trend", np.where(dn, "bearish_trend", "sideways")), index=c.index)
    trend = trend.where(ind["ema_50"].notna(), None)
    # causal expanding percentile of EWMA vol (needs 250 obs)
    pct = ewma.expanding(min_periods=250).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    vol = pd.Series(np.where(pct >= 0.8, "high", np.where(pct <= 0.2, "low", "normal")), index=c.index)
    vol = vol.where(pct.notna(), None)
    return pd.DataFrame({"trend": trend, "vol": vol, "vol_percentile": pct})


def describe(reg: pd.DataFrame) -> dict:
    last = reg.iloc[-1]
    run = 1
    t = reg["trend"].to_numpy()
    for i in range(len(t) - 2, -1, -1):
        if t[i] == t[-1]:
            run += 1
        else:
            break
    return dict(trend=last["trend"], trend_fa=TREND_FA.get(last["trend"]), vol=last["vol"],
                vol_fa=VOL_FA.get(last["vol"]), vol_percentile=None if pd.isna(last["vol_percentile"]) else round(float(last["vol_percentile"]), 3),
                days_in_trend_state=run)
