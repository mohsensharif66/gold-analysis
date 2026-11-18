"""Standard causal feature layer. Every column at row t only uses data <= t.
Scaling/selection happen inside model pipelines, fit on training rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import technical as ta
from app.analytics.volatility import ewma_vol, log_returns

THEORETICAL_K = 0.75 / 31.1034768   # 18k purity / grams per troy ounce
CORE = ["r1", "r3", "r5", "r10", "r20", "vol_ewma", "vol20", "range_pct", "d_ema20", "d_ema50",
        "d_ema200", "mom60", "rsi", "macd_n", "pctb", "adx", "di_diff", "usd_r1", "usd_r5", "usd_r20"]
FULL = CORE + ["xau_r1", "xau_r5", "xau_r20", "prem", "prem_z"]


def theoretical_gold(xau, usd):
    return xau * usd * THEORETICAL_K


def build(panel: pd.DataFrame, ind: pd.DataFrame | None = None) -> pd.DataFrame:
    close = panel["gold_close"]
    df = pd.DataFrame({"close": close, "high": panel["gold_high"], "low": panel["gold_low"]})
    ind = ind if ind is not None else ta.compute_indicators(df)
    r = log_returns(close)
    X = pd.DataFrame(index=panel.index)
    lc = np.log(close)
    for n in (1, 3, 5, 10, 20):
        X[f"r{n}"] = lc.diff(n)
    X["vol_ewma"] = ewma_vol(r)
    X["vol20"] = r.rolling(20, min_periods=20).std()
    X["range_pct"] = (df["high"] - df["low"]) / close
    for n in (20, 50, 200):
        X[f"d_ema{n}"] = np.log(close / ind[f"ema_{n}"])
    X["mom60"] = lc.diff(60)
    X["rsi"] = (ind["rsi_14"] - 50) / 50
    X["macd_n"] = ind["macd_hist"] / close
    X["pctb"] = ind["bb_pctb"]
    X["adx"] = ind["adx"] / 100
    X["di_diff"] = (ind["di_plus"] - ind["di_minus"]) / 100
    usd = np.log(panel["usd"])
    for n in (1, 5, 20):
        X[f"usd_r{n}"] = usd.diff(n)
    xau = np.log(panel["xau"])
    for n in (1, 5, 20):
        X[f"xau_r{n}"] = xau.diff(n)
    theo = theoretical_gold(panel["xau"], panel["usd"])
    X["prem"] = np.log(close / theo)
    X["prem_z"] = (X["prem"] - X["prem"].rolling(60, min_periods=40).mean()) / X["prem"].rolling(60, min_periods=40).std()
    return X.replace([np.inf, -np.inf], np.nan)
