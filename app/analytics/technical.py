"""Technical analysis + explainable 0-100 technical score.

Everything is causal: a value at row t uses only rows <= t. Divergences use
confirmed pivots (a pivot at i is only known at i+order). Ichimoku's cloud is
the span computed 26 bars earlier (what is drawn at t); the lagging span is
expressed as close_t vs close_{t-26}, never close_{t+26}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"EMA": 0.20, "MACD": 0.15, "Ichimoku": 0.15, "RSI": 0.10,
           "Bollinger": 0.10, "Divergence": 0.10, "ADX": 0.10, "Volume": 0.10}
LABELS_FA = {"EMA": "میانگین‌های متحرک", "MACD": "MACD", "Ichimoku": "ایچیموکو", "RSI": "RSI",
             "Bollinger": "باندهای بولینگر", "Divergence": "واگرایی", "ADX": "قدرت/جهت روند (ADX)",
             "Volume": "حجم"}
MIN_OBS = {"RSI": 15, "MACD": 35, "EMA": 200, "BB": 20, "ATR": 15, "ADX": 30, "ICHI": 78}


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(~((dn == 0) & (up > 0)), 100.0)


def macd(close, fast=12, slow=26, sig=9):
    m = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - \
        close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    s = m.ewm(span=sig, adjust=False, min_periods=sig).mean()
    return m, s, m - s


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def adx_di(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    atr = wilder(true_range(h, l, c), n)
    pdi, mdi = 100 * wilder(pdm, n) / atr, 100 * wilder(mdm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi, atr


def ichimoku(h, l, c, t=9, k=26, b=52):
    mid = lambda n: (h.rolling(n, min_periods=n).max() + l.rolling(n, min_periods=n).min()) / 2
    tenkan, kijun = mid(t), mid(k)
    span_a = ((tenkan + kijun) / 2).shift(k)       # cloud as displayed at t
    span_b = mid(b).shift(k)
    return tenkan, kijun, span_a, span_b, c - c.shift(k)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df needs close; high/low enable ATR/ADX/DI/Ichimoku (flag `has_hl`)."""
    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["rsi_14"] = rsi(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    for n in (20, 50, 100, 200):
        out[f"ema_{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"], out["bb_up"], out["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    out["bb_pctb"] = (c - out["bb_lo"]) / (out["bb_up"] - out["bb_lo"]).replace(0, np.nan)
    has_hl = {"high", "low"} <= set(df.columns) and df["high"].notna().mean() > 0.8 \
        and bool((df["high"] > df["low"]).mean() > 0.8)
    out.attrs["has_hl"] = has_hl
    if has_hl:
        h, l = df["high"].astype(float), df["low"].astype(float)
        out["adx"], out["di_plus"], out["di_minus"], out["atr_14"] = adx_di(h, l, c)
        out["atr_pct"] = out["atr_14"] / c
        (out["tenkan"], out["kijun"], out["span_a"], out["span_b"], out["chikou_diff"]) = ichimoku(h, l, c)
    else:
        for k in ("adx", "di_plus", "di_minus", "atr_14", "atr_pct", "tenkan", "kijun",
                  "span_a", "span_b", "chikou_diff"):
            out[k] = np.nan
    return out


def indicator_validity(ind: pd.DataFrame) -> dict:
    n = int(ind["close"].notna().sum())
    has_hl = bool(ind.attrs.get("has_hl"))
    need = {"RSI14": MIN_OBS["RSI"], "MACD": MIN_OBS["MACD"], "EMA200": MIN_OBS["EMA"],
            "Bollinger": MIN_OBS["BB"]}
    if has_hl:
        need.update({"ATR14": MIN_OBS["ATR"], "ADX/DI": MIN_OBS["ADX"], "Ichimoku": MIN_OBS["ICHI"]})
    res = {k: {"valid": n >= v, "min_obs": v} for k, v in need.items()}
    if not has_hl:
        for k in ("ATR14", "ADX/DI", "Ichimoku"):
            res[k] = {"valid": False, "reason": "high/low not available"}
    res["Volume"] = {"valid": False, "reason": "no volume data"}
    return res


# ---------------- divergences (confirmed pivots only) ----------------
def confirmed_pivots(x: np.ndarray, order: int = 3):
    """Return [(pivot_idx, confirm_idx, 'H'|'L')]; confirm_idx = pivot_idx + order."""
    out = []
    for i in range(order, len(x) - order):
        w = x[i - order:i + order + 1]
        if np.isnan(w).any():
            continue
        if x[i] == w.max() and (w == x[i]).sum() == 1:
            out.append((i, i + order, "H"))
        elif x[i] == w.min() and (w == x[i]).sum() == 1:
            out.append((i, i + order, "L"))
    return out


def find_divergences(price: pd.Series, indicator: pd.Series, name: str, order=3, min_gap=5, max_gap=60):
    p, v = price.to_numpy(float), indicator.to_numpy(float)
    piv = confirmed_pivots(p, order)
    events = []
    for kind in ("L", "H"):
        pts = [t for t in piv if t[2] == kind]
        for (a, _, _), (b, cb, _) in zip(pts, pts[1:]):
            if not (min_gap <= b - a <= max_gap) or np.isnan(v[a]) or np.isnan(v[b]):
                continue
            if kind == "L" and p[b] < p[a] and v[b] > v[a]:
                events.append(dict(type="bullish", indicator=name, confirm_idx=cb, pivot_idx=b, from_idx=a))
            if kind == "H" and p[b] > p[a] and v[b] < v[a]:
                events.append(dict(type="bearish", indicator=name, confirm_idx=cb, pivot_idx=b, from_idx=a))
    return [e for e in events if e["confirm_idx"] < len(p)]


def all_divergences(ind: pd.DataFrame):
    ev = find_divergences(ind["close"], ind["rsi_14"], "RSI") + \
         find_divergences(ind["close"], ind["macd"], "MACD")
    idx = ind.index
    for e in ev:
        e["date"] = str(idx[e["confirm_idx"]].date())
        e["pivot_date"] = str(idx[e["pivot_idx"]].date())
    return sorted(ev, key=lambda e: e["confirm_idx"])


# ---------------- evidence + score ----------------
def _hump(x, lo=-1.0, hi=1.0, ok=0.4, fade=0.4):
    """Linear evidence in [-ok, ok] -> [-1,1], fading to 0 then reversing beyond."""
    x = np.asarray(x, float)
    base = np.clip(x / ok, -1, 1)
    over = np.abs(x) - ok
    scale = np.where(over > 0, np.clip(1 - over / fade, -1, 1), 1.0)
    return base * scale


def evidence_frame(ind: pd.DataFrame, divs: list | None = None, div_validity_bars: int = 10) -> pd.DataFrame:
    c = ind["close"]
    ev = pd.DataFrame(index=ind.index)
    pairs = [np.sign(c - ind["ema_20"]), np.sign(ind["ema_20"] - ind["ema_50"]),
             np.sign(ind["ema_50"] - ind["ema_100"]), np.sign(ind["ema_100"] - ind["ema_200"])]
    stack = pd.concat(pairs, axis=1)
    ev["EMA"] = stack.mean(axis=1).where(stack.notna().sum(axis=1) >= 2)
    m = 0.5 * np.sign(ind["macd"] - ind["macd_signal"]) + 0.5 * np.sign(ind["macd"])
    ev["MACD"] = m.where(ind["macd_signal"].notna())
    ev["RSI"] = pd.Series(_hump((ind["rsi_14"] - 50) / 50, ok=0.4, fade=0.2), index=ind.index).where(ind["rsi_14"].notna())
    ev["Bollinger"] = pd.Series(_hump((ind["bb_pctb"] - 0.5), ok=0.4, fade=0.2), index=ind.index).where(ind["bb_pctb"].notna())
    if ind["adx"].notna().any():
        ev["ADX"] = (np.sign(ind["di_plus"] - ind["di_minus"]) * ((ind["adx"] - 15) / 25).clip(0, 1)).where(ind["adx"].notna())
        top = np.maximum(ind["span_a"], ind["span_b"]); bot = np.minimum(ind["span_a"], ind["span_b"])
        cloud = pd.Series(np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0)), index=ind.index)
        ichi = pd.concat([cloud.where(top.notna()), np.sign(ind["tenkan"] - ind["kijun"]).where(ind["kijun"].notna()),
                          np.sign(ind["chikou_diff"]).where(ind["chikou_diff"].notna())], axis=1)
        ev["Ichimoku"] = ichi.mean(axis=1).where(top.notna())
    else:
        ev["ADX"] = np.nan; ev["Ichimoku"] = np.nan
    d = np.zeros(len(ind)); dset = np.zeros(len(ind), bool)
    for e in (divs or []):
        s, t = e["confirm_idx"], min(len(ind), e["confirm_idx"] + div_validity_bars)
        d[s:t] = 1.0 if e["type"] == "bullish" else -1.0
        dset[s:t] = True
    ev["Divergence"] = pd.Series(np.where(dset, d, 0.0), index=ind.index).where(ind["rsi_14"].notna())
    ev["Volume"] = np.nan
    return ev[list(WEIGHTS)]


def score_series(ev: pd.DataFrame) -> pd.Series:
    w = pd.Series(WEIGHTS)
    valid = ev.notna()
    wsum = (valid * w).sum(axis=1)
    raw = (ev.fillna(0) * w).sum(axis=1)
    return (50 + 50 * raw / wsum.replace(0, np.nan)).clip(0, 100)


def explain_last(ev: pd.DataFrame, score: pd.Series) -> dict:
    row = ev.iloc[-1]
    valid = row.dropna()
    wsum = sum(WEIGHTS[k] for k in valid.index) or np.nan
    comps = []
    for k, v in row.items():
        if pd.isna(v):
            comps.append(dict(name=k, label=LABELS_FA[k], valid=False, weight=WEIGHTS[k], effective_weight=0.0,
                              evidence=None, contribution=0.0))
        else:
            ew = WEIGHTS[k] / wsum
            comps.append(dict(name=k, label=LABELS_FA[k], valid=True, weight=WEIGHTS[k],
                              effective_weight=round(ew, 4), evidence=round(float(v), 3),
                              contribution=round(float(50 * ew * v), 2),
                              stance="bullish" if v > 0.15 else "bearish" if v < -0.15 else "neutral"))
    s = float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else None
    return dict(score=None if s is None else round(s, 1),
                label=None if s is None else ("bullish" if s >= 60 else "bearish" if s <= 40 else "neutral"),
                components=sorted(comps, key=lambda x: -abs(x["contribution"])),
                dropped=[k for k, v in row.items() if pd.isna(v)],
                note="وزن‌های اولیه ثابت نیستند؛ اعتبار پیش‌بینی امتیاز در بخش بک‌تست سنجیده می‌شود. "
                     "مؤلفه‌های روندی همبسته‌اند و یک سیگنال را چندبار نمی‌شمارند چون وزن کل آن‌ها محدود است.")


def support_resistance(df: pd.DataFrame, lookback=300, order=5, tol=0.01, top=3):
    d = df.tail(lookback)
    last = float(d["close"].iloc[-1])
    hi = d["high"] if "high" in d and d["high"].notna().any() else d["close"]
    lo = d["low"] if "low" in d and d["low"].notna().any() else d["close"]
    pts = [(float(hi.iloc[i]), "R") for i, _, k in confirmed_pivots(hi.to_numpy(float), order) if k == "H"] + \
          [(float(lo.iloc[i]), "S") for i, _, k in confirmed_pivots(lo.to_numpy(float), order) if k == "L"]
    clusters = []
    for price, _ in sorted(pts):
        if clusters and abs(price - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= tol:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    levels = [dict(price=round(float(np.mean(c)), 2), touches=len(c)) for c in clusters if len(c) >= 2]
    sup = sorted([l for l in levels if l["price"] < last], key=lambda l: -l["price"])[:top]
    res = sorted([l for l in levels if l["price"] > last], key=lambda l: l["price"])[:top]
    return dict(supports=sup, resistances=res)


def analyze(df: pd.DataFrame) -> dict:
    ind = compute_indicators(df)
    divs = all_divergences(ind)
    ev = evidence_frame(ind, divs)
    sc = score_series(ev)
    last = ind.iloc[-1]
    snap = {k: (None if pd.isna(v) else float(v)) for k, v in last.items()}
    return dict(indicators=ind, evidence=ev, score_series=sc, divergences=divs,
                explain=explain_last(ev, sc), validity=indicator_validity(ind),
                snapshot=snap, levels=support_resistance(df))
