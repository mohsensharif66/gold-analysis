"""Historical pattern matching: KNN on the normalised return path, re-ranked with
banded DTW. Output is descriptive; the unconditional base rate is always shown
next to it because gold in IRR drifts upward and a high 'up %' may just be that."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 7, 10)


def _path(logc: np.ndarray, end: int, w: int) -> np.ndarray:
    """Cumulative log return path over the window ending at `end` (w points)."""
    return logc[end - w + 1:end + 1] - logc[end - w]


def dtw(a: np.ndarray, b: np.ndarray, band: int) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def find_similar(close: pd.Series, window: int = 20, k: int = 10, horizons=HORIZONS,
                 prefilter: int = 120, use_dtw: bool = True) -> dict:
    logc = np.log(close.to_numpy(float)); N = len(logc); maxh = max(horizons)
    last = N - 1
    last_cand = last - max(window, maxh)          # no overlap with the query, full future known
    if last_cand < window + 50:
        return dict(valid=False, reason="not enough history")
    ends = np.arange(window, last_cand + 1)
    P = np.vstack([_path(logc, e, window) for e in ends])
    q = _path(logc, last, window)
    # normalisation statistics come from the candidate pool only (history, never the query)
    mu, sd = P.mean(0), P.std(0); sd[sd == 0] = 1
    dist = np.sqrt((((P - mu) / sd - (q - mu) / sd) ** 2).sum(1))
    order = np.argsort(dist)[:prefilter]
    qz = _z(q)
    cand = []
    for i in order:
        d_shape = dtw(qz, _z(P[i]), max(2, window // 5)) if use_dtw else dist[i]
        cand.append((ends[i], float(dist[i]), d_shape))
    cand.sort(key=lambda t: (t[2] if use_dtw else t[1]))
    chosen = []
    for e, de, dd in cand:                        # drop heavily overlapping matches
        if all(abs(e - c[0]) >= window // 2 for c in chosen):
            chosen.append((e, de, dd))
        if len(chosen) == k:
            break
    ref = float(np.median(dist))
    idx = close.index
    matches = []
    for e, de, dd in chosen:
        fut = {h: float(logc[e + h] - logc[e]) for h in horizons}
        matches.append(dict(end_date=str(idx[e].date()), start_date=str(idx[e - window + 1].date()),
                            euclid=de, dtw=dd, similarity=float(100 / (1 + de / ref)),
                            future_returns=fut, path=(_path(logc, e, window) * 100).round(3).tolist()))
    summary = {}
    for h in horizons:
        arr = np.array([m["future_returns"][h] for m in matches])
        allr = logc[h:] - logc[:-h]
        summary[h] = dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)), std=float(arr.std()),
                          up=int((arr > 0).sum()), down=int((arr < 0).sum()), up_pct=float((arr > 0).mean() * 100),
                          base_rate_up_pct=float((allr > 0).mean() * 100), base_mean=float(allr.mean()))
    return dict(valid=True, window=window, k=len(matches), method="KNN(Euclid)→DTW" if use_dtw else "KNN(Euclid)",
                query_path=(q * 100).round(3).tolist(), query_end=str(idx[last].date()),
                matches=matches, summary=summary,
                caveat="نتایج توصیفی‌اند و نشان‌دهنده‌ی پیش‌بینی نیستند؛ درصد صعودی را با نرخ پایه مقایسه کنید.")
