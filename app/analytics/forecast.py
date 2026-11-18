"""Multi-model forecasting of h-day log returns (not raw prices).

Walk-forward only: a model used at time t was fit on samples whose label is fully
known at the fit time T <= t (sample i needs i+h <= T), which is the embargo that
prevents leakage through overlapping horizons. Scaling lives inside each fit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analytics.features import CORE, FULL

HORIZONS = (1, 3, 5, 7, 10)
MODEL_VERSION = "gt-1.0"
FEATURE_VERSION = "f1"
LAGS = 5


@dataclass
class Ctx:
    logc: np.ndarray            # log close
    r: np.ndarray               # daily log returns (r[0]=nan)
    F: pd.DataFrame             # causal features
    ewma_alpha_grid: tuple = (0.005, 0.01, 0.02, 0.04, 0.08, 0.15)
    cache: dict = field(default_factory=dict)

    def lags(self):
        if "lags" not in self.cache:
            L = np.full((len(self.r), LAGS), np.nan)
            for j in range(LAGS):
                L[j:, j] = self.r[: len(self.r) - j]
            self.cache["lags"] = L
        return self.cache["lags"]

    def ewma(self, alpha):
        key = ("ewma", alpha)
        if key not in self.cache:
            self.cache[key] = pd.Series(self.r).ewm(alpha=alpha, adjust=False, min_periods=20).mean().to_numpy()
        return self.cache[key]


def _y(ctx: Ctx, h: int) -> np.ndarray:
    y = np.full(len(ctx.logc), np.nan)
    y[:-h] = ctx.logc[h:] - ctx.logc[:-h]
    return y


class Base:
    name = "base"; label = "base"; kind = "stat"
    refit_every = 21; min_train = 250; max_train = None
    def available(self) -> bool: return True
    def fit(self, ctx, h, T): self.h = h
    def predict(self, ctx, idx: np.ndarray) -> np.ndarray: raise NotImplementedError


class RandomWalk(Base):
    name, label = "rw", "Random Walk (بازده صفر)"
    def predict(self, ctx, idx): return np.zeros(len(idx))


class MovingAverage(Base):
    name, label = "ma", "میانگین متحرک بازده (۶۰ روز)"
    def predict(self, ctx, idx):
        m = pd.Series(ctx.r).rolling(60, min_periods=60).mean().to_numpy()
        return m[idx] * self.h


class ExpSmoothing(Base):
    name, label = "es", "هموارسازی نمایی بازده"
    def fit(self, ctx, h, T):
        self.h = h
        best, self.alpha = np.inf, 0.02
        for a in ctx.ewma_alpha_grid:        # one-step SSE on data known at T
            e = ctx.ewma(a)
            err = ctx.r[2:T + 1] - e[1:T]
            err = err[np.isfinite(err)]
            if len(err) > 100 and (err ** 2).mean() < best:
                best, self.alpha = (err ** 2).mean(), a
    def predict(self, ctx, idx): return ctx.ewma(self.alpha)[idx] * self.h


class ARRecursive(Base):
    name, label, kind = "ar_rec", "AR(5) بازگشتی", "recursive"
    def fit(self, ctx, h, T):
        self.h = h
        L, r = ctx.lags(), ctx.r
        rows = np.arange(LAGS, T)            # predict r[i+1] from lags at i, i+1<=T
        X, y = L[rows], r[rows + 1]
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        A = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.solve(A.T @ A + 1e-9 * np.eye(A.shape[1]), A.T @ y) if len(X) > 50 else None
    def predict(self, ctx, idx):
        if self.coef is None:
            return np.full(len(idx), np.nan)
        state = ctx.lags()[idx].copy()       # columns: r_t, r_{t-1}, ...
        total = np.zeros(len(idx))
        for _ in range(self.h):
            nxt = self.coef[0] + state @ self.coef[1:]
            total += nxt
            state = np.column_stack([nxt, state[:, :-1]])
        return np.where(np.isfinite(state).all(1), total, np.nan)


class SklearnDirect(Base):
    kind = "direct"
    cols: list = CORE
    max_train = 1500
    def make(self): raise NotImplementedError
    def fit(self, ctx, h, T):
        self.h = h
        X = ctx.F[self.cols].to_numpy()
        y = _y(ctx, h)
        rows = np.arange(0, T - h + 1)       # label known at T
        ok = np.isfinite(X[rows]).all(1) & np.isfinite(y[rows])
        rows = rows[ok][-self.max_train:]
        self.model = None
        if len(rows) >= self.min_train:
            self.model = self.make().fit(X[rows], y[rows])
    def predict(self, ctx, idx):
        if self.model is None:
            return np.full(len(idx), np.nan)
        X = ctx.F[self.cols].to_numpy()[idx]
        out = np.full(len(idx), np.nan)
        ok = np.isfinite(X).all(1)
        if ok.any():
            out[ok] = self.model.predict(X[ok])
        return out


class Ridge_(SklearnDirect):
    name, label = "ridge", "Ridge (ویژگی‌های پایه)"
    def make(self): return make_pipeline(StandardScaler(), Ridge(alpha=300.0))


class RidgeX(SklearnDirect):
    name, label, cols, min_train = "ridge_x", "Ridge + دلار/انس/پریمیوم", FULL, 300
    def make(self): return make_pipeline(StandardScaler(), Ridge(alpha=300.0))


class ElasticNet_(SklearnDirect):
    name, label = "enet", "ElasticNet"
    def make(self): return make_pipeline(StandardScaler(), ElasticNet(alpha=2e-4, l1_ratio=0.5, max_iter=3000))


class ARDirect(SklearnDirect):
    name, label = "ar_dir", "AR(5) مستقیم (Ridge)"
    def fit(self, ctx, h, T):
        self.h = h
        L = ctx.lags(); y = _y(ctx, h)
        rows = np.arange(LAGS, T - h + 1)
        ok = np.isfinite(L[rows]).all(1) & np.isfinite(y[rows])
        rows = rows[ok]
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=300.0)).fit(L[rows], y[rows]) if len(rows) > 250 else None
    def predict(self, ctx, idx):
        if self.model is None: return np.full(len(idx), np.nan)
        X = ctx.lags()[idx]; out = np.full(len(idx), np.nan); ok = np.isfinite(X).all(1)
        if ok.any(): out[ok] = self.model.predict(X[ok])
        return out


class GBRT(SklearnDirect):
    name, label, refit_every, min_train = "gbrt", "Gradient Boosting (درخت کم‌عمق)", 84, 600
    def make(self):
        try:
            from xgboost import XGBRegressor  # optional, preferred when installed
            self.label = "XGBoost"
            return XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8,
                                min_child_weight=20, reg_lambda=5.0, verbosity=0)
        except Exception:
            return HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.06,
                                                 min_samples_leaf=40, l2_regularization=5.0, random_state=0)


class ARIMA_(Base):
    """Optional: requires statsmodels. ARIMA(1,0,1) on daily log returns."""
    name, label, kind, refit_every = "arima", "ARIMA(1,0,1) بر بازده", "stat", 63
    def available(self):
        try:
            import statsmodels.tsa.arima.model  # noqa
            return True
        except Exception:
            return False
    def fit(self, ctx, h, T):
        self.h = h
        from statsmodels.tsa.arima.model import ARIMA
        y = pd.Series(ctx.r[1:T + 1]).dropna().to_numpy()[-1500:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ARIMA(y, order=(1, 0, 1), trend="c").fit()
        self.T = T
    def predict(self, ctx, idx):
        # Multi-step from the fit origin; later rows are refreshed via apply() on new data.
        out = np.full(len(idx), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, i in enumerate(idx):
                r_new = ctx.r[self.T + 1:i + 1]
                res = self.res.append(r_new, refit=False) if len(r_new) else self.res
                out[k] = float(np.sum(res.forecast(self.h)))
        return out


def registry(include_optional=True) -> list[Base]:
    models = [RandomWalk(), MovingAverage(), ExpSmoothing(), ARRecursive(), ARDirect(), Ridge_(),
              ElasticNet_(), RidgeX(), GBRT()]
    if include_optional:
        models.append(ARIMA_())
    return [m for m in models if m.available()]


def walk_forward(ctx: Ctx, models: list[Base], horizons=HORIZONS, init: int = 750, live: bool = True,
                 progress: Callable | None = None) -> dict:
    """Returns {h: {model_name: np.ndarray preds aligned to index}} (NaN where unavailable)."""
    N = len(ctx.logc)
    out = {h: {m.name: np.full(N, np.nan) for m in models} for h in horizons}
    for h in horizons:
        for m in models:
            step = m.refit_every
            Ts = list(range(init, N, step))
            if live and Ts[-1] != N - 1:
                Ts.append(N - 1)
            for j, T in enumerate(Ts):
                end = Ts[j + 1] if j + 1 < len(Ts) else N
                if live and T == N - 1:
                    end = N
                if T == end:
                    continue
                m.fit(ctx, h, T)
                idx = np.arange(T, end)
                out[h][m.name][idx] = m.predict(ctx, idx)
            if progress:
                progress(h, m.name)
    return out


def realized(ctx: Ctx, h: int) -> np.ndarray:
    return _y(ctx, h)
