"""Orchestrates every engine into one JSON-able snapshot. The heavy walk-forward
backtest is cached per data cutoff; everything else is recomputed (fast)."""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.analytics import (alerts, evaluation as ev, features as ft, forecast as fc, intermarket as im,
                           patterns as pt, quality as qa, regime as rg, risk as rk, technical as ta,
                           tracking as tr, volatility as vol)
from app.core.logsafe import logger
from app.data_layer.panel import aligned_panel, complete_bars, current_quotes, load_ohlc

HORIZONS = fc.HORIZONS
_lock = threading.Lock()
STATUS = {"state": "idle", "message": "", "progress": 0.0}


def clean(o):
    """Recursively make JSON-safe: numpy -> python, NaN/inf -> None, drop '_private' keys."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return o


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Context:
    """Everything derived from the DB for one analysis run."""
    def __init__(self, db):
        self.db = db
        self.panel, self.panel_info = aligned_panel(db)
        self.df = self.panel[["gold_open", "gold_high", "gold_low", "gold_close"]].rename(
            columns=lambda c: c.replace("gold_", ""))
        self.ind_pack = ta.analyze(self.df)
        self.ind = self.ind_pack["indicators"]
        self.vt = vol.vol_table(self.df)
        self.reg = rg.classify(self.ind, self.vt["ewma"])
        self.X = ft.build(self.panel, self.ind)
        self.cutoff = f"{self.df.index.max().date()}"
        self.key = f"{self.cutoff}:{len(self.df)}:{fc.MODEL_VERSION}"


def run_forecast_bundle(c: Context, progress=None) -> dict:
    ctx = fc.Ctx(np.log(c.df["close"].to_numpy()), c.vt["ret"].to_numpy(), c.X)
    models = fc.registry()
    labels = {m.name: m.label for m in models}; labels["ensemble"] = "Ensemble (وزن‌دهی بر اساس عملکرد)"
    P = fc.walk_forward(ctx, models, HORIZONS, init=750, live=True, progress=progress)
    Y = {h: fc.realized(ctx, h) for h in HORIZONS}
    sigma = c.vt["ewma"].to_numpy()
    trend = c.reg["trend"].to_numpy()
    close = c.df["close"].to_numpy()
    rep = ev.run_backtest(c.df.index, close, P, Y, sigma, trend, labels)
    base = float(close[-1]); sig_now = float(sigma[-1])
    live = {}
    for h in HORIZONS:
        r = rep[h]
        enabled = r["ensemble_enabled"]
        ens = r["_ens_pred"]; act = r["_active"]
        final = ens if enabled else np.zeros(len(ens))
        lo, hi = ev.causal_interval(final, Y[h], sigma, h)
        pred = float(final[-1]) if np.isfinite(final[-1]) else 0.0
        s_h = sig_now * math.sqrt(h)
        lo_r = float(lo[-1]) if np.isfinite(lo[-1]) else float(-1.2816 * s_h)
        hi_r = float(hi[-1]) if np.isfinite(hi[-1]) else float(1.2816 * s_h)
        preds = {n: (None if not np.isfinite(P[h][n][-1]) else float(P[h][n][-1])) for n in P[h] if n != "rw"}
        vals = np.array([v for v in preds.values() if v is not None])
        cov = r["models"]["ensemble"].get("interval_80_coverage")
        ens_cov = None if cov is None else round(cov * 100, 1)
        live[h] = dict(
            horizon=h, base_price=base, sigma_h=s_h, pred_return=pred, pred_price=float(base * math.exp(pred)),
            lower=float(base * math.exp(lo_r)), upper=float(base * math.exp(hi_r)),
            lower_return=lo_r, upper_return=hi_r,
            ensemble_active=bool(enabled and act[-1]), ensemble_enabled=enabled,
            ensemble_return=None if not enabled else pred,
            model_preds=preds, best_single=r["best_single"],
            weights={k: float(v) for k, v in r["_weights"].iloc[-1].items() if v > 0},
            share_models_up=None if not len(vals) else float((vals > 0).mean()),
            dispersion=None if len(vals) < 2 else float(vals.std() / s_h),
            trend_state=("up" if pred > 0.25 * s_h else "down" if pred < -0.25 * s_h else "flat") if enabled else "no_edge",
            uncertainty_pct=float((math.exp(hi_r) - math.exp(lo_r)) * 100),
            calibration_status=f"interval 80% backtest coverage {ens_cov}%" if ens_cov else "uncalibrated")
    dis = [live[h]["dispersion"] for h in HORIZONS if live[h]["dispersion"] is not None]
    bundle = dict(model_version=fc.MODEL_VERSION, feature_version=fc.FEATURE_VERSION, base_price=base,
                  data_cutoff=c.cutoff, horizons=live, disagreement=float(np.mean(dis)) if dis else None,
                  models_available=[m.name for m in models],
                  models_unavailable=[n for n in ("arima",) if n not in [m.name for m in models]],
                  recursive_vs_direct={h: dict(recursive=rep[h]["models"]["ar_rec"].get("rmse_ratio"),
                                               direct=rep[h]["models"]["ar_dir"].get("rmse_ratio")) for h in HORIZONS})
    backtest = {h: {k: v for k, v in rep[h].items() if not k.startswith("_")} for h in HORIZONS}
    # recent absolute errors of the final series, for the error chart
    err = {}
    for h in HORIZONS:
        r = rep[h]; final = r["_ens_pred"] if r["ensemble_enabled"] else np.zeros(len(c.df))
        e = (Y[h] - final)[-250:]
        err[h] = dict(dates=[str(d.date()) for d in c.df.index[-250:]], err=[None if not np.isfinite(x) else float(x) for x in e])
    return dict(forecast=bundle, backtest=backtest, errors=err, computed_at=_now(),
                skill_note="هیچ‌کدام از مدل‌ها نباید صرفاً به دلیل پیچیدگی مورد اعتماد باشند؛ "
                           "نسبت RMSE برابر یک یعنی هم‌سطح Random Walk.")


def get_bundle(c: Context, force=False, progress=None) -> dict:
    with c.db.connect() as conn:
        row = conn.execute("SELECT data_cutoff,payload_json FROM analysis_cache WHERE key='bundle'").fetchone()
    if row and row["data_cutoff"] == c.key and not force:
        return json.loads(row["payload_json"])
    with _lock:
        STATUS.update(state="running", message="در حال اجرای بک‌تست Walk-Forward...", progress=0.0)
        total = len(HORIZONS) * len(fc.registry()); done = [0]
        def prog(h, name):
            done[0] += 1; STATUS["progress"] = done[0] / total
            STATUS["message"] = f"بک‌تست: افق {h} / مدل {name}"
        try:
            b = clean(run_forecast_bundle(c, progress=prog))
            with c.db.connect() as conn:
                conn.execute("INSERT INTO analysis_cache(key,data_cutoff,computed_at_utc,payload_json) VALUES('bundle',?,?,?) "
                             "ON CONFLICT(key) DO UPDATE SET data_cutoff=excluded.data_cutoff, computed_at_utc=excluded.computed_at_utc,"
                             " payload_json=excluded.payload_json", (c.key, _now(), json.dumps(b)))
            STATUS.update(state="idle", message="بک‌تست کامل شد", progress=1.0)
            return b
        except Exception as exc:
            STATUS.update(state="error", message=str(exc))
            logger.exception("bundle failed")
            raise


def cached_bundle(c: Context):
    with c.db.connect() as conn:
        row = conn.execute("SELECT data_cutoff,payload_json FROM analysis_cache WHERE key='bundle'").fetchone()
    return json.loads(row["payload_json"]) if row and row["data_cutoff"] == c.key else None


def data_health(c: Context, quality: dict) -> dict:
    last = c.df.index.max()
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - last).days
    blocked = None
    if age > 10:
        blocked = f"آخرین کندل کامل طلا {age} روز قدیمی است؛ تحلیل و سیگنال معتبر نیست. ابتدا داده را به‌روزرسانی کنید."
    elif quality["critical"]:
        blocked = "خطای بحرانی در کیفیت داده وجود دارد."
    q = current_quotes(c.db)
    return dict(blocked=blocked, history_age_days=int(age), quotes=q, status=quality["status"])


def snapshot(db, compute_backtest=True, force=False, progress=None) -> dict:
    c = Context(db)
    bundle = get_bundle(c, force=force, progress=progress) if compute_backtest else cached_bundle(c)
    quality = qa.run_quality(db)
    health = data_health(c, quality)
    last = c.ind.iloc[-1]
    reg = rg.describe(c.reg)
    tech = c.ind_pack
    tech_validation = rk.validate_score(tech["score_series"], c.df["close"], c.vt["ewma"])
    pats = {w: pt.find_similar(c.df["close"], w) for w in (20, 30)}
    inter = im.analyze(db)
    prem_df = inter.pop("premium_df")
    cal = vol.calibrate(c.df["close"], c.vt["ewma"], regimes=c.reg["vol"])
    vrow = c.vt.iloc[-1]
    volatility = dict(ewma=float(vrow["ewma"]), std20=float(vrow["std20"]), std60=float(vrow["std60"]),
                      parkinson20=None if pd.isna(vrow.get("parkinson20")) else float(vrow["parkinson20"]),
                      atr_pct=None if pd.isna(vrow.get("atr_pct")) else float(vrow["atr_pct"]),
                      last_ret=float(vrow["ret"]), baseline_sqrt_h={h: float(vrow["ewma"] * math.sqrt(h)) for h in HORIZONS},
                      calibration=cal, vol_percentile=reg["vol_percentile"])
    quotes = health["quotes"]
    snap = dict(
        meta=dict(generated_at=_now(), data_cutoff=c.cutoff, bars=len(c.df), panel=c.panel_info,
                  model_version=fc.MODEL_VERSION, backtest_ready=bundle is not None),
        quotes=quotes,
        last_bar=dict(date=c.cutoff, close=float(c.df["close"].iloc[-1]),
                      change_pct=float((c.df["close"].iloc[-1] / c.df["close"].iloc[-2] - 1) * 100)),
        technical=dict(explain=tech["explain"], validity=tech["validity"], divergences=tech["divergences"][-8:],
                       levels=tech["levels"], snapshot=tech["snapshot"], validation=tech_validation),
        volatility=volatility, regime=reg, intermarket=inter, patterns=pats, quality=quality, data_health=health,
        forecast=bundle["forecast"] if bundle else None, backtest=bundle["backtest"] if bundle else None,
        errors=bundle["errors"] if bundle else None)
    if bundle:
        f5 = bundle["forecast"]["horizons"]["5"]
        snap["decision"] = rk.decide(tech["explain"], tech_validation, {"horizons": {5: f5}}, pats[20], reg, health)
        fh = {int(h): v for h, v in bundle["forecast"]["horizons"].items()}
        snap["scenarios"] = rk.scenarios(bundle["forecast"]["base_price"], 5, f5, tech["levels"], tech["snapshot"], reg)
        snap["risk"] = rk.risk_block(bundle["forecast"]["base_price"], volatility["ewma"],
                                     {"horizons": fh, "disagreement": bundle["forecast"]["disagreement"]}, 5)
        snap["prediction_ids"] = tr.store_predictions(db, snap)
    else:
        snap["decision"] = dict(signal="HOLD", reasons=["بک‌تست هنوز اجرا نشده است"], evidence=[], validated_votes=0,
                                confidence_status="no_signal_insufficient_data")
    tr.evaluate_due(db, c.df["close"])
    snap["tracking"] = tr.performance(db)
    if bundle:
        snap["new_alerts"] = alerts.evaluate(db, snap)
    snap["alerts"] = alerts.list_alerts(db)
    return clean(snap)


# ---------------- data for the UI ----------------
SERIES_COLS = ["open", "high", "low", "close", "ema_20", "ema_50", "ema_100", "ema_200", "bb_up", "bb_mid", "bb_lo",
               "rsi_14", "macd", "macd_signal", "macd_hist", "adx", "di_plus", "di_minus", "atr_pct",
               "tenkan", "kijun", "span_a", "span_b"]
PERIOD_DAYS = {"1m": 30, "3m": 91, "6m": 182, "1y": 365, "2y": 730, "4y": 1461, "5y": 1826, "10y": 3652}


def series_payload(db, symbol: str, days: int | None) -> dict:
    df = load_ohlc(db, symbol, include_provisional=True)
    if df.empty:
        return dict(symbol=symbol, dates=[])
    d = df[["open", "high", "low", "close"]].copy()
    ind = ta.compute_indicators(d)
    out = ind.copy(); out["open"] = d["open"]; out["high"] = d["high"]; out["low"] = d["low"]
    if symbol == "GOLD18_IRR":
        comp = complete_bars(df)
        pack = ta.analyze(comp[["open", "high", "low", "close"]])
        out["score"] = pack["score_series"].reindex(out.index)
        divs = [dict(e, date=e["date"]) for e in pack["divergences"][-30:]]
    else:
        divs = []
    if days:
        out = out.tail(days)
    has_ohlc = bool(out["high"].notna().any() and (out["high"] > out["low"]).mean() > 0.8)
    payload = dict(symbol=symbol, dates=[str(i.date()) for i in out.index], has_ohlc=has_ohlc,
                   provisional_last=bool(df["provisional"].iloc[-1]), divergences=divs,
                   source=str(df["source"].iloc[-1]), candles_allowed=has_ohlc)
    for col in SERIES_COLS + (["score"] if "score" in out else []):
        payload[col] = out[col].tolist()
    return clean(payload)


def market_overview(db) -> dict:
    bars = {s: load_ohlc(db, s, include_provisional=True)["close"] for s in ("GOLD18_IRR", "USD_IRR", "XAU_USD")}
    names = {"GOLD18_IRR": "طلای ۱۸ عیار (تومان/گرم)", "USD_IRR": "دلار آزاد (تومان)", "XAU_USD": "انس جهانی (دلار)"}
    rows = []
    for s, c in bars.items():
        c = c.dropna()
        if c.empty:
            continue
        last = c.index.max(); rets = {}
        for k, d in PERIOD_DAYS.items():
            ref = last - pd.Timedelta(days=d)
            if c.index.min() > ref + pd.Timedelta(days=7):      # history does not reach back far enough
                rets[k] = None
            else:
                rets[k] = float(c.iloc[-1] / c.loc[:ref].iloc[-1] - 1) * 100
        rows.append(dict(symbol=s, name=names[s], last=float(c.iloc[-1]), last_date=str(last.date()),
                         change_1d=float((c.iloc[-1] / c.iloc[-2] - 1) * 100), returns=rets,
                         first_date=str(c.index.min().date()), n=int(len(c))))
    # rebased (=100) comparison on the common period
    common_start = max(c.dropna().index.min() for c in bars.values())
    rb = {}
    for s, c in bars.items():
        cc = c.dropna(); cc = cc[cc.index >= common_start]
        rb[s] = dict(dates=[str(i.date()) for i in cc.index], values=(cc / cc.iloc[0] * 100).tolist())
    return clean(dict(assets=rows, rebased=rb, common_start=str(common_start.date()),
                      note="دارایی‌هایی که تاریخچه کافی برای یک بازه ندارند با «—» نشان داده می‌شوند. نقره، مس، آلومینیوم، بیت‌کوین و سکه "
                           "هنوز منبع داده‌ی معتبر ندارند و بدون داده‌ی واقعی اضافه نشده‌اند."))


def sources_payload(db) -> dict:
    with db.connect() as conn:
        src = [dict(r) for r in conn.execute("""SELECT s.name, s.base_url, s.priority, s.is_active,
            COUNT(r.id) AS observations, MIN(r.ts_utc) AS first_ts, MAX(r.ts_utc) AS last_ts, MAX(r.fetched_at_utc) AS last_fetch
            FROM sources s LEFT JOIN raw_observations r ON r.source_id=s.id GROUP BY s.id ORDER BY s.priority""")]
        runs = [dict(r) for r in conn.execute("SELECT started_at_utc,source,status,inserted,error FROM collection_runs ORDER BY id DESC LIMIT 25")]
    errors = []
    from app.core.config import PROJECT_ROOT
    try:
        for f in sorted((PROJECT_ROOT / "logs").glob("goldtrend_*.log"))[-3:]:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if " | ERROR" in line or " | WARNING" in line or line.startswith("ERROR"):
                    errors.append(line[:300])
    except Exception:
        pass
    return clean(dict(sources=src, runs=runs, log_errors=errors[-20:], quotes=current_quotes(db)))


def premium_payload(db) -> dict:
    d = im.premium_series(db).tail(1000)
    return clean(dict(dates=[str(i.date()) for i in d.index], premium=d["premium_pct"].tolist(),
                      market=d["market"].tolist(), theoretical=d["theoretical"].tolist()))
