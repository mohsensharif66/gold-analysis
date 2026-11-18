"""Persistent prediction tracking: every forecast is stored with its context and
later compared with the realised price once the horizon has passed."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def store_predictions(db, snap: dict) -> list[str]:
    fc, cutoff = snap["forecast"], snap["meta"]["data_cutoff"]
    ids = []
    with db.connect() as conn:
        for h, f in fc["horizons"].items():
            row = conn.execute("SELECT prediction_id FROM predictions WHERE asset_symbol=? AND data_cutoff=? "
                               "AND model_version=? AND horizon_days=?",
                               ("GOLD18_IRR", cutoff, fc["model_version"], int(h))).fetchone()
            if row:
                ids.append(row[0]); continue
            pid = uuid.uuid4().hex[:16]
            conn.execute("""INSERT INTO predictions(prediction_id,created_at_utc,data_cutoff,asset_symbol,model_version,
                feature_version,horizon_days,base_price,predicted_return,predicted_price,lower_bound,upper_bound,
                market_regime,technical_score,ensemble_output,ensemble_active,signal,confidence_calibration_status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, _now(), cutoff, "GOLD18_IRR", fc["model_version"], fc["feature_version"], int(h),
                 fc["base_price"], f["pred_return"], f["pred_price"], f["lower"], f["upper"],
                 snap["regime"]["trend"], snap["technical"]["explain"]["score"], f.get("ensemble_return"),
                 int(f["ensemble_active"]), snap["decision"]["signal"], f["calibration_status"]))
            ids.append(pid)
    return ids


def evaluate_due(db, gold_close) -> int:
    """Resolve predictions whose horizon has elapsed. gold_close: Series indexed by date."""
    resolved = 0
    with db.connect() as conn:
        rows = conn.execute("SELECT prediction_id,data_cutoff,horizon_days,base_price,predicted_price,lower_bound,"
                            "upper_bound FROM predictions WHERE resolved_at_utc IS NULL").fetchall()
        idx = gold_close.index
        for r in rows:
            try:
                import pandas as pd
                pos = idx.get_loc(pd.Timestamp(r["data_cutoff"]))
            except Exception:
                continue
            tgt = pos + r["horizon_days"]
            if tgt >= len(idx):
                continue
            actual = float(gold_close.iloc[tgt])
            conn.execute("""UPDATE predictions SET target_date=?, actual_price=?, actual_return=?, in_interval=?,
                abs_error=?, resolved_at_utc=? WHERE prediction_id=?""",
                (str(idx[tgt].date()), actual, float(np.log(actual / r["base_price"])),
                 int(r["lower_bound"] <= actual <= r["upper_bound"]), abs(actual - r["predicted_price"]), _now(),
                 r["prediction_id"]))
            resolved += 1
    return resolved


def performance(db) -> dict:
    with db.connect() as conn:
        tot = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        res = conn.execute("""SELECT horizon_days, COUNT(*) n, AVG(in_interval) cov, AVG(abs_error/actual_price) mape,
            AVG(CASE WHEN predicted_return != 0 THEN (predicted_return*actual_return>0) END) dir_acc
            FROM predictions WHERE resolved_at_utc IS NOT NULL GROUP BY horizon_days ORDER BY 1""").fetchall()
        recent = conn.execute("""SELECT prediction_id,data_cutoff,horizon_days,predicted_price,lower_bound,upper_bound,
            signal,actual_price,in_interval,target_date,ensemble_active FROM predictions ORDER BY created_at_utc DESC, horizon_days LIMIT 15""").fetchall()
    return dict(total=tot, resolved=[dict(r) for r in res], recent=[dict(r) for r in recent],
                note="تا زمانی که تعداد پیش‌بینی‌های تسویه‌شده کم است، این آمار معنادار نیست.")
