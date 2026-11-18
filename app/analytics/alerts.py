"""Rule-based alerts, de-duplicated per (rule, data point) in `alert_events`."""
from __future__ import annotations

import json
from datetime import datetime, timezone

DEFAULTS = dict(move_sigma=2.5, premium_z=2.0, notify_signal_change=True, notify_stale=True)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_settings(db) -> dict:
    out = dict(DEFAULTS)
    with db.connect() as conn:
        r = conn.execute("SELECT value_json FROM app_settings WHERE key='alerts'").fetchone()
    if r:
        try:
            out.update(json.loads(r[0]))
        except Exception:
            pass
    return out


def evaluate(db, snap: dict) -> list[dict]:
    cfg = get_settings(db)
    cutoff = snap["meta"]["data_cutoff"]
    ev = []
    t = snap["technical"]["snapshot"]
    vr = snap["volatility"]
    if vr.get("last_ret") is not None and vr.get("ewma"):
        z = vr["last_ret"] / vr["ewma"]
        if abs(z) >= cfg["move_sigma"]:
            ev.append((f"move:{cutoff}", "move", "warning",
                       f"حرکت روزانه‌ی غیرعادی: {vr['last_ret']*100:.2f}٪ (حدود {z:.1f} برابر نوسان معمول)"))
    pz = snap["intermarket"]["premium"].get("z60")
    if pz is not None and abs(pz) >= cfg["premium_z"]:
        ev.append((f"prem:{cutoff}", "premium", "warning",
                   f"پریمیوم/دیسکانت نسبت به ارزش نظری غیرعادی است (z={pz:.1f}، مقدار {snap['intermarket']['premium']['last_close_based']:.1f}٪)"))
    if snap["quality"]["status"] != "ok":
        ev.append((f"quality:{cutoff}:{snap['quality']['status']}", "quality", snap["quality"]["status"],
                   f"کیفیت داده: {snap['quality']['critical']} مورد بحرانی و {snap['quality']['warnings']} هشدار"))
    if cfg["notify_stale"] and snap["data_health"].get("blocked"):
        ev.append((f"stale:{cutoff}", "stale", "critical", snap["data_health"]["blocked"]))
    with db.connect() as conn:
        prev = conn.execute("SELECT signal FROM predictions WHERE horizon_days=5 AND data_cutoff<? "
                            "ORDER BY data_cutoff DESC LIMIT 1", (cutoff,)).fetchone()
    cur = snap["decision"]["signal"]
    if cfg["notify_signal_change"] and prev and prev[0] != cur:
        ev.append((f"signal:{cutoff}:{cur}", "signal", "info", f"سیگنال از {prev[0]} به {cur} تغییر کرد"))
    new = []
    with db.connect() as conn:
        for key, kind, sev, msg in ev:
            cur_ = conn.execute("INSERT OR IGNORE INTO alert_events(dedupe_key,created_at_utc,kind,severity,message_fa,data_json)"
                                " VALUES(?,?,?,?,?,?)", (key, _now(), kind, sev, msg, None))
            if cur_.rowcount:
                new.append(dict(kind=kind, severity=sev, message=msg))
    return new


def list_alerts(db, limit=30) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id,created_at_utc,kind,severity,message_fa,acknowledged FROM alert_events "
                            "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]
