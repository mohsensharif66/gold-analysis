"""Local web UI server (stdlib only). Binds to 127.0.0.1 by default.

* Heavy work (collection, backtest) runs in background threads: the UI never blocks.
* Every endpoint reads real engine output; nothing here fabricates data.
* POST requires the custom header X-GT-Client (blocks cross-site form posts) and a
  same-origin Origin; input is whitelisted; bodies are size-limited; rate-limited.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.agents import agents, assistant
from app.analytics import alerts as al, service
from app.core.logsafe import logger
from app.db.database import Database

STATIC = Path(__file__).parent / "static"
SYMBOLS = {"GOLD18_IRR", "USD_IRR", "XAU_USD"}
DAYS = {"30": 30, "90": 90, "130": 130, "180": 180, "365": 365, "730": 730, "1460": 1460, "1825": 1825, "3650": 3650, "all": None}
MAX_BODY = 8192
RATE = {}          # ip -> [timestamps]
STATE = {"snapshot": None, "status": "starting", "message": "", "updated": None, "refreshing": False}
_state_lock = threading.Lock()


def compute_state(db: Database, force: bool = False):
    with _state_lock:
        STATE.update(status="computing", message="در حال تحلیل...")
    try:
        snap = service.snapshot(db, compute_backtest=True, force=force)
        with _state_lock:
            STATE.update(snapshot=snap, status="ready", message="", updated=time.time())
    except Exception as exc:
        logger.exception("analysis failed")
        with _state_lock:
            STATE.update(status="error", message=str(exc)[:300])


def refresh_data(db: Database):
    from app.services.collection_service import collect_all
    with _state_lock:
        if STATE["refreshing"]:
            return
        STATE["refreshing"] = True
    try:
        STATE.update(message="در حال دریافت داده...")
        collect_all(db)
    except Exception:
        logger.exception("refresh failed")
    finally:
        STATE["refreshing"] = False
    compute_state(db)


def rate_ok(ip: str, limit=60) -> bool:
    now = time.time()
    ts = [t for t in RATE.get(ip, []) if now - t < 60]
    ts.append(now); RATE[ip] = ts
    return len(ts) <= limit


def make_handler(db: Database):
    class H(BaseHTTPRequestHandler):
        server_version = "GoldTrendPro/1.0"

        def log_message(self, fmt, *args):  # quiet
            logger.debug("http " + fmt, *args)

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(service.clean(obj), ensure_ascii=False).encode("utf-8"))

        def _err(self, code, msg):
            self._json({"error": msg}, code)

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if u.path == "/api/status":
                    return self._json({k: v for k, v in STATE.items() if k != "snapshot"} | {"engine": dict(service.STATUS)})
                if u.path == "/api/snapshot":
                    return self._json({"status": STATE["status"], "engine": dict(service.STATUS), "snapshot": STATE["snapshot"]})
                if u.path == "/api/series":
                    sym = (q.get("symbol") or ["GOLD18_IRR"])[0]; days = (q.get("days") or ["365"])[0]
                    if sym not in SYMBOLS or days not in DAYS:
                        return self._err(400, "invalid parameter")
                    return self._json(service.series_payload(db, sym, DAYS[days]))
                if u.path == "/api/overview":
                    return self._json(service.market_overview(db))
                if u.path == "/api/premium":
                    return self._json(service.premium_payload(db))
                if u.path == "/api/sources":
                    return self._json(service.sources_payload(db))
                if u.path == "/api/alerts":
                    return self._json({"alerts": al.list_alerts(db, 60), "settings": al.get_settings(db)})
                if u.path == "/api/report":
                    s = STATE["snapshot"]
                    return self._json({"report": agents.daily_report(s) if s else None})
                if u.path == "/api/agents":
                    s = STATE["snapshot"]
                    return self._json(agents.supervisor(s) if s else {"error": "not ready"})
                return self._err(404, "not found")
            except Exception as exc:
                logger.exception("GET failed")
                return self._err(500, "internal error")

        def do_POST(self):
            ip = self.client_address[0]
            if not rate_ok(ip):
                return self._err(429, "too many requests")
            if self.headers.get("X-GT-Client") != "1":
                return self._err(403, "forbidden")
            origin = self.headers.get("Origin")
            if origin and origin.split("://")[-1] != self.headers.get("Host"):
                return self._err(403, "bad origin")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._err(413, "body too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/refresh":
                    threading.Thread(target=refresh_data, args=(db,), daemon=True).start()
                    return self._json({"started": True})
                if path == "/api/rerun":
                    threading.Thread(target=compute_state, args=(db, True), daemon=True).start()
                    return self._json({"started": True})
                if path == "/api/assistant":
                    qn = str(body.get("question", ""))[:500].strip()
                    if not qn or not STATE["snapshot"]:
                        return self._err(400, "question missing or analysis not ready")
                    return self._json(assistant.answer(qn, STATE["snapshot"]))
                if path == "/api/settings":
                    cur = al.get_settings(db)
                    new = {}
                    if "move_sigma" in body: new["move_sigma"] = max(1.0, min(6.0, float(body["move_sigma"])))
                    if "premium_z" in body: new["premium_z"] = max(1.0, min(5.0, float(body["premium_z"])))
                    for k in ("notify_signal_change", "notify_stale"):
                        if k in body: new[k] = bool(body[k])
                    cur.update(new)
                    with db.connect() as conn:
                        conn.execute("INSERT INTO app_settings(key,value_json,updated_at_utc) VALUES('alerts',?,?) ON CONFLICT(key) "
                                     "DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc",
                                     (json.dumps(cur), service._now()))
                    return self._json({"settings": cur})
                if path == "/api/alerts/ack":
                    with db.connect() as conn:
                        conn.execute("UPDATE alert_events SET acknowledged=1")
                    return self._json({"ok": True})
                return self._err(404, "not found")
            except (ValueError, TypeError, json.JSONDecodeError):
                return self._err(400, "bad request")
            except Exception:
                logger.exception("POST failed")
                return self._err(500, "internal error")
    return H


def serve(host="127.0.0.1", port=8765):
    db = Database(); db.initialize()
    threading.Thread(target=compute_state, args=(db,), daemon=True).start()
    srv = ThreadingHTTPServer((host, port), make_handler(db))
    logger.info("GoldTrend Pro UI: http://{}:{}", host, port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
