import json
import threading
import urllib.request as u
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd

from app.agents import assistant
from app.analytics import quality, tracking
from app.web import server
from tests.helpers import make_db, put_yf


def fake_snapshot(cutoff="2024-03-10"):
    hz = {str(h): dict(pred_return=0.0, pred_price=100.0, lower=95.0, upper=108.0, ensemble_active=False,
                       ensemble_return=None, calibration_status="x") for h in (1, 3, 5, 7, 10)}
    return dict(meta=dict(data_cutoff=cutoff), forecast=dict(horizons=hz, model_version="t", feature_version="f", base_price=100.0),
                regime=dict(trend="sideways"), technical=dict(explain=dict(score=55.0)), decision=dict(signal="HOLD"))


def test_prediction_tracking_is_idempotent_and_resolves_against_reality(tmp_path):
    db = make_db(tmp_path)
    snap = fake_snapshot("2024-01-03")
    ids1 = tracking.store_predictions(db, snap); ids2 = tracking.store_predictions(db, snap)
    assert ids1 == ids2 and len(ids1) == 5
    idx = pd.date_range("2024-01-03", periods=8)
    close = pd.Series([100, 101, 102, 103, 104, 110, 106, 107.0], index=idx)
    n = tracking.evaluate_due(db, close)
    assert n == 4                                   # horizons 1,3,5,7 have elapsed; 10 has not
    perf = tracking.performance(db)
    assert perf["total"] == 5 and sum(r["n"] for r in perf["resolved"]) == 4
    with db.connect() as conn:
        row = conn.execute("SELECT actual_price,in_interval FROM predictions WHERE horizon_days=5").fetchone()
    assert row[0] == 110.0 and row[1] == 0          # 110 is above the 108 upper bound


def test_quality_engine_flags_outlier_without_modifying_history(tmp_path):
    db = make_db(tmp_path)
    rng = np.random.default_rng(0); price = 100.0
    for i in range(200):
        price *= np.exp(rng.normal(0, 0.005) if i != 150 else 0.5)
        d = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
        put_yf(db, "GOLD18_IRR", d, price, price * 1.002, price * 0.998, price)
    with db.connect() as conn:
        before = conn.execute("SELECT COUNT(*), SUM(price) FROM raw_observations").fetchone()
    rep = quality.run_quality(db)
    flags = [f for s in rep["symbols"] for f in s["flags"] if s["symbol"] == "GOLD18_IRR"]
    assert any(f["type"] == "q_outlier" for f in flags)
    with db.connect() as conn:
        after = conn.execute("SELECT COUNT(*), SUM(price) FROM raw_observations").fetchone()
        stored = conn.execute("SELECT COUNT(*) FROM quality_flags WHERE flag_type='q_outlier'").fetchone()[0]
    assert tuple(before) == tuple(after) and stored >= 1


def test_assistant_numbers_must_be_grounded():
    ctx = {"price": 24557700.0, "ret": 0.0123}
    assert assistant.numbers_grounded("قیمت 24,557,700 است و بازده 1.23٪", ctx)
    assert not assistant.numbers_grounded("قیمت 31,000,000 خواهد شد", ctx)


def test_llm_answer_with_invented_number_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k"); monkeypatch.setenv("LLM_BASE_URL", "http://x")
    class R:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "قیمت به 99,999,999 می‌رسد"}}]}
    monkeypatch.setattr(assistant.requests, "post", lambda *a, **k: R())
    assert assistant.llm_rephrase("قیمت 24,557,700", "q", {"a": 24557700.0}) is None


def test_server_security_and_validation(tmp_path):
    db = make_db(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(db))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    def call(path, body=None, headers=None):
        req = u.Request(base + path, None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
        try:
            r = u.urlopen(req); return r.status
        except u.HTTPError as e:
            return e.code
    try:
        assert call("/") == 200
        assert call("/api/series?symbol=BAD;DROP") == 400
        assert call("/api/series?symbol=GOLD18_IRR&days=999") == 400
        assert call("/api/assistant", {"question": "x"}) == 403                              # no custom header
        assert call("/api/assistant", {"question": "x"}, {"X-GT-Client": "1", "Origin": "http://evil.example"}) == 403
        assert call("/api/assistant", {"question": "x" * 50}, {"X-GT-Client": "1"}) == 400   # analysis not ready
        assert call("/api/settings", {"move_sigma": "abc"}, {"X-GT-Client": "1"}) == 400
        big = {"question": "x" * 20000}
        assert call("/api/assistant", big, {"X-GT-Client": "1"}) == 413
    finally:
        srv.shutdown()
