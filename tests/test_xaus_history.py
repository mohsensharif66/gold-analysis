import json
from app.collectors.xau_history import XauHistoryCollector


class FakeDB:
    def __init__(self):
        self.rows = []

    def source_id(self, name):
        assert name == "XAUS_HISTORY"
        return 17

    def insert_observation(self, source_id, symbol, ts, price, raw, fetched):
        self.rows.append((source_id, symbol, ts, price, json.loads(raw), fetched))
        return True


def test_xaus_history_maps_daily_close_and_preserves_hilo():
    db = FakeDB()
    collector = XauHistoryCollector(db)
    collector.get_json = lambda url: {
        "points": [
            {"d": "2026-11-14", "c": 4100.5, "h": 4120.0, "l": 4080.0},
            {"d": "2026-11-15", "c": 4110.5, "h": 4130.0, "l": 4090.0},
        ],
        "data_state": {"status": "fresh"},
    }

    inserted = collector.collect()

    assert inserted == 2
    assert db.rows[0][1] == "XAU_USD"
    assert db.rows[0][2] == "2026-11-14T00:00:00Z"
    assert db.rows[0][3] == 4100.5
    assert db.rows[0][4]["high"] == 4120.0
    assert db.rows[0][4]["low"] == 4080.0
    assert db.rows[0][4]["open"] is None


def test_xaus_history_rejects_empty_valid_points():
    db = FakeDB()
    collector = XauHistoryCollector(db)
    collector.get_json = lambda url: {"points": [{"d": "bad-date", "c": "n/a"}]}

    try:
        collector.collect()
    except ValueError as exc:
        assert "no valid daily close" in str(exc)
    else:
        raise AssertionError("Expected invalid history payload to fail")
