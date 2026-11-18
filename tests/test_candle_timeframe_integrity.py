from datetime import datetime, timedelta, timezone

from app.db.database import Database
from app.engines.candles import build_candles


def test_daily_history_is_not_fabricated_into_four_hour_candles(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    source_id = db.source_id("YFINANCE_IR")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(30):
        ts = (start + timedelta(days=i)).isoformat().replace("+00:00", "Z")
        db.insert_observation(source_id, "GOLD18_IRR", ts, 1000 + i, "{}", ts)

    assert build_candles(db, "GOLD18_IRR", "4h") == 0
    assert build_candles(db, "GOLD18_IRR", "1d") == 30
    with db.connect() as conn:
        four_hour_count = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol_code='GOLD18_IRR' AND timeframe='4h'"
        ).fetchone()[0]
    assert four_hour_count == 0
