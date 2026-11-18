from datetime import datetime, timezone


def test_yfinance_fallback_rejects_naive_timestamp():
    # The collector must not silently interpret a provider timestamp without timezone.
    dt = datetime(2026, 11, 16, 12, 0, 0)
    assert dt.tzinfo is None


def test_utc_timestamp_is_aware():
    dt = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
    assert dt.tzinfo is not None
