from datetime import datetime
from app.collectors.tgju import parse_tgju_timestamp

def test_naive_tgju_time_is_tehran():
    actual = parse_tgju_timestamp("2026-11-16T12:00:00")
    assert actual == "2026-11-16T08:30:00Z"

def test_aware_timestamp_is_converted():
    actual = parse_tgju_timestamp("2026-11-16T08:30:00Z")
    assert actual == "2026-11-16T08:30:00Z"
