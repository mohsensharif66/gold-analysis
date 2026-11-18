"""Daily XAU/USD history from the documented XAUS history endpoint.

XAUS documents up to 1,825 daily closes (approximately five years). The API
returns daily close/high/low, but no open or volume. We persist the close as
the observation price and retain the full source point in raw_json.
"""
import json
from datetime import date, datetime, time, timezone
from loguru import logger
from app.collectors.base import BaseCollector

URL = "https://xaus.com/api/v1/history?days=1825"


class XauHistoryCollector(BaseCollector):
    source_name = "XAUS_HISTORY"

    def collect(self) -> int:
        payload = self.get_json(URL)
        points = payload.get("points") if isinstance(payload, dict) else None
        if not isinstance(points, list):
            raise ValueError("XAUS history response has no points array")

        source_id = self.db.source_id(self.source_name)
        fetched = self.fetched_at()
        inserted = 0
        accepted = 0
        for point in points:
            if not isinstance(point, dict):
                continue
            day = point.get("d")
            close = point.get("c")
            try:
                day_value = date.fromisoformat(str(day))
                price = float(close)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            # Daily date is represented at 00:00 UTC as a date bucket key,
            # not as a claim about the exact market close timestamp.
            dt = datetime.combine(day_value, time.min, tzinfo=timezone.utc)
            ts = dt.isoformat(timespec="seconds").replace("+00:00", "Z")
            raw = json.dumps({
                "provider": "XAUS",
                "endpoint": URL,
                "interval": "1d",
                "date": day_value.isoformat(),
                "close": price,
                "high": _positive_or_none(point.get("h")),
                "low": _positive_or_none(point.get("l")),
                "open": None,
                "volume": None,
                "data_state": payload.get("data_state"),
            }, ensure_ascii=False, separators=(",", ":"))
            accepted += 1
            inserted += int(self.db.insert_observation(
                source_id, "XAU_USD", ts, price, raw, fetched
            ))

        if not accepted:
            raise ValueError("XAUS history returned no valid daily close points")
        logger.info(
            "XAUS historical XAU/USD: accepted={} inserted={} first={} last={}",
            accepted, inserted, points[0].get("d"), points[-1].get("d")
        )
        return inserted


def _positive_or_none(value):
    try:
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None
