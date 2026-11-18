import json
from datetime import datetime, timezone
from loguru import logger
from app.collectors.base import BaseCollector

URL = "https://xaus.com/api/v1/spot?compact=1"

def _walk_find(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
        for value in obj.values():
            found = _walk_find(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _walk_find(value, keys)
            if found is not None:
                return found
    return None

class XausCollector(BaseCollector):
    source_name = "XAUS"

    def collect(self) -> int:
        payload = self.get_json(URL)
        price_raw = _walk_find(payload, ("price", "spot", "xau_usd", "value", "last"))
        if price_raw is None:
            raise ValueError("XAUS response does not contain a recognized spot price")
        price = float(str(price_raw).replace(",", ""))
        if price <= 0:
            raise ValueError(f"Invalid XAUS price: {price}")
        status = _walk_find(payload, ("status",))
        state = str(status).lower() if status is not None else "unknown"
        timestamp = _walk_find(payload, ("timestamp", "ts", "updated_at", "time"))
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(float(timestamp) / (1000 if timestamp > 1e10 else 1), timezone.utc)
        elif timestamp:
            dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        ts_utc = dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        raw = json.dumps({"payload": payload, "data_state_status": state}, ensure_ascii=False, default=str)
        inserted = self.db.insert_observation(self.db.source_id(self.source_name),
                    "XAU_USD", ts_utc, price, raw, self.fetched_at())
        logger.info("XAUS status={} inserted={}", state, inserted)
        if state not in ("fresh", "unknown"):
            logger.warning("XAUS quote is not fresh: {}", state)
        return int(inserted)
