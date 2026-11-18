import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from loguru import logger
from app.collectors.base import BaseCollector
from app.db.database import utc_now_iso

URL = "https://call1.tgju.org/ajax.json?rev=2"
TEHRAN = ZoneInfo("Asia/Tehran")

def _find_symbol_payload(obj, wanted):
    """Find a TGJU symbol record regardless of wrapper nesting."""
    if isinstance(obj, dict):
        if wanted in obj and isinstance(obj[wanted], (dict, list, str, int, float)):
            return obj[wanted]
        for value in obj.values():
            found = _find_symbol_payload(value, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_symbol_payload(value, wanted)
            if found is not None:
                return found
    return None

def _first_value(record, names):
    if isinstance(record, dict):
        for key in names:
            if key in record and record[key] not in (None, ""):
                return record[key]
    return None

def parse_tgju_timestamp(value):
    if value is None:
        return utc_now_iso()
    if isinstance(value, (int, float)):
        # TGJU commonly supplies Unix seconds; tolerate milliseconds.
        epoch = float(value) / (1000 if value > 10_000_000_000 else 1)
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TEHRAN)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except ValueError:
        # Numeric timestamp encoded as text.
        try:
            return parse_tgju_timestamp(float(text))
        except ValueError:
            logger.warning("Unrecognized TGJU timestamp {!r}; using fetch time", value)
            return utc_now_iso()

def parse_price(value):
    if value is None:
        raise ValueError("Missing price")
    cleaned = str(value).replace(",", "").replace("٬", "").strip()
    price = float(cleaned)
    if price <= 0:
        raise ValueError(f"Invalid non-positive price: {price}")
    return price

class TgjuCollector(BaseCollector):
    source_name = "TGJU"
    SYMBOLS = {
        "geram18": ("GOLD18_IRR", 0.1),
        "price_dollar_rl": ("USD_IRR", 0.1),
    }

    def collect(self) -> int:
        payload = self.get_json(URL)
        source_id = self.db.source_id(self.source_name)
        fetched = self.fetched_at()
        inserted = 0
        for provider_symbol, (symbol, factor) in self.SYMBOLS.items():
            record = _find_symbol_payload(payload, provider_symbol)
            if record is None:
                logger.warning("TGJU symbol missing: {}", provider_symbol)
                continue
            price_raw = _first_value(record, ("p", "price", "value", "last", "close"))
            ts_raw = _first_value(record, ("ts", "timestamp", "time", "date"))
            try:
                price = parse_price(price_raw) * factor
                ts_utc = parse_tgju_timestamp(ts_raw)
                raw = json.dumps({"provider_symbol": provider_symbol, "record": record},
                                 ensure_ascii=False, default=str)
                inserted += int(self.db.insert_observation(source_id, symbol, ts_utc, price, raw, fetched))
            except (TypeError, ValueError) as exc:
                logger.warning("TGJU parse failed for {}: {}", provider_symbol, exc)
        logger.info("TGJU inserted {} new observations", inserted)
        return inserted
