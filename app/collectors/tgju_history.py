import json
from datetime import datetime, timezone
from loguru import logger
from app.collectors.base import BaseCollector

class TgjuHistoryCollector(BaseCollector):
    """Historical backfill via the unofficial yfinance-ir package."""
    source_name = "YFINANCE_IR"

    def collect(self) -> int:
        try:
            import yfinance_ir as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance-ir before running historical backfill") from exc

        source_id = self.db.source_id(self.source_name)
        fetched = self.fetched_at()
        inserted = 0
        for ticker, symbol, factor in (("geram18", "GOLD18_IRR", 0.1),
                                       ("USD", "USD_IRR", 0.1)):
            hist = yf.Ticker(ticker).history(period="max")
            if hist is None or hist.empty:
                logger.warning("No history returned for {}", ticker)
                continue
            hist = hist.dropna(subset=["Close"])
            for index, row in hist.iterrows():
                dt = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
                if dt.tzinfo is None:
                    # Provider's historical index is expected to represent Tehran local time.
                    from zoneinfo import ZoneInfo
                    dt = dt.replace(tzinfo=ZoneInfo("Asia/Tehran"))
                ts = dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                price = float(row["Close"]) * factor
                raw = json.dumps({"ticker": ticker, "date": str(index),
                                  "open": _safe(row.get("Open")), "high": _safe(row.get("High")),
                                  "low": _safe(row.get("Low")), "close": _safe(row.get("Close")),
                                  "volume": _safe(row.get("Volume"))}, default=str)
                inserted += int(self.db.insert_observation(source_id, symbol, ts, price, raw, fetched))
        logger.info("Historical backfill inserted {} new observations", inserted)
        return inserted

def _safe(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
