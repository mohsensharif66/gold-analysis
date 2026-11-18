"""Fallback XAU/USD spot collector using Yahoo Finance's public market data."""
import json
from datetime import timezone

from loguru import logger

from app.collectors.base import BaseCollector


class YFinanceSpotCollector(BaseCollector):
    source_name = "YFINANCE"
    ticker = "XAUUSD=X"

    def collect(self) -> int:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use the XAU/USD fallback") from exc

        frame = yf.download(
            self.ticker,
            period="5d",
            interval="1h",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if frame is None or frame.empty or "Close" not in frame:
            raise RuntimeError(f"Yahoo Finance returned no usable data for {self.ticker}")

        close = frame["Close"]
        # yfinance may return a one-column DataFrame with MultiIndex columns.
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            raise RuntimeError(f"Yahoo Finance returned no closing price for {self.ticker}")

        index = close.index[-1]
        dt = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
        if dt.tzinfo is None:
            raise RuntimeError("Yahoo Finance timestamp is timezone-naive; refusing ambiguous XAU/USD time")
        ts_utc = dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        price = float(close.iloc[-1])
        if price <= 0:
            raise ValueError(f"Invalid Yahoo Finance XAU/USD price: {price}")

        raw = json.dumps({
            "provider": "Yahoo Finance",
            "ticker": self.ticker,
            "interval": "1h",
            "close": price,
            "provider_timestamp": str(index),
            "fallback": True,
        }, ensure_ascii=False, default=str)
        inserted = self.db.insert_observation(
            self.db.source_id(self.source_name), "XAU_USD", ts_utc,
            price, raw, self.fetched_at(),
        )
        logger.info("Yahoo Finance XAU/USD fallback price={} inserted={}", price, inserted)
        return int(inserted)
