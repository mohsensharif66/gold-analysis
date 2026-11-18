from datetime import datetime, timezone
from loguru import logger

from app.collectors.tgju import TgjuCollector
from app.collectors.xaus import XausCollector
from app.collectors.yfinance_spot import YFinanceSpotCollector
from app.collectors.tgju_history import TgjuHistoryCollector
from app.collectors.xau_history import XauHistoryCollector


def _record(db, source, count, error=None):
    try:
        with db.connect() as conn:
            conn.execute("INSERT INTO collection_runs(started_at_utc,source,status,inserted,error) VALUES(?,?,?,?,?)",
                         (datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), source,
                          "ok" if error is None else "error", count, error))
    except Exception:  # never let bookkeeping break collection
        logger.warning("could not record collection run for {}", source)


def collect_all(db, include_history=False):
    totals = {}

    # Iranian market quotes are collected independently from the international quote.
    for collector in (TgjuCollector(db),):
        try:
            totals[collector.source_name] = collector.collect()
            _record(db, collector.source_name, totals[collector.source_name])
        except Exception as exc:
            logger.exception("Collector failed: {}", collector.source_name)
            totals[collector.source_name] = None
            _record(db, collector.source_name, None, str(exc)[:300])

    # XAUS is the primary XAU/USD source. Yahoo Finance is attempted only on failure.
    try:
        xaus_count = XausCollector(db).collect()
        totals["XAUS"] = xaus_count
        _record(db, "XAUS", xaus_count)
    except Exception as exc:
        _record(db, "XAUS", None, str(exc)[:300])
        logger.warning("XAUS unavailable ({}); trying Yahoo Finance fallback", exc)
        totals["XAUS"] = None
        try:
            totals["YFINANCE"] = YFinanceSpotCollector(db).collect()
            _record(db, "YFINANCE", totals["YFINANCE"])
        except Exception as exc2:
            _record(db, "YFINANCE", None, str(exc2)[:300])
            logger.exception("Both XAUS and Yahoo Finance XAU/USD collection failed")
            totals["YFINANCE"] = None

    if include_history:
        collector = TgjuHistoryCollector(db)
        try:
            totals[collector.source_name] = collector.collect()
            _record(db, collector.source_name, totals[collector.source_name])
        except Exception as exc:
            logger.exception("Collector failed: {}", collector.source_name)
            totals[collector.source_name] = None
            _record(db, collector.source_name, None, str(exc)[:300])

        xau_history = XauHistoryCollector(db)
        try:
            totals[xau_history.source_name] = xau_history.collect()
            _record(db, xau_history.source_name, totals[xau_history.source_name])
        except Exception as exc:
            logger.exception("Collector failed: {}", xau_history.source_name)
            totals[xau_history.source_name] = None
            _record(db, xau_history.source_name, None, str(exc)[:300])

    return totals
