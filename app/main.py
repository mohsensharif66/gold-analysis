import argparse
from loguru import logger
from app.core.logging import configure_logging
from app.db.database import Database
from app.services.collection_service import collect_all
from app.engines.candles import build_candles
from app.engines.indicators import persist_indicators
from app.engines.audit import run_data_audit

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="GoldTrend Pro backend")
    parser.add_argument("command", choices=["init-db","collect","backfill","build-candles","indicators","pipeline","audit","serve","analyze","report","backtest","quality"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    db = Database()
    if args.command == "init-db":
        db.initialize()
        logger.success("Database initialized: {}", db.path)
        return
    if args.command == "audit":
        db.initialize()
        run_data_audit(db)
        return
    db.initialize()
    if args.command == "serve":
        from app.web.server import serve
        serve(args.host, args.port)
        return
    if args.command in ("analyze", "report", "backtest", "quality"):
        from app.analytics import service, quality
        from app.agents import agents
        if args.command == "quality":
            q = quality.run_quality(db)
            logger.info("Quality: {} (critical={}, warnings={})", q["status"], q["critical"], q["warnings"])
            return
        snap = service.snapshot(db, compute_backtest=True, force=(args.command == "backtest"))
        print(agents.daily_report(snap))
        return
    if args.command in ("collect","pipeline"):
        logger.info("Collection results: {}", collect_all(db))
    if args.command == "backfill":
        logger.info("Backfill results: {}", collect_all(db, include_history=True))
    if args.command in ("build-candles","pipeline"):
        for symbol in ("GOLD18_IRR","USD_IRR","XAU_USD"):
            for tf in ("4h","1d","1w"):
                count = build_candles(db,symbol,tf)
                logger.info("{} {} candles upserted: {}",symbol,tf,count)
    if args.command in ("indicators","pipeline"):
        for symbol in ("GOLD18_IRR","USD_IRR","XAU_USD"):
            for tf in ("4h","1d","1w"):
                count = persist_indicators(db,symbol,tf)
                logger.info("{} {} indicators upserted: {}",symbol,tf,count)

if __name__ == "__main__":
    main()
