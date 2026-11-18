from loguru import logger


def run_data_audit(db) -> None:
    logger.info("========== GOLDTREND DATA AUDIT ==========")
    with db.connect() as conn:
        raw = conn.execute("""
            SELECT y.symbol_code, s.name, COUNT(*) AS n, MIN(y.ts_utc) AS first_ts,
                   MAX(y.ts_utc) AS last_ts
            FROM raw_observations y JOIN sources s ON s.id=y.source_id
            GROUP BY y.symbol_code, s.name ORDER BY y.symbol_code, s.name
        """).fetchall()
        candles = conn.execute("""
            SELECT symbol_code, timeframe, COUNT(*) AS n, MIN(open_time_utc) AS first_ts,
                   MAX(open_time_utc) AS last_ts, MIN(sample_count) AS min_samples,
                   MAX(sample_count) AS max_samples
            FROM candles GROUP BY symbol_code, timeframe
            ORDER BY symbol_code, timeframe
        """).fetchall()
        indicators = conn.execute("""
            SELECT symbol_code, timeframe, COUNT(*) AS n,
                   SUM(CASE WHEN rsi_14 IS NOT NULL THEN 1 ELSE 0 END) AS valid_rsi,
                   SUM(CASE WHEN ema_200 IS NOT NULL THEN 1 ELSE 0 END) AS valid_ema200
            FROM indicators GROUP BY symbol_code, timeframe
            ORDER BY symbol_code, timeframe
        """).fetchall()

    logger.info("RAW OBSERVATIONS BY SYMBOL/SOURCE")
    for r in raw:
        logger.info("{} | {} | rows={} | first={} | last={}",
                    r["symbol_code"], r["name"], r["n"], r["first_ts"], r["last_ts"])
    logger.info("CANDLES BY SYMBOL/TIMEFRAME")
    for r in candles:
        logger.info("{} | {} | rows={} | first={} | last={} | samples_per_bar={}..{}",
                    r["symbol_code"], r["timeframe"], r["n"], r["first_ts"], r["last_ts"],
                    r["min_samples"], r["max_samples"])
    logger.info("INDICATOR VALIDITY")
    for r in indicators:
        logger.info("{} | {} | rows={} | valid_RSI={} | valid_EMA200={}",
                    r["symbol_code"], r["timeframe"], r["n"], r["valid_rsi"], r["valid_ema200"])
    logger.info("========== END DATA AUDIT ==========")
