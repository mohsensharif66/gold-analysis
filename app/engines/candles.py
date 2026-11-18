import pandas as pd

TIMEFRAMES = {"4h": "4h", "1d": "1D", "1w": "W-MON"}
FOUR_HOURS = pd.Timedelta(hours=4)
MIN_INTRADAY_SAMPLES = 8
MIN_INTRADAY_BUCKETS = 3


def _load_observations(db, symbol_code: str) -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute("""
            SELECT r.ts_utc, r.price, s.name AS source_name, s.priority
            FROM raw_observations r
            JOIN sources s ON s.id = r.source_id
            WHERE r.symbol_code=?
            ORDER BY r.ts_utc ASC, s.priority ASC, r.id ASC
        """, (symbol_code,)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "price"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"]).sort_values(["ts", "priority"])
    # One observation per timestamp; the source with the best configured priority wins.
    df = df.drop_duplicates("ts", keep="first").set_index("ts").sort_index()
    return df[["price"]]


def _intraday_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep observations belonging to clusters with <=4h spacing.

    Daily historical observations must never be converted into synthetic 4h candles.
    A minimum sample/bucket threshold avoids treating one isolated pair as a usable
    intraday history.
    """
    if len(df) < MIN_INTRADAY_SAMPLES:
        return df.iloc[0:0]
    gaps = df.index.to_series().diff()
    adjacent = gaps.le(FOUR_HOURS) | gaps.shift(-1).le(FOUR_HOURS)
    intraday = df.loc[adjacent.fillna(False)]
    if len(intraday) < MIN_INTRADAY_SAMPLES:
        return intraday.iloc[0:0]
    bucket_count = intraday.resample("4h", label="left", closed="left").size().gt(0).sum()
    if bucket_count < MIN_INTRADAY_BUCKETS:
        return intraday.iloc[0:0]
    return intraday


def build_candles(db, symbol_code: str, timeframe: str) -> int:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    df = _load_observations(db, symbol_code)

    if timeframe == "4h":
        df = _intraday_only(df)
        if df.empty:
            # Remove legacy 4h bars previously fabricated from daily observations.
            with db.connect() as conn:
                conn.execute("DELETE FROM indicators WHERE symbol_code=? AND timeframe='4h'", (symbol_code,))
                conn.execute("DELETE FROM candles WHERE symbol_code=? AND timeframe='4h'", (symbol_code,))
            return 0

    if df.empty:
        return 0

    grouped = df["price"].resample(TIMEFRAMES[timeframe], label="left", closed="left")
    ohlc = grouped.ohlc().dropna()
    counts = grouped.count().reindex(ohlc.index).fillna(0).astype(int)
    now = pd.Timestamp.now(tz="UTC")
    duration = pd.Timedelta(hours=4) if timeframe == "4h" else (
        pd.Timedelta(days=1) if timeframe == "1d" else pd.Timedelta(days=7))
    values = []
    for ts, row in ohlc.iterrows():
        open_time = ts.isoformat().replace("+00:00", "Z")
        complete = int(ts + duration <= now)
        values.append((symbol_code, timeframe, open_time, float(row.open), float(row.high),
                       float(row.low), float(row.close), int(counts.loc[ts]), None, complete))
    with db.connect() as conn:
        conn.executemany("""
            INSERT INTO candles(symbol_code,timeframe,open_time_utc,open,high,low,close,
                                sample_count,volume,is_complete)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol_code,timeframe,open_time_utc) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
              sample_count=excluded.sample_count, is_complete=excluded.is_complete
        """, values)
    return len(values)
