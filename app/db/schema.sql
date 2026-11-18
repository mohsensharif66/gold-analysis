PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1))
);

CREATE TABLE IF NOT EXISTS symbols (
    code TEXT PRIMARY KEY,
    name_fa TEXT NOT NULL,
    unit TEXT NOT NULL,
    currency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    ts_utc TEXT NOT NULL,
    price REAL NOT NULL CHECK (price > 0),
    raw_json TEXT,
    fetched_at_utc TEXT NOT NULL,
    UNIQUE(source_id, symbol_code, ts_utc)
);
CREATE INDEX IF NOT EXISTS ix_raw_symbol_ts ON raw_observations(symbol_code, ts_utc);

CREATE TABLE IF NOT EXISTS candles (
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    volume REAL,
    is_complete INTEGER NOT NULL DEFAULT 0 CHECK (is_complete IN (0,1)),
    PRIMARY KEY(symbol_code, timeframe, open_time_utc)
);

CREATE TABLE IF NOT EXISTS quality_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,
    flag_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,
    close REAL NOT NULL,
    rsi_14 REAL, macd REAL, macd_signal REAL, macd_hist REAL,
    ema_20 REAL, ema_50 REAL, ema_200 REAL,
    bb_middle REAL, bb_upper REAL, bb_lower REAL,
    tenkan_sen REAL, kijun_sen REAL,
    senkou_span_a REAL, senkou_span_b REAL, chikou_span REAL,
    computed_at_utc TEXT NOT NULL,
    UNIQUE(symbol_code, timeframe, open_time_utc)
);

CREATE TABLE IF NOT EXISTS levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    level_type TEXT NOT NULL,
    source_layer TEXT NOT NULL,
    price REAL NOT NULL,
    strength REAL,
    distance_pct REAL,
    meta_json TEXT,
    computed_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bubble_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,
    market_price REAL NOT NULL,
    theoretical_price REAL NOT NULL,
    bubble_pct REAL NOT NULL,
    zscore_60 REAL,
    computed_at_utc TEXT NOT NULL,
    UNIQUE(symbol_code, timeframe, open_time_utc)
);

CREATE TABLE IF NOT EXISTS decomposition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_code TEXT NOT NULL REFERENCES symbols(code),
    timeframe TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,
    ret_gold REAL, ret_usd REAL, ret_xau REAL,
    alpha REAL, beta_usd REAL, beta_xau REAL, residual REAL,
    computed_at_utc TEXT NOT NULL,
    UNIQUE(symbol_code, timeframe, open_time_utc)
);

CREATE TABLE IF NOT EXISTS economic_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES sources(id),
    event_uid TEXT NOT NULL,
    currency TEXT,
    event_name TEXT,
    scheduled_utc TEXT,
    impact TEXT,
    actual TEXT,
    forecast TEXT,
    previous TEXT,
    raw_json TEXT,
    fetched_at_utc TEXT NOT NULL,
    UNIQUE(source_id, event_uid)
);

-- ===== v5 additions (additive; existing tables untouched) =====
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    asset_symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    base_price REAL NOT NULL,
    predicted_return REAL NOT NULL,
    predicted_price REAL NOT NULL,
    lower_bound REAL NOT NULL,
    upper_bound REAL NOT NULL,
    market_regime TEXT,
    technical_score REAL,
    ensemble_output REAL,
    ensemble_active INTEGER NOT NULL DEFAULT 0,
    signal TEXT,
    confidence_calibration_status TEXT,
    target_date TEXT,
    actual_price REAL,
    actual_return REAL,
    in_interval INTEGER,
    abs_error REAL,
    resolved_at_utc TEXT,
    UNIQUE(asset_symbol, data_cutoff, model_version, horizon_days)
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    key TEXT PRIMARY KEY,
    data_cutoff TEXT NOT NULL,
    computed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    inserted INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key TEXT NOT NULL UNIQUE,
    created_at_utc TEXT NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message_fa TEXT NOT NULL,
    data_json TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
