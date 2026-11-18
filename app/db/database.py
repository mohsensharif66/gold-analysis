import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import settings, PROJECT_ROOT

SEED_SYMBOLS = [
    ("GOLD18_IRR", "طلای ۱۸ عیار", "toman/gram", "IRR"),
    ("USD_IRR", "دلار آزاد", "toman", "IRR"),
    ("XAU_USD", "انس جهانی طلا", "USD/oz", "USD"),
]

SEED_SOURCES = [
    ("TGJU", "https://call1.tgju.org/ajax.json?rev=2", 1),
    ("XAUS", "https://xaus.com/api/v1/spot?compact=1", 2),
    ("YFINANCE_IR", "yfinance-ir", 3),
    ("XAUS_HISTORY", "https://xaus.com/api/v1/history", 3),
    ("YFINANCE", "yfinance", 4),
]

class Database:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.resolved_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def initialize(self):
        schema = (PROJECT_ROOT / "app" / "db" / "schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)
            conn.executemany(
                "INSERT OR IGNORE INTO symbols(code,name_fa,unit,currency) VALUES(?,?,?,?)",
                SEED_SYMBOLS,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO sources(name,base_url,priority) VALUES(?,?,?)",
                SEED_SOURCES,
            )

    def source_id(self, name: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM sources WHERE name=? AND is_active=1", (name,)).fetchone()
            if not row:
                raise RuntimeError(f"Active source not found: {name}")
            return int(row["id"])

    def insert_observation(self, source_id: int, symbol: str, ts_utc: str,
                           price: float, raw_json: str, fetched_at_utc: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO raw_observations
                (source_id,symbol_code,ts_utc,price,raw_json,fetched_at_utc)
                VALUES(?,?,?,?,?,?)
            """, (source_id, symbol, ts_utc, price, raw_json, fetched_at_utc))
            return cur.rowcount == 1

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
