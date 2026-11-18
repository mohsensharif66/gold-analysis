from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_path: Path = Path(os.getenv("DATABASE_PATH", "data/goldtrend.db"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    http_connect_timeout: int = int(os.getenv("HTTP_CONNECT_TIMEOUT", "8"))
    http_read_timeout: int = int(os.getenv("HTTP_READ_TIMEOUT", "15"))
    user_agent: str = os.getenv("HTTP_USER_AGENT", "GoldTrendPro/0.1")

    def resolved_database_path(self) -> Path:
        p = self.database_path
        return p if p.is_absolute() else PROJECT_ROOT / p

settings = Settings()
