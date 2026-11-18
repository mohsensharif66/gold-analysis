from loguru import logger
from app.core.config import settings, PROJECT_ROOT

def configure_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=settings.log_level)
    logger.add(log_dir / "goldtrend_{time:YYYY-MM-DD}.log",
               rotation="10 MB", retention="14 days",
               level=settings.log_level, encoding="utf-8")
