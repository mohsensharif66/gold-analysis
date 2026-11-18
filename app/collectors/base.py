from abc import ABC, abstractmethod
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import settings


class BaseCollector(ABC):
    source_name: str

    def __init__(self, db):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.user_agent,
            "Accept": "application/json,*/*",
        })
        retry = Retry(
            total=2,
            connect=2,
            read=1,
            status=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_json(self, url: str):
        response = self.session.get(
            url,
            timeout=(settings.http_connect_timeout, settings.http_read_timeout),
        )
        response.raise_for_status()
        return response.json()

    @abstractmethod
    def collect(self) -> int:
        raise NotImplementedError

    @staticmethod
    def fetched_at() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
