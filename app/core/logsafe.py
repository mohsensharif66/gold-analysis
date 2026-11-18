"""Logger that uses loguru when installed and falls back to stdlib logging.

New modules import `logger` from here so the analytics layer stays importable
(and testable) even if loguru is missing. Messages use loguru's "{}" style.
"""
try:  # pragma: no cover
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    class _Adapter:
        def __init__(self):
            self._l = logging.getLogger("goldtrend")

        def _fmt(self, msg, args):
            try:
                return msg.format(*args) if args else msg
            except Exception:
                return f"{msg} {args}"

        def debug(self, m, *a): self._l.debug(self._fmt(m, a))
        def info(self, m, *a): self._l.info(self._fmt(m, a))
        def success(self, m, *a): self._l.info(self._fmt(m, a))
        def warning(self, m, *a): self._l.warning(self._fmt(m, a))
        def error(self, m, *a): self._l.error(self._fmt(m, a))
        def exception(self, m, *a): self._l.exception(self._fmt(m, a))

    logger = _Adapter()
