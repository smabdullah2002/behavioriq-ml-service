"""Centralized logging configuration for BehaviorIQ ML Service.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("Something happened", extra={"endpoint": "/ml/intent-score"})

Logs are sent to:
  1. stdout (always) — for local development and docker logs
  2. Grafana Loki (when LOKI_URL env var is set) — for the observability dashboard
"""

import logging
import os
import sys
from typing import Optional

# ── Loki labels attached to every log line ──────────────────────────────────
_SERVICE_LABEL = "ml-service"
_ENV_LABEL = os.getenv("APP_ENV", "hackathon")
_LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# ── Module-level sentinel so setup only runs once ───────────────────────────
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with stdout + optional Loki handlers.

    Call this once at application startup (in main.py).
    Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    # ── 1. Stdout handler ────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    # ── 2. Loki handler (optional) ───────────────────────────────────────────
    _attach_loki_handler(root)

    logging.getLogger("uvicorn.access").propagate = False  # suppress noisy access logs from root


def _loki_reachable() -> bool:
    """Return True if Loki's HTTP port responds within 1 second."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(_LOKI_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 3100
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _attach_loki_handler(root: logging.Logger) -> None:
    """Try to attach python-logging-loki handler. Skips gracefully if unavailable or unreachable."""
    try:
        import logging_loki  # type: ignore
    except ImportError:
        root.warning(
            "python-logging-loki not installed — logs will only go to stdout. "
            "Run: pip install python-logging-loki"
        )
        return

    if not _loki_reachable():
        print(
            f"[logger] Loki not reachable at {_LOKI_URL} — stdout-only logging active. "
            "Start Docker (docker compose up) to enable Loki.",
            flush=True,
        )
        return

    try:
        loki_handler = logging_loki.LokiHandler(
            url=_LOKI_URL,
            tags={
                "service": _SERVICE_LABEL,
                "environment": _ENV_LABEL,
            },
            version="1",
        )
        loki_handler.setLevel(logging.DEBUG)
        root.addHandler(loki_handler)
        print(f"[logger] Loki handler attached → {_LOKI_URL}", flush=True)
    except Exception as exc:  # noqa: BLE001
        root.warning("Could not attach Loki handler: %s — continuing with stdout only.", exc)



def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger.  Call setup_logging() first at startup.

    Args:
        name: typically __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name or _SERVICE_LABEL)
