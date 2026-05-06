"""Centralized logging via loguru. Import `log` from anywhere."""

from __future__ import annotations

import sys

from loguru import logger

from packages.common.config import LOGS_DIR, settings

# Remove the default handler and reconfigure.
logger.remove()

# Pretty stderr handler for development.
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Rolling file handler for persistent logs.
logger.add(
    LOGS_DIR / "ta_agent_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    enqueue=True,
)

log = logger
