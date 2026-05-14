"""Structured logging via loguru."""
from __future__ import annotations

import sys

from loguru import logger as _logger

_logger.remove()
_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)

logger = _logger
