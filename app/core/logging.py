"""Centralized logging configuration for the application."""

import logging
import logging.config
from typing import Final


DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_level: str | int = logging.INFO) -> None:
    """Configure consistent console logging for the application.

    The function is safe to call more than once. Each call replaces the root
    logger configuration instead of adding duplicate handlers.
    """

    if isinstance(log_level, str):
        normalized_level = log_level.upper()
        if normalized_level not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid log level: {log_level}")
        log_level = normalized_level

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": DEFAULT_LOG_FORMAT,
                    "datefmt": DEFAULT_DATE_FORMAT,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": log_level,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger managed by the centralized configuration."""

    return logging.getLogger(name)
