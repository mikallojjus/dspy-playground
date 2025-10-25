"""
Logging utility for the claim-quote extraction system.

Provides structured logging with multiple levels, file output, and console output.
Each run creates a new timestamped log file.

Usage:
    from src.infrastructure.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing episode 123")
    logger.warning("Reranker unavailable, using fallback")
    logger.error("Failed to extract claims", exc_info=True)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


# ANSI color codes for console output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console."""
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{RESET}"
        return super().format(record)


# Initialize logging on module import
def _init_logging():
    """Initialize logging with file and console handlers."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{timestamp}.log"

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Remove any existing handlers

    # File handler - logs everything (DEBUG and above)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler - configurable level (default INFO)
    console_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_map.get(console_level, logging.INFO))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initialization
    init_logger = logging.getLogger(__name__)
    init_logger.info(f"Logger initialized. Logs saved to: {log_file}")


# Initialize logging when module is imported
_init_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance configured with file and console handlers

    Example:
        ```python
        from src.infrastructure.logger import get_logger

        logger = get_logger(__name__)
        logger.debug("Detailed debug information")
        logger.info("Processing episode 123")
        logger.warning("Reranker service unavailable, using fallback")
        logger.error("Failed to process claim", exc_info=True)
        ```
    """
    return logging.getLogger(name)
