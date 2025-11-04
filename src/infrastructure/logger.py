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

from rich.logging import RichHandler
from rich.console import Console


# Shared console instance for coordinating rich output (Progress, logging, etc.)
# This ensures all rich components write to the same console and don't overlap
_shared_console = Console()


def get_shared_console() -> Console:
    """
    Get the shared Console instance used by logger and other rich components.

    This should be used by Progress bars and other rich displays to ensure
    proper coordination with logging output.

    Returns:
        Shared Console instance

    Example:
        ```python
        from src.infrastructure.logger import get_shared_console
        from rich.progress import Progress

        console = get_shared_console()
        with Progress(console=console) as progress:
            # Progress bar and logging are now coordinated
            task = progress.add_task("Processing...")
        ```
    """
    return _shared_console


# Initialize logging on module import
def _init_logging():
    """Initialize logging with file and console handlers."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{timestamp}.log"

    # Get log level from environment (default INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    configured_level = level_map.get(log_level, logging.INFO)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(configured_level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # File handler - respects LOG_LEVEL from .env
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(configured_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler - use RichHandler with shared console to coordinate with rich Progress displays
    console_handler = RichHandler(
        level=configured_level,
        console=_shared_console,  # Use shared console for coordination
        show_time=True,
        show_level=True,
        show_path=True,
        markup=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(configured_level)
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
