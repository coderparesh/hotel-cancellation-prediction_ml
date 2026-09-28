"""
Logging configuration for the project.

Provides a pre-configured logger with both file and console output,
using a timestamped format for easy debugging and audit trails.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def get_logger(name: str, log_file: str = "project.log") -> logging.Logger:
    """
    Return a logger with console + file handlers.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    log_file : str
        Log filename, written to ``PROJECT_ROOT / log_file``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler ──
    log_path = PROJECT_ROOT / log_file
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
