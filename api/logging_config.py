"""Structured logging configuration for Forensic Signature AI.

Configures standard Python logging with ISO timestamps, appropriate levels,
and formatted outputs suitable for containerized and cloud runtimes.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return the root application logger.

    Args:
        log_level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured root logger instance.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("forensic_ai")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logging()
