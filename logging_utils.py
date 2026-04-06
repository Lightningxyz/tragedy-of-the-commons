"""
Logging configuration utilities for commons simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure app-wide logging with optional file sink.

    Log format is key-value oriented to simplify downstream parsing.
    """
    parsed_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=parsed_level,
        format=(
            "%(asctime)s level=%(levelname)s logger=%(name)s "
            "message=%(message)s"
        ),
        handlers=handlers,
        force=force,
    )
