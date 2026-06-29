"""Logging consistente con `rich` para todo el paquete."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


_CONFIGURED = False


def _build_file_handler() -> Optional[logging.Handler]:
    """FileHandler durable si MICORIZAE_LOG_FILE está definido.

    Escribe cada registro a disco con flush inmediato (independiente del
    buffering de la consola/PowerShell), de modo que nunca se pierdan logs
    aunque el proceso muera abruptamente.
    """
    path = os.environ.get("MICORIZAE_LOG_FILE")
    if not path:
        return None
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        return fh
    except Exception:  # noqa: BLE001 - el log a archivo es best-effort
        return None


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [handler]
    file_handler = _build_file_handler()
    if file_handler is not None:
        handlers.append(file_handler)
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name or "micorizae")
