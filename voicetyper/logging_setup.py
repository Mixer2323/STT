"""Logging setup: rotation, crash dumps and privacy-aware transcript logging."""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import Optional

from . import paths

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_configured = False
_fault_file = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    global _configured
    root = logging.getLogger()
    if _configured:
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logging.getLogger("voicetyper")

    log_dir = paths.logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "voice_typer.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    handlers.append(file_handler)

    stream = sys.stdout
    if stream is not None and getattr(stream, "isatty", lambda: False)():
        console = logging.StreamHandler(stream)
        console.setFormatter(logging.Formatter(_FORMAT))
        handlers.append(console)

    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = handlers
    for noisy in ("urllib3", "huggingface_hub", "filelock", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _install_exception_hooks(log_dir)
    _configured = True
    logging.getLogger("voicetyper").info("Логи: %s", log_dir / "voice_typer.log")
    return logging.getLogger("voicetyper")


def _install_exception_hooks(log_dir: Path) -> None:
    global _fault_file
    logger = logging.getLogger("voicetyper.crash")

    def handle_exception(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: Optional[TracebackType],
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Необработанное исключение", exc_info=(exc_type, exc, tb))

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        name = args.thread.name if args.thread else "?"
        logger.critical(
            "Необработанное исключение в потоке %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception

    try:
        import faulthandler

        _fault_file = open(log_dir / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        logger.debug("faulthandler недоступен", exc_info=True)


def transcript_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:10]


def log_transcript(text: str, *, store_text: bool, engine: str = "") -> None:
    """Log a transcript without leaking its content unless explicitly enabled."""
    logger = logging.getLogger("voicetyper.transcript")
    if store_text:
        logger.info("Распознано (%s, %d симв.): %s", engine, len(text), text)
    else:
        logger.info(
            "Распознано (%s, %d симв., sha=%s)",
            engine,
            len(text),
            transcript_fingerprint(text),
        )
