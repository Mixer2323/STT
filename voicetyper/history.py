"""Local transcription history backed by SQLite."""

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryEntry:
    id: int
    created_at: str
    text: str
    engine: str = ""
    language: str = ""
    duration: float = 0.0
    app: str = ""
    status: str = "inserted"

    @property
    def created_display(self) -> str:
        try:
            return datetime.fromisoformat(self.created_at).strftime("%d.%m.%Y %H:%M")
        except ValueError:
            return self.created_at


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    app TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'inserted'
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at DESC);
"""


class HistoryStore:
    def __init__(self, path: Path | str | None = None, limit: int = 200, enabled: bool = True):
        if path is None:
            from . import paths

            path = paths.history_path()
        self.path = Path(path)
        self.limit = max(0, int(limit))
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except Exception:
            logger.exception("Не удалось инициализировать историю %s", self.path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version=1")

    def add(
        self,
        text: str,
        *,
        engine: str = "",
        language: str = "",
        duration: float = 0.0,
        app: str = "",
        status: str = "inserted",
    ) -> Optional[int]:
        if not self.enabled or not text.strip():
            return None
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    "INSERT INTO entries (created_at, text, engine, language, duration, app, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        created_at,
                        text,
                        engine[:80],
                        language[:16],
                        float(duration),
                        app[:200],
                        status[:32],
                    ),
                )
                entry_id = int(cursor.lastrowid)
                if self.limit > 0:
                    connection.execute(
                        "DELETE FROM entries WHERE id NOT IN "
                        "(SELECT id FROM entries ORDER BY id DESC LIMIT ?)",
                        (self.limit,),
                    )
                return entry_id
        except Exception:
            logger.exception("Не удалось добавить запись в историю")
            return None

    def list(self, limit: int = 50, offset: int = 0, query: str = "") -> list[HistoryEntry]:
        limit = max(1, min(1_000_000, int(limit)))
        offset = max(0, int(offset))
        try:
            with self._lock, closing(self._connect()) as connection:
                if query.strip():
                    pattern = f"%{query.strip()}%"
                    rows = connection.execute(
                        "SELECT * FROM entries WHERE text LIKE ? "
                        "ORDER BY id DESC LIMIT ? OFFSET ?",
                        (pattern, limit, offset),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT * FROM entries ORDER BY id DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    ).fetchall()
                return [self._row(row) for row in rows]
        except Exception:
            logger.exception("Не удалось прочитать историю")
            return []

    def count(self) -> int:
        try:
            with self._lock, closing(self._connect()) as connection:
                row = connection.execute("SELECT COUNT(*) AS total FROM entries").fetchone()
                return int(row["total"]) if row else 0
        except Exception:
            logger.exception("Не удалось посчитать историю")
            return 0

    @staticmethod
    def _row(row: sqlite3.Row) -> HistoryEntry:
        return HistoryEntry(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            text=str(row["text"]),
            engine=str(row["engine"] or ""),
            language=str(row["language"] or ""),
            duration=float(row["duration"] or 0.0),
            app=str(row["app"] or ""),
            status=str(row["status"] or ""),
        )

    def delete(self, entry_id: int) -> bool:
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                cursor = connection.execute("DELETE FROM entries WHERE id = ?", (int(entry_id),))
                return cursor.rowcount > 0
        except Exception:
            logger.exception("Не удалось удалить запись истории")
            return False

    def clear(self) -> bool:
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("DELETE FROM entries")
                connection.commit()
                connection.execute("VACUUM")
            return True
        except Exception:
            logger.exception("Не удалось очистить историю")
            return False

    def export_csv(self, destination: Path | str) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = self.list(limit=1000000)
        with open(destination, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                ["id", "created_at", "engine", "language", "duration", "app", "status", "text"]
            )
            for entry in entries:
                writer.writerow(
                    [
                        entry.id,
                        entry.created_at,
                        entry.engine,
                        entry.language,
                        f"{entry.duration:.1f}",
                        entry.app,
                        entry.status,
                        entry.text,
                    ]
                )
        return destination

    def update_settings(self, *, enabled: bool, limit: int) -> None:
        self.enabled = bool(enabled)
        self.limit = max(0, int(limit))
        if self.limit > 0:
            try:
                with self._lock, closing(self._connect()) as connection, connection:
                    connection.execute(
                        "DELETE FROM entries WHERE id NOT IN "
                        "(SELECT id FROM entries ORDER BY id DESC LIMIT ?)",
                        (self.limit,),
                    )
            except Exception:
                logger.exception("Не удалось применить лимит истории")


def delete_entries(store: HistoryStore, ids: Iterable[int]) -> int:
    removed = 0
    for entry_id in ids:
        if store.delete(entry_id):
            removed += 1
    return removed
