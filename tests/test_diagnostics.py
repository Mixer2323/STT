"""Diagnostics must stay useful without leaking who generated them."""

from __future__ import annotations

from pathlib import Path

from voicetyper import diagnostics
from voicetyper.config import ConfigStore
from voicetyper.history import HistoryStore
from voicetyper.models import ModelManager


def test_redact_hides_the_user_profile(monkeypatch) -> None:
    monkeypatch.setenv("USERPROFILE", r"C:\Users\ivan")
    text = r"Python: C:\Users\ivan\AppData\Local\Programs\Python\python.exe"

    redacted = diagnostics._redact(text)

    assert "ivan" not in redacted
    assert "%USERPROFILE%" in redacted


def test_diagnostics_do_not_expose_the_profile_path(tmp_path, data_dir) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    history = HistoryStore(tmp_path / "history.db")
    models = ModelManager(store)
    try:
        text = diagnostics.collect_diagnostics(store, models, history)
    finally:
        models.shutdown()

    assert "Содержимое распознавания" in text
    assert "Пути к профилю" in text
    assert str(Path.home()).lower() not in text.lower()


def test_diagnostics_are_written_to_the_data_dir(data_dir) -> None:
    destination = diagnostics.write_diagnostics("пример", prefix="test")

    assert destination.exists()
    assert destination.name.startswith("test_")
    assert destination.read_text(encoding="utf-8").strip() == "пример"
