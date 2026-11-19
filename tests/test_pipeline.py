from __future__ import annotations

import threading
from pathlib import Path

import pytest

import voicetyper.pipeline as pipeline_module
from voicetyper.engines.base import TranscriptionResult
from voicetyper.history import HistoryStore
from voicetyper.insertion import InsertionResult
from voicetyper.models import EngineNotReady
from voicetyper.pipeline import TranscriptionPipeline


class FakeLease:
    def __init__(self, engine):
        self.engine = engine
        self.name = "faster_whisper"
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeEngine:
    def transcribe(self, path: Path, language=None) -> TranscriptionResult:
        return TranscriptionResult(text="привет   мир", language="ru", engine="faster_whisper")


class FakeModels:
    def acquire(self, timeout=0, preferred_name=None) -> FakeLease:
        return FakeLease(FakeEngine())


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from voicetyper.config import ConfigStore

        self.store = ConfigStore(tmp_path / "config.json")
        self.store.load()
        self.store.update(model_type="faster_whisper")
        self.history = HistoryStore(tmp_path / "history.db")
        self.results = []
        self.states = []
        self.done = threading.Event()
        self.pipeline = TranscriptionPipeline(
            FakeModels(),
            self.store,
            self.history,
            on_state=lambda state, detail: self.states.append((state, detail)),
            on_result=self._on_result,
        )

    def _on_result(self, result) -> None:
        self.results.append(result)
        self.done.set()

    def submit(self, tmp_path: Path) -> Path:
        path = tmp_path / "audio.wav"
        path.write_bytes(b"fake")
        self.done.clear()
        assert self.pipeline.submit(path, 1.0) is not None
        assert self.done.wait(5)
        return path


def test_pipeline_transcribes_inserts_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "insert_text",
        lambda text, cfg, target=None: InsertionResult(method="paste", inserted=True),
    )
    path = harness.submit(tmp_path)
    result = harness.results[-1]
    assert result.text == "Привет мир"
    assert result.insertion is not None and result.insertion.inserted
    assert result.history_id is not None
    assert harness.history.count() == 1
    assert not path.exists()
    harness.pipeline.stop()


def test_history_is_kept_when_insertion_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path, monkeypatch)

    def boom(text, cfg, target=None):
        raise RuntimeError("clipboard unavailable")

    monkeypatch.setattr(pipeline_module, "insert_text", boom)
    harness.submit(tmp_path)
    result = harness.results[-1]
    assert "clipboard unavailable" in (result.insertion.error or "")
    assert harness.history.count() == 1
    harness.pipeline.stop()


def test_excluded_phrase_is_not_inserted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.store.update(exclude_phrases=["Привет мир"])
    called = []
    monkeypatch.setattr(
        pipeline_module,
        "insert_text",
        lambda *a, **k: called.append(True) or InsertionResult(method="paste", inserted=True),
    )
    harness.submit(tmp_path)
    result = harness.results[-1]
    assert result.excluded_by == "Привет мир"
    assert not called
    assert harness.history.count() == 0
    harness.pipeline.stop()


def test_job_reloads_a_model_unloaded_by_the_idle_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from voicetyper.config import ConfigStore

    class ReloadingModels:
        def __init__(self) -> None:
            self.calls = 0
            self.requested = 0
            self.loading = False
            self.status = SimpleNamespace(error=None)

        def request(self, *_args, **_kwargs) -> None:
            self.requested += 1

        def acquire(self, timeout=0, preferred_name=None):
            self.calls += 1
            if self.calls == 1:
                raise EngineNotReady("Модель ещё не загружена")
            return FakeLease(FakeEngine())

    models = ReloadingModels()
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    history = HistoryStore(tmp_path / "history.db")
    results = []
    done = threading.Event()
    pipeline = TranscriptionPipeline(
        models,
        store,
        history,
        on_result=lambda result: (results.append(result), done.set()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "insert_text",
        lambda text, cfg, target=None: InsertionResult(method="paste", inserted=True),
    )
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake")

    try:
        assert pipeline.submit(path, 1.0) is not None
        assert done.wait(5)
        assert results[-1].error is None
        assert models.requested == 1
        assert models.calls == 2
    finally:
        pipeline.stop()
