from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import voicetyper.models as models_module
from voicetyper.config import ConfigStore
from voicetyper.constants import GIGASTT_ASSETS, GIGASTT_SOURCES
from voicetyper.engines.base import Engine, TranscriptionResult
from voicetyper.hardware import HardwareInfo
from voicetyper.models import EngineNotReady, ModelManager


class FakeEngine(Engine):
    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.released = threading.Event()

    def transcribe(self, path: Path, language=None) -> TranscriptionResult:
        return TranscriptionResult(text="ok", language=language or "ru", engine="faster_whisper")

    def release(self) -> None:
        self.released.set()

    def runtime_description(self) -> str:
        return f"fake-{self.index}"


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models_module, "cached_hardware", lambda *_a, **_k: HardwareInfo(cpu_cores=4, ram_gb=16))
    created: list[FakeEngine] = []

    def fake_create(name, cfg, hw, *, progress=None, cancel_event=None):
        engine = FakeEngine(len(created))
        created.append(engine)
        return engine

    monkeypatch.setattr(models_module, "create_engine", fake_create)
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    store.update(model_type="faster_whisper")
    instance = ModelManager(store)
    yield instance, created, store
    instance.shutdown()


def test_acquire_before_load_fails(manager) -> None:
    manager_instance, _created, _store = manager
    with pytest.raises(EngineNotReady):
        manager_instance.acquire(timeout=0.1)


def test_background_load_and_runtime(manager) -> None:
    manager_instance, created, _store = manager
    manager_instance.request()
    assert manager_instance.wait_ready(5)
    assert manager_instance.current_name == "faster_whisper"
    assert manager_instance.current_runtime() == "fake-0"
    assert len(created) == 1


def test_old_engine_survives_until_lease_release(manager) -> None:
    manager_instance, created, _store = manager
    manager_instance.request()
    assert manager_instance.wait_ready(5)

    lease = manager_instance.acquire(timeout=2)
    first_engine = lease.engine

    manager_instance.request(force=True)
    deadline = time.time() + 5
    while time.time() < deadline and manager_instance.current_runtime() != "fake-1":
        time.sleep(0.02)
    assert len(created) == 2
    assert not first_engine.released.is_set()

    second = manager_instance.acquire(timeout=2)
    assert second.engine is created[1]
    second.release()

    assert not first_engine.released.is_set()
    lease.release()
    assert first_engine.released.wait(2)


def test_shutdown_releases_current(manager) -> None:
    manager_instance, created, _store = manager
    manager_instance.request()
    assert manager_instance.wait_ready(5)
    manager_instance.shutdown()
    assert created[0].released.is_set()


def test_unload_releases_an_idle_engine(manager) -> None:
    manager_instance, created, _store = manager
    manager_instance.request()
    assert manager_instance.wait_ready(5)

    assert manager_instance.unload() is True

    assert created[0].released.is_set()
    assert manager_instance.ready is False
    assert manager_instance.status.state == "idle"


def test_unload_refuses_while_a_lease_is_held(manager) -> None:
    manager_instance, created, _store = manager
    manager_instance.request()
    assert manager_instance.wait_ready(5)

    lease = manager_instance.acquire(timeout=2)
    assert manager_instance.unload() is False
    assert not created[0].released.is_set()
    lease.release()

    assert manager_instance.unload() is True
    assert created[0].released.is_set()


def test_pinned_revisions_cover_every_whisper_model() -> None:
    from voicetyper.constants import TONE_REVISION, WHISPER_MODELS, WHISPER_REVISIONS

    assert set(WHISPER_REVISIONS) == set(WHISPER_MODELS)
    assert all(len(revision) == 40 for revision in WHISPER_REVISIONS.values())
    assert len(TONE_REVISION) == 40


def test_gigastt_sources_cover_every_asset() -> None:
    for label, base_url, assets in GIGASTT_SOURCES:
        assert base_url.startswith("https://"), label
        for name in GIGASTT_ASSETS:
            remote_name, size, digest = assets[name]
            assert remote_name, (label, name)
            assert size and size > 0, (label, name)
            assert len(digest) == 64, (label, name)


def test_loader_does_not_get_stuck_when_loading_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unexpected failure outside _load_one's own error handling used to kill
    # the loader thread with _loading stuck True, so every acquire() waited for
    # its whole timeout. The worker must publish the failure and stay usable.
    monkeypatch.setattr(
        models_module, "cached_hardware",
        lambda *_a, **_k: HardwareInfo(cpu_cores=2, ram_gb=8),
    )
    monkeypatch.setattr(models_module, "resolve_engine_name", lambda *_a: "fake")
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    manager = ModelManager(store)

    def exploding_load(*_args, **_kwargs):
        raise RuntimeError("unexpected loader failure")

    monkeypatch.setattr(manager, "_load_one", exploding_load)
    try:
        manager.request()
        deadline = time.time() + 5
        while time.time() < deadline and manager.loading:
            time.sleep(0.02)
        assert manager.loading is False
        assert manager.status.state == "error"
        assert "unexpected loader failure" in (manager.status.error or "")
        with pytest.raises(EngineNotReady):
            manager.acquire(timeout=0.1)
    finally:
        manager.shutdown()


def test_forced_reload_during_a_load_is_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "Перезагрузить модель" while that engine is already loading used to be
    # silently ignored: the user saw a stuck download and no way to retry.
    monkeypatch.setattr(
        models_module,
        "cached_hardware",
        lambda *_a, **_k: HardwareInfo(cpu_cores=4, ram_gb=16),
    )
    monkeypatch.setattr(models_module, "resolve_engine_name", lambda *_a: "faster_whisper")
    created: list[FakeEngine] = []
    started = threading.Event()
    release = threading.Event()

    def slow_create(name, cfg, hw, *, progress=None, cancel_event=None):
        started.set()
        assert release.wait(5), "загрузка не была отпущена"
        engine = FakeEngine(len(created))
        created.append(engine)
        return engine

    monkeypatch.setattr(models_module, "create_engine", slow_create)
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    manager = ModelManager(store)
    try:
        manager.request()
        assert started.wait(5)
        manager.request(force=True)
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and len(created) < 2:
            time.sleep(0.02)
        assert len(created) == 2
        assert manager.current_name == "faster_whisper"
    finally:
        manager.shutdown()
