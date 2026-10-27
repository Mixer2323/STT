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
    monkeypatch.setattr(models_module, "detect_hardware", lambda *_a, **_k: HardwareInfo(cpu_cores=4, ram_gb=16))
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


def test_gigastt_sources_cover_every_asset() -> None:
    for label, base_url, assets in GIGASTT_SOURCES:
        assert base_url.startswith("https://"), label
        for name in GIGASTT_ASSETS:
            remote_name, size, digest = assets[name]
            assert remote_name, (label, name)
            assert size and size > 0, (label, name)
            assert len(digest) == 64, (label, name)
