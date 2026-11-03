from __future__ import annotations

from types import SimpleNamespace

import pytest

from voicetyper import audio
from voicetyper.config import Config


class FakeStream:
    def __init__(self, device, samplerate, channels):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    """Minimal sounddevice stand-in for the device/open retry logic."""

    def __init__(self, devices, default_index, *, failing=None, fail_until_refresh=None):
        self._devices = devices
        self.default = SimpleNamespace(device=[default_index, -1])
        self.failing = failing or set()
        self.fail_until_refresh = fail_until_refresh
        self.opened: list[tuple[object, int, int]] = []
        self.refreshed = 0

    def query_hostapis(self):
        return [{"name": "MME"}]

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return self._devices
        index = self.default.device[0] if device is None else int(device)
        for item in self._devices:
            if item["index"] == index:
                return item
        raise ValueError(f"Invalid device: {device}")

    def InputStream(self, *, device, samplerate, channels, dtype, blocksize, callback):
        self.opened.append((device, samplerate, channels))
        if not self._devices:
            raise RuntimeError("Error opening InputStream: no input device")
        if samplerate in self.failing:
            raise RuntimeError("Error opening InputStream: MME error 11")
        if self.fail_until_refresh is not None and self.refreshed == 0:
            raise RuntimeError("Error opening InputStream: MME error 11")
        return FakeStream(device, samplerate, channels)

    def _terminate(self):
        self.refreshed += 1

    def _initialize(self):
        pass


def _device(index: int, name: str, *, channels: int = 1, rate: float = 44100.0) -> dict:
    return {
        "index": index,
        "name": name,
        "max_input_channels": channels,
        "default_samplerate": rate,
        "hostapi": 0,
    }


@pytest.fixture()
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSoundDevice:
    fake = FakeSoundDevice([_device(1, "Микрофон (2- USB PnP Audio Device)")], 1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    return fake


def test_candidate_formats_prefer_native_rate(fake_sd: FakeSoundDevice) -> None:
    fake_sd._devices = [_device(1, "Микрофон", channels=2, rate=48000.0)]

    formats = audio.RecordingSession(Config())._candidate_formats(1)

    assert formats[0] == (48000, 1)
    assert formats[1] == (48000, 2)
    assert (16000, 1) in formats


def test_start_falls_back_when_mme_rejects_rate(fake_sd: FakeSoundDevice) -> None:
    fake_sd.failing = {16000, 48000, 44100}
    session = audio.RecordingSession(Config())

    session.start()

    assert session.capture_rate == 8000
    assert fake_sd.opened[-1][1] == 8000
    session.stop()


def test_start_refreshes_devices_before_giving_up(fake_sd: FakeSoundDevice) -> None:
    fake_sd.fail_until_refresh = True
    session = audio.RecordingSession(Config())

    session.start()

    assert fake_sd.refreshed == 1
    assert session.is_recording
    session.stop()


def test_missing_index_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice([_device(2, "Перенаправление звука в RDP")], 2)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    assert audio.resolve_input_device(99) == 2


def test_no_devices_reports_rdp_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice([], -1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    with pytest.raises(audio.AudioError) as excinfo:
        audio.RecordingSession(Config()).start()

    message = str(excinfo.value)
    assert "Микрофоны не найдены" in message
    assert "RDP" in message
