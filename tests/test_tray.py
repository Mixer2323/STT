from __future__ import annotations

import types

import voicetyper.tray as tray_module
from voicetyper.config import ConfigStore


class _FakeIcon:
    def __init__(self, *args):
        self.visible = False
        self.args = args
        self.stopped = False

    def run(self, setup=None):
        if setup is not None:
            setup(self)

    def stop(self):
        self.stopped = True


def test_tray_icon_is_made_visible(tmp_path, monkeypatch) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    controller = tray_module.TrayController(object(), store)
    monkeypatch.setattr(controller, "build_menu", lambda: None)
    monkeypatch.setattr(
        tray_module, "pystray", types.SimpleNamespace(Icon=_FakeIcon)
    )

    controller.run()

    assert controller.icon is not None
    assert controller.icon.visible is True


class _MenuApp:
    def __init__(self, recording: bool):
        self.is_recording = recording

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _default_text(menu) -> str:
    return next(item.text for item in menu.items if item.default)


def test_left_click_starts_settings_and_stops_recording(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tray_module, "describe_engines", lambda: {})
    store = ConfigStore(tmp_path / "config.json")
    store.load()

    idle = tray_module.TrayController(_MenuApp(recording=False), store)
    assert _default_text(idle.build_menu()) == "Настройки…"

    recording = tray_module.TrayController(_MenuApp(recording=True), store)
    assert _default_text(recording.build_menu()) == "Остановить запись"


def _language_items(menu):
    item = next(item for item in menu.items if item.text == "Язык")
    return list(item.submenu.items)


def test_language_menu_is_limited_without_whisper(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tray_module, "describe_engines", lambda: {})
    store = ConfigStore(tmp_path / "config.json")
    store.load()

    for engine in ("auto", "gigaam", "tone"):
        store.update(model_type=engine)
        items = _language_items(tray_module.TrayController(_MenuApp(False), store).build_menu())
        assert len(items) == 1
        assert items[0].text.startswith("Русский")
        assert items[0].enabled is False
        assert items[0].checked is True

    store.update(model_type="faster_whisper")
    items = _language_items(tray_module.TrayController(_MenuApp(False), store).build_menu())
    assert len(items) > 1
    assert all(item.enabled for item in items)
