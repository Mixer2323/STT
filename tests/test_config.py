from __future__ import annotations

import json
from pathlib import Path

from voicetyper.config import Config, ConfigStore, validate_mapping


def test_defaults_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    assert store.load() == []
    assert path.exists()
    assert store.get().hotkey == "ctrl+alt+space"

    store.update(
        model_type="faster_whisper", language="en", beam_size=3, history_limit=50
    )
    reloaded = ConfigStore(path)
    assert reloaded.load() == []
    cfg = reloaded.get()
    assert cfg.language == "en"
    assert cfg.beam_size == 3
    assert cfg.history_limit == 50


def test_invalid_values_are_coerced() -> None:
    cfg, warnings = validate_mapping(
        {
            "language": "klingon",
            "beam_size": 99,
            "sample_rate": 10,
            "hotkey": "",
            "model_type": "unknown",
            "exclude_phrases": ["  one  ", "one", 42, None],
            "history_limit": 0,
        }
    )
    assert cfg.language == "ru"
    assert cfg.beam_size == 10
    assert cfg.sample_rate == 8000
    assert cfg.hotkey == "ctrl+alt+space"
    assert cfg.model_type == "auto"
    assert cfg.exclude_phrases == ["one", "42"]
    assert cfg.history_enabled is False
    assert warnings


def test_legacy_force_cpu_is_migrated() -> None:
    single, warnings = validate_mapping({"force_cpu": True})
    assert single.device == "auto"
    assert warnings

    explicit, _ = validate_mapping({"force_cpu": True, "model_type": "faster_whisper"})
    assert explicit.device == "cpu"


def test_corrupt_config_is_backed_up(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    store = ConfigStore(path)
    warnings = store.load()
    assert warnings
    assert store.get().hotkey == "ctrl+alt+space"
    backups = list(tmp_path.glob("config.corrupt-*.json"))
    assert backups


def test_listener_receives_validated_config(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    seen: list[Config] = []
    store.subscribe(seen.append)
    store.update(model_type="faster_whisper", language="EN", beam_size=0)
    assert len(seen) == 1
    assert seen[0].language == "en"
    assert seen[0].beam_size == 1


def test_language_is_russian_only_without_whisper() -> None:
    for engine in ("auto", "gigaam", "tone"):
        cfg, warnings = validate_mapping({"model_type": engine, "language": "en"})
        assert cfg.language == "ru"
        assert warnings

    whisper, _ = validate_mapping({"model_type": "faster_whisper", "language": "en"})
    assert whisper.language == "en"
    whisper_auto, _ = validate_mapping({"model_type": "faster_whisper", "language": "auto"})
    assert whisper_auto.language == "auto"

    default, warnings = validate_mapping({})
    assert default.language == "ru"
    assert not warnings


def test_persisted_json_has_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.load()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 3
