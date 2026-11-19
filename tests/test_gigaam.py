"""GigaAM model readiness: a half-downloaded set must not look complete."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from voicetyper.constants import GIGASTT_ASSETS
from voicetyper.engines import gigaam
from voicetyper.engines.gigaam import _models_ready


def _write_all(directory: Path, *, sizes: dict[str, int] | None = None) -> None:
    sizes = sizes or {}
    directory.mkdir(parents=True, exist_ok=True)
    for name in GIGASTT_ASSETS:
        (directory / name).write_bytes(b"x" * sizes.get(name, 1))


def test_all_assets_are_required(tmp_path: Path) -> None:
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir)
    assert _models_ready(model_dir)

    for name in GIGASTT_ASSETS:
        name_path = model_dir / name
        original = name_path.read_bytes()
        name_path.unlink()
        assert not _models_ready(model_dir), name
        name_path.write_bytes(original)


def test_empty_asset_is_not_ready(tmp_path: Path) -> None:
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir, sizes={GIGASTT_ASSETS[0]: 0})
    assert not _models_ready(model_dir)


def test_missing_directory_is_not_ready(tmp_path: Path) -> None:
    assert not _models_ready(tmp_path / "absent")


def test_unrelated_files_do_not_mark_the_model_ready(tmp_path: Path) -> None:
    model_dir = tmp_path / "gigastt"
    model_dir.mkdir()
    (model_dir / "some_encoder.onnx").write_bytes(b"x")
    (model_dir / "some_vocab.txt").write_bytes(b"x")
    assert not _models_ready(model_dir)


def test_custom_mirror_is_verified_like_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOICETYPER_GIGASTT_MIRROR", "https://mirror.example/models")

    sources = gigaam._sources()

    label, base, assets = sources[0]
    assert label == "зеркало"
    assert base == "https://mirror.example/models"
    for name in GIGASTT_ASSETS:
        remote_name, size, digest = assets[name]
        assert remote_name == name
        assert size and size > 0
        assert digest and len(digest) == 64


def test_truncated_asset_is_not_considered_complete(tmp_path: Path) -> None:
    # A non-empty file is not enough: an interrupted download leaves a short
    # ONNX file that the native engine cannot load.
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir)

    assert _models_ready(model_dir)
    assert not gigaam._sizes_ok(model_dir)


def test_sizes_ok_accepts_any_published_size(tmp_path: Path) -> None:
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir)
    published = {name: {1} for name in GIGASTT_ASSETS}

    assert gigaam._sizes_ok(model_dir, published)


def test_truncated_asset_is_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir)
    monkeypatch.setattr(
        gigaam,
        "download_pack",
        lambda *_a, **_k: (_ for _ in ()).throw(gigaam.DownloadError("offline")),
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        gigaam,
        "download_file",
        lambda url, destination, **kwargs: (
            downloaded.append(Path(destination).name) or Path(destination)
        ),
    )
    engine = object.__new__(gigaam.GigaAMEngine)
    engine._model_dir = model_dir
    engine._cancel = threading.Event()
    engine._progress = None

    engine._download()

    assert set(downloaded) == set(GIGASTT_ASSETS)


def test_complete_asset_set_is_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "gigastt"
    _write_all(model_dir)
    monkeypatch.setattr(
        gigaam, "_published_sizes", lambda: {name: {1} for name in GIGASTT_ASSETS}
    )
    monkeypatch.setattr(
        gigaam,
        "download_pack",
        lambda *_a, **_k: (_ for _ in ()).throw(gigaam.DownloadError("offline")),
    )
    calls: list[object] = []
    monkeypatch.setattr(gigaam, "download_file", lambda *a, **k: calls.append(a))
    engine = object.__new__(gigaam.GigaAMEngine)
    engine._model_dir = model_dir
    engine._cancel = threading.Event()
    engine._progress = None

    engine._download()

    assert calls == []
