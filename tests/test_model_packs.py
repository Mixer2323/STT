"""Tests for GitHub model pack metadata and extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from voicetyper import model_packs
from voicetyper.downloads import DownloadError


def test_pack_metadata_is_complete() -> None:
    assert set(model_packs.PACKS) == {"gigastt", "tone", "whisper-large-v3-turbo"}
    assets = set()
    for key, pack in model_packs.PACKS.items():
        assert pack.key == key
        assert pack.asset.endswith(".zip")
        assert pack.asset not in assets
        assets.add(pack.asset)
        assert pack.size > 0
        assert len(pack.sha256) == 64
        int(pack.sha256, 16)
        assert pack.files
        assert pack.url.startswith(model_packs.MODELS_DOWNLOAD_BASE + "/")
        assert key in model_packs.BACKUP_URLS


def test_pack_ready_requires_every_file(tmp_path: Path) -> None:
    pack = model_packs.PACKS["tone"]
    assert not model_packs.pack_ready(pack, tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"x")
    assert model_packs.pack_ready(pack, tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"")
    assert not model_packs.pack_ready(pack, tmp_path)


def test_download_pack_extracts_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = model_packs.PACKS["tone"]
    calls = {}

    def fake_download(url, destination, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("model.onnx", b"model-bytes")
        return Path(destination)

    monkeypatch.setattr(model_packs, "download_file", fake_download)

    model_packs.download_pack("tone", tmp_path)
    assert calls["url"] == pack.url
    assert calls["expected_size"] == pack.size
    assert calls["expected_sha256"] == pack.sha256
    assert (tmp_path / "model.onnx").read_bytes() == b"model-bytes"
    assert not (tmp_path / f".{pack.asset}.download").exists()


def test_extract_rejects_paths_outside_target(tmp_path: Path) -> None:
    pack = model_packs.PACKS["tone"]
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.onnx", b"x")
    with pytest.raises(DownloadError):
        model_packs._extract(archive_path, tmp_path / "target", pack, None)


def test_extract_reports_missing_files(tmp_path: Path) -> None:
    pack = model_packs.PACKS["tone"]
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("other.bin", b"x")
    with pytest.raises(DownloadError):
        model_packs._extract(archive_path, tmp_path / "target", pack, None)
