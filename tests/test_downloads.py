from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from voicetyper.downloads import DownloadCancelled, DownloadError, _part_path, download_file


class _Handler(BaseHTTPRequestHandler):
    payload = b"voice-typer" * 1000
    requests: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests.append(self.path)
        if self.path != "/data.bin":
            self.send_error(404)
            return
        data = self.payload
        start = 0
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            try:
                start = int(range_header[len("bytes="):].split("-", 1)[0])
            except ValueError:
                start = 0
        if start >= len(data):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(data)}")
            self.end_headers()
            return
        body = data[start:]
        if start:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture()
def server():
    _Handler.requests.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/data.bin"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_download_file(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    download_file(server, destination, retries=1)
    assert destination.read_bytes() == _Handler.payload


def test_checksum_mismatch_removes_partial_file(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    with pytest.raises(DownloadError):
        download_file(
            server,
            destination,
            expected_sha256="0" * 64,
            retries=1,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_resume_keeps_existing_bytes(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    part = _part_path(destination, server)
    part.write_bytes(_Handler.payload[:1000])
    download_file(
        server,
        destination,
        expected_size=len(_Handler.payload),
        retries=1,
    )
    assert destination.read_bytes() == _Handler.payload


def test_complete_partial_file_is_promoted(server: str, tmp_path: Path) -> None:
    # A crash after the last byte leaves a complete .part file; the server then
    # answers the resume request with 416 and the file must still be finished.
    destination = tmp_path / "data.bin"
    part = _part_path(destination, server)
    part.write_bytes(_Handler.payload)
    download_file(
        server,
        destination,
        expected_size=len(_Handler.payload),
        retries=1,
    )
    assert destination.read_bytes() == _Handler.payload


def test_partial_file_from_another_source_is_ignored(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    foreign = _part_path(destination, "https://mirror.invalid/data.bin")
    foreign.write_bytes(b"\0" * 1000)
    download_file(server, destination, retries=1)
    assert destination.read_bytes() == _Handler.payload


def test_cancellation(server: str, tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadCancelled):
        download_file(server, tmp_path / "data.bin", cancel_event=cancel)


def test_existing_file_is_reused(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    destination.write_bytes(_Handler.payload)
    result = download_file(
        server,
        destination,
        expected_sha256=hashlib.sha256(_Handler.payload).hexdigest(),
        retries=1,
    )
    assert result == destination


def test_corrupted_existing_file_is_downloaded_again(server: str, tmp_path: Path) -> None:
    destination = tmp_path / "data.bin"
    # Same size, wrong bytes: the old code trusted the size and kept a
    # corrupted model file.
    destination.write_bytes(b"x" * len(_Handler.payload))
    download_file(
        server,
        destination,
        expected_size=len(_Handler.payload),
        expected_sha256=hashlib.sha256(_Handler.payload).hexdigest(),
        retries=1,
    )
    assert destination.read_bytes() == _Handler.payload


def test_missing_url_is_not_retried(server: str, tmp_path: Path) -> None:
    missing = server.rsplit("/", 1)[0] + "/missing.bin"
    with pytest.raises(DownloadError):
        download_file(missing, tmp_path / "x.bin", retries=3)
    # A 404 cannot succeed on retry; retrying only slows down failover.
    assert len(_Handler.requests) == 1


def test_hf_offline_uses_the_local_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voicetyper.downloads as downloads

    (tmp_path / "model.bin").write_bytes(b"model")
    monkeypatch.setattr(
        downloads,
        "_hf_remote_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    result = downloads.download_hf_files(
        "owner/repo",
        tmp_path,
        ["model.bin", "config.json"],
        required="model.bin",
    )

    assert result == tmp_path


def test_hf_lfs_checksum_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    import voicetyper.downloads as downloads

    monkeypatch.setattr(
        downloads,
        "_hf_remote_metadata",
        lambda *_args, **_kwargs: {"model.bin": (5, "0" * 64)},
    )

    def fake_download(repo_id, filename, local_dir, revision=None):
        target = Path(local_dir) / filename
        target.write_bytes(b"hello")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    with pytest.raises(DownloadError, match="Контрольная сумма"):
        downloads.download_hf_files(
            "owner/repo", tmp_path, ["model.bin"], required="model.bin"
        )
    assert not (tmp_path / "model.bin").exists()


def _payload() -> bytes:
    return b"0123456789"


def test_hf_truncated_file_is_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-empty but short model.bin used to be accepted forever: the engine
    # then failed on every start with a native "invalid model" error.
    import huggingface_hub

    import voicetyper.downloads as downloads

    data = _payload()
    (tmp_path / "model.bin").write_bytes(data[:3])
    monkeypatch.setattr(
        downloads,
        "_hf_remote_metadata",
        lambda *_args, **_kwargs: {
            "model.bin": (len(data), hashlib.sha256(data).hexdigest())
        },
    )
    downloaded: list[str] = []

    def fake_download(repo_id, filename, local_dir, revision=None):
        downloaded.append(filename)
        target = Path(local_dir) / filename
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    downloads.download_hf_files(
        "owner/repo", tmp_path, ["model.bin"], required="model.bin"
    )

    assert downloaded == ["model.bin"]
    assert (tmp_path / "model.bin").read_bytes() == data
    assert downloads.manifest_path(tmp_path).exists()


def test_hf_verified_files_do_not_need_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voicetyper.downloads as downloads

    data = _payload()
    (tmp_path / "model.bin").write_bytes(data)
    monkeypatch.setattr(
        downloads,
        "_hf_remote_metadata",
        lambda *_args, **_kwargs: {
            "model.bin": (len(data), hashlib.sha256(data).hexdigest())
        },
    )
    downloads.download_hf_files(
        "owner/repo", tmp_path, ["model.bin"], required="model.bin"
    )
    assert downloads.manifest_path(tmp_path).exists()

    def offline(*_args, **_kwargs):
        raise AssertionError("сеть не должна использоваться повторно")

    monkeypatch.setattr(downloads, "_hf_remote_metadata", offline)
    downloads.download_hf_files(
        "owner/repo", tmp_path, ["model.bin"], required="model.bin"
    )


def test_hf_manifest_detects_a_file_truncated_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    import voicetyper.downloads as downloads

    data = _payload()
    path = tmp_path / "model.bin"
    path.write_bytes(data)
    downloads._write_manifest(
        tmp_path, {"model.bin": {"size": len(data), "sha256": None}}
    )
    path.write_bytes(data[:3])
    monkeypatch.setattr(
        downloads,
        "_hf_remote_metadata",
        lambda *_args, **_kwargs: {"model.bin": (len(data), None)},
    )
    downloaded: list[str] = []

    def fake_download(repo_id, filename, local_dir, revision=None):
        downloaded.append(filename)
        target = Path(local_dir) / filename
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    downloads.download_hf_files(
        "owner/repo", tmp_path, ["model.bin"], required="model.bin"
    )

    assert downloaded == ["model.bin"]
    assert path.read_bytes() == data
