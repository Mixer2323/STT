from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from voicetyper.downloads import DownloadCancelled, DownloadError, _part_path, download_file


class _Handler(BaseHTTPRequestHandler):
    payload = b"voice-typer" * 1000

    def do_GET(self) -> None:  # noqa: N802
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
