"""Resumable downloads with progress reporting and cancellation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_CHUNK = 256 * 1024
_USER_AGENT = "VoiceTyper/1.0 (+https://github.com/Mixer2323/STT)"


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


@dataclass
class DownloadProgress:
    filename: str
    downloaded: int = 0
    total: Optional[int] = None
    file_index: int = 0
    file_count: int = 1
    message: str = ""
    speed_bps: float = 0.0

    @property
    def percent(self) -> Optional[float]:
        if not self.total or self.total <= 0:
            return None
        return min(100.0, self.downloaded * 100.0 / self.total)

    @property
    def overall_percent(self) -> Optional[float]:
        file_part = self.percent
        if file_part is None:
            return None
        return ((self.file_index + file_part / 100.0) / max(1, self.file_count)) * 100.0


ProgressCallback = Callable[[DownloadProgress], None]


def _report(
    callback: Optional[ProgressCallback],
    progress: DownloadProgress,
) -> None:
    if callback is None:
        return
    try:
        callback(progress)
    except Exception:
        logger.debug("Ошибка колбэка прогресса", exc_info=True)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("Загрузка отменена")


def human_size(size: Optional[float]) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(value) < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024
    return f"{value:.1f} ТБ"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _part_path(destination: Path, url: str) -> Path:
    """Per-source partial file so a mirror switch never resumes foreign bytes."""
    tag = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return destination.with_name(f"{destination.name}.{tag}.part")


def download_file(
    url: str,
    destination: Path | str,
    *,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> Path:
    """Download ``url`` to ``destination`` atomically and resumably."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(destination, url)

    if destination.exists() and destination.stat().st_size > 0:
        size_ok = expected_size is None or destination.stat().st_size == expected_size
        digest_ok = True
        if size_ok and expected_sha256:
            # A same-size but corrupted file must not be accepted silently.
            digest_ok = sha256_file(destination).lower() == expected_sha256.lower()
            if not digest_ok:
                logger.warning(
                    "Контрольная сумма %s не совпала — файл будет скачан заново",
                    destination.name,
                )
                try:
                    destination.unlink()
                except OSError:
                    logger.debug("Не удалось удалить повреждённый файл", exc_info=True)
        if size_ok and digest_ok:
            _report(progress, DownloadProgress(destination.name, 1, 1, message="уже загружено"))
            return destination

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        _check_cancel(cancel_event)
        existing = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": _USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status == 200 and existing:
                    # Server ignored Range: restart from scratch.
                    existing = 0
                    part.unlink(missing_ok=True)
                content_length = response.headers.get("Content-Length")
                total = existing + int(content_length) if content_length else None
                if expected_size:
                    total = expected_size

                _report(
                    progress,
                    DownloadProgress(
                        destination.name,
                        existing,
                        total,
                        message=f"попытка {attempt}" if attempt > 1 else "загрузка",
                    ),
                )
                mode = "ab" if existing else "wb"
                started = time.monotonic()
                downloaded = existing
                last_report = 0.0
                with open(part, mode) as handle:
                    while True:
                        _check_cancel(cancel_event)
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 0.2:
                            elapsed = max(0.001, now - started)
                            _report(
                                progress,
                                DownloadProgress(
                                    destination.name,
                                    downloaded,
                                    total,
                                    speed_bps=(downloaded - existing) / elapsed,
                                    message="загрузка",
                                ),
                            )
                            last_report = now
                    handle.flush()
                    os.fsync(handle.fileno())

                actual_size = part.stat().st_size
                if total is not None and actual_size != total:
                    raise DownloadError(
                        f"Размер {destination.name} не совпал: {actual_size} из {total} байт"
                    )
                if expected_size is not None and actual_size != expected_size:
                    raise DownloadError(
                        f"Размер {destination.name} не совпал: {actual_size} из {expected_size} байт"
                    )
                if expected_sha256:
                    digest = sha256_file(part)
                    if digest.lower() != expected_sha256.lower():
                        part.unlink(missing_ok=True)
                        raise DownloadError(f"Контрольная сумма {destination.name} не совпала")
                os.replace(part, destination)
                _report(
                    progress,
                    DownloadProgress(
                        destination.name,
                        actual_size,
                        actual_size,
                        message="готово",
                    ),
                )
                return destination
        except DownloadCancelled:
            raise
        except urllib.error.HTTPError as exc:
            # urllib raises HTTPError for 416, so the status check above never
            # sees it: a complete .part file must be promoted, not retried.
            if (
                exc.code == 416
                and existing
                and (expected_size is None or existing == expected_size)
            ):
                os.replace(part, destination)
                _report(
                    progress,
                    DownloadProgress(
                        destination.name, existing, existing, message="готово"
                    ),
                )
                return destination
            last_error = exc
            logger.warning(
                "Загрузка %s (попытка %d/%d) не удалась: %s",
                destination.name,
                attempt,
                retries,
                exc,
            )
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                # A missing file or a forbidden URL will not appear on retry.
                raise DownloadError(f"Не удалось скачать {destination.name}: {exc}") from exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 6.0))
        except (urllib.error.URLError, TimeoutError, OSError, DownloadError) as exc:
            last_error = exc
            logger.warning(
                "Загрузка %s (попытка %d/%d) не удалась: %s",
                destination.name,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 6.0))
    raise DownloadError(
        f"Не удалось скачать {destination.name}: {last_error}"
    ) from last_error


def _hf_remote_metadata(
    repo_id: str, revision: Optional[str] = None, timeout: float = 10.0
) -> dict[str, tuple[int, Optional[str]]]:
    """File sizes plus LFS SHA-256 fingerprints for one HF repository.

    ``timeout`` keeps an offline machine from stalling the model load: the Hub
    metadata endpoint is only needed to confirm files that are already on disk.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.model_info(
            repo_id, files_metadata=True, revision=revision, timeout=timeout
        )
    except TypeError:
        # Older huggingface_hub releases do not accept a timeout here.
        info = api.model_info(repo_id, files_metadata=True, revision=revision)
    result: dict[str, tuple[int, Optional[str]]] = {}
    for sibling in info.siblings or []:
        name = getattr(sibling, "rfilename", None)
        if not name:
            continue
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None else None
        result[name] = (
            int(size) if size is not None else -1,
            str(digest).lower() if digest else None,
        )
    return result


_MANIFEST_NAME = ".voicetyper-manifest.json"


def manifest_path(local_dir: Path | str) -> Path:
    """Where the verified-file manifest of a model folder lives."""
    return Path(local_dir) / _MANIFEST_NAME


def _read_manifest(local_dir: Path) -> dict[str, dict]:
    try:
        with open(manifest_path(local_dir), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return {}
    entries: dict[str, dict] = {}
    for name, item in files.items():
        if not isinstance(item, dict):
            continue
        size = item.get("size")
        if not isinstance(size, int) or size <= 0:
            continue
        digest = item.get("sha256")
        entries[str(name)] = {
            "size": size,
            "sha256": str(digest).lower() if isinstance(digest, str) and digest else None,
        }
    return entries


def _write_manifest(local_dir: Path, entries: dict[str, dict]) -> None:
    """Record sizes (and known hashes) of verified files, atomically."""
    payload = {
        "version": 1,
        "files": {
            name: {"size": entry["size"], "sha256": entry.get("sha256")}
            for name, entry in sorted(entries.items())
        },
    }
    destination = manifest_path(local_dir)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    except OSError:
        logger.debug("Не удалось записать манифест моделей в %s", local_dir, exc_info=True)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _manifest_covers(manifest: dict[str, dict], local_dir: Path, names: list[str]) -> bool:
    """True when every file is present with exactly the size that was verified."""
    for name in names:
        entry = manifest.get(name)
        if not entry:
            return False
        try:
            size = (local_dir / name).stat().st_size
        except OSError:
            return False
        if size <= 0 or size != entry["size"]:
            return False
    return True


def download_hf_files(
    repo_id: str,
    local_dir: Path | str,
    filenames: Iterable[str],
    *,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    allow_offline: bool = True,
    required: Optional[str] = None,
    ignore_missing: bool = False,
    revision: Optional[str] = None,
    verify_hashes: bool = True,
) -> Path:
    """Download selected files from a Hugging Face repository into ``local_dir``.

    Files are stored as regular files (no cache symlinks), which is required for
    packaging and avoids ``WinError 1314`` on non-elevated Windows accounts.
    ``revision`` pins the exact repository commit so a later model update cannot
    silently change what the application loads; LFS files are additionally
    verified by their SHA-256.

    Sizes of files that are already on disk are checked against the repository
    instead of trusting any non-empty file: a truncated ``model.bin`` used to be
    accepted forever, and the engine then failed with a native error on every
    start. The first notification of a verified file set is stored in a local
    manifest so later starts do not need the network at all.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name in filenames if name]
    if required and required not in names:
        raise DownloadError(f"Обязательный файл {required} отсутствует в списке загрузки")

    manifest = _read_manifest(local_dir)
    existing = {name: (local_dir / name).stat().st_size for name in names if (local_dir / name).exists()}
    missing = [name for name in names if name not in existing or existing[name] <= 0]
    if not missing and _manifest_covers(manifest, local_dir, names):
        # Sizes were verified against the pinned revision once; no network round
        # trip and no second multi-gigabyte hash are needed on every start.
        _report(
            progress,
            DownloadProgress(
                local_dir.name,
                len(names),
                len(names),
                file_count=len(names),
                message="файлы уже загружены",
            ),
        )
        return local_dir

    try:
        remote = _hf_remote_metadata(repo_id, revision)
    except Exception as exc:
        required_present = bool(
            required
            and (local_dir / required).is_file()
            and (local_dir / required).stat().st_size > 0
        )
        if allow_offline and required_present:
            # The engine can start: the required file is on disk. Optional
            # helpers may be missing, but a multi-gigabyte model must not be
            # rejected just because the Hub metadata endpoint is unreachable.
            absent = [name for name in names if name not in existing]
            logger.warning(
                "Не удалось проверить %s, работаем с локальными файлами%s: %s",
                repo_id,
                f" (нет: {', '.join(absent)})" if absent else "",
                exc,
            )
            return local_dir
        raise DownloadError(f"Не удалось получить список файлов {repo_id}: {exc}") from exc

    for name in names:
        if name not in remote:
            if required and name == required:
                raise DownloadError(f"В репозитории {repo_id} нет обязательного файла {name}")
            if ignore_missing:
                continue
            raise DownloadError(f"В репозитории {repo_id} нет файла {name}")
    names = [name for name in names if name in remote]

    from huggingface_hub import hf_hub_download

    total_files = len(names)
    downloaded_files = 0
    entries = dict(manifest)
    manifest_changed = False
    for index, name in enumerate(names):
        _check_cancel(cancel_event)
        remote_size, remote_sha256 = remote.get(name, (-1, None))
        local_path = local_dir / name
        local_size = local_path.stat().st_size if local_path.exists() else 0
        recorded = entries.get(name) or {}
        size_matches = local_size > 0 and (remote_size < 0 or local_size == remote_size)
        # A file recorded with a different hash belongs to another pinned
        # revision; trusting its size alone would silently mix model versions.
        same_revision = not (
            remote_sha256
            and recorded.get("sha256")
            and recorded["sha256"] != remote_sha256.lower()
        )
        if size_matches and same_revision:
            downloaded_files += 1
            if recorded.get("size") != local_size or (
                remote_sha256 and recorded.get("sha256") != remote_sha256
            ):
                entries[name] = {
                    "size": local_size,
                    "sha256": remote_sha256 or recorded.get("sha256"),
                }
                manifest_changed = True
            _report(
                progress,
                DownloadProgress(
                    name,
                    index + 1,
                    total_files,
                    file_index=index,
                    file_count=total_files,
                    message="уже загружено",
                ),
            )
            continue
        if local_size > 0:
            logger.warning(
                "Файл %s не совпадает с репозиторием (на диске %d байт, ожидалось %s) — "
                "скачиваю заново",
                name,
                local_size,
                remote_size if remote_size > 0 else "?",
            )
        _report(
            progress,
            DownloadProgress(
                name,
                index,
                total_files,
                file_index=index,
                file_count=total_files,
                message=f"{human_size(remote_size)} · скачивание",
            ),
        )
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(local_dir),
                revision=revision,
            )
        except Exception as exc:
            raise DownloadError(f"Не удалось скачать {name} из {repo_id}: {exc}") from exc
        _check_cancel(cancel_event)
        local_size = local_path.stat().st_size if local_path.exists() else 0
        if local_size <= 0:
            raise DownloadError(f"После загрузки файл {name} отсутствует или пуст")
        if remote_size > 0 and local_size != remote_size:
            raise DownloadError(
                f"Размер {name} не совпал: {local_size} из {remote_size} байт"
            )
        digest: Optional[str] = None
        if verify_hashes and remote_sha256:
            digest = sha256_file(local_path)
            if digest.lower() != remote_sha256:
                local_path.unlink(missing_ok=True)
                raise DownloadError(
                    f"Контрольная сумма {name} не совпала: файл удалён"
                )
        entries[name] = {
            "size": local_size,
            "sha256": (digest or remote_sha256 or recorded.get("sha256")),
        }
        manifest_changed = True
        downloaded_files += 1
        _report(
            progress,
            DownloadProgress(
                name,
                index + 1,
                total_files,
                file_index=index,
                file_count=total_files,
                message="готово",
            ),
        )
    if downloaded_files == 0 and not all((local_dir / name).exists() for name in names):
        raise DownloadError(f"Не удалось подготовить файлы {repo_id}")
    if manifest_changed:
        _write_manifest(local_dir, entries)
    return local_dir


def directory_size(path: Path | str) -> int:
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def cleanup_partials(path: Path | str, max_age_seconds: float = 24 * 3600) -> int:
    """Remove stale ``*.part`` files. Returns the number removed."""
    root = Path(path)
    if not root.exists():
        return 0
    removed = 0
    now = time.time()
    for item in root.rglob("*.part"):
        try:
            if now - item.stat().st_mtime > max_age_seconds:
                item.unlink()
                removed += 1
        except OSError:
            pass
    return removed
