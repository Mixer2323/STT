from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "release_assets"
OUT.mkdir(exist_ok=True)

PACKS = {
    "gigastt": {
        "asset": "VoiceTyper-models-gigastt-v3.zip",
        "source": MODELS / "gigastt",
        "files": [
            "v3_e2e_rnnt_encoder_int8.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ],
    },
    "tone": {
        "asset": "VoiceTyper-models-tone-model.zip",
        "source": MODELS / "tone",
        "files": ["model.onnx"],
    },
    "whisper-large-v3-turbo": {
        "asset": "VoiceTyper-models-whisper-large-v3-turbo.zip",
        "source": MODELS / "faster_whisper" / "large-v3-turbo",
        "files": [
            "config.json",
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.json",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


result = {}
for key, pack in PACKS.items():
    source = pack["source"]
    missing = [name for name in pack["files"] if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"{key}: missing {missing}")
    target = OUT / pack["asset"]
    print(f"packing {key} -> {target.name}", flush=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in pack["files"]:
            archive.write(source / name, arcname=name)
    size = target.stat().st_size
    digest = sha256(target)
    result[key] = {"asset": pack["asset"], "size": size, "sha256": digest}
    print(f"{target.name}: {size} bytes sha256={digest}", flush=True)

(OUT / "model_packs.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(result, indent=2, ensure_ascii=False))
