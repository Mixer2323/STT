"""Process-wide environment defaults that must be set before optional imports.

``huggingface_hub``, ``tokenizers`` and ``onnxruntime`` read these variables
when they are imported, so they are applied from ``voicetyper/__init__.py``:
every supported entry point (``voice_typer.py`` and ``python -m voicetyper``)
imports the package before any third-party module.
"""

from __future__ import annotations

import os
from typing import Mapping, MutableMapping

# Xet downloads and telemetry are disabled (unreliable behind proxies and not
# wanted here), symlink warnings are silenced because Windows without the
# privilege falls back to copying, and tokenizers must not fork threads inside
# a GUI process.
DEFAULTS: Mapping[str, str] = {
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def prepare(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Apply :data:`DEFAULTS` without overriding values the user set explicitly.

    Returns the names that were actually applied.
    """
    target = os.environ if environ is None else environ
    applied: list[str] = []
    for name, value in DEFAULTS.items():
        if name in target:
            continue
        target[name] = value
        applied.append(name)
    return applied
