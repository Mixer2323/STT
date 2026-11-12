"""VoiceTyper - local speech-to-text for Windows."""

import sys

from .environment import prepare as _prepare_environment

# huggingface_hub / tokenizers / onnxruntime read these on import, so they are
# applied as early as possible: every supported entry point (voice_typer.py and
# python -m voicetyper) imports this package before any third-party module.
_prepare_environment()

__version__ = "1.0.13"
APP_NAME = "VoiceTyper"


class _NullStream:
    encoding = "utf-8"

    def write(self, _data):
        return len(_data)

    def flush(self):
        pass

    def isatty(self):
        return False


# A GUI build of the app has no console; libraries that write to stdout must not
# crash because of that.
if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()
