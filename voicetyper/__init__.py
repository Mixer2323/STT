"""VoiceTyper - local speech-to-text for Windows."""

import os
import sys

# These variables must be set before huggingface_hub / tokenizers / onnxruntime
# are imported anywhere in the process.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

__version__ = "1.0.2"
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
