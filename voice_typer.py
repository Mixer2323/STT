"""VoiceTyper entry point.

The application lives in the ``voicetyper`` package; this file stays as the
executable/PyInstaller entry point.
"""

from voicetyper.app import main

if __name__ == "__main__":
    raise SystemExit(main())
