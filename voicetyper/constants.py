"""Application-wide constants."""

APP_ID = "VoiceTyper"
APP_NAME = "VoiceTyper"
APP_TITLE = "Voice Typer"
APP_VERSION = "1.0.0"
CONFIG_VERSION = 3
HISTORY_VERSION = 1

DEFAULT_HOTKEY = "ctrl+alt+space"

LANGUAGES = (
    "auto",
    "ru",
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "zh",
    "ja",
)

WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
)

# Mirrors faster-whisper 1.2.0 so that the downloaded snapshot can be loaded
# directly from disk instead of relying on the Hugging Face cache.
WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

WHISPER_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
    "model.bin",
)

ENGINE_LABELS = {
    "auto": "Авто",
    "gigaam": "GigaAM v3",
    "faster_whisper": "Faster-Whisper",
    "tone": "T-one",
}

ENGINE_ORDER = ("auto", "gigaam", "faster_whisper", "tone")

DEFAULT_EXCLUDE_PHRASES = (
    "Thank you.",
    "Thank you for watching.",
    "We'll be right back.",
    "Спасибо за просмотр.",
    "Продолжение следует.",
)

GIGASTT_RELEASE = "models-v3-2026-10-09"
GIGASTT_ASSETS = (
    "v3_e2e_rnnt_encoder_int8.onnx",
    "v3_e2e_rnnt_decoder.onnx",
    "v3_e2e_rnnt_joint.onnx",
    "v3_e2e_rnnt_vocab.txt",
)
GIGASTT_BASE_URL = (
    "https://github.com/ekhodzitsky/gigastt/releases/download/" + GIGASTT_RELEASE
)

# Hugging Face serves the same GigaAM v3 export through a much faster CDN in
# most regions, so it is tried before GitHub release assets. Each source maps
# the local asset name to ``(remote name, size in bytes, SHA-256)``. Decoder,
# joint and vocab are byte-identical between both sources; the int8 encoder is
# quantized differently but produces the same transcriptions.
GIGASTT_HF_REPO = "istupakov/gigaam-v3-onnx"
GIGASTT_HF_BASE_URL = f"https://huggingface.co/{GIGASTT_HF_REPO}/resolve/main"
GIGASTT_SOURCES = (
    (
        "Hugging Face",
        GIGASTT_HF_BASE_URL,
        {
            "v3_e2e_rnnt_encoder_int8.onnx": (
                "v3_e2e_rnnt_encoder.int8.onnx",
                224_570_477,
                "4e0e076a6076cd110277e529b8ac8f32cd5297f7fbebad5341ae8ddb7d00817b",
            ),
            "v3_e2e_rnnt_decoder.onnx": (
                "v3_e2e_rnnt_decoder.onnx",
                4_599_910,
                "7b0a16d67fd2cb37061decc93c69e364a9ab27afee3c57495d55b1c974cf7231",
            ),
            "v3_e2e_rnnt_joint.onnx": (
                "v3_e2e_rnnt_joint.onnx",
                2_712_896,
                "602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c",
            ),
            "v3_e2e_rnnt_vocab.txt": (
                "v3_e2e_rnnt_vocab.txt",
                13_354,
                "39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7",
            ),
        },
    ),
    (
        "GitHub",
        GIGASTT_BASE_URL,
        {
            "v3_e2e_rnnt_encoder_int8.onnx": (
                "v3_e2e_rnnt_encoder_int8.onnx",
                319_665_270,
                "16805903b102d8b27bb0253de044f25e33a8a5bf07fce0c4da4ac1a5a7544091",
            ),
            "v3_e2e_rnnt_decoder.onnx": (
                "v3_e2e_rnnt_decoder.onnx",
                4_599_910,
                "7b0a16d67fd2cb37061decc93c69e364a9ab27afee3c57495d55b1c974cf7231",
            ),
            "v3_e2e_rnnt_joint.onnx": (
                "v3_e2e_rnnt_joint.onnx",
                2_712_896,
                "602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c",
            ),
            "v3_e2e_rnnt_vocab.txt": (
                "v3_e2e_rnnt_vocab.txt",
                13_354,
                "39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7",
            ),
        },
    ),
)

TONE_REPO = "t-tech/T-one"
TONE_FILES = ("model.onnx", "kenlm.bin")

TONE_DECODERS = ("auto", "beam_search", "greedy")
DEVICES = ("auto", "cpu", "cuda")
COMPUTE_TYPES = ("auto", "int8", "int8_float16", "float16", "float32")
INSERT_MODES = ("paste", "clipboard", "type")
HOTKEY_MODES = ("hold", "toggle")

# Status codes shared by the tray, pipeline and notifications.
STATUS_IDLE = "idle"
STATUS_LOADING = "loading"
STATUS_READY = "ready"
STATUS_RECORDING = "recording"
STATUS_TRANSCRIBING = "transcribing"
STATUS_ERROR = "error"

MUTEX_NAME = "Local\\VoiceTyper.SingleInstance"
ACTIVATION_EVENT_NAME = "Local\\VoiceTyper.Activate"
