"""Constants for the Qwen3-TTS text to speech service."""

from typing import Final

DOMAIN: Final = "qwen3_tts"

CONF_MODEL_PATH: Final = "model_path"
CONF_VOICE: Final = "voice"
CONF_LANGUAGE: Final = "language"

DEFAULT_MODEL_PATH: Final = "/config/models/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICE: Final = "default"
DEFAULT_LANGUAGE: Final = "it"

SUPPORTED_LANGUAGES: Final[list[str]] = [
    "it",
    "en",
    "zh",
    "de",
    "pt",
    "es",
    "ja",
    "ko",
    "fr",
    "ru",
]

SUPPORTED_OPTIONS: Final[list[str]] = [CONF_VOICE, CONF_LANGUAGE]

SAMPLE_RATE: Final = 24000

# Config flow step IDs
STEP_USER: Final = "user"
