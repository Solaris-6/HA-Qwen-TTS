"""Support for the Qwen3-TTS text to speech service."""

import io
import logging
import os
import wave
from typing import Any, Final

import numpy as np
import torch
import probatio

from homeassistant.components.tts import (
    Provider,
    TtsAudioType,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import (
    CONF_LANGUAGE,
    CONF_MODEL_PATH,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    DOMAIN,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)

_LOGGER: Final = logging.getLogger(__name__)

# Cache globale per il modello e il processor
_model: Any = None
_processor: Any = None
_model_loaded: bool = False


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Qwen3-TTS platform (legacy YAML setup)."""
    model_path = config.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
    voice = config.get(CONF_VOICE, DEFAULT_VOICE)
    language = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

    _async_setup_provider(hass, model_path, voice, language, async_add_entities)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Qwen3-TTS from a config entry."""
    model_path = entry.data.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
    voice = entry.data.get(CONF_VOICE, DEFAULT_VOICE)
    language = entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

    _async_setup_provider(hass, model_path, voice, language, async_add_entities)


def _async_setup_provider(
    hass: HomeAssistant,
    model_path: str,
    voice: str,
    language: str,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Qwen3-TTS provider."""
    if model_path and not os.path.exists(model_path):
        _LOGGER.error(
            "Il percorso del modello Qwen3-TTS non esiste: %s", model_path
        )
        return

    try:
        _load_model(model_path)
    except Exception:
        _LOGGER.exception("Impossibile caricare il modello Qwen3-TTS")
        return

    async_add_entities([Qwen3TTSProvider(hass, model_path, voice, language)])


def _load_model(model_path: str) -> tuple[Any, Any]:
    """Carica il modello Qwen3-TTS e il processor."""
    global _model, _processor, _model_loaded

    if _model_loaded:
        return _model, _processor

    try:
        from transformers import Qwen3TTSForConditionalGeneration, AutoProcessor

        _LOGGER.info("Caricamento modello Qwen3-TTS da %s", model_path)

        _processor = AutoProcessor.from_pretrained(model_path)
        _model = Qwen3TTSForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        _model_loaded = True
        _LOGGER.info("Modello Qwen3-TTS caricato con successo")

    except ImportError as err:
        _LOGGER.error("Impossibile importare transformers: %s", err)
        raise
    except Exception as err:
        _LOGGER.error("Errore durante il caricamento del modello Qwen3-TTS: %s", err)
        raise

    return _model, _processor


def _synthesize_audio(
    model: Any,
    processor: Any,
    text: str,
    language: str,
    voice: str,
) -> bytes:
    """Sintetizza audio da testo usando Qwen3-TTS."""
    try:
        # Prepara il testo con i token speciali TTS
        tts_text = f"<tts_text_bos>{text}<tts_text_eod>"

        # Tokenizza l'input
        inputs = processor(
            text=tts_text,
            language=language,
            return_tensors="pt",
        )

        # Sposta su GPU se disponibile
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Genera l'audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
            )

        # Decodifica l'output audio
        audio = processor.decode(outputs[0], skip_special_tokens=False)

        # Converti in bytes WAV
        audio_array = np.array(audio, dtype=np.float32)
        audio_array = (audio_array * 32767).astype(np.int16)

        # Crea WAV in memoria
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_array.tobytes())

        return buffer.getvalue()

    except Exception as err:
        _LOGGER.error("Errore durante la sintesi audio Qwen3-TTS: %s", err)
        raise


class Qwen3TTSProvider(Provider):
    """Qwen3-TTS speech api provider."""

    def __init__(
        self,
        hass: HomeAssistant,
        model_path: str,
        voice: str,
        language: str,
    ) -> None:
        """Initialize Qwen3-TTS provider for TTS."""
        self.hass = hass
        self.model_path = model_path
        self._attr_name = "Qwen3-TTS"
        self._attr_unique_id = "qwen3_tts"
        self.default_voice = voice
        self.default_language = language

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def default_language(self) -> str | None:
        """Return the default language."""
        return self.default_language

    @property
    def default_options(self) -> dict[str, str]:
        """Return dict include default options."""
        return {
            CONF_VOICE: self.default_voice,
            CONF_LANGUAGE: self.default_language,
        }

    @property
    def supported_options(self) -> list[str]:
        """Return a list of supported options."""
        return [CONF_VOICE, CONF_LANGUAGE]

    def get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Request TTS file from Qwen3-TTS."""
        voice = options.get(CONF_VOICE, self.default_voice)
        lang = options.get(CONF_LANGUAGE, self.default_language)

        if lang not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Lingua %s non supportata da Qwen3-TTS", lang)
            return None, None

        _LOGGER.debug("Richiesta TTS per testo: %s", message)

        try:
            model, processor = _load_model(self.model_path)
            audio_data = _synthesize_audio(
                model=model,
                processor=processor,
                text=message,
                language=lang,
                voice=voice,
            )

            _LOGGER.debug("Risposta TTS ricevuta per: %s", message)
            return ("wav", audio_data)

        except Exception:
            _LOGGER.exception("Errore durante la generazione TTS Qwen3-TTS")
            return None, None
