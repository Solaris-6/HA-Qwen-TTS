"""Qwen3-TTS integration for Home Assistant."""

from __future__ import annotations

__version__ = "1.0.0"

from .tts import async_setup_entry


async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    # Nessuna risorsa da pulire, ma la funzione deve esistere
    return True