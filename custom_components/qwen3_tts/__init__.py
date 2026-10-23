"""Qwen3-TTS integration for Home Assistant."""

from __future__ import annotations

# Rimossa la riga "from . import __version__" che causava ImportError
# La costante è definita direttamente qui sotto

__version__ = "1.0.0"

from .tts import async_setup_entry
