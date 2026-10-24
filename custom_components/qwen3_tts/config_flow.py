"""Config flow for Qwen3-TTS integration."""

from __future__ import annotations

import logging
import os
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv

from .const import (
    CONF_LANGUAGE,
    CONF_MODEL_PATH,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_VOICE,
    DOMAIN,
    SUPPORTED_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)


class Qwen3TTSConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Qwen3-TTS."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model_path = user_input[CONF_MODEL_PATH]

            if model_path and not await self.hass.async_add_executor_job(
                os.path.exists, model_path
            ):
                errors["base"] = "model_path_not_found"
            else:
                return self.async_create_entry(
                    title="Qwen3-TTS",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL_PATH,
                        default=DEFAULT_MODEL_PATH,
                    ): str,
                    vol.Optional(
                        CONF_VOICE,
                        default=DEFAULT_VOICE,
                    ): str,
                    vol.Optional(
                        CONF_LANGUAGE,
                        default=DEFAULT_LANGUAGE,
                    ): vol.In(SUPPORTED_LANGUAGES),
                }
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return Qwen3TTSOptionsFlow()


class Qwen3TTSOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Qwen3-TTS."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model_path = user_input.get(CONF_MODEL_PATH, "")

            if model_path and not await self.hass.async_add_executor_job(
                os.path.exists, model_path
            ):
                errors[CONF_MODEL_PATH] = "model_path_not_found"
            else:
                return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL_PATH,
                        default=self.config_entry.options.get(
                            CONF_MODEL_PATH,
                            self.config_entry.data.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
                        ),
                    ): str,
                    vol.Optional(
                        CONF_VOICE,
                        default=self.config_entry.options.get(
                            CONF_VOICE,
                            self.config_entry.data.get(CONF_VOICE, DEFAULT_VOICE)
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGUAGE,
                        default=self.config_entry.options.get(
                            CONF_LANGUAGE,
                            self.config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
                        ),
                    ): vol.In(SUPPORTED_LANGUAGES),
                }
            ),
            errors=errors,
        )