"""Reusable LangChain integration for the local Ollama chat service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama


logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """Generate text through Ollama without coupling generation to retrieval."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = (
            settings.OLLAMA_TEMPERATURE
            if temperature is None
            else temperature
        )
        self.max_tokens = (
            settings.OLLAMA_MAX_TOKENS if max_tokens is None else max_tokens
        )
        self.request_timeout = (
            settings.OLLAMA_REQUEST_TIMEOUT
            if request_timeout is None
            else request_timeout
        )
        self._validate_configuration()
        self.chat_model = chat_model or self._create_chat_model()

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Generate a response for a user prompt and optional system prompt."""

        user_text = self._validate_prompt(prompt, "prompt")
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(
                SystemMessage(
                    content=self._validate_prompt(system_prompt, "system_prompt")
                )
            )
        messages.append(HumanMessage(content=user_text))
        return self.generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def generate_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Generate from LangChain messages for reuse by conversation services."""

        if not messages:
            raise ValueError("at least one message is required")
        if any(not isinstance(message, BaseMessage) for message in messages):
            raise TypeError("messages must contain LangChain BaseMessage instances")

        generation_temperature = (
            self.temperature if temperature is None else temperature
        )
        generation_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        self._validate_temperature(generation_temperature)
        self._validate_max_tokens(generation_max_tokens)
        stop_sequences = self._validate_stop_sequences(stop)

        invocation_options: dict[str, Any] = {
            "temperature": generation_temperature,
            "num_predict": generation_max_tokens,
        }
        if stop_sequences is not None:
            invocation_options["stop"] = stop_sequences

        try:
            response = self.chat_model.invoke(
                list(messages),
                **invocation_options,
            )
            text = self._extract_text(response.content)
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "Ollama generation failed using model %s at %s",
                self.model_name,
                self.base_url,
            )
            raise LLMServiceError(
                "The Ollama language model service could not generate a response."
            ) from exc

        logger.info("Generated an Ollama response using model %s", self.model_name)
        return text

    def _create_chat_model(self) -> ChatOllama:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise LLMServiceError(
                "The LangChain Ollama integration is not installed."
            ) from exc

        try:
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                client_kwargs={"timeout": self.request_timeout},
            )
        except Exception as exc:
            logger.exception("Could not configure the Ollama chat model")
            raise LLMServiceError(
                "The Ollama language model service could not be configured."
            ) from exc

    def _validate_configuration(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must contain non-whitespace text")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must contain non-whitespace text")
        self._validate_temperature(self.temperature)
        self._validate_max_tokens(self.max_tokens)
        if (
            isinstance(self.request_timeout, bool)
            or not isinstance(self.request_timeout, (int, float))
            or self.request_timeout <= 0
        ):
            raise ValueError("request_timeout must be greater than zero")

    @staticmethod
    def _validate_prompt(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-whitespace text")
        return value.strip()

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number between 0 and 2")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    @staticmethod
    def _validate_max_tokens(max_tokens: int) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

    @staticmethod
    def _validate_stop_sequences(stop: Sequence[str] | None) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str) or not stop:
            raise ValueError("stop must be a non-empty sequence of strings")
        sequences = list(stop)
        if any(not isinstance(item, str) or not item for item in sequences):
            raise ValueError("stop must contain non-empty strings")
        return sequences

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            text = "".join(parts).strip()
        else:
            text = ""

        if not text:
            raise LLMServiceError(
                "The Ollama language model returned an empty response."
            )
        return text
