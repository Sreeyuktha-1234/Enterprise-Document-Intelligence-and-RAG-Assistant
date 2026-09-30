"""Tests for reusable LangChain Ollama generation."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.exceptions import LLMServiceError
from app.services.llm_service import LLMService


class FakeChatModel:
    """Small injectable chat model that records LangChain invocations."""

    def __init__(
        self,
        content: Any = "Generated answer.",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.content)


def _service(model: FakeChatModel, **kwargs: Any) -> LLMService:
    return LLMService(
        chat_model=model,  # type: ignore[arg-type]
        model_name="test-model",
        base_url="http://ollama.test:11434",
        **kwargs,
    )


def test_generate_supports_system_prompt_and_generation_overrides() -> None:
    model = FakeChatModel("  Enterprise answer.  ")
    service = _service(model, temperature=0.2, max_tokens=500)

    answer = service.generate(
        "  Explain the policy.  ",
        system_prompt="  Answer using verified facts.  ",
        temperature=0.7,
        max_tokens=200,
        stop=["END"],
    )

    assert answer == "Enterprise answer."
    messages, options = model.calls[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Answer using verified facts."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Explain the policy."
    assert options == {
        "temperature": 0.7,
        "num_predict": 200,
        "stop": ["END"],
    }


def test_generate_uses_configured_defaults_without_system_prompt() -> None:
    model = FakeChatModel()
    service = _service(model, temperature=0.3, max_tokens=321)

    assert service.generate("Summarize this.") == "Generated answer."

    messages, options = model.calls[0]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert options == {"temperature": 0.3, "num_predict": 321}


def test_generate_messages_accepts_conversation_history() -> None:
    model = FakeChatModel()
    service = _service(model)
    history = [
        SystemMessage(content="Be concise."),
        HumanMessage(content="What is the retention period?"),
        AIMessage(content="Seven years."),
        HumanMessage(content="Which policy states that?"),
    ]

    service.generate_messages(history)

    assert model.calls[0][0] == history


def test_generation_extracts_text_content_blocks() -> None:
    model = FakeChatModel(
        [
            {"type": "text", "text": "First "},
            {"type": "text", "text": "second."},
        ]
    )

    assert _service(model).generate("Answer") == "First second."


def test_connection_failure_is_wrapped_without_leaking_details() -> None:
    model = FakeChatModel(error=ConnectionError("internal host detail"))

    with pytest.raises(LLMServiceError, match="could not generate") as error:
        _service(model).generate("Answer")

    assert "internal host detail" not in str(error.value)


def test_empty_model_response_raises_controlled_error() -> None:
    with pytest.raises(LLMServiceError, match="empty response"):
        _service(FakeChatModel("   ")).generate("Answer")


@pytest.mark.parametrize("prompt", ["", "   "])
def test_generate_rejects_blank_prompts(prompt: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        _service(FakeChatModel()).generate(prompt)


@pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "warm"])
def test_generate_rejects_invalid_temperature(temperature: Any) -> None:
    with pytest.raises(ValueError, match="temperature"):
        _service(FakeChatModel()).generate("Answer", temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_generate_rejects_invalid_max_tokens(max_tokens: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _service(FakeChatModel()).generate("Answer", max_tokens=max_tokens)


@pytest.mark.parametrize("stop", [[], "END", [""], [1]])
def test_generate_rejects_invalid_stop_sequences(stop: Any) -> None:
    with pytest.raises(ValueError, match="stop"):
        _service(FakeChatModel()).generate("Answer", stop=stop)


def test_generate_messages_rejects_empty_or_invalid_messages() -> None:
    service = _service(FakeChatModel())

    with pytest.raises(ValueError, match="at least one"):
        service.generate_messages([])
    with pytest.raises(TypeError, match="BaseMessage"):
        service.generate_messages(["not a message"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_name": "   "}, "model_name"),
        ({"base_url": "   "}, "base_url"),
        ({"request_timeout": 0}, "request_timeout"),
        ({"request_timeout": True}, "request_timeout"),
    ],
)
def test_service_rejects_invalid_configuration(
    overrides: dict[str, Any],
    message: str,
) -> None:
    configuration = {
        "model_name": "test-model",
        "base_url": "http://ollama.test:11434",
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        LLMService(
            chat_model=FakeChatModel(),  # type: ignore[arg-type]
            **configuration,
        )
