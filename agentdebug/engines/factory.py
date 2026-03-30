"""Engine factory helpers."""

from typing import Any, Callable, Optional

from .anthropic import ChatAnthropic
from .base import BaseChatModel
from .gemini import ChatGemini
from .openai import ChatOpenAI
from .together import ChatTogether

TOGETHER_MODEL_PREFIXES = ("meta-llama/", "qwen/", "mistralai/", "nousresearch/", "teknium/")


def normalize_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "openai": "openai",
        "openai-compatible": "openai",
        "compatible": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "gemini": "gemini",
        "google": "gemini",
        "together": "together",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported provider '{provider}'")
    return aliases[normalized]


def infer_provider(
    model_string: str,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    normalized = normalize_provider(provider)
    if normalized != "auto":
        return normalized

    if base_url:
        return "openai"

    lowered = model_string.strip().lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith(TOGETHER_MODEL_PREFIXES):
        return "together"
    return "openai"


def create_chat_model(
    model_string: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """Create a chat model instance for the given provider."""
    resolved_provider = infer_provider(
        model_string=model_string,
        base_url=base_url,
        provider=provider,
    )
    providers = {
        "openai": ChatOpenAI,
        "anthropic": ChatAnthropic,
        "gemini": ChatGemini,
        "together": ChatTogether,
    }
    chat_cls = providers[resolved_provider]
    return chat_cls(
        model=model_string,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def create_llm_engine(
    model_string: str = "gpt-4o-mini",
    is_multimodal: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> Callable[[Any], Any]:
    """Create a callable LLM engine (backward-compatible wrapper)."""
    chat = create_chat_model(
        model_string=model_string,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        provider=provider,
    )

    def _engine(prompt: Any, *args: Any, **kwargs: Any) -> Any:
        return chat(prompt, *args, **kwargs)

    return _engine
