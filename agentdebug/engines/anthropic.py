"""Anthropic chat wrapper."""

import os
from typing import Any, List, Optional

try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore

from .base import BaseChatModel, Message, flatten_message_content


class ChatAnthropic(BaseChatModel):
    """Thin wrapper around Anthropic's Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if Anthropic is None:
            raise RuntimeError("anthropic package is not installed")

        super().__init__(
            model=model,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
        )
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = Anthropic(**client_kwargs)

    def _prepare_messages(self, messages: List[Message]) -> tuple[List[dict[str, str]], Optional[str]]:
        system_parts: List[str] = []
        converted: List[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            text = flatten_message_content(message.get("content")).strip()
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
                continue
            if role not in ("user", "assistant"):
                role = "user"
            converted.append({"role": role, "content": text})
        system_prompt = "\n\n".join(system_parts).strip() or None
        return converted, system_prompt

    def _generate_text(self, messages: List[Message], **kwargs: Any) -> str:
        converted, system_prompt = self._prepare_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": self.temperature,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            request["system"] = system_prompt

        response = self.client.messages.create(**request)
        blocks = getattr(response, "content", None) or []
        texts: List[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()
