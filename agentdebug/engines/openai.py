"""OpenAI chat wrapper."""

import os
from typing import Any, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore

from .base import BaseChatModel, Message


class ChatOpenAI(BaseChatModel):
    """Thin wrapper around OpenAI's Chat Completions API."""

    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        super().__init__(
            model=model,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            temperature=temperature,
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _generate_text(self, messages: List[Message], **kwargs: Any) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "n": 1,
        }
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()
