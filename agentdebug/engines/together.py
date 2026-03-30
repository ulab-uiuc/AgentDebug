"""Together AI chat wrapper."""

import os
from typing import Any, List, Optional

try:
    from together import Together  # type: ignore
except Exception:  # pragma: no cover
    Together = None  # type: ignore

from .base import BaseChatModel, Message


class ChatTogether(BaseChatModel):
    """Thin wrapper around Together's chat completions API."""

    provider = "together"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if Together is None:
            raise RuntimeError("together package is not installed")

        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.getenv("TOGETHER_API_KEY", ""),
            temperature=temperature,
        )
        self.client = Together(api_key=self.api_key)

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
