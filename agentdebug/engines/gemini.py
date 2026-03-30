"""Google Gemini chat wrapper."""

import os
from typing import Any, List, Optional

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

from .base import BaseChatModel, Message, flatten_message_content


class ChatGemini(BaseChatModel):
    """Thin wrapper around Google's Gemini generate_content API."""

    provider = "gemini"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if genai is None:
            raise RuntimeError("google-genai package is not installed")

        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _messages_to_prompt(messages: List[Message]) -> str:
        chunks: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            text = flatten_message_content(message.get("content")).strip()
            if not text:
                continue
            chunks.append(f"{role.upper()}:\n{text}")
        return "\n\n".join(chunks)

    def _generate_text(self, messages: List[Message], **kwargs: Any) -> str:
        prompt = self._messages_to_prompt(messages)
        config = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config or None,
            )
        except TypeError:
            response = self.client.models.generate_content(model=self.model, contents=prompt)

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None) if content else None
            for part in candidate_parts or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts).strip()
