"""Provider-agnostic chat model interface.

Providers implement ``_generate_text`` while the base class handles
prompt-to-messages conversion and best-effort structured JSON parsing.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore


Message = Dict[str, Any]


def flatten_message_content(content: Any) -> str:
    """Convert provider-specific message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_structured_response(text: str, response_format: Optional[Type] = None) -> Any:
    """Best-effort parsing for pydantic response models."""
    if not response_format or not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
        return text

    try:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return response_format(**data)
        except Exception:
            pass

        # Fallback: try to fill known fields with the raw text
        payload = {}
        for field in ("analysis", "text"):
            if (hasattr(response_format, "model_fields") and field in response_format.model_fields) or (
                hasattr(response_format, "__fields__") and field in getattr(response_format, "__fields__")
            ):
                payload[field] = text
        if payload:
            return response_format(**payload)
    except Exception:
        pass

    return text


class BaseChatModel(ABC):
    """Provider-agnostic chat model interface."""

    provider = "base"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    def build_messages(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> List[Message]:
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})

        if not images:
            messages.append({"role": "user", "content": prompt})
            return messages

        content = prompt
        for image_path in images:
            content += f"\n[Image: {image_path}]"
        messages.append({"role": "user", "content": content})
        return messages

    def __call__(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
        response_format: Optional[Type] = None,
        **kwargs: Any,
    ) -> Any:
        messages = self.build_messages(prompt=prompt, images=images, system=system)
        text = self.generate(messages, **kwargs)
        return parse_structured_response(text, response_format=response_format)

    def generate(self, messages: List[Message], **kwargs: Any) -> str:
        return self._generate_text(messages, **kwargs)

    @abstractmethod
    def _generate_text(self, messages: List[Message], **kwargs: Any) -> str:
        """Generate a plain-text response from a list of chat messages."""
