"""Anthropic implementation of ``LLMClient``.

Credentials are read from an explicit ``api_key`` argument or the
``ANTHROPIC_API_KEY`` env var — this is the stubbed credentials boundary the
roadmap calls for; OAuth/token management UI replaces it in M6 without the
loop or tools needing to change.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from metis.agent.client import AgentMessage, LLMClient, ToolCall
from metis.agent.tools import ToolSpec

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class CredentialError(RuntimeError):
    """Raised when no Anthropic API key is available."""


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise CredentialError(
            "No Anthropic API key found. Pass api_key= or set ANTHROPIC_API_KEY "
            "(token management UI lands in M6)."
        )
    return key


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key))

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str,
    ) -> AgentMessage:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return AgentMessage(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
        )
