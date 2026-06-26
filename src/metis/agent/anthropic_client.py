"""Anthropic implementation of ``LLMClient``.

Credentials are resolved through the M6 auth boundary
(``metis.agent.credentials``): an explicit ``api_key`` argument wins, otherwise
a ``CredentialProvider`` chain (``ANTHROPIC_API_KEY`` env var, then the local
``0600`` credentials file written by the token-management UI) supplies the key.
The loop and tools never see the secret — they only hold a constructed client.
"""

from __future__ import annotations

from typing import Any

import anthropic

from metis.agent.client import AgentMessage, LLMClient, ToolCall
from metis.agent.credentials import CredentialProvider, default_credential_provider
from metis.agent.tools import ToolSpec

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class CredentialError(RuntimeError):
    """Raised when no Anthropic API key is available."""


def _resolve_api_key(
    api_key: str | None,
    provider: CredentialProvider | None = None,
) -> str:
    """Resolve a key: explicit argument > credential provider chain.

    The error message never echoes any partial secret.
    """
    key = api_key or (provider or default_credential_provider()).get_api_key()
    if not key:
        raise CredentialError(
            "No Anthropic API key found. Set one in the TUI token manager, pass api_key=, "
            "or export ANTHROPIC_API_KEY."
        )
    return key


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key, credential_provider))

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
