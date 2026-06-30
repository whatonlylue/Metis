"""Anthropic implementation of ``LLMClient``.

Credentials are resolved through the M6 auth boundary
(``metis.agent.credentials``): an explicit ``api_key`` argument wins, otherwise
a ``CredentialProvider`` chain (``ANTHROPIC_API_KEY`` env var, then the local
``0600`` credentials file written by the token-management UI) supplies the key.
The loop and tools never see the secret — they only hold a constructed client.
"""

from __future__ import annotations

import copy
from typing import Any

import anthropic

from metis.agent.client import AgentMessage, LLMClient, ToolCall, Usage
from metis.agent.credentials import CredentialProvider, credential_provider_for
from metis.agent.tools import ToolSpec

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096

# 1-hour cache TTL (vs. the 5-minute default): model training turns are long-running,
# so a turn's prefix is often re-read well after the 5-minute window would have expired.
# The doubled write cost (2x vs 1.25x) pays off after ~3 reads, which a training loop
# easily clears. 1h TTL is GA on first-party Claude models — no beta header required.
_EPHEMERAL = {"type": "ephemeral", "ttl": "1h"}


class CredentialError(RuntimeError):
    """Raised when no Anthropic API key is available."""


def _with_rolling_cache_breakpoint(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return ``messages`` with a cache breakpoint on the last message's final block.

    Only the last message is copied (deep) so the caller's list — which the loop
    folds back into ``session.history`` — is never mutated. A string ``content`` is
    promoted to a single text block; structured blocks get ``cache_control`` directly.
    """
    if not messages:
        return messages
    out = list(messages)
    last = copy.deepcopy(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": dict(_EPHEMERAL)}]
    elif isinstance(content, list) and content:
        # content blocks may be plain strings or dicts; normalise the final one.
        tail = content[-1]
        if isinstance(tail, dict):
            tail["cache_control"] = dict(_EPHEMERAL)
        else:
            content[-1] = {"type": "text", "text": str(tail), "cache_control": dict(_EPHEMERAL)}
    else:
        return out  # empty content — nothing to cache
    out[-1] = last
    return out


def _resolve_api_key(
    api_key: str | None,
    provider: CredentialProvider | None = None,
) -> str:
    """Resolve a key: explicit argument > credential provider chain.

    The error message never echoes any partial secret.
    """
    key = api_key or (provider or credential_provider_for("anthropic")).get_api_key()
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

    @property
    def model(self) -> str:
        """The model id this client drives (used by the harness to price usage)."""
        return self._model

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str,
    ) -> AgentMessage:
        # Prompt caching: mark the most-stable-first prefixes (tools, system) and a
        # rolling breakpoint on the conversation-so-far. Repeated prefixes then bill
        # at ~0.1x on subsequent turns instead of full rate every turn.
        tool_params: list[dict[str, Any]] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        if tool_params:
            tool_params[-1] = {**tool_params[-1], "cache_control": dict(_EPHEMERAL)}

        system_blocks = [{"type": "text", "text": system, "cache_control": dict(_EPHEMERAL)}]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_blocks,  # type: ignore[arg-type]
            messages=_with_rolling_cache_breakpoint(messages),  # type: ignore[arg-type]
            tools=tool_params,  # type: ignore[arg-type]
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        raw = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw, "input_tokens", 0) or 0,
            output_tokens=getattr(raw, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        )

        return AgentMessage(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=usage,
        )
