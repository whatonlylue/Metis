"""litellm implementation of ``LLMClient`` — the one provider client Metis ships.

litellm unifies every provider behind a single OpenAI-shaped ``completion`` call,
so Metis is provider-agnostic without a per-provider translator: the user supplies
one API key and a free-form model string (``anthropic/claude-opus-4-8``, ``gpt-4o``,
``gemini/gemini-1.5-pro``, …) and litellm routes it.

The agent loop speaks one internal message shape — Anthropic-style content blocks
(``text`` / ``tool_use`` / ``tool_result``). This client translates that to
litellm's OpenAI wire format on the way out and normalises the response back into
an ``AgentMessage`` on the way in, so the loop, tools, and session never learn
which provider is behind the call.

Prompt caching: when the target model supports it (litellm's
``supports_prompt_caching``), we mark the stable prefixes (system, the last tool)
and a rolling breakpoint on the final message with a 1-hour ``cache_control``.
litellm forwards these to Anthropic; for providers that cache transparently
(OpenAI) we skip the markers and still surface reused tokens from the usage block.

Credentials resolve through ``metis.agent.credentials``: an explicit ``api_key``
wins, else the credential chain (``METIS_API_KEY`` env, then the local ``0600``
file). When nothing is stored we pass ``None`` and let litellm fall back to the
provider's own env var. The secret never reaches the loop or tools.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import litellm
from litellm.utils import supports_prompt_caching

from metis.agent.client import AgentMessage, LLMClient, ToolCall, Usage
from metis.agent.credentials import CredentialProvider, resolve_api_key
from metis.agent.tools import ToolSpec

DEFAULT_MAX_TOKENS = 4096

# 1-hour cache TTL (vs. the 5-minute default): training turns are long-running, so
# a turn's prefix is often re-read well after the 5-minute window would expire. 1h
# TTL is GA on first-party Claude models — no beta header required.
_EPHEMERAL = {"type": "ephemeral", "ttl": "1h"}


# --------------------------------------------------------------- translation


def _text_of(content: Any) -> str:
    """Flatten an Anthropic-style content value to plain text (best-effort)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _stringify_result(content: Any) -> str:
    """Tool results in the loop are strings; coerce anything else defensively."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_of(content)
    return json.dumps(content) if content is not None else ""


def _to_openai_messages(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    """Translate the loop's Anthropic-style transcript into OpenAI chat messages.

    * ``user``/``assistant`` text becomes a normal chat turn.
    * assistant ``tool_use`` blocks become an assistant message with ``tool_calls``.
    * user ``tool_result`` blocks become individual ``role: tool`` messages keyed
      by ``tool_call_id`` (OpenAI requires one message per tool result).
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
            else:
                text_parts.append(_text_of(content))
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            text = "".join(text_parts)
            # OpenAI requires content to be present; use null only when tool calls exist.
            assistant_msg["content"] = text or (None if tool_calls else "")
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        # role == "user" (or anything else): may carry tool_result blocks.
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            leading_text: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": _stringify_result(block.get("content")),
                        }
                    )
                elif isinstance(block, dict) and block.get("type") == "text":
                    leading_text.append(block.get("text", ""))
            if leading_text:
                out.append({"role": "user", "content": "".join(leading_text)})
        else:
            out.append({"role": "user", "content": _text_of(content)})
    return out


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


# ------------------------------------------------------------- prompt caching


def _cache_last_block(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *message* with a cache breakpoint on its final content block.

    A string ``content`` is promoted to a single cached text block; a list gets
    ``cache_control`` on its last element. The input is not mutated so the caller's
    transcript stays clean.
    """
    out = copy.deepcopy(message)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = [{"type": "text", "text": content, "cache_control": dict(_EPHEMERAL)}]
    elif isinstance(content, list) and content:
        tail = content[-1]
        if isinstance(tail, dict):
            tail["cache_control"] = dict(_EPHEMERAL)
        else:
            content[-1] = {"type": "text", "text": str(tail), "cache_control": dict(_EPHEMERAL)}
    return out


def _apply_prompt_caching(
    oai_messages: list[dict[str, Any]], oai_tools: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark the stable prefixes (system, last tool) and a rolling breakpoint on the
    final message so repeated prefixes bill at cache-read rates on later turns."""
    messages = list(oai_messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = _cache_last_block(messages[0])
    if messages:
        messages[-1] = _cache_last_block(messages[-1])

    tools = list(oai_tools)
    if tools:
        tools[-1] = {**tools[-1], "cache_control": dict(_EPHEMERAL)}
    return messages, tools


# ------------------------------------------------------------------- usage


def _usage_from_response(response: Any) -> Usage:
    """Map litellm's OpenAI-shaped usage (with Anthropic cache buckets) to ``Usage``.

    ``prompt_tokens`` is the full prompt total; ``cached_tokens`` are cache reads
    and ``cache_creation_tokens`` are cache writes, so the uncached remainder is
    ``prompt_tokens`` minus both — matching ``Usage.input_tokens`` semantics.
    """
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0 if details is not None else 0
    creation = getattr(details, "cache_creation_tokens", 0) or 0 if details is not None else 0
    if not creation:
        creation = getattr(raw, "_cache_creation_input_tokens", 0) or 0
    prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
    return Usage(
        input_tokens=max(prompt_tokens - cached - creation, 0),
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
        cache_creation_input_tokens=creation,
        cache_read_input_tokens=cached,
    )


# ------------------------------------------------------------------ client


class LiteLLMClient(LLMClient):
    """Drives any litellm-supported model through a single ``completion`` call."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        credential_provider: CredentialProvider | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        # Resolve now so a missing-key situation is knowable up front, but keep None
        # (litellm falls back to the provider's own env var) rather than failing hard.
        self._api_key = resolve_api_key(api_key, credential_provider)

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
        oai_messages = _to_openai_messages(messages, system)
        oai_tools = _to_openai_tools(tools)
        if supports_prompt_caching(self._model):
            oai_messages, oai_tools = _apply_prompt_caching(oai_messages, oai_tools)

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": oai_messages,
            "api_key": self._api_key,
        }
        if oai_tools:
            request["tools"] = oai_tools

        response: Any = litellm.completion(**request)
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, input=args))

        return AgentMessage(
            text=text,
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(choice.finish_reason or "stop", "end_turn"),
            usage=_usage_from_response(response),
        )


def build_client(
    model: str,
    *,
    api_key: str | None = None,
    credential_provider: CredentialProvider | None = None,
    max_tokens: int | None = None,
) -> LLMClient:
    """Construct the litellm-backed client for *model* (a free-form litellm id)."""
    return LiteLLMClient(
        model,
        api_key=api_key,
        credential_provider=credential_provider,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
    )
