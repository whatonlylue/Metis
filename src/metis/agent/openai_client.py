"""OpenAI implementation of ``LLMClient``.

The agent loop speaks one internal message shape — Anthropic-style content blocks
(``text`` / ``tool_use`` / ``tool_result``) — regardless of provider. This client
translates that shape to OpenAI's Chat Completions wire format on the way out and
back on the way in, so the loop, tools, and session never learn there is more than
one provider.

Credentials resolve through the same boundary as the Anthropic client
(``metis.agent.credentials``): an explicit ``api_key`` wins, else a
``CredentialProvider`` chain (``OPENAI_API_KEY`` env var, then the local
credentials file) supplies the key. The secret never reaches the loop or tools.
"""

from __future__ import annotations

import json
from typing import Any

from metis.agent.client import AgentMessage, LLMClient, ToolCall, Usage
from metis.agent.credentials import CredentialProvider, credential_provider_for
from metis.agent.tools import ToolSpec

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 4096


class CredentialError(RuntimeError):
    """Raised when no OpenAI API key is available."""


def _resolve_api_key(
    api_key: str | None,
    provider: CredentialProvider | None = None,
) -> str:
    """Resolve a key: explicit argument > credential provider chain.

    The error message never echoes any partial secret.
    """
    key = api_key or (provider or credential_provider_for("openai")).get_api_key()
    if not key:
        raise CredentialError(
            "No OpenAI API key found. Set one in the TUI token manager, pass api_key=, "
            "or export OPENAI_API_KEY."
        )
    return key


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


def _stringify_result(content: Any) -> str:
    """Tool results in the loop are strings; coerce anything else defensively."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_of(content)
    return json.dumps(content) if content is not None else ""


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


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        import openai

        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=_resolve_api_key(api_key, credential_provider))

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
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _to_openai_messages(messages, system),
        }
        if tools:
            request["tools"] = _to_openai_tools(tools)

        response = self._client.chat.completions.create(**request)
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

        raw = getattr(response, "usage", None)
        # OpenAI reports cached prompt tokens inside prompt_tokens_details; surface
        # them as cache reads so the TUI's "reused" readout works across providers.
        cached = 0
        details = getattr(raw, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
        usage = Usage(
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=getattr(raw, "completion_tokens", 0) or 0,
            cache_read_input_tokens=cached,
        )

        return AgentMessage(
            text=text,
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(choice.finish_reason or "stop", "end_turn"),
            usage=usage,
        )
