"""Provider-agnostic contract the agent loop drives.

The single concrete implementation (``litellm_client.LiteLLMClient``) translates
litellm's OpenAI-shaped API into ``AgentMessage``s and back, so every provider is
reached through one client. The loop in ``loop.py`` never touches a provider SDK
directly — it only depends on this ``LLMClient`` contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metis.agent.tools import ToolSpec


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    """Token usage for one model turn, including prompt-cache accounting.

    ``input_tokens`` is the *uncached* remainder only — the full prompt size is
    ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``.
    A non-zero ``cache_read_input_tokens`` across turns is the signal that prompt
    caching is actually working (vs. silently invalidated every turn)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
        )


@dataclass
class AgentMessage:
    """The model's next turn: some text, and zero or more tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


class LLMClient(ABC):
    """Drives a single model turn. Auth/credentials are handled in ``__init__``."""

    @abstractmethod
    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str,
    ) -> AgentMessage:
        """Send the conversation so far and return the model's next turn."""
