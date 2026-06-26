"""Provider-agnostic contract the agent loop drives.

Concrete providers (Anthropic first, see ``anthropic_client.py``) translate their
native API into ``AgentMessage``s and back. The loop in ``loop.py`` never touches
a provider SDK directly, so adding a second provider means writing one new
``LLMClient`` implementation, not touching the loop.
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
class AgentMessage:
    """The model's next turn: some text, and zero or more tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


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
