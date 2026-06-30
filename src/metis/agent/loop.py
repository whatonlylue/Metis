"""Tool-use loop: drives an ``LLMClient`` against a set of ``ToolSpec``s.

This is provider-agnostic — it only depends on the ``LLMClient``/``AgentMessage``
contract in ``client.py``, not on any specific SDK. Tool handler exceptions
(including ``LockboxViolation`` from the sandbox layer) are caught and turned
into ``"error: ..."`` tool results, so a rejected action is feedback the model
can react to rather than a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from metis.agent.client import LLMClient
from metis.agent.tools import ToolSpec

AgentEvent = dict[str, Any]
EventCallback = Callable[[AgentEvent], None]


class TurnBudgetExceeded(RuntimeError):
    """Raised when ``max_turns`` is exhausted without the model stopping.

    Turn count is the M1 stand-in for the resource budgets described in
    CLAUDE.md ("enforced by the harness, not trusted to the agent"); time/$
    budgets are enforced separately later (M4).
    """


#: Callback invoked after each loop step with a snapshot of the transcript so far,
#: so the caller can persist partial progress (e.g. if the user quits mid-run).
StepCallback = Callable[[list[dict[str, Any]]], None]


@dataclass
class AgentLoop:
    client: LLMClient
    system: str
    tools: list[ToolSpec]
    max_turns: int = 20
    on_event: EventCallback | None = None
    on_step: StepCallback | None = None

    def run(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the loop to completion, returning the full message transcript.

        Pass ``history`` (a prior transcript returned by an earlier ``run``) to
        continue a multi-turn conversation — the new ``user_message`` is appended
        after it, so the model keeps the full context of the chat. Omit it for a
        fresh, single-shot run.
        """
        tools_by_name = {t.name: t for t in self.tools}
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_message})
        self._emit({"type": "user", "text": user_message})

        for _ in range(self.max_turns):
            reply = self.client.send(messages, self.tools, self.system)
            if reply.text:
                self._emit({"type": "text", "text": reply.text})
            self._emit({"type": "usage", "usage": reply.usage})

            assistant_content: list[dict[str, Any]] = []
            if reply.text:
                assistant_content.append({"type": "text", "text": reply.text})
            for call in reply.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            if not reply.tool_calls:
                self._persist(messages)
                return messages

            tool_results = []
            for call in reply.tool_calls:
                self._emit({"type": "tool_call", "name": call.name, "input": call.input})
                tool = tools_by_name.get(call.name)
                if tool is None:
                    result = f"error: unknown tool {call.name!r}"
                else:
                    try:
                        result = tool.handler(call.input)
                    except Exception as exc:
                        result = f"error: {exc}"
                self._emit({"type": "tool_result", "name": call.name, "result": result})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})
            # Persist after every tool round so a quit mid-training still leaves a
            # recent transcript on disk (the model's reply + the tool results).
            self._persist(messages)

        raise TurnBudgetExceeded(f"Agent did not finish within {self.max_turns} turns")

    def _emit(self, event: AgentEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _persist(self, messages: list[dict[str, Any]]) -> None:
        if self.on_step is not None:
            self.on_step(messages)
