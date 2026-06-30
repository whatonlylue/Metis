"""Tests for AgentLoop: a fake LLMClient drives scripted turns through tools."""

from __future__ import annotations

from typing import Any

import pytest

from metis.agent.client import AgentMessage, LLMClient, ToolCall
from metis.agent.loop import AgentLoop, TurnBudgetExceeded
from metis.agent.tools import ToolSpec


class ScriptedClient(LLMClient):
    """Replays a fixed sequence of AgentMessages, one per call to ``send``."""

    def __init__(self, replies: list[AgentMessage]) -> None:
        self._replies = list(replies)
        self.seen_messages: list[list[dict[str, Any]]] = []

    def send(self, messages, tools, system) -> AgentMessage:
        self.seen_messages.append(messages)
        return self._replies.pop(0)


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echoes back its input",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda args: f"echo: {args['text']}",
    )


def test_loop_returns_immediately_with_no_tool_calls() -> None:
    client = ScriptedClient([AgentMessage(text="hello", tool_calls=[])])
    loop = AgentLoop(client=client, system="sys", tools=[])

    messages = loop.run("hi")

    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["role"] == "assistant"
    assert len(client.seen_messages) == 1


def test_loop_dispatches_tool_call_and_feeds_result_back() -> None:
    call = ToolCall(id="call-1", name="echo", input={"text": "hi"})
    client = ScriptedClient(
        [
            AgentMessage(text="", tool_calls=[call]),
            AgentMessage(text="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(client=client, system="sys", tools=[_echo_tool()])

    messages = loop.run("go")

    tool_result_message = messages[2]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["content"] == "echo: hi"
    assert len(client.seen_messages) == 2


def test_loop_turns_handler_exception_into_error_tool_result() -> None:
    failing_tool = ToolSpec(
        name="boom",
        description="always fails",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: (_ for _ in ()).throw(ValueError("nope")),
    )
    call = ToolCall(id="call-1", name="boom", input={})
    client = ScriptedClient(
        [
            AgentMessage(text="", tool_calls=[call]),
            AgentMessage(text="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(client=client, system="sys", tools=[failing_tool])

    messages = loop.run("go")

    assert messages[2]["content"][0]["content"] == "error: nope"


def test_loop_unknown_tool_name_reported_as_error() -> None:
    call = ToolCall(id="call-1", name="missing", input={})
    client = ScriptedClient(
        [
            AgentMessage(text="", tool_calls=[call]),
            AgentMessage(text="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(client=client, system="sys", tools=[])

    messages = loop.run("go")

    assert "error: unknown tool" in messages[2]["content"][0]["content"]


def test_loop_raises_when_turn_budget_exhausted() -> None:
    call = ToolCall(id="call-1", name="echo", input={"text": "hi"})
    client = ScriptedClient([AgentMessage(text="", tool_calls=[call]) for _ in range(3)])
    loop = AgentLoop(client=client, system="sys", tools=[_echo_tool()], max_turns=3)

    with pytest.raises(TurnBudgetExceeded):
        loop.run("go")


def test_loop_emits_events_via_on_event() -> None:
    call = ToolCall(id="call-1", name="echo", input={"text": "hi"})
    client = ScriptedClient(
        [
            AgentMessage(text="thinking", tool_calls=[call]),
            AgentMessage(text="done", tool_calls=[]),
        ]
    )
    events: list[dict[str, Any]] = []
    loop = AgentLoop(client=client, system="sys", tools=[_echo_tool()], on_event=events.append)

    loop.run("go")

    # A "usage" event is emitted after each model turn (drives the live token/cost
    # readout), so two turns contribute two usage events.
    event_types = [e["type"] for e in events]
    assert event_types == [
        "user",
        "text",
        "usage",
        "tool_call",
        "tool_result",
        "text",
        "usage",
    ]
