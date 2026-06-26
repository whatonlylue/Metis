"""Provider-agnostic agent client + tool-use loop driving the sandbox."""

from __future__ import annotations

from metis.agent.client import AgentMessage, LLMClient, ToolCall
from metis.agent.credentials import (
    CredentialProvider,
    FileCredentialStore,
    default_credential_provider,
    looks_like_api_key,
    mask_key,
)
from metis.agent.define import run_define_step
from metis.agent.loop import AgentLoop, TurnBudgetExceeded
from metis.agent.tools import (
    ToolSpec,
    build_data_tools,
    build_define_tool,
    build_sandbox_tools,
)

__all__ = [
    "AgentMessage",
    "LLMClient",
    "ToolCall",
    "AgentLoop",
    "TurnBudgetExceeded",
    "ToolSpec",
    "build_define_tool",
    "build_sandbox_tools",
    "build_data_tools",
    "run_define_step",
    "CredentialProvider",
    "FileCredentialStore",
    "default_credential_provider",
    "looks_like_api_key",
    "mask_key",
]
