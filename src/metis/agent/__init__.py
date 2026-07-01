"""Provider-agnostic agent client (litellm-backed) + tool-use loop driving the sandbox."""

from __future__ import annotations

from metis.agent.client import AgentMessage, LLMClient, ToolCall, Usage
from metis.agent.credentials import (
    CredentialProvider,
    FileCredentialStore,
    default_credential_provider,
    looks_like_api_key,
    mask_key,
    resolve_api_key,
)
from metis.agent.define import run_define_step
from metis.agent.litellm_client import LiteLLMClient, build_client
from metis.agent.loop import AgentLoop, TurnBudgetExceeded
from metis.agent.model_config import (
    ModelSelection,
    load_selection,
    save_selection,
)
from metis.agent.pricing import cost_usd
from metis.agent.session import MAIN_SYSTEM_PROMPT, AgentSession
from metis.agent.tools import (
    ToolSpec,
    build_agent_tools,
    build_data_tools,
    build_define_tool,
    build_sandbox_tools,
)

__all__ = [
    "AgentMessage",
    "LLMClient",
    "ToolCall",
    "Usage",
    "AgentLoop",
    "TurnBudgetExceeded",
    "AgentSession",
    "MAIN_SYSTEM_PROMPT",
    "ToolSpec",
    "build_define_tool",
    "build_sandbox_tools",
    "build_data_tools",
    "build_agent_tools",
    "run_define_step",
    "CredentialProvider",
    "FileCredentialStore",
    "default_credential_provider",
    "looks_like_api_key",
    "mask_key",
    "resolve_api_key",
    "LiteLLMClient",
    "build_client",
    "cost_usd",
    "ModelSelection",
    "load_selection",
    "save_selection",
]
