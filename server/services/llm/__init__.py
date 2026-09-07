"""LLM client seam -- every model call in OpenPoke routes through here."""

from .base import LLMClient, get_llm_client, set_llm_client, use_llm_client
from .openrouter import OpenRouterClient
from .scripted import (
    RecordedCall,
    ScriptExhausted,
    ScriptedLLM,
    raw_tool_call_response,
    text_response,
    tool_call_response,
)

__all__ = [
    "LLMClient",
    "get_llm_client",
    "set_llm_client",
    "use_llm_client",
    "OpenRouterClient",
    "ScriptedLLM",
    "ScriptExhausted",
    "RecordedCall",
    "text_response",
    "tool_call_response",
    "raw_tool_call_response",
]
