"""The LLM client seam.

Every model call in OpenPoke goes through the ``LLMClient`` protocol rather than
importing :func:`request_chat_completion` directly. That indirection is what
makes the orchestration layer testable: a test installs a scripted client and
the interaction/execution loops run to completion without a network call, an API
key, or a source of non-determinism.

The client is held in a :class:`~contextvars.ContextVar` rather than a plain
module global so that concurrent tasks (and concurrent tests) can each install
their own client without stepping on one another.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a chat completion request into a response payload.

    The signature deliberately mirrors ``request_chat_completion`` so that
    swapping the call sites over is a mechanical change, and so that the
    returned payload keeps the OpenAI/OpenRouter shape the runtimes already
    parse (``{"choices": [{"message": {...}}]}``).
    """

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ...


_client_var: ContextVar[Optional[LLMClient]] = ContextVar("openpoke_llm_client", default=None)


def get_llm_client() -> LLMClient:
    """Return the active client, defaulting to the real OpenRouter one."""

    client = _client_var.get()
    if client is not None:
        return client

    # Imported lazily so that installing a test client never pulls in httpx or
    # touches configuration.
    from .openrouter import OpenRouterClient

    client = OpenRouterClient()
    _client_var.set(client)
    return client


def set_llm_client(client: Optional[LLMClient]) -> None:
    """Install a client process-wide. Pass ``None`` to fall back to the default."""

    _client_var.set(client)


@contextmanager
def use_llm_client(client: LLMClient) -> Iterator[LLMClient]:
    """Install ``client`` for the duration of the block, then restore.

    This is the entry point tests and the evaluation harness use::

        with use_llm_client(ScriptedLLM([...])) as llm:
            await runtime.execute("hello")
        assert llm.tool_names() == ["send_message_to_user"]
    """

    token = _client_var.set(client)
    try:
        yield client
    finally:
        _client_var.reset(token)


__all__ = ["LLMClient", "get_llm_client", "set_llm_client", "use_llm_client"]
