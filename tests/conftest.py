"""Shared fixtures.

Two jobs here:

1. **Isolate on-disk state.** OpenPoke builds its stores as module-level
   singletons pointed at ``server/data/`` at import time. Tests must not read or
   write a developer's real conversation log, so every test gets fresh stores
   rooted in a tmp directory. (That the singletons have to be swapped out by
   hand is itself a finding -- see the single-tenant note in the write-up.)

2. **Make model calls deterministic** via the ``ScriptedLLM`` seam.
"""

from __future__ import annotations

import pytest

from server.config import get_settings
from server.services.llm import ScriptedLLM, use_llm_client


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point every persistent store at a per-test tmp directory.

    Ordering matters: ``ConversationLog.__init__`` resolves the working-memory
    singleton, so working memory has to be swapped first.
    """

    from server.services.conversation import log as log_module
    from server.services.conversation.summarization import working_memory_log as wm_module
    from server.services.execution import log_store as log_store_module
    from server.services.execution import roster as roster_module

    data_dir = tmp_path / "data"

    # -- working memory (must precede the conversation log) ----------------
    working_memory = wm_module.WorkingMemoryLog(data_dir / "conversation" / "working_memory.log")
    monkeypatch.setattr(wm_module, "_working_memory_log", working_memory)

    # -- conversation log --------------------------------------------------
    conversation = log_module.ConversationLog(data_dir / "conversation" / "conversation.log")
    monkeypatch.setattr(log_module, "_conversation_log", conversation)

    # -- execution agent roster + logs -------------------------------------
    roster = roster_module.AgentRoster(data_dir / "execution_agents" / "roster.json")
    monkeypatch.setattr(roster_module, "_agent_roster", roster)

    logs = log_store_module.ExecutionAgentLogStore(data_dir / "execution_agents")
    monkeypatch.setattr(log_store_module, "_execution_agent_logs", logs)

    # -- attention layer: agent registry -----------------------------------
    from server.services.attention import registry as registry_module

    registry = registry_module.AgentRegistry(data_dir / "execution_agents" / "registry.json")
    monkeypatch.setattr(registry_module, "_registry", registry)

    yield


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """A settings object safe for tests.

    ``summarization_enabled`` is derived from the threshold, so setting the
    threshold to 0 switches off the background summariser -- otherwise a test
    that writes 100+ log entries would fire a real model call.
    """

    settings = get_settings()
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key", raising=False)
    monkeypatch.setattr(settings, "conversation_summary_threshold", 0, raising=False)
    monkeypatch.setattr(settings, "interaction_agent_model", "test/interaction", raising=False)
    monkeypatch.setattr(settings, "execution_agent_model", "test/execution", raising=False)
    monkeypatch.setattr(settings, "summarizer_model", "test/summarizer", raising=False)
    monkeypatch.setattr(settings, "email_classifier_model", "test/classifier", raising=False)
    return settings


@pytest.fixture
def scripted_llm():
    """Install a ``ScriptedLLM`` for the duration of a test.

    Usage::

        def test_x(scripted_llm):
            scripted_llm.queue(text_response("hi"))
    """

    client = ScriptedLLM()
    with use_llm_client(client):
        yield client
