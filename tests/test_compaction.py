"""Per-agent history compaction.

Upstream loads an execution agent's entire log into its system prompt on every
invocation, so an agent driven by a recurring trigger grows its own prompt
without bound. These tests pin the replacement: a rolling summary plus a recent
verbatim tail, with the full transcript as the fallback.
"""

from __future__ import annotations

import pytest

from server.services.attention import get_agent_registry
from server.services.attention.compaction import (
    COMPACT_THRESHOLD,
    TAIL_SIZE,
    compact_agent_history,
    needs_compaction,
    render_agent_history,
)
from server.services.execution import get_execution_agent_logs
from server.services.llm import ScriptedLLM, text_response, use_llm_client

AGENT = "Email to Alice"


def seed_log(entry_count: int, *, agent: str = AGENT) -> None:
    """Write a realistic request/action/response cycle to the agent's log."""

    logs = get_execution_agent_logs()
    for index in range(entry_count):
        logs.record_request(agent, f"instruction number {index}")


def register(agent: str = AGENT):
    registry = get_agent_registry()
    registry.upsert(agent, purpose="lunch thread", entities=["Alice"])
    return registry


# ---------------------------------------------------------------------------
# When compaction runs
# ---------------------------------------------------------------------------

def test_short_histories_are_left_alone():
    register()
    seed_log(5)

    assert not needs_compaction(AGENT)


def test_threshold_accounts_for_the_preserved_tail():
    """Compaction triggers only once there is more than a tail's worth to fold."""

    register()
    seed_log(COMPACT_THRESHOLD + TAIL_SIZE)
    assert not needs_compaction(AGENT)

    seed_log(1)
    assert needs_compaction(AGENT)


async def test_no_model_call_below_the_threshold():
    register()
    seed_log(5)
    scripted = ScriptedLLM()

    with use_llm_client(scripted):
        compacted = await compact_agent_history(AGENT)

    assert compacted is False
    assert scripted.call_count == 0, "compaction must not spend a model call it doesn't need"


# ---------------------------------------------------------------------------
# What compaction produces
# ---------------------------------------------------------------------------

async def test_compaction_stores_a_summary_and_advances_the_cursor():
    register()
    total = COMPACT_THRESHOLD + TAIL_SIZE + 20
    seed_log(total)
    scripted = ScriptedLLM([text_response("Drafted a lunch invite to Alice; awaiting her reply.")])

    with use_llm_client(scripted):
        assert await compact_agent_history(AGENT) is True

    record = get_agent_registry().get(AGENT)
    assert "Alice" in record.summary
    assert record.compacted_through == total - TAIL_SIZE, "the tail is never compacted"


async def test_the_summariser_sees_the_previous_summary_and_the_new_entries():
    register()
    seed_log(COMPACT_THRESHOLD + TAIL_SIZE + 5)
    get_agent_registry().set_compaction(AGENT, summary="earlier context", compacted_through=0)
    scripted = ScriptedLLM([text_response("merged summary")])

    with use_llm_client(scripted):
        await compact_agent_history(AGENT)

    prompt = scripted.last_request().messages[0]["content"]
    assert "earlier context" in prompt
    assert "instruction number 0" in prompt


async def test_rendered_history_is_summary_plus_tail():
    register()
    total = COMPACT_THRESHOLD + TAIL_SIZE + 20
    seed_log(total)
    scripted = ScriptedLLM([text_response("rolling summary text")])

    with use_llm_client(scripted):
        await compact_agent_history(AGENT)

    rendered = render_agent_history(AGENT)

    assert "rolling summary text" in rendered
    assert "<history_summary" in rendered
    assert rendered.count("<agent_request") == TAIL_SIZE
    # The oldest entry is summarised away; the newest is still verbatim.
    assert "instruction number 0<" not in rendered
    assert f"instruction number {total - 1}<" in rendered


async def test_history_cost_stops_growing_with_invocation_count():
    """The property this component exists for."""

    register()
    seed_log(COMPACT_THRESHOLD + TAIL_SIZE + 20)
    scripted = ScriptedLLM([text_response("summary one")])
    with use_llm_client(scripted):
        await compact_agent_history(AGENT)
    after_first = len(render_agent_history(AGENT))

    # Another 200 invocations' worth of log.
    seed_log(200)
    scripted = ScriptedLLM([text_response("summary two")])
    with use_llm_client(scripted):
        await compact_agent_history(AGENT)
    after_many = len(render_agent_history(AGENT))

    assert after_many < after_first * 2, (
        f"history should stay bounded across invocations, {after_first} -> {after_many} chars"
    )


# ---------------------------------------------------------------------------
# Failure and edge behaviour
# ---------------------------------------------------------------------------

async def test_a_failed_summariser_falls_back_to_the_full_transcript():
    """A degraded prompt beats a failed task: compaction is best-effort."""

    register()
    seed_log(COMPACT_THRESHOLD + TAIL_SIZE + 20)

    class ExplodingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("summariser unavailable")

    with use_llm_client(ExplodingLLM()):
        assert await compact_agent_history(AGENT) is False

    record = get_agent_registry().get(AGENT)
    assert record.compacted_through == 0
    assert "instruction number 0" in render_agent_history(AGENT), "nothing was lost"


async def test_an_empty_summary_is_not_stored():
    register()
    seed_log(COMPACT_THRESHOLD + TAIL_SIZE + 20)
    scripted = ScriptedLLM([text_response("   ")])

    with use_llm_client(scripted):
        assert await compact_agent_history(AGENT) is False

    assert get_agent_registry().get(AGENT).compacted_through == 0


async def test_compaction_is_a_no_op_for_an_unregistered_agent():
    seed_log(200, agent="Ghost Agent")
    scripted = ScriptedLLM()

    with use_llm_client(scripted):
        assert await compact_agent_history("Ghost Agent") is False

    assert scripted.call_count == 0


def test_uncompacted_agents_render_exactly_as_before():
    register()
    seed_log(5)

    rendered = render_agent_history(AGENT)

    assert rendered == get_execution_agent_logs().load_transcript(AGENT)


def test_the_compaction_cursor_never_moves_backwards():
    """Guards against a stale writer un-summarising entries."""

    registry = register()
    registry.set_compaction(AGENT, summary="later", compacted_through=50)
    registry.set_compaction(AGENT, summary="stale", compacted_through=10)

    assert registry.get(AGENT).compacted_through == 50
