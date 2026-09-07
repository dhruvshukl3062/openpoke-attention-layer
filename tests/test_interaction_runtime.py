"""The interaction agent's tool loop, driven by a scripted model.

This is the loop that decides everything the user sees, and until now none of it
could be exercised without a live API key. Each test below pins one branch of
``_run_interaction_loop``.
"""

from __future__ import annotations

import asyncio

import pytest

from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.services.conversation import get_conversation_log
from server.services.execution import get_agent_roster
from server.services.llm import (
    ScriptedLLM,
    raw_tool_call_response,
    text_response,
    tool_call_response,
    use_llm_client,
)


def _quiet_execution_agents(scripted: ScriptedLLM) -> ScriptedLLM:
    """Answer any stray call from a spawned execution agent.

    ``send_message_to_agent`` fires a background task that runs a real execution
    agent. These tests are about the *interaction* loop, so the handler keeps
    those background calls from draining the script or raising.
    """

    scripted._handler = lambda call: text_response("execution agent finished")
    return scripted


# ---------------------------------------------------------------------------
# The happy paths
# ---------------------------------------------------------------------------

async def test_plain_reply_ends_the_loop_and_is_recorded():
    scripted = ScriptedLLM([text_response("hey, what's up")])

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("hello")

    assert result.success
    assert result.response == "hey, what's up"
    assert scripted.call_count == 1, "a reply with no tool calls should end the loop immediately"

    transcript = get_conversation_log().load_transcript()
    assert "hello" in transcript
    assert "hey, what's up" in transcript


async def test_send_message_to_user_becomes_the_final_response():
    scripted = ScriptedLLM(
        [
            tool_call_response(("send_message_to_user", {"message": "on it"})),
            text_response(""),
        ]
    )

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("check my email")

    assert result.response == "on it"
    # The tool result must be fed back before the model is asked again.
    second_call_roles = [message["role"] for message in scripted.calls[1].messages]
    assert "tool" in second_call_roles


async def test_parallel_agent_dispatch_registers_every_agent():
    scripted = _quiet_execution_agents(
        ScriptedLLM(
            [
                tool_call_response(
                    ("send_message_to_user", {"message": "emailing both of them"}),
                    ("send_message_to_agent", {"agent_name": "Email to Alice", "instructions": "lunch"}),
                    ("send_message_to_agent", {"agent_name": "Email to Bob", "instructions": "lunch"}),
                ),
                text_response(""),
            ]
        )
    )

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("email alice and bob about lunch")
        await asyncio.sleep(0)  # let the spawned tasks start

    assert result.execution_agents_used == 2
    assert set(get_agent_roster().get_agents()) == {"Email to Alice", "Email to Bob"}


async def test_existing_agent_is_reused_not_duplicated():
    roster = get_agent_roster()
    roster.add_agent("Email to Alice")

    scripted = _quiet_execution_agents(
        ScriptedLLM(
            [
                tool_call_response(
                    ("send_message_to_agent", {"agent_name": "Email to Alice", "instructions": "reply"}),
                ),
                text_response("replied"),
            ]
        )
    )

    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute("reply to alice")
        await asyncio.sleep(0)

    assert get_agent_roster().get_agents() == ["Email to Alice"]


# ---------------------------------------------------------------------------
# Context: what the model is actually shown
# ---------------------------------------------------------------------------

async def test_every_roster_entry_is_pasted_into_the_prompt():
    """Baseline for the Attention Layer.

    Upstream renders the whole roster into every turn, so prompt size grows with
    the number of agents. This test documents that behaviour so the change is
    visible when the shortlist replaces it.
    """

    roster = get_agent_roster()
    for index in range(40):
        roster.add_agent(f"Agent Number {index}")

    scripted = ScriptedLLM([text_response("ok")])
    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute("hi")

    prompt = scripted.last_request().messages[0]["content"]
    assert prompt.count("<agent name=") == 40
    assert "Agent Number 0" in prompt and "Agent Number 39" in prompt


async def _prompt_chars_with_roster(agent_count: int) -> int:
    roster = get_agent_roster()
    roster.clear()
    for index in range(agent_count):
        roster.add_agent(f"Agent Number {index}")

    scripted = ScriptedLLM([text_response("ok")])
    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute("hi")
    return scripted.last_request().prompt_chars


async def test_roster_cost_per_turn_grows_linearly_and_without_bound():
    """The growth curve, measured. The Attention Layer should flatten this.

    Measured as the *marginal* cost of the roster rather than total prompt size:
    the static system prompt is ~10k characters, so at small roster sizes it
    swamps the signal. What matters is that each additional agent adds a fixed
    cost to every turn forever, which is what makes this unbounded rather than
    merely large.
    """

    baseline = await _prompt_chars_with_roster(0)
    at_50 = await _prompt_chars_with_roster(50) - baseline
    at_200 = await _prompt_chars_with_roster(200) - baseline

    assert at_50 > 0, "the roster should contribute to the prompt at all"

    # Linear in agent count: 4x the agents costs ~4x the characters.
    ratio = at_200 / at_50
    assert 3.5 < ratio < 4.5, f"expected linear growth, got a {ratio:.2f}x ratio"

    # And the per-agent cost is real, not rounding noise.
    per_agent = at_200 / 200
    assert per_agent > 20, f"each agent costs {per_agent:.1f} chars of every single turn"


# ---------------------------------------------------------------------------
# The failure paths
# ---------------------------------------------------------------------------

async def test_malformed_tool_arguments_do_not_crash_the_loop():
    broken = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "send_message_to_user", "arguments": "{not json"},
    }
    scripted = ScriptedLLM(
        [raw_tool_call_response(broken), text_response("recovered")]
    )

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("hi")

    assert result.success
    assert result.response == "recovered"
    # The model is told what went wrong, so it can correct itself.
    tool_reply = [m for m in scripted.calls[1].messages if m["role"] == "tool"][0]
    assert "invalid json" in tool_reply["content"]


async def test_unknown_tool_is_reported_rather_than_raised():
    scripted = ScriptedLLM(
        [tool_call_response(("no_such_tool", {})), text_response("moving on")]
    )

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("hi")

    assert result.success
    tool_reply = [m for m in scripted.calls[1].messages if m["role"] == "tool"][0]
    assert "Unknown tool" in tool_reply["content"]


async def test_iteration_cap_fails_closed():
    """A model that never stops calling tools must not loop forever.

    Upstream raises past ``MAX_TOOL_ITERATIONS`` and the runtime converts it into
    a failed result. Worth pinning: the cap is the only thing bounding cost per
    turn.
    """

    cap = InteractionAgentRuntime.MAX_TOOL_ITERATIONS
    scripted = ScriptedLLM(
        [tool_call_response(("wait", {"reason": "still thinking"}))] * (cap + 1)
    )

    with use_llm_client(scripted):
        result = await InteractionAgentRuntime().execute("hi")

    assert not result.success
    assert "iteration limit" in (result.error or "")
    assert scripted.call_count == cap


async def test_model_failure_is_surfaced_not_swallowed():
    class ExplodingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("upstream 502")

    with use_llm_client(ExplodingLLM()):
        result = await InteractionAgentRuntime().execute("hi")

    assert not result.success
    assert "502" in (result.error or "")
