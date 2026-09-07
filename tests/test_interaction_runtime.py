"""The interaction agent's tool loop, driven by a scripted model.

This is the loop that decides everything the user sees, and until now none of it
could be exercised without a live API key. Each test below pins one branch of
``_run_interaction_loop``.
"""

from __future__ import annotations

import asyncio

import pytest

from server.agents.interaction_agent.agent import SHORTLIST_SIZE
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.services.conversation import get_conversation_log
from server.services.attention import get_agent_registry
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
    assert set(get_agent_registry().names()) == {"Email to Alice", "Email to Bob"}


async def test_existing_agent_is_reused_not_duplicated():
    get_agent_registry().upsert("Email to Alice", entities=["Alice"])

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

    assert get_agent_registry().names() == ["Email to Alice"]


# ---------------------------------------------------------------------------
# Context: what the model is actually shown
# ---------------------------------------------------------------------------

async def _prompt_with_roster(agent_count: int, request: str = "hi") -> str:
    # Each call appends to the conversation log, so it has to be reset or the
    # growing transcript is mistaken for growing roster cost.
    get_conversation_log().clear()
    registry = get_agent_registry()
    registry.clear()
    for index in range(agent_count):
        # Fixed-width names and a constant purpose, so the only thing that can
        # change between roster sizes is how many agents are rendered.
        registry.upsert(f"Agent Number {index:04d}", purpose="handles a topic")

    scripted = ScriptedLLM([text_response("ok")])
    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute(request)
    return scripted.last_request().messages[0]["content"]


async def test_the_model_sees_a_bounded_shortlist_not_the_whole_roster():
    prompt = await _prompt_with_roster(40)

    assert prompt.count("<agent ") == SHORTLIST_SIZE
    assert 'total="40"' in prompt, "the model should know the list is a selection"
    assert "name an agent directly" in prompt, "and how to reach one that is not listed"


async def test_roster_cost_per_turn_is_flat_regardless_of_roster_size():
    """The claim the whole Attention Layer rests on, measured.

    Upstream rendered every agent, so the marginal prompt cost grew linearly and
    without bound. With a shortlist the per-turn cost is set by SHORTLIST_SIZE,
    so a 40x bigger roster costs the same.
    """

    baseline = len(await _prompt_with_roster(0))
    at_10 = len(await _prompt_with_roster(10)) - baseline
    at_400 = len(await _prompt_with_roster(400)) - baseline

    assert at_10 > 0, "the shortlist should still contribute to the prompt"
    # The only legitimate difference is the digits in total="N".
    assert at_400 - at_10 <= 4, (
        f"prompt cost should be flat in roster size, grew {at_10} -> {at_400} chars"
    )


async def test_an_agent_named_in_the_request_is_never_ranked_away():
    registry = get_agent_registry()
    registry.clear()
    for index in range(40):
        registry.upsert(f"Agent Number {index:04d}", purpose="noise")
    registry.upsert("Vercel Job Offer", purpose="offer negotiation")

    scripted = ScriptedLLM([text_response("ok")])
    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute("any update on the Vercel Job Offer?")
    prompt = scripted.last_request().messages[0]["content"]

    assert "Vercel Job Offer" in prompt


async def test_shortlist_carries_the_metadata_that_supports_the_choice():
    registry = get_agent_registry()
    registry.clear()
    registry.upsert("Email to Alice", purpose="lunch thread", entities=["Alice"])
    registry.touch("Email to Alice")

    scripted = ScriptedLLM([text_response("ok")])
    with use_llm_client(scripted):
        await InteractionAgentRuntime().execute("ping Alice")
    prompt = scripted.last_request().messages[0]["content"]

    assert 'purpose="lunch thread"' in prompt
    assert "last_used=" in prompt
    assert 'runs="1"' in prompt


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
