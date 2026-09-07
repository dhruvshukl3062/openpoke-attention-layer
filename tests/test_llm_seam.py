"""Tests for the LLM seam itself.

If these fail, nothing else in the suite can be trusted -- every other test
depends on model calls being interceptable and deterministic.
"""

from __future__ import annotations

import json

import pytest

from server.services.llm import (
    OpenRouterClient,
    ScriptExhausted,
    ScriptedLLM,
    get_llm_client,
    text_response,
    tool_call_response,
    use_llm_client,
)


def test_default_client_is_the_real_one():
    assert isinstance(get_llm_client(), OpenRouterClient)


def test_use_llm_client_installs_and_restores():
    scripted = ScriptedLLM()
    with use_llm_client(scripted):
        assert get_llm_client() is scripted
    assert isinstance(get_llm_client(), OpenRouterClient)


def test_use_llm_client_restores_after_an_exception():
    scripted = ScriptedLLM()
    with pytest.raises(ValueError):
        with use_llm_client(scripted):
            raise ValueError("boom")
    assert isinstance(get_llm_client(), OpenRouterClient)


async def test_scripted_llm_replays_in_order():
    scripted = ScriptedLLM([text_response("first"), text_response("second")])

    one = await scripted.complete(model="m", messages=[{"role": "user", "content": "a"}])
    two = await scripted.complete(model="m", messages=[{"role": "user", "content": "b"}])

    assert one["choices"][0]["message"]["content"] == "first"
    assert two["choices"][0]["message"]["content"] == "second"
    assert scripted.call_count == 2


async def test_scripted_llm_records_the_request():
    scripted = ScriptedLLM([text_response("ok")])
    tools = [{"type": "function", "function": {"name": "send_message_to_user"}}]

    await scripted.complete(
        model="test/interaction",
        messages=[{"role": "user", "content": "hello"}],
        system="SYSTEM",
        tools=tools,
    )

    call = scripted.last_request()
    assert call.model == "test/interaction"
    assert call.system == "SYSTEM"
    assert call.tool_names_offered == ["send_message_to_user"]
    assert call.prompt_chars == len("SYSTEM") + len("hello")


async def test_exhausted_script_raises_a_useful_assertion():
    scripted = ScriptedLLM([text_response("only one")])
    await scripted.complete(model="m", messages=[])

    with pytest.raises(ScriptExhausted) as excinfo:
        await scripted.complete(model="m", messages=[])

    # The message has to say which call ran dry, or debugging a long agent loop
    # becomes guesswork.
    assert "call #2" in str(excinfo.value)


async def test_handler_covers_unscripted_calls():
    scripted = ScriptedLLM(
        [text_response("scripted")],
        handler=lambda call: text_response(f"handled {call.model}"),
    )

    first = await scripted.complete(model="m", messages=[])
    second = await scripted.complete(model="other", messages=[])

    assert first["choices"][0]["message"]["content"] == "scripted"
    assert second["choices"][0]["message"]["content"] == "handled other"


async def test_per_model_queues_route_by_substring():
    scripted = ScriptedLLM()
    scripted.queue_for("interaction", text_response("from interaction"))
    scripted.queue_for("execution", text_response("from execution"))

    execution = await scripted.complete(model="test/execution", messages=[])
    interaction = await scripted.complete(model="test/interaction", messages=[])

    assert execution["choices"][0]["message"]["content"] == "from execution"
    assert interaction["choices"][0]["message"]["content"] == "from interaction"


def test_tool_call_response_matches_the_openrouter_shape():
    payload = tool_call_response(
        ("send_message_to_agent", {"agent_name": "Email to Alice", "instructions": "draft it"}),
        ("send_message_to_agent", {"agent_name": "Email to Bob", "instructions": "draft it"}),
    )

    calls = payload["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == [
        "send_message_to_agent",
        "send_message_to_agent",
    ]
    # Arguments arrive as a JSON *string*, exactly as the API delivers them --
    # the runtimes' parsing code depends on that.
    assert json.loads(calls[0]["function"]["arguments"])["agent_name"] == "Email to Alice"
    assert calls[0]["id"] != calls[1]["id"]
