"""Batching behaviour of ``ExecutionBatchManager``.

The manager exists so that a fan-out ("email Alice and Bob") produces one reply
to the user rather than two. It does that by holding results until the batch
drains. The problem is that a batch is a single piece of global state with no
notion of which user turn opened it, so unrelated work joins whatever batch
happens to be open.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from server.agents.execution_agent import batch_manager as batch_module
from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult


class _FakeRuntime:
    """Stands in for a real execution agent, with controllable duration."""

    delays = {}

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def execute(self, instructions: str) -> ExecutionResult:
        await asyncio.sleep(self.delays.get(self.agent_name, 0))
        return ExecutionResult(
            agent_name=self.agent_name,
            success=True,
            response=f"{self.agent_name} done",
            tools_executed=[],
        )


@pytest.fixture
def captured_dispatches(monkeypatch) -> List[str]:
    """Record what the manager sends on to the interaction agent."""

    dispatched: List[str] = []

    async def _capture(self, payload: str) -> None:
        dispatched.append(payload)

    monkeypatch.setattr(batch_module, "ExecutionAgentRuntime", _FakeRuntime)
    monkeypatch.setattr(
        ExecutionBatchManager, "_dispatch_to_interaction_agent", _capture, raising=True
    )
    _FakeRuntime.delays = {}
    return dispatched


async def test_a_fan_out_is_delivered_as_one_message(captured_dispatches):
    """The behaviour the manager is *for*: two agents, one reply."""

    manager = ExecutionBatchManager()

    await asyncio.gather(
        manager.execute_agent("Email to Alice", "draft lunch"),
        manager.execute_agent("Email to Bob", "draft lunch"),
    )

    assert len(captured_dispatches) == 1
    payload = captured_dispatches[0]
    assert "Email to Alice" in payload and "Email to Bob" in payload
    assert payload.count("[SUCCESS]") == 2


async def test_a_failed_agent_still_reports(captured_dispatches):
    manager = ExecutionBatchManager(timeout_seconds=0.05)
    _FakeRuntime.delays = {"Slow Agent": 5}

    result = await manager.execute_agent("Slow Agent", "something long")

    assert not result.success
    assert result.error == "Timeout"
    assert "[FAILED]" in captured_dispatches[0]


@pytest.mark.known_bug
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Batch state is a single global with no turn identity: "
        "_register_pending_execution joins whatever batch is currently open. "
        "Work from a later user turn is absorbed into an earlier turn's batch "
        "and withheld until the slowest agent in it finishes."
    ),
)
async def test_a_later_turn_does_not_get_swallowed_by_an_open_batch(captured_dispatches):
    """Two logically separate turns should produce two deliveries.

    Scenario: the user asks for something slow, then sends an unrelated second
    message while the first is still running. The second answer should come back
    on its own, not wait behind the first and arrive glued to it.
    """

    manager = ExecutionBatchManager()
    _FakeRuntime.delays = {"Slow Research": 0.2}

    slow = asyncio.create_task(manager.execute_agent("Slow Research", "turn one"))
    await asyncio.sleep(0.02)  # the user sends a second message

    await manager.execute_agent("Quick Lookup", "turn two")

    # The quick turn is finished; its answer should already be on its way.
    assert len(captured_dispatches) == 1, (
        "the fast, unrelated result should dispatch on its own rather than "
        "being held behind the slow one"
    )

    await slow
    assert len(captured_dispatches) == 2


# The second batching defect -- TriggerScheduler building a fresh
# ExecutionBatchManager per trigger, so simultaneous reminders never batch --
# is no longer user-visible: every result now passes through the attention
# broker, which coalesces across sources. See
# tests/test_attention_wiring.py::test_simultaneous_triggers_reach_the_user_once.
# The underlying duplication remains and is documented in the README.
