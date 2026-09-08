"""End-to-end: every interrupt source now reaches the user through the broker.

The unit tests in ``test_broker.py`` prove the policy. These prove the wiring --
that the email watcher, execution batches and fired triggers actually go through
it rather than around it, which is the part that would silently regress.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from server.agents.execution_agent import batch_manager as batch_module
from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult
from server.services.attention import service as attention_service
from server.services.attention.broker import AttentionBroker, Policy, Route
from server.services.attention.scoring import (
    urgency_for_email,
    urgency_for_execution_result,
    urgency_for_trigger,
)


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock(datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc))


@pytest.fixture
def delivered(monkeypatch, clock) -> List[str]:
    """Install a broker whose deliveries we can see, in place of the singleton."""

    payloads: List[str] = []

    async def deliver(payload: str) -> None:
        payloads.append(payload)

    broker = AttentionBroker(
        deliver=deliver,
        policy=Policy(),
        now=clock,
        local_now=clock,  # 14:00, outside quiet hours
    )
    monkeypatch.setattr(attention_service, "_broker", broker)
    return payloads


class _FakeRuntime:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def execute(self, instructions: str) -> ExecutionResult:
        return ExecutionResult(
            agent_name=self.agent_name,
            success=True,
            response=f"{self.agent_name} finished",
            tools_executed=[],
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_an_urgent_email_outranks_a_routine_one():
    urgent = urgency_for_email("Your policy renewal expires today - action required")
    routine = urgency_for_email("Weekly newsletter: unsubscribe at any time")

    assert urgent > routine
    assert urgent >= Policy().interrupt_threshold
    assert routine < Policy().interrupt_threshold


def test_an_email_the_classifier_declined_scores_low_but_not_zero():
    """It should reach a digest, not vanish -- the classifier can be wrong."""

    score = urgency_for_email("some ordinary update", classified_important=False)

    assert 0 < score < Policy().interrupt_threshold


def test_a_recurring_digest_trigger_does_not_punch_through_quiet_hours():
    daily = urgency_for_trigger("Send the daily summary")
    urgent = urgency_for_trigger("URGENT: renewal deadline")

    assert daily < Policy().escalation_threshold
    assert urgent >= Policy().escalation_threshold


def test_a_failed_execution_outranks_a_routine_success():
    failure = urgency_for_execution_result("could not reach Gmail", success=False)
    success = urgency_for_execution_result("drafted the reply", success=True)

    assert failure > success


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------

async def test_execution_results_reach_the_user_through_the_broker(delivered, clock, monkeypatch):
    monkeypatch.setattr(batch_module, "ExecutionAgentRuntime", _FakeRuntime)
    manager = ExecutionBatchManager()

    await manager.execute_agent("Email to Alice", "draft lunch")

    assert delivered == [], "nothing should be delivered before the window closes"

    clock.advance(seconds=91)
    await attention_service.get_attention_broker().flush()

    assert len(delivered) == 1
    assert "Email to Alice" in delivered[0]


async def test_simultaneous_triggers_reach_the_user_once(delivered, clock, monkeypatch):
    """Three reminders firing in the same tick used to mean three interruptions.

    The underlying cause -- TriggerScheduler building a fresh batch manager per
    trigger -- is untouched. Fixing it at the broker means the user-visible
    behaviour is right regardless of how many batch managers exist upstream,
    which is the more robust place for the fix.
    """

    monkeypatch.setattr(batch_module, "ExecutionAgentRuntime", _FakeRuntime)

    await asyncio.gather(
        *[
            ExecutionBatchManager().execute_agent(f"Reminder {index}", "fire")
            for index in range(3)
        ]
    )

    clock.advance(seconds=91)
    await attention_service.get_attention_broker().flush()

    assert len(delivered) == 1, (
        f"three simultaneous triggers produced {len(delivered)} interruptions"
    )
    for index in range(3):
        assert f"Reminder {index}" in delivered[0]


async def test_the_email_watcher_submits_instead_of_interrupting(delivered, clock):
    from server.services.gmail.importance_watcher import ImportantEmailWatcher

    watcher = ImportantEmailWatcher()
    await watcher._dispatch_summary(
        "Renewal expires today - action required", email_id="msg-1"
    )

    broker = attention_service.get_attention_broker()
    assert broker.decisions[-1].route is Route.INTERRUPT
    assert delivered == [], "the broker decides when, not the watcher"

    clock.advance(seconds=91)
    await broker.flush()
    assert len(delivered) == 1


async def test_a_repeated_email_notification_is_dropped(delivered, clock):
    from server.services.gmail.importance_watcher import ImportantEmailWatcher

    watcher = ImportantEmailWatcher()
    await watcher._dispatch_summary("Alice replied about lunch", email_id="msg-1")
    await watcher._dispatch_summary("Alice replied about lunch", email_id="msg-1")

    broker = attention_service.get_attention_broker()
    assert broker.decisions[-1].route is Route.DUPLICATE


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

async def test_the_loop_flushes_a_closed_window(delivered, clock, monkeypatch):
    from server.services.attention.service import AttentionLoop
    from server.services.attention.broker import Candidate

    monkeypatch.setattr(attention_service, "_local_now", clock)
    attention_service.submit_candidate(
        Candidate(
            source="email_watcher", key="k", summary="something", urgency=0.8,
            created_at=clock.now,
        )
    )

    loop = AttentionLoop()
    await loop.tick()
    assert delivered == []

    clock.advance(seconds=91)
    await loop.tick()
    assert len(delivered) == 1


async def test_the_digest_is_delivered_once_per_day(delivered, clock, monkeypatch):
    from server.services.attention.service import AttentionLoop
    from server.services.attention.broker import Candidate

    monkeypatch.setattr(attention_service, "_local_now", clock)
    attention_service.submit_candidate(
        Candidate(
            source="email_watcher", key="k1", summary="fyi", urgency=0.4,
            created_at=clock.now,
        )
    )

    loop = AttentionLoop(digest_hour=8)  # clock is at 14:00, so it is due
    await loop.tick()
    assert len(delivered) == 1 and "Digest" in delivered[0]

    attention_service.submit_candidate(
        Candidate(
            source="email_watcher", key="k2", summary="also fyi", urgency=0.4,
            created_at=clock.now,
        )
    )
    clock.advance(hours=2)
    await loop.tick()

    assert len(delivered) == 1, "the digest should not fire twice in one day"
