"""The attention broker: dedupe, coalesce, route, budget, quiet hours.

Entirely deterministic -- no model, no sleeping, injected clocks. Every
assertion is a policy claim the evaluation harness later measures at scale.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import List

import pytest

from server.services.attention.broker import (
    AttentionBroker,
    Candidate,
    Policy,
    Route,
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
    # Mid-afternoon, comfortably outside quiet hours.
    return Clock(datetime(2026, 9, 7, 14, 0, tzinfo=timezone.utc))


@pytest.fixture
def delivered() -> List[str]:
    return []


@pytest.fixture
def broker(clock, delivered) -> AttentionBroker:
    async def deliver(payload: str) -> None:
        delivered.append(payload)

    return AttentionBroker(deliver=deliver, now=clock, policy=Policy())


def candidate(key="thread-1", *, urgency=0.8, summary=None, source="email_watcher") -> Candidate:
    return Candidate(
        source=source,
        key=key,
        summary=summary or f"something happened on {key}",
        urgency=urgency,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_high_urgency_interrupts(broker):
    assert broker.submit(candidate(urgency=0.85)).route is Route.INTERRUPT


def test_middling_urgency_is_held_for_the_digest(broker):
    decision = broker.submit(candidate(urgency=0.5))

    assert decision.route is Route.DIGEST
    assert broker.held_count() == 1


def test_trivial_items_are_suppressed_entirely(broker):
    decision = broker.submit(candidate(urgency=0.1))

    assert decision.route is Route.SUPPRESS
    assert broker.held_count() == 0, "suppressed items must not leak into the digest"


def test_every_decision_carries_a_reason(broker):
    """Routing decisions are logged and evaluated, so they have to be explicable."""

    broker.submit(candidate(urgency=0.9))
    broker.submit(candidate(key="t2", urgency=0.4))

    assert all(decision.reason for decision in broker.decisions)


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def test_the_same_event_twice_is_dropped(broker):
    first = broker.submit(candidate())
    second = broker.submit(candidate())

    assert first.route is Route.INTERRUPT
    assert second.route is Route.DUPLICATE


def test_a_genuine_update_on_the_same_thread_still_gets_through(broker):
    broker.submit(candidate(summary="Alice replied"))
    second = broker.submit(candidate(summary="Alice replied again with the contract"))

    assert second.route is not Route.DUPLICATE


def test_dedupe_ignores_whitespace_and_case(broker):
    broker.submit(candidate(summary="Alice replied"))
    second = broker.submit(candidate(summary="  alice   REPLIED  "))

    assert second.route is Route.DUPLICATE


def test_dedupe_expires(broker, clock):
    broker.submit(candidate())
    clock.advance(hours=7)  # past the 6h window

    assert broker.submit(candidate()).route is not Route.DUPLICATE


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------

async def test_nothing_is_delivered_before_the_window_closes(broker, clock, delivered):
    broker.submit(candidate(urgency=0.8))

    assert await broker.flush() is None
    assert delivered == []

    clock.advance(seconds=91)
    assert await broker.flush() is not None
    assert len(delivered) == 1


async def test_items_arriving_together_become_one_message(broker, clock, delivered):
    broker.submit(candidate("thread-1", urgency=0.8, summary="Alice replied"))
    clock.advance(seconds=20)
    broker.submit(candidate("thread-2", urgency=0.9, summary="Bob sent the contract"))
    clock.advance(seconds=80)

    await broker.flush()

    assert len(delivered) == 1, "two events, one interruption"
    assert "Alice replied" in delivered[0]
    assert "Bob sent the contract" in delivered[0]


async def test_the_most_urgent_item_is_listed_first(broker, clock, delivered):
    broker.submit(candidate("t1", urgency=0.75, summary="lower"))
    broker.submit(candidate("t2", urgency=0.95, summary="higher"))
    clock.advance(seconds=91)

    await broker.flush()

    assert delivered[0].index("higher") < delivered[0].index("lower")


async def test_a_lone_item_is_delivered_without_batch_framing(broker, clock, delivered):
    broker.submit(candidate(urgency=0.8, summary="Alice replied"))
    clock.advance(seconds=91)

    await broker.flush()

    assert delivered[0] == "Alice replied", "one item should not be dressed up as a list"


async def test_flush_with_nothing_pending_is_a_no_op(broker, delivered):
    assert await broker.flush() is None
    assert delivered == []


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

async def test_the_hourly_budget_pushes_overflow_into_the_digest(broker, clock):
    for index in range(broker.policy.max_interrupts_per_hour):
        broker.submit(candidate(f"t{index}", urgency=0.8))
        clock.advance(seconds=91)
        await broker.flush()

    overflow = broker.submit(candidate("overflow", urgency=0.8))

    assert overflow.route is Route.DIGEST
    assert "budget" in overflow.reason


async def test_the_budget_is_a_rolling_hour(broker, clock):
    for index in range(broker.policy.max_interrupts_per_hour):
        broker.submit(candidate(f"t{index}", urgency=0.8))
        clock.advance(seconds=91)
        await broker.flush()

    clock.advance(hours=1, seconds=1)

    assert broker.submit(candidate("later", urgency=0.8)).route is Route.INTERRUPT


async def test_an_escalation_breaks_through_a_spent_budget(broker, clock):
    """The failure that matters most: never bury something that cannot wait."""

    for index in range(broker.policy.max_interrupts_per_hour):
        broker.submit(candidate(f"t{index}", urgency=0.8))
        clock.advance(seconds=91)
        await broker.flush()

    urgent = broker.submit(candidate("emergency", urgency=0.95))

    assert urgent.route is Route.INTERRUPT


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------

def _night_broker(delivered, local_hour=23):
    clock = Clock(datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc))
    local = Clock(datetime(2026, 9, 7, local_hour, 0, tzinfo=timezone.utc))

    async def deliver(payload: str) -> None:
        delivered.append(payload)

    return AttentionBroker(deliver=deliver, now=clock, local_now=local)


def test_quiet_hours_hold_ordinary_interruptions(delivered):
    broker = _night_broker(delivered, local_hour=23)

    decision = broker.submit(candidate(urgency=0.8))

    assert decision.route is Route.DIGEST
    assert decision.reason == "quiet hours"


def test_quiet_hours_wrap_midnight(delivered):
    """22:00-07:00 spans midnight, which naive comparisons get wrong."""

    broker = _night_broker(delivered, local_hour=3)

    assert broker.submit(candidate(urgency=0.8)).route is Route.DIGEST


def test_daytime_is_not_quiet(delivered):
    broker = _night_broker(delivered, local_hour=14)

    assert broker.submit(candidate(urgency=0.8)).route is Route.INTERRUPT


def test_an_escalation_breaks_through_quiet_hours(delivered):
    broker = _night_broker(delivered, local_hour=3)

    assert broker.submit(candidate(urgency=0.95)).route is Route.INTERRUPT


def test_quiet_hours_can_be_disabled(delivered):
    broker = _night_broker(delivered, local_hour=3)
    broker.policy.quiet_hours_enabled = False

    assert broker.submit(candidate(urgency=0.8)).route is Route.INTERRUPT


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

async def test_the_digest_collects_held_items_and_clears(broker, delivered):
    broker.submit(candidate("t1", urgency=0.4, summary="newsletter arrived"))
    broker.submit(candidate("t2", urgency=0.5, summary="calendar invite"))

    payload = await broker.flush_digest()

    assert "newsletter arrived" in payload and "calendar invite" in payload
    assert broker.held_count() == 0
    assert len(delivered) == 1


async def test_an_empty_digest_delivers_nothing(broker, delivered):
    assert await broker.flush_digest() is None
    assert delivered == [], "silence is a valid outcome; never send an empty digest"


# ---------------------------------------------------------------------------
# The headline property
# ---------------------------------------------------------------------------

async def test_a_burst_of_email_becomes_one_interruption(broker, clock, delivered):
    """The morning-inbox scenario, which upstream turns into ten pings.

    Ten emails arrive together. Two are genuinely urgent, five are worth
    knowing, three are noise.
    """

    for index in range(2):
        broker.submit(candidate(f"urgent-{index}", urgency=0.85))
    for index in range(5):
        broker.submit(candidate(f"middling-{index}", urgency=0.5))
    for index in range(3):
        broker.submit(candidate(f"noise-{index}", urgency=0.1))

    clock.advance(seconds=91)
    await broker.flush()

    assert len(delivered) == 1, "one interruption, not ten"
    assert broker.held_count() == 5, "the middling ones wait for the digest"
