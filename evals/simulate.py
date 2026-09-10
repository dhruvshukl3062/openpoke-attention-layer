"""Run a configuration against a simulated month and measure what happened.

The baseline is not a separate implementation: it is the broker with every
policy disabled, which reproduces upstream's behaviour exactly (interrupt on
anything the classifier flags, immediately, with no dedupe, budget or quiet
hours). One code path, config-driven, so ablations fall out for free -- and the
comparison can't drift because there is only one thing being compared.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from server.services.attention.broker import AttentionBroker, Candidate, Policy, Route
from server.services.attention.ranking import find_duplicate, shortlist
from server.services.attention.registry import AgentRecord
from server.services.attention.scoring import urgency_for_email

from .world import Email, SimulatedClassifier, World

#: Local timezone offset used for quiet-hours decisions in the simulation.
LOCAL_OFFSET = timedelta(hours=-4)


def baseline_policy() -> Policy:
    """Upstream behaviour, expressed as policy.

    Everything the classifier flags interrupts, immediately, forever.
    """

    return Policy(
        coalesce_seconds=0.0,
        interrupt_threshold=0.0,
        digest_threshold=0.0,
        max_interrupts_per_hour=10**9,
        quiet_hours_enabled=False,
        dedupe_window=timedelta(0),
    )


def treatment_policy() -> Policy:
    return Policy()


@dataclass
class Metrics:
    config: str
    days: int
    interrupts: int = 0
    interrupt_events: int = 0
    items_in_interrupts: int = 0
    digest_items: int = 0
    suppressed: int = 0
    duplicates_dropped: int = 0
    urgent_total: int = 0
    urgent_surfaced: int = 0
    urgent_delays: List[float] = field(default_factory=list)
    #: Urgent items surfaced within an hour. Plain recall counts a digest
    #: delivery the next morning as a success, which flatters the system --
    #: for something genuinely urgent, 20 hours late is a miss.
    urgent_surfaced_within_hour: int = 0
    true_positives: int = 0
    false_positives: int = 0
    prompt_chars_per_turn: float = 0.0
    duplicate_agents: int = 0
    agents_created: int = 0

    @property
    def interrupts_per_day(self) -> float:
        return self.interrupt_events / self.days if self.days else 0.0

    @property
    def precision(self) -> float:
        surfaced = self.true_positives + self.false_positives
        return self.true_positives / surfaced if surfaced else 0.0

    @property
    def urgent_recall(self) -> float:
        return self.urgent_surfaced / self.urgent_total if self.urgent_total else 0.0

    @property
    def timely_urgent_recall(self) -> float:
        """The metric to lead with: urgent mail surfaced while it still mattered."""

        return (
            self.urgent_surfaced_within_hour / self.urgent_total
            if self.urgent_total
            else 0.0
        )

    @property
    def median_urgent_delay_minutes(self) -> float:
        return statistics.median(self.urgent_delays) / 60 if self.urgent_delays else 0.0

    @property
    def p95_urgent_delay_minutes(self) -> float:
        if not self.urgent_delays:
            return 0.0
        ordered = sorted(self.urgent_delays)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index] / 60

    def as_row(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "interrupts/day": round(self.interrupts_per_day, 2),
            "precision@interrupt": round(self.precision, 3),
            "urgent recall": round(self.urgent_recall, 3),
            "timely urgent recall": round(self.timely_urgent_recall, 3),
            "median urgent delay (min)": round(self.median_urgent_delay_minutes, 1),
            "p95 urgent delay (min)": round(self.p95_urgent_delay_minutes, 1),
            "digest items": self.digest_items,
            "dupes dropped": self.duplicates_dropped,
            "prompt chars/turn": round(self.prompt_chars_per_turn),
            "duplicate agents": self.duplicate_agents,
        }


class _Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


async def run_attention_sim(
    world: World,
    policy: Policy,
    *,
    config_name: str,
    dedupe: bool = True,
) -> Metrics:
    """Replay the world's inbox through a broker and record what the user got."""

    metrics = Metrics(config=config_name, days=world.days)
    classifier = SimulatedClassifier(seed=world.seed)

    clock = _Clock(world.emails[0].arrived_at if world.emails else datetime.now(timezone.utc))
    local_clock = _Clock(clock.now + LOCAL_OFFSET)

    delivered: List[tuple[datetime, List[Candidate]]] = []

    async def deliver(payload: str) -> None:
        delivered.append((clock.now, list(in_flight)))

    broker = AttentionBroker(
        deliver=deliver, policy=policy, now=clock, local_now=local_clock
    )

    if not dedupe:
        broker.policy.dedupe_window = timedelta(0)

    in_flight: List[Candidate] = []
    pending: Dict[str, tuple[Email, Candidate]] = {}
    surfaced_at: Dict[str, datetime] = {}
    #: Digest deliveries are a surface too, just not an interruption.
    digested_at: Dict[str, datetime] = {}
    last_digest_date = None

    DIGEST_HOUR = 8

    async def _run_digest(at: datetime) -> None:
        nonlocal last_digest_date
        clock.now = at
        local_clock.now = at + LOCAL_OFFSET
        held = list(broker.held_items())
        if held:
            await broker.flush_digest()
            for candidate in held:
                digested_at.setdefault(candidate.key, at)
        last_digest_date = local_clock.now.date()

    async def advance_to(moment: datetime) -> None:
        """Move the virtual clock, flushing anything due on the way.

        Both a closed coalescing window and the daily digest can fall between
        two arrivals, so both have to be checked as time is advanced rather
        than only at the end.
        """

        # Daily digest boundaries crossed since the last advance.
        while True:
            local_target = moment + LOCAL_OFFSET
            candidate_date = local_clock.now.date()
            boundary_local = datetime.combine(
                candidate_date, datetime.min.time(), tzinfo=timezone.utc
            ) + timedelta(hours=DIGEST_HOUR)
            if last_digest_date == candidate_date or boundary_local < local_clock.now:
                boundary_local += timedelta(days=1)
            if boundary_local > local_target:
                break
            await _run_digest(boundary_local - LOCAL_OFFSET)

        broker.promote_stale()

        while broker.window_is_open():
            due = broker.window_due_at()
            if due is None or due > moment:
                break
            clock.now = due
            local_clock.now = due + LOCAL_OFFSET
            in_flight[:] = broker.pending_interrupts()
            payload = await broker.flush()
            if payload is not None:
                for candidate in in_flight:
                    surfaced_at.setdefault(candidate.key, clock.now)
                metrics.interrupt_events += 1
                metrics.items_in_interrupts += len(in_flight)
            in_flight.clear()
        clock.now = moment
        local_clock.now = moment + LOCAL_OFFSET

    for email in world.emails:
        await advance_to(email.arrived_at)

        if email.truly_urgent:
            metrics.urgent_total += 1

        if not classifier.is_important(email):
            continue

        candidate = Candidate(
            source="email_watcher",
            key=email.thread_id or email.id,
            summary=email.subject,
            urgency=urgency_for_email(email.subject, classified_important=True),
            created_at=email.arrived_at,
        )
        pending[email.id] = (email, candidate)
        decision = broker.submit(candidate)

        if decision.route is Route.DUPLICATE:
            metrics.duplicates_dropped += 1
        elif decision.route is Route.SUPPRESS:
            metrics.suppressed += 1
        elif decision.route is Route.DIGEST:
            metrics.digest_items += 1

    # Close out anything still in flight at the end of the run.
    if world.emails:
        await advance_to(world.emails[-1].arrived_at + timedelta(hours=2))
    in_flight[:] = broker.pending_interrupts()
    payload = await broker.flush(force=True)
    if payload is not None:
        for candidate in in_flight:
            surfaced_at.setdefault(candidate.key, clock.now)
        metrics.interrupt_events += 1
        metrics.items_in_interrupts += len(in_flight)
    in_flight.clear()

    # Anything still held at the end of the run is delivered, so nothing is
    # counted as permanently lost purely because the simulation stopped.
    await _run_digest(clock.now)

    for email_id, (email, candidate) in pending.items():
        key = candidate.key
        interrupted = key in surfaced_at
        digested = key in digested_at

        if interrupted:
            if email.truly_urgent:
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1

        if email.truly_urgent and (interrupted or digested):
            metrics.urgent_surfaced += 1
            when = surfaced_at.get(key) or digested_at.get(key, clock.now)
            delay = (when - email.arrived_at).total_seconds()
            metrics.urgent_delays.append(delay)
            if delay <= 3600:
                metrics.urgent_surfaced_within_hour += 1

    metrics.interrupts = metrics.items_in_interrupts
    return metrics


def measure_context_cost(agent_count: int, *, shortlist_size: Optional[int]) -> float:
    """Characters of roster context per turn.

    ``shortlist_size=None`` models upstream, which renders every agent.
    """

    now = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    records = [
        AgentRecord(
            # Fixed-width names: otherwise the measurement picks up label length
            # drift rather than roster size.
            name=f"Email to {name}",
            purpose="ongoing thread",
            entities=(name,),
            created_at=now,
            last_used_at=now - timedelta(hours=index % 72),
        )
        for index, name in enumerate(
            (f"Person{i:04d}" for i in range(agent_count))
        )
    ]

    if shortlist_size is None:
        rendered = "\n".join(f'<agent name="{record.name}" />' for record in records)
    else:
        picked = shortlist(records, "follow up with Person0007", limit=shortlist_size, now=now)
        rendered = "\n".join(
            f'<agent name="{item.record.name}" purpose="{item.record.purpose}" '
            f'last_used="recent" />'
            for item in picked
        )
    return float(len(rendered))


def measure_duplicate_agents(world: World, *, dedupe: bool) -> tuple[int, int]:
    """How many redundant agents accumulate over the world's user requests.

    Models the interaction agent naming an agent per request. Without dedupe,
    a reworded name for the same person creates a second agent.
    """

    records: List[AgentRecord] = []
    now = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    duplicates = 0

    phrasings = ["Email to {}", "{} Follow Up", "Message {}", "{} thread"]

    for index, request in enumerate(world.requests):
        proposed = phrasings[index % len(phrasings)].format(request.subject)

        if any(record.name == proposed for record in records):
            continue

        match = find_duplicate(records, proposed) if dedupe else None
        if match is not None:
            continue

        if any(request.subject in record.entities for record in records):
            duplicates += 1

        records.append(
            AgentRecord(
                name=proposed,
                entities=(request.subject,),
                created_at=now,
                last_used_at=now,
            )
        )

    return duplicates, len(records)


__all__ = [
    "Metrics",
    "baseline_policy",
    "measure_context_cost",
    "measure_duplicate_agents",
    "run_attention_sim",
    "treatment_policy",
]
