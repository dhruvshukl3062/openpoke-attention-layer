"""The attention broker: one place that decides what reaches the user.

Upstream, anything that wants the user's attention gets it immediately and
independently. The email watcher classifies each new message and calls
``handle_agent_message`` per important one. The trigger scheduler fires on its
own timer. Completed execution batches dispatch on their own. Nothing dedupes,
nothing batches across sources, nothing knows what time it is where the user
is, and nothing caps how often this can happen. Ten interesting emails at 7am is
ten separate interruptions and ten model calls.

Every source now publishes a :class:`Candidate` here instead, and the broker
decides: interrupt now, hold for the digest, or suppress.

Two design choices worth stating plainly:

**Scoring is not done here.** A candidate arrives carrying an ``urgency`` its
source already determined -- the email classifier's judgement, a trigger's
configured priority. The broker turns urgency into a *routing* decision using
policy. Keeping the model's judgement separate from the policy that acts on it
means each can be tested and evaluated on its own, and the broker itself stays
fully deterministic.

**Time is injected and flushing is explicit.** ``submit`` queues and decides;
``flush`` delivers whatever the coalescing window has gathered. Nothing sleeps.
Production drives it from a timer; the evaluation harness drives it from a
virtual clock, which is what makes a simulated month run in seconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from ...logging_config import logger


class Route(str, Enum):
    """What the broker decided to do with a candidate."""

    INTERRUPT = "interrupt"
    DIGEST = "digest"
    SUPPRESS = "suppress"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Candidate:
    """Something that would like the user's attention.

    ``key`` is the dedupe identity -- an email thread id, a trigger id, an agent
    name. Two candidates sharing a key inside the dedupe window are the same
    event arriving twice, not two events.
    """

    source: str
    key: str
    summary: str
    urgency: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None

    def fingerprint(self) -> str:
        """Identity for dedupe: the key plus the gist of the text.

        Including the text means a genuinely updated status on the same thread
        still gets through, while a verbatim repeat does not.
        """

        normalized = " ".join(self.summary.lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{self.key}:{digest}"


@dataclass(frozen=True)
class Decision:
    candidate: Candidate
    route: Route
    reason: str

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"[{self.route.value}] {self.candidate.key}: {self.reason}"


@dataclass
class Policy:
    """The knobs. Every one of these is a product decision, not a technical one."""

    #: How long to gather candidates before delivering, so things arriving
    #: together become one message. The cost is latency; the benefit is quiet.
    coalesce_seconds: float = 90.0

    #: At or above this urgency, interrupt now.
    interrupt_threshold: float = 0.7

    #: Below this, never surface at all.
    digest_threshold: float = 0.25

    #: Interruptions allowed in any rolling hour.
    max_interrupts_per_hour: int = 4

    #: Urgency high enough to override the budget and quiet hours. Without this,
    #: a quiet period could bury something that genuinely cannot wait -- the
    #: failure mode that matters most.
    escalation_threshold: float = 0.9

    #: Local-time window in which only escalations get through.
    quiet_start: time = time(22, 0)
    quiet_end: time = time(7, 0)

    #: A repeat of the same fingerprint inside this window is dropped.
    dedupe_window: timedelta = timedelta(hours=6)

    #: Skip quiet hours entirely (useful in tests and for users who opt out).
    quiet_hours_enabled: bool = True


Deliver = Callable[[str], Awaitable[None]]


class AttentionBroker:
    """Dedupe, coalesce, route and budget everything that wants the user."""

    def __init__(
        self,
        *,
        deliver: Deliver,
        policy: Optional[Policy] = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        local_now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._deliver = deliver
        self.policy = policy or Policy()
        self._now = now
        #: Quiet hours are a wall-clock question in the user's timezone, which is
        #: not the same as UTC. Injected separately so tests can pin it.
        self._local_now = local_now or now
        self._seen: Dict[str, datetime] = {}
        self._pending_interrupts: List[Candidate] = []
        self._held: List[Candidate] = []
        self._interrupt_times: List[datetime] = []
        self._window_opened_at: Optional[datetime] = None
        self.decisions: List[Decision] = []

    # -- intake ------------------------------------------------------------

    def submit(self, candidate: Candidate) -> Decision:
        """Route a candidate. Does not deliver -- see :meth:`flush`."""

        decision = self._decide(candidate)
        self.decisions.append(decision)

        if decision.route is Route.INTERRUPT:
            self._pending_interrupts.append(candidate)
            if self._window_opened_at is None:
                self._window_opened_at = self._now()
        elif decision.route is Route.DIGEST:
            self._held.append(candidate)

        return decision

    def _decide(self, candidate: Candidate) -> Decision:
        now = self._now()

        # 1. Duplicate?
        fingerprint = candidate.fingerprint()
        last_seen = self._seen.get(fingerprint)
        if last_seen is not None and now - last_seen < self.policy.dedupe_window:
            return Decision(candidate, Route.DUPLICATE, "already surfaced recently")
        self._seen[fingerprint] = now

        # 2. Too trivial to ever surface?
        if candidate.urgency < self.policy.digest_threshold:
            return Decision(candidate, Route.SUPPRESS, "below the digest threshold")

        # 3. Not urgent enough to interrupt for?
        if candidate.urgency < self.policy.interrupt_threshold:
            return Decision(candidate, Route.DIGEST, "worth knowing, not worth interrupting")

        escalating = candidate.urgency >= self.policy.escalation_threshold

        # 4. Quiet hours -- escalations still get through.
        if self._in_quiet_hours() and not escalating:
            return Decision(candidate, Route.DIGEST, "quiet hours")

        # 5. Interruption budget -- escalations still get through.
        if not self._budget_available(now) and not escalating:
            return Decision(candidate, Route.DIGEST, "hourly interruption budget spent")

        return Decision(candidate, Route.INTERRUPT, "urgent enough to interrupt")

    # -- policy helpers ----------------------------------------------------

    def _in_quiet_hours(self) -> bool:
        if not self.policy.quiet_hours_enabled:
            return False
        current = self._local_now().time()
        start, end = self.policy.quiet_start, self.policy.quiet_end
        if start <= end:
            return start <= current < end
        # Window wraps midnight (the usual case: 22:00 -> 07:00).
        return current >= start or current < end

    def _budget_available(self, now: datetime) -> bool:
        cutoff = now - timedelta(hours=1)
        self._interrupt_times = [t for t in self._interrupt_times if t > cutoff]
        return len(self._interrupt_times) < self.policy.max_interrupts_per_hour

    # -- delivery ----------------------------------------------------------

    def window_is_open(self) -> bool:
        return self._window_opened_at is not None

    def window_elapsed(self) -> bool:
        if self._window_opened_at is None:
            return False
        age = (self._now() - self._window_opened_at).total_seconds()
        return age >= self.policy.coalesce_seconds

    async def flush(self, *, force: bool = False) -> Optional[str]:
        """Deliver the coalesced batch if its window has closed.

        Returns the delivered payload, or None when there was nothing to send.
        """

        if not self._pending_interrupts:
            self._window_opened_at = None
            return None
        if not force and not self.window_elapsed():
            return None

        batch = self._pending_interrupts
        self._pending_interrupts = []
        self._window_opened_at = None
        self._interrupt_times.append(self._now())

        payload = self._render(batch)
        try:
            await self._deliver(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Attention broker delivery failed: {exc}")
        logger.info(f"Broker delivered {len(batch)} coalesced item(s)")
        return payload

    async def flush_digest(self) -> Optional[str]:
        """Deliver everything held back, as one digest, and clear the queue."""

        if not self._held:
            return None
        batch = self._held
        self._held = []

        lines = [f"Digest of {len(batch)} item(s) held back:"]
        lines.extend(self._render_lines(batch))
        payload = "\n".join(lines)
        try:
            await self._deliver(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Attention broker digest delivery failed: {exc}")
        logger.info(f"Broker delivered a digest of {len(batch)} item(s)")
        return payload

    def held_count(self) -> int:
        return len(self._held)

    def _render(self, batch: Sequence[Candidate]) -> str:
        if len(batch) == 1:
            return batch[0].summary
        lines = [f"{len(batch)} things came in together:"]
        lines.extend(self._render_lines(batch))
        return "\n".join(lines)

    def _render_lines(self, batch: Sequence[Candidate]) -> List[str]:
        # Most urgent first: if the user only reads the top line, it should be
        # the one that mattered.
        ordered = sorted(batch, key=lambda item: -item.urgency)
        return [f"- [{item.source}] {item.summary}" for item in ordered]


__all__ = ["AttentionBroker", "Candidate", "Decision", "Policy", "Route"]
