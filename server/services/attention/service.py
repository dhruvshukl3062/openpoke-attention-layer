"""Wiring: the process-wide broker, and the loop that drains it.

Sources call :func:`submit_candidate` instead of reaching for the interaction
agent directly. A background loop closes coalescing windows and delivers the
digest on a schedule.

Delivery still goes through ``InteractionAgentRuntime.handle_agent_message`` --
the broker changes *what* and *when*, never how the assistant speaks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone
from typing import Optional

from ...logging_config import logger
from ...utils.timezones import convert_to_user_timezone
from .broker import AttentionBroker, Candidate, Decision, Policy

#: How often the loop checks whether a coalescing window has closed. Well under
#: the window itself, so the effective delay is the window and not this.
TICK_SECONDS = 10.0

#: Local hour at which anything held back is delivered as one digest.
DIGEST_HOUR = 8


def _local_now() -> datetime:
    """The user's wall clock, which is what quiet hours are actually about."""

    try:
        return convert_to_user_timezone(datetime.now(timezone.utc))
    except Exception:  # pragma: no cover - defensive
        return datetime.now(timezone.utc)


async def _deliver_to_interaction_agent(payload: str) -> None:
    from ...agents.interaction_agent.runtime import InteractionAgentRuntime

    await InteractionAgentRuntime().handle_agent_message(payload)


_broker: Optional[AttentionBroker] = None


def get_attention_broker() -> AttentionBroker:
    global _broker
    if _broker is None:
        _broker = AttentionBroker(
            deliver=_deliver_to_interaction_agent,
            policy=Policy(),
            local_now=_local_now,
        )
    return _broker


def submit_candidate(candidate: Candidate) -> Decision:
    """Single entry point for anything that wants the user's attention."""

    decision = get_attention_broker().submit(candidate)
    logger.info(f"Attention broker: {decision}")
    return decision


class AttentionLoop:
    """Closes coalescing windows and delivers the daily digest."""

    def __init__(self, tick_seconds: float = TICK_SECONDS, digest_hour: int = DIGEST_HOUR):
        self._tick = tick_seconds
        self._digest_hour = digest_hour
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._last_digest_date = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            self._running = True
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="attention-loop"
            )
            logger.info("Attention loop started", extra={"tick_seconds": self._tick})

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
                logger.info("Attention loop stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                try:
                    await self.tick()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(f"Attention loop tick failed: {exc}")
                await asyncio.sleep(self._tick)
        except asyncio.CancelledError:
            raise

    async def tick(self) -> None:
        """One pass: flush a closed window, and the digest when it's due."""

        broker = get_attention_broker()
        broker.promote_stale()
        await broker.flush()

        local = _local_now()
        if local.hour >= self._digest_hour and self._last_digest_date != local.date():
            if broker.held_count():
                await broker.flush_digest()
            # Marked regardless, so an empty morning doesn't retry all day.
            self._last_digest_date = local.date()


_loop: Optional[AttentionLoop] = None


def get_attention_loop() -> AttentionLoop:
    global _loop
    if _loop is None:
        _loop = AttentionLoop()
    return _loop


__all__ = [
    "AttentionLoop",
    "DIGEST_HOUR",
    "TICK_SECONDS",
    "get_attention_broker",
    "get_attention_loop",
    "submit_candidate",
]
