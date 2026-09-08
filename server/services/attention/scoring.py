"""Turning each source's signal into an urgency the broker can route on.

The broker deliberately does no scoring (see ``broker.py``): it receives an
urgency and applies policy. This module is where urgency comes from.

Everything here is rule-based and deterministic, for three reasons. It runs on
every inbound item, so a model call per candidate would be the most expensive
thing in the system. It is the part most likely to need tuning against real
data, and rules can be tuned by reading them. And a deterministic scorer means
the routing metrics in the evaluation harness measure the *broker's* behaviour
rather than sampling noise from a scorer.

The obvious next step is a small cheap model scoring the same inputs, validated
against the labels these rules produce. That swap is a one-line change at each
call site because the interface is just ``str -> float``.
"""

from __future__ import annotations

import re
from typing import Optional

#: Words that reliably indicate something cannot wait. Kept short on purpose --
#: a long list of weak signals produces confident-looking noise.
_CRITICAL = (
    "urgent", "asap", "emergency", "immediately", "critical", "expires today",
    "final notice", "action required", "time sensitive", "deadline today",
)

_ELEVATED = (
    "deadline", "tomorrow", "today", "eod", "end of day", "waiting on",
    "reminder", "follow up", "overdue", "please confirm", "needs your",
)

_ROUTINE = (
    "newsletter", "unsubscribe", "digest", "no-reply", "noreply",
    "notification", "receipt", "invoice attached", "promotion",
)


def _matches(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def urgency_for_email(summary: str, *, classified_important: bool = True) -> float:
    """Score an inbound email the watcher has already surfaced.

    The classifier's binary judgement is the prior; the text shifts it. An email
    the classifier declined is scored low but not zero, so it can still reach a
    digest rather than vanishing.
    """

    score = 0.72 if classified_important else 0.2

    if _matches(summary, _CRITICAL):
        score += 0.22
    elif _matches(summary, _ELEVATED):
        score += 0.08

    if _matches(summary, _ROUTINE):
        score -= 0.35

    # A direct question addressed to the user is a weak but real signal that a
    # reply is expected.
    if "?" in summary:
        score += 0.03

    return _clamp(score)


def urgency_for_trigger(payload: str) -> float:
    """Score a fired reminder.

    A trigger is something the user explicitly asked to be reminded about, so it
    starts above the interrupt threshold -- but a recurring digest-style trigger
    ("daily summary") should not punch through quiet hours.
    """

    score = 0.75

    if _matches(payload, _CRITICAL):
        score += 0.2
    if re.search(r"\b(daily|weekly|monthly)\b", payload, re.IGNORECASE):
        score -= 0.2

    return _clamp(score)


def urgency_for_execution_result(response: str, *, success: bool) -> float:
    """Score a finished execution agent's report.

    A failure the user needs to know about outranks a routine success. A
    successful result is usually the answer to something they asked for moments
    ago, so it should reach them promptly but need not break through quiet hours.
    """

    if not success:
        return 0.8

    score = 0.72
    if _matches(response, _CRITICAL):
        score += 0.15
    return _clamp(score)


def urgency_for(source: str, text: str, *, success: bool = True, important: bool = True) -> float:
    """Dispatch by source name, for call sites that only know the source string."""

    if source == "email_watcher":
        return urgency_for_email(text, classified_important=important)
    if source == "trigger":
        return urgency_for_trigger(text)
    if source == "execution_batch":
        return urgency_for_execution_result(text, success=success)
    return 0.5


__all__ = [
    "urgency_for",
    "urgency_for_email",
    "urgency_for_execution_result",
    "urgency_for_trigger",
]
