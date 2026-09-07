"""Choosing which agents the orchestrator gets to see.

The interaction agent has to answer one question per turn: *is there an existing
agent that already owns this?* Upstream answers it by pasting every agent name
into the prompt and hoping the model infers the rest. That degrades in both
directions as the roster grows -- the prompt inflates without bound, and picking
the right name out of hundreds of bare strings gets harder, so duplicate agents
accumulate for the same subject.

Here the answer is a small ranked shortlist. Four signals, in order of how much
they are trusted:

1. **Entity match** -- the request names something this agent tracks (a person,
   an address, a thread). Strongest signal by far: "reply to Alice" should find
   the Alice agent even when no other word overlaps.
2. **Name overlap** -- tokens shared with the agent's own name.
3. **Lexical overlap** -- IDF-weighted tokens shared with purpose, tags and
   summary, so common words like "email" (which appear in most agents) count for
   little and distinctive ones count for a lot.
4. **Recency** -- exponential decay on last use. A tiebreaker, never a driver:
   weighting it heavily would make the most recent agent win everything.

No embeddings. For a few hundred short structured records, exact entity matching
plus IDF-weighted overlap outperforms vector similarity on the cases that
matter (names and identifiers), needs no model call, and can be explained line
by line when it gets something wrong. Semantic similarity earns its place when
the corpus is large and unstructured; this one is neither.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from .registry import AgentRecord

# Weights. Entity match dominates on purpose; recency only separates ties.
W_ENTITY = 3.0
W_NAME = 1.5
W_LEXICAL = 1.0
W_RECENCY = 0.5

#: Days after which a recency score has halved.
RECENCY_HALF_LIFE_DAYS = 7.0

#: Above this combined score, a proposed new agent is treated as a duplicate of
#: an existing one. Tuned so "Email to Alice" / "Alice lunch email" collide but
#: "Email to Alice" / "Email to Bob" do not.
DUPLICATE_THRESHOLD = 0.55

_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have i in is it its of on or that
    the to was were will with about please can could would should need needs
    my me you your we our us this these those there their them they he she him
    her his hers do does did done get got make made send sent reply replied
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'@.\-]*", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]+\b")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{1,})\b")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""

    return [
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOPWORDS and len(token) > 1
    ]


def extract_entities(text: str) -> List[str]:
    """Best-effort pull of the people and identifiers a piece of text is about.

    Email addresses plus capitalised words that are not sentence-initial and not
    common words. Deliberately crude -- this feeds a ranking signal, not a
    decision, so a false positive costs a slightly worse shortlist and nothing
    else. A model call here would be more accurate and far more expensive on a
    path that runs every turn.
    """

    if not text:
        return []

    found: List[str] = []
    seen: set[str] = set()

    for match in _EMAIL_RE.findall(text):
        key = match.casefold()
        if key not in seen:
            seen.add(key)
            found.append(match)

    stripped = _EMAIL_RE.sub(" ", text)
    for sentence in re.split(r"(?<=[.!?])\s+|\n", stripped):
        words = sentence.strip().split()
        for index, word in enumerate(words):
            cleaned = word.strip(".,;:!?\"'()[]")
            if not _PROPER_NOUN_RE.fullmatch(cleaned):
                continue
            if index == 0:  # sentence-initial capitals carry no signal
                continue
            if cleaned.lower() in _STOPWORDS:
                continue
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                found.append(cleaned)

    return found


@dataclass(frozen=True)
class ScoreBreakdown:
    """Why a record scored what it did.

    Kept as structured data rather than a bare float so a bad shortlist can be
    debugged, and so the evaluation harness can attribute routing errors to a
    specific signal instead of to "the ranker".
    """

    record: AgentRecord
    entity: float = 0.0
    name: float = 0.0
    lexical: float = 0.0
    recency: float = 0.0

    @property
    def total(self) -> float:
        return (
            W_ENTITY * self.entity
            + W_NAME * self.name
            + W_LEXICAL * self.lexical
            + W_RECENCY * self.recency
        )

    def explain(self) -> str:
        return (
            f"{self.record.name}: total={self.total:.2f} "
            f"(entity={self.entity:.2f} name={self.name:.2f} "
            f"lexical={self.lexical:.2f} recency={self.recency:.2f})"
        )


def _idf(records: Sequence[AgentRecord]) -> Dict[str, float]:
    """Inverse document frequency across the candidate set.

    A term in most agent names ("email") should barely move the ranking; a term
    in one ("vercel") should move it a lot.
    """

    total = max(len(records), 1)
    counts: Counter[str] = Counter()
    for record in records:
        terms = set(tokenize(f"{record.name} {record.purpose} {' '.join(record.tags)}"))
        counts.update(terms)
    return {
        term: math.log(1 + total / (1 + count))
        for term, count in counts.items()
    }


def _recency_score(record: AgentRecord, now: datetime) -> float:
    age_days = max((now - record.last_used_at).total_seconds() / 86400.0, 0.0)
    return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def _mentions(entity: str, text: str) -> bool:
    """Whole-word containment.

    Substring matching would make "Alice" match "Alicia" and quietly merge two
    people's threads, which is the worst failure this component can produce.
    """

    entity = entity.strip()
    if not entity:
        return False
    pattern = rf"(?<![\w@.]){re.escape(entity)}(?![\w@.])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _entity_score(record: AgentRecord, request: str) -> float:
    if not record.entities:
        return 0.0
    hits = sum(1 for entity in record.entities if _mentions(entity, request))
    return min(hits / len(record.entities), 1.0) if hits else 0.0


def _overlap(request_terms: set[str], target_terms: set[str], idf: Dict[str, float]) -> float:
    if not request_terms or not target_terms:
        return 0.0
    shared = request_terms & target_terms
    if not shared:
        return 0.0
    numerator = sum(idf.get(term, 1.0) for term in shared)
    denominator = sum(idf.get(term, 1.0) for term in target_terms)
    return min(numerator / denominator, 1.0) if denominator else 0.0


def score_records(
    records: Sequence[AgentRecord],
    request: str,
    *,
    now: Optional[datetime] = None,
) -> List[ScoreBreakdown]:
    """Score every record against a request, best first."""

    now = now or datetime.now(timezone.utc)
    idf = _idf(records)
    request_terms = set(tokenize(request))

    scored = [
        ScoreBreakdown(
            record=record,
            entity=_entity_score(record, request),
            name=_overlap(request_terms, set(tokenize(record.name)), idf),
            lexical=_overlap(
                request_terms,
                set(tokenize(f"{record.purpose} {' '.join(record.tags)} {record.summary}")),
                idf,
            ),
            recency=_recency_score(record, now),
        )
        for record in records
    ]
    # Name as the final tiebreaker keeps ordering stable across runs, which
    # matters for reproducible evaluation.
    scored.sort(key=lambda item: (-item.total, item.record.name))
    return scored


def shortlist(
    records: Sequence[AgentRecord],
    request: str,
    *,
    limit: int = 8,
    now: Optional[datetime] = None,
    always_include: Iterable[str] = (),
) -> List[ScoreBreakdown]:
    """The top ``limit`` records for this request.

    ``always_include`` forces specific agents in regardless of score -- used for
    agents the request names outright, so an explicit reference can never be
    ranked away.
    """

    scored = score_records(records, request, now=now)
    forced = {name.casefold() for name in always_include}

    if not forced:
        return scored[:limit]

    picked = [item for item in scored if item.record.name.casefold() in forced]
    picked_names = {item.record.name for item in picked}
    for item in scored:
        if len(picked) >= limit:
            break
        if item.record.name not in picked_names:
            picked.append(item)
            picked_names.add(item.record.name)
    return picked[:limit]


def find_duplicate(
    records: Sequence[AgentRecord],
    proposed_name: str,
    *,
    threshold: float = DUPLICATE_THRESHOLD,
) -> Optional[AgentRecord]:
    """The existing agent a proposed new one would duplicate, if any.

    Compares on name tokens and the existing agent's tracked entities -- not
    recency, not purpose. A proposed agent has no history yet, so the question
    is purely "is this the same subject". Letting recency contribute would make
    every new agent collide with whatever happened to run most recently.

    Entity extraction is *not* applied to the proposed name: a two-or-three word
    agent label has no sentence structure to key off, so capitalisation there is
    noise. Instead the existing record's known entities are matched against the
    proposed name, which is both more reliable and case-insensitive.
    """

    proposed_tokens = set(tokenize(proposed_name))
    if not proposed_tokens:
        return None

    best: Optional[AgentRecord] = None
    best_score = 0.0

    for record in records:
        record_tokens = set(tokenize(record.name))
        if not record_tokens:
            continue

        union = proposed_tokens | record_tokens
        jaccard = len(proposed_tokens & record_tokens) / len(union) if union else 0.0

        if record.entities:
            hits = sum(1 for entity in record.entities if _mentions(entity, proposed_name))
            entity_overlap = hits / len(record.entities)
        else:
            entity_overlap = 0.0

        combined = 0.7 * jaccard + 0.3 * entity_overlap
        if combined > best_score:
            best_score = combined
            best = record

    return best if best_score >= threshold else None


__all__ = [
    "ScoreBreakdown",
    "DUPLICATE_THRESHOLD",
    "extract_entities",
    "find_duplicate",
    "score_records",
    "shortlist",
    "tokenize",
]
