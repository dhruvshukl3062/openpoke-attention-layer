"""Shortlist ranking and spawn-time duplicate detection.

Pure functions over fixed inputs -- no model, no clock drift, no I/O. Every
assertion here is a claim about routing behaviour that the evaluation harness
later measures at scale.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.services.attention.ranking import (
    extract_entities,
    find_duplicate,
    score_records,
    shortlist,
    tokenize,
)
from server.services.attention.registry import AgentRecord

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def record(name: str, *, purpose="", entities=(), tags=(), summary="", days_ago=0) -> AgentRecord:
    used = NOW - timedelta(days=days_ago)
    return AgentRecord(
        name=name,
        purpose=purpose,
        entities=tuple(entities),
        tags=tuple(tags),
        summary=summary,
        created_at=used,
        last_used_at=used,
    )


# ---------------------------------------------------------------------------
# Tokenizing and entity extraction
# ---------------------------------------------------------------------------

def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("Please send the email to Alice") == ["email", "alice"]


def test_entities_pull_names_and_addresses():
    found = extract_entities("Ask Alice about the Vercel offer, cc bob@example.com")

    assert "Alice" in found
    assert "Vercel" in found
    assert "bob@example.com" in found


def test_sentence_initial_capitals_are_not_treated_as_entities():
    """Otherwise every request starting with a capital yields a junk entity."""

    assert extract_entities("Draft a reply to the landlord") == []


# ---------------------------------------------------------------------------
# Ranking signals
# ---------------------------------------------------------------------------

def test_entity_match_beats_everything_else():
    """A named person should find their agent even with zero word overlap."""

    records = [
        record("Q3 Planning Notes", purpose="planning docs", days_ago=0),
        record("Email to Alice", purpose="lunch thread", entities=["Alice"], days_ago=30),
    ]

    ranked = score_records(records, "can you ping Alice again", now=NOW)

    assert ranked[0].record.name == "Email to Alice"
    assert ranked[0].entity > 0


def test_recency_only_breaks_ties():
    """Two equally relevant agents: the recent one wins. It must not beat a
    genuinely more relevant stale one -- that is the previous test."""

    records = [
        record("Email to Alice", entities=["Alice"], days_ago=20),
        record("Alice Follow Up", entities=["Alice"], days_ago=0),
    ]

    ranked = score_records(records, "message Alice", now=NOW)

    assert ranked[0].record.name == "Alice Follow Up"
    assert ranked[0].recency > ranked[1].recency


def test_common_terms_are_discounted():
    """'email' appears in most agents, so it should barely discriminate."""

    records = [
        record("Email to Alice", entities=["Alice"]),
        record("Email to Bob", entities=["Bob"]),
        record("Email to Carol", entities=["Carol"]),
        record("Vercel Job Offer", purpose="offer negotiation"),
    ]

    ranked = score_records(records, "vercel", now=NOW)

    assert ranked[0].record.name == "Vercel Job Offer"


def test_scores_are_reproducible_and_deterministically_ordered():
    records = [record("Alpha Agent"), record("Beta Agent"), record("Gamma Agent")]

    first = [item.record.name for item in score_records(records, "unrelated", now=NOW)]
    second = [item.record.name for item in score_records(records, "unrelated", now=NOW)]

    assert first == second
    # Nothing matches, so the alphabetical tiebreaker decides.
    assert first == ["Alpha Agent", "Beta Agent", "Gamma Agent"]


def test_breakdown_explains_itself():
    records = [record("Email to Alice", entities=["Alice"])]

    explanation = score_records(records, "email Alice", now=NOW)[0].explain()

    assert "entity=" in explanation and "recency=" in explanation


# ---------------------------------------------------------------------------
# Shortlisting
# ---------------------------------------------------------------------------

def test_shortlist_is_capped():
    records = [record(f"Agent {index}") for index in range(50)]

    assert len(shortlist(records, "anything", limit=8, now=NOW)) == 8


def test_shortlist_size_does_not_depend_on_roster_size():
    """The property the whole Attention Layer rests on."""

    small = shortlist([record(f"Agent {i}") for i in range(10)], "hello", limit=8, now=NOW)
    large = shortlist([record(f"Agent {i}") for i in range(500)], "hello", limit=8, now=NOW)

    assert len(small) == len(large) == 8


def test_always_include_survives_a_low_score():
    """An agent the user named outright must never be ranked away."""

    records = [record(f"Agent {index}") for index in range(20)]
    records.append(record("Obscure Legacy Agent", days_ago=400))

    picked = shortlist(
        records, "unrelated request", limit=5, now=NOW,
        always_include=["Obscure Legacy Agent"],
    )

    assert "Obscure Legacy Agent" in [item.record.name for item in picked]
    assert len(picked) == 5


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_a_reworded_name_for_the_same_subject_is_a_duplicate():
    existing = [record("Email to Alice", entities=["Alice"])]

    match = find_duplicate(existing, "Alice Email")

    assert match is not None and match.name == "Email to Alice"


def test_different_people_are_not_duplicates():
    """The failure that would matter most: merging Alice's thread into Bob's."""

    existing = [record("Email to Alice", entities=["Alice"])]

    assert find_duplicate(existing, "Email to Bob") is None


def test_unrelated_subjects_are_not_duplicates():
    existing = [record("Email to Alice", entities=["Alice"])]

    assert find_duplicate(existing, "Vercel Job Offer") is None


def test_no_duplicate_against_an_empty_registry():
    assert find_duplicate([], "Email to Alice") is None


@pytest.mark.parametrize(
    "proposed",
    ["Email to Alice", "email to alice", "EMAIL TO ALICE"],
)
def test_duplicate_detection_ignores_case(proposed):
    existing = [record("Email to Alice", entities=["Alice"])]

    assert find_duplicate(existing, proposed) is not None


def test_similar_names_for_different_people_are_not_merged():
    """The nastiest false positive available: Alice vs Alicia.

    Whole-word entity matching is what prevents it; a substring check would
    quietly merge two people's threads.
    """

    existing = [record("Email to Alice", entities=["Alice"])]

    assert find_duplicate(existing, "Email to Alicia") is None


def test_a_looser_rewording_still_collides():
    existing = [record("Email to Alice", entities=["Alice"])]

    match = find_duplicate(existing, "Alice lunch email")

    assert match is not None and match.name == "Email to Alice"
