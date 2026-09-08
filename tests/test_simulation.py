"""Tests for the evaluation harness.

An eval suite nobody tests is a number generator. These check the properties the
results depend on: that runs are reproducible, that the synthetic world is not
secretly aligned with the scorer, and that the headline comparison is measuring
what it claims to.
"""

from __future__ import annotations

import pytest

from evals.simulate import (
    baseline_policy,
    measure_context_cost,
    measure_duplicate_agents,
    run_attention_sim,
    treatment_policy,
)
from evals.world import SimulatedClassifier, WorldGenerator
from server.services.attention.scoring import urgency_for_email


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_the_same_seed_gives_the_same_world():
    first = WorldGenerator(seed=7).build(days=5)
    second = WorldGenerator(seed=7).build(days=5)

    assert [e.id for e in first.emails] == [e.id for e in second.emails]
    assert [e.subject for e in first.emails] == [e.subject for e in second.emails]
    assert [r.text for r in first.requests] == [r.text for r in second.requests]


def test_different_seeds_give_different_worlds():
    first = WorldGenerator(seed=1).build(days=5)
    second = WorldGenerator(seed=2).build(days=5)

    assert [e.subject for e in first.emails] != [e.subject for e in second.emails]


async def test_a_run_is_reproducible():
    world = WorldGenerator(seed=3).build(days=7)

    first = await run_attention_sim(world, treatment_policy(), config_name="a")
    second = await run_attention_sim(world, treatment_policy(), config_name="b")

    assert first.interrupt_events == second.interrupt_events
    assert first.precision == second.precision
    assert first.urgent_recall == second.urgent_recall


# ---------------------------------------------------------------------------
# The world is honest
# ---------------------------------------------------------------------------

def test_urgent_mail_does_not_always_announce_itself():
    """If it did, precision and recall would be 1.0 by construction."""

    world = WorldGenerator(seed=0).build(days=20)
    urgent = [e for e in world.emails if e.truly_urgent]

    scores = [urgency_for_email(e.subject) for e in urgent]
    missed_cues = [s for s in scores if s < treatment_policy().interrupt_threshold]

    assert urgent, "the world should contain urgent mail"
    assert missed_cues, "some urgent mail must lack an urgency cue, or the eval is rigged"


def test_routine_mail_sometimes_carries_a_false_cue():
    world = WorldGenerator(seed=0).build(days=20)
    routine = [e for e in world.emails if not e.truly_urgent]

    false_alarms = [
        e for e in routine
        if urgency_for_email(e.subject) >= treatment_policy().interrupt_threshold
    ]

    assert false_alarms, "routine mail must sometimes look urgent, or precision is meaningless"


def test_the_simulated_classifier_is_imperfect():
    """A perfect upstream classifier would flatter the baseline."""

    world = WorldGenerator(seed=0).build(days=20)
    classifier = SimulatedClassifier(seed=0)
    verdicts = [(e.truly_urgent, classifier.is_important(e)) for e in world.emails]

    misses = [truth for truth, flagged in verdicts if truth and not flagged]
    false_positives = [truth for truth, flagged in verdicts if not truth and flagged]

    assert misses, "the classifier should miss some urgent mail"
    assert false_positives, "and flag some mail that does not matter"


def test_the_world_contains_bursts():
    """Coalescing and the budget cannot be measured without them."""

    world = WorldGenerator(seed=0).build(days=5)
    gaps = [
        (b.arrived_at - a.arrived_at).total_seconds()
        for a, b in zip(world.emails, world.emails[1:])
    ]

    assert sum(1 for gap in gaps if gap < 90) > len(gaps) * 0.1


# ---------------------------------------------------------------------------
# The headline comparison
# ---------------------------------------------------------------------------

async def test_the_attention_layer_reduces_interruptions():
    world = WorldGenerator(seed=0).build(days=14)

    baseline = await run_attention_sim(
        world, baseline_policy(), config_name="baseline", dedupe=False
    )
    treatment = await run_attention_sim(world, treatment_policy(), config_name="treatment")

    assert treatment.interrupts_per_day < baseline.interrupts_per_day


async def test_quiet_is_not_bought_by_losing_urgent_mail():
    """The safety property. Silence only counts as a win if nothing was buried."""

    world = WorldGenerator(seed=0).build(days=14)

    baseline = await run_attention_sim(
        world, baseline_policy(), config_name="baseline", dedupe=False
    )
    treatment = await run_attention_sim(world, treatment_policy(), config_name="treatment")

    assert treatment.urgent_recall >= baseline.urgent_recall


async def test_interrupting_less_means_interrupting_better():
    world = WorldGenerator(seed=0).build(days=14)

    baseline = await run_attention_sim(
        world, baseline_policy(), config_name="baseline", dedupe=False
    )
    treatment = await run_attention_sim(world, treatment_policy(), config_name="treatment")

    assert treatment.precision > baseline.precision


async def test_the_baseline_never_holds_anything_back():
    """Confirms the baseline really does model upstream's behaviour."""

    world = WorldGenerator(seed=0).build(days=7)

    baseline = await run_attention_sim(
        world, baseline_policy(), config_name="baseline", dedupe=False
    )

    assert baseline.digest_items == 0
    assert baseline.suppressed == 0
    assert baseline.duplicates_dropped == 0


# ---------------------------------------------------------------------------
# Context cost and routing
# ---------------------------------------------------------------------------

def test_context_cost_grows_with_the_roster_upstream_but_not_with_a_shortlist():
    upstream_small = measure_context_cost(20, shortlist_size=None)
    upstream_large = measure_context_cost(400, shortlist_size=None)
    ours_small = measure_context_cost(20, shortlist_size=8)
    ours_large = measure_context_cost(400, shortlist_size=8)

    assert upstream_large > upstream_small * 15, "upstream should grow roughly linearly"
    assert ours_large == ours_small, "a shortlist costs the same at any roster size"


def test_dedupe_prevents_redundant_agents():
    world = WorldGenerator(seed=0).build(days=20)

    without, _ = measure_duplicate_agents(world, dedupe=False)
    with_dedupe, _ = measure_duplicate_agents(world, dedupe=True)

    assert without > 0, "rewordings should create duplicates when nothing stops them"
    assert with_dedupe == 0
