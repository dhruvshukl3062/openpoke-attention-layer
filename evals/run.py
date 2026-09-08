"""Entry point: baseline vs treatment, plus the ablation sweep.

    python -m evals.run
    python -m evals.run --days 30 --seeds 5

Every configuration runs over the same seeded worlds, so differences between
rows are differences in policy and nothing else. Multiple seeds are run because
a single run is an anecdote -- the spread across seeds is reported alongside the
mean.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
from dataclasses import replace
from datetime import timedelta
from typing import Dict, List

from .simulate import (
    Metrics,
    baseline_policy,
    measure_context_cost,
    measure_duplicate_agents,
    run_attention_sim,
    treatment_policy,
)
from .world import WorldGenerator


def _configs() -> Dict[str, tuple]:
    """(policy, dedupe_enabled) per configuration.

    The ablations turn off exactly one thing each, so a metric that moves can be
    attributed to that component rather than to the system as a whole.
    """

    treatment = treatment_policy()
    return {
        "baseline (upstream)": (baseline_policy(), False),
        "attention layer": (treatment, True),
        "  ablate: no coalescing": (replace(treatment, coalesce_seconds=0.0), True),
        "  ablate: no dedupe": (treatment, False),
        "  ablate: no budget": (replace(treatment, max_interrupts_per_hour=10**9), True),
        "  ablate: no quiet hours": (replace(treatment, quiet_hours_enabled=False), True),
    }


async def _run_config(
    name: str, policy, dedupe: bool, *, days: int, seeds: int, per_day: int
) -> List[Metrics]:
    runs: List[Metrics] = []
    for seed in range(seeds):
        world = WorldGenerator(seed=seed, emails_per_day=per_day).build(days=days)
        metrics = await run_attention_sim(world, policy, config_name=name, dedupe=dedupe)
        duplicates, created = measure_duplicate_agents(world, dedupe=dedupe)
        metrics.duplicate_agents = duplicates
        metrics.agents_created = created
        metrics.prompt_chars_per_turn = measure_context_cost(
            created, shortlist_size=None if "baseline" in name else 8
        )
        runs.append(metrics)
    return runs


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _spread(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _print_table(results: Dict[str, List[Metrics]]) -> None:
    columns = [
        ("config", 26),
        ("intr/day", 9),
        ("prec@intr", 10),
        ("recall", 8),
        ("timely recall", 14),
        ("med delay", 10),
        ("p95 delay", 10),
        ("digest", 7),
        ("dupes", 6),
        ("ctx chars", 10),
        ("dup agents", 11),
    ]
    header = "".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))

    for name, runs in results.items():
        row = [
            name.ljust(columns[0][1]),
            f"{_mean([m.interrupts_per_day for m in runs]):.2f}".ljust(columns[1][1]),
            f"{_mean([m.precision for m in runs]):.3f}".ljust(columns[2][1]),
            f"{_mean([m.urgent_recall for m in runs]):.3f}".ljust(columns[3][1]),
            f"{_mean([m.timely_urgent_recall for m in runs]):.3f}".ljust(columns[4][1]),
            f"{_mean([m.median_urgent_delay_minutes for m in runs]):.1f}".ljust(columns[5][1]),
            f"{_mean([m.p95_urgent_delay_minutes for m in runs]):.1f}".ljust(columns[6][1]),
            f"{_mean([m.digest_items for m in runs]):.0f}".ljust(columns[7][1]),
            f"{_mean([m.duplicates_dropped for m in runs]):.0f}".ljust(columns[8][1]),
            f"{_mean([m.prompt_chars_per_turn for m in runs]):.0f}".ljust(columns[9][1]),
            f"{_mean([m.duplicate_agents for m in runs]):.1f}".ljust(columns[10][1]),
        ]
        print("".join(row))

    print()
    interrupts = {
        name: [m.interrupts_per_day for m in runs] for name, runs in results.items()
    }
    for name, values in interrupts.items():
        print(f"  spread across seeds, {name.strip()}: sd={_spread(values):.2f}")


async def main_async(days: int, seeds: int, per_day: int) -> Dict[str, List[Metrics]]:
    results: Dict[str, List[Metrics]] = {}
    for name, (policy, dedupe) in _configs().items():
        results[name] = await _run_config(
            name, policy, dedupe, days=days, seeds=seeds, per_day=per_day
        )
    return results


def main() -> None:
    # The broker logs every routing decision; useful in the server, noise here.
    logging.getLogger("openpoke.server").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description="Attention Layer evaluation")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="Run the hold-ceiling sweep")
    parser.add_argument(
        "--emails-per-day",
        type=int,
        default=30,
        help="Inbox volume. The interruption budget only binds under load.",
    )
    args = parser.parse_args()

    print(
        f"Simulated {args.days} days x {args.seeds} seeds "
        f"at {args.emails_per_day} emails/day\n"
    )
    results = asyncio.run(main_async(args.days, args.seeds, args.emails_per_day))
    _print_table(results)

    baseline = results["baseline (upstream)"]
    treatment = results["attention layer"]
    before = _mean([m.interrupts_per_day for m in baseline])
    after = _mean([m.interrupts_per_day for m in treatment])
    recall_before = _mean([m.urgent_recall for m in baseline])
    recall_after = _mean([m.urgent_recall for m in treatment])
    timely_before = _mean([m.timely_urgent_recall for m in baseline])
    timely_after = _mean([m.timely_urgent_recall for m in treatment])
    delay_before = _mean([m.median_urgent_delay_minutes for m in baseline])
    delay_after = _mean([m.median_urgent_delay_minutes for m in treatment])

    print()
    print(f"Interruptions per day: {before:.2f} -> {after:.2f} "
          f"({(1 - after / before) * 100:.0f}% fewer)" if before else "")
    print(f"Urgent recall (ever):  {recall_before:.3f} -> {recall_after:.3f}")
    print(f"Urgent recall (<1h):   {timely_before:.3f} -> {timely_after:.3f}  <-- the honest one")
    print(f"Median urgent delay:   {delay_before:.1f} -> {delay_after:.1f} min "
          f"(the price paid for the quiet)")

    if args.sweep:
        asyncio.run(sweep_hold(args.days, args.seeds, args.emails_per_day))



async def sweep_hold(days: int, seeds: int, per_day: int) -> None:
    """How the hold ceiling trades quiet against timeliness.

    3 hours was not picked by taste. This is the curve it was picked from, and
    it is also where the rule-based scorer's ceiling becomes visible: no hold
    setting recovers the urgent mail the keyword scorer never rated urgent in
    the first place.
    """

    from datetime import timedelta as _td

    print("\nSensitivity: hold ceiling\n")
    print("max_hold   intr/day  prec@intr  timely recall  med delay (min)")
    print("-" * 63)

    for hours in (1, 2, 3, 6, 12, 24):
        policy = replace(treatment_policy(), max_hold=_td(hours=hours))
        runs = await _run_config(
            f"hold={hours}h", policy, True, days=days, seeds=seeds, per_day=per_day
        )
        print(
            f"{hours:>2}h        "
            f"{_mean([m.interrupts_per_day for m in runs]):<10.2f}"
            f"{_mean([m.precision for m in runs]):<11.3f}"
            f"{_mean([m.timely_urgent_recall for m in runs]):<15.3f}"
            f"{_mean([m.median_urgent_delay_minutes for m in runs]):.1f}"
        )


if __name__ == "__main__":
    main()
