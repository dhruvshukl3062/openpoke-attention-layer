"""The Attention Layer.

Two halves of one idea: attention is a finite budget, and OpenPoke spends it on
both sides without one.

* ``registry`` -- structured records for execution agents, so the orchestrator
  can be shown a ranked shortlist instead of the entire roster on every turn.
* ``broker`` -- everything that would interrupt the user goes through one place
  that dedupes, batches and budgets interruptions.
* ``scoring`` -- where each source's urgency comes from, kept out of the broker
  so policy and judgement can be evaluated separately.
"""

from .broker import AttentionBroker, Candidate, Decision, Policy, Route
from .compaction import compact_agent_history, needs_compaction, render_agent_history
from .registry import (
    DEFAULT_DORMANT_AFTER,
    AgentRecord,
    AgentRegistry,
    AgentStatus,
    get_agent_registry,
)

__all__ = [
    "AttentionBroker",
    "Candidate",
    "Decision",
    "Policy",
    "Route",
    "compact_agent_history",
    "needs_compaction",
    "render_agent_history",
    "AgentRecord",
    "AgentRegistry",
    "AgentStatus",
    "DEFAULT_DORMANT_AFTER",
    "get_agent_registry",
]
