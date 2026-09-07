"""The Attention Layer.

Two halves of one idea: attention is a finite budget, and OpenPoke spends it on
both sides without one.

* ``registry`` -- structured records for execution agents, so the orchestrator
  can be shown a ranked shortlist instead of the entire roster on every turn.
* (upcoming) ``broker`` -- everything that would interrupt the user goes through
  one place that dedupes, batches, scores and budgets interruptions.
"""

from .registry import (
    DEFAULT_DORMANT_AFTER,
    AgentRecord,
    AgentRegistry,
    AgentStatus,
    get_agent_registry,
)

__all__ = [
    "AgentRecord",
    "AgentRegistry",
    "AgentStatus",
    "DEFAULT_DORMANT_AFTER",
    "get_agent_registry",
]
