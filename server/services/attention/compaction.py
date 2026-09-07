"""Keeping a long-lived execution agent's own prompt bounded.

OpenPoke summarises the *user* conversation once it passes 100 entries, but
execution agent logs have no equivalent: ``build_system_prompt_with_history``
loads the entire transcript -- every request, tool call and truncated tool
response -- into the system prompt, with ``conversation_limit`` defaulting to
``None``. An agent driven by a daily recurring trigger therefore grows its own
prompt without bound until the request fails outright.

This applies the same shape the conversation log already uses: compress
everything except a recent tail into a rolling summary, and render
``summary + tail`` instead of the full history. The summary lives on the agent's
registry record, where it doubles as shortlist metadata -- one artefact, two
uses.

Two deliberate properties:

* **Compaction is not on the critical path for correctness.** If the model call
  fails, the caller falls back to the full transcript. A degraded prompt is
  better than a failed task.
* **Only whole entries are compacted**, never a partial tail, so what the agent
  sees is always a valid prefix-summary plus intact recent history.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ...config import get_settings
from ...logging_config import logger
from ..execution import get_execution_agent_logs
from ..llm import get_llm_client
from .registry import get_agent_registry

#: Compact once this many un-summarised entries have built up beyond the tail.
COMPACT_THRESHOLD = 40

#: Entries kept verbatim. Recent tool calls and their results are what the agent
#: needs in full; older ones only need their outcome preserved.
TAIL_SIZE = 12

_SUMMARY_SYSTEM_PROMPT = """\
You maintain a running summary of one execution agent's work log.

You will be given the previous summary (possibly empty) and a batch of newer log
entries. Produce a single replacement summary that folds the new entries into
the old one.

Preserve, in priority order:
1. Identifiers that later work depends on -- draft ids, thread ids, message ids,
   email addresses, trigger ids.
2. People and organisations involved, and their relationship to the task.
3. Outcomes: what was actually done, what succeeded, what failed and why.
4. Open loops: anything awaiting confirmation, a reply, or a scheduled time.

Drop: reasoning narration, retries that later succeeded, and verbose tool
payloads whose outcome you have already captured.

Write compact prose, under 250 words. No preamble, no headings, no bullet list
of every event. This text is read by the agent itself as background, so write it
as a factual record rather than a report to a person.
"""


def _entries(agent_name: str) -> List[Tuple[str, str, str]]:
    return list(get_execution_agent_logs().iter_entries(agent_name))


def _render_entries(entries: Sequence[Tuple[str, str, str]]) -> str:
    from html import escape

    parts: List[str] = []
    for tag, timestamp, payload in entries:
        body = escape(payload, quote=False)
        if timestamp:
            parts.append(f'<{tag} timestamp="{timestamp}">{body}</{tag}>')
        else:
            parts.append(f"<{tag}>{body}</{tag}>")
    return "\n".join(parts)


def needs_compaction(agent_name: str) -> bool:
    """True when enough un-summarised entries have accumulated to be worth a call."""

    record = get_agent_registry().get(agent_name)
    already = record.compacted_through if record else 0
    pending = len(_entries(agent_name)) - already
    return pending > COMPACT_THRESHOLD + TAIL_SIZE


async def compact_agent_history(agent_name: str) -> bool:
    """Fold older log entries into the agent's rolling summary.

    Returns True when a compaction actually happened. Failures are logged and
    swallowed: the caller falls back to rendering the full transcript, which is
    expensive but correct.
    """

    registry = get_agent_registry()
    record = registry.get(agent_name)
    if record is None:
        return False

    entries = _entries(agent_name)
    already = record.compacted_through
    pending = entries[already:]
    if len(pending) <= COMPACT_THRESHOLD + TAIL_SIZE:
        return False

    # Everything except the tail gets folded in. Cutting on a whole-entry
    # boundary keeps the rendered history a valid summary + intact tail.
    batch = pending[: len(pending) - TAIL_SIZE]
    cutoff = already + len(batch)

    settings = get_settings()
    user_content = (
        f"<previous_summary>\n{record.summary or '(none)'}\n</previous_summary>\n\n"
        f"<new_entries>\n{_render_entries(batch)}\n</new_entries>"
    )

    try:
        response = await get_llm_client().complete(
            model=settings.summarizer_model,
            messages=[{"role": "user", "content": user_content}],
            system=_SUMMARY_SYSTEM_PROMPT,
            api_key=settings.openrouter_api_key,
        )
        choices = response.get("choices") or []
        summary = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    except Exception as exc:
        logger.warning(f"[{agent_name}] History compaction failed: {exc}")
        return False

    if not summary:
        logger.warning(f"[{agent_name}] History compaction returned nothing; leaving log intact")
        return False

    registry.set_compaction(agent_name, summary=summary, compacted_through=cutoff)
    logger.info(
        f"[{agent_name}] Compacted {len(batch)} log entries "
        f"({cutoff}/{len(entries)} now summarised)"
    )
    return True


def render_agent_history(agent_name: str) -> str:
    """The history an execution agent should see: rolling summary plus recent tail.

    Falls back to the full transcript when nothing has been compacted, so an
    agent that has never crossed the threshold behaves exactly as before.
    """

    record = get_agent_registry().get(agent_name)
    entries = _entries(agent_name)

    if record is None or not record.compacted_through:
        return _render_entries(entries)

    tail = entries[record.compacted_through:]
    sections: List[str] = []
    if record.summary:
        sections.append(
            f"<history_summary covering=\"{record.compacted_through} earlier entries\">\n"
            f"{record.summary}\n"
            f"</history_summary>"
        )
    rendered_tail = _render_entries(tail)
    if rendered_tail:
        sections.append(rendered_tail)
    return "\n".join(sections)


__all__ = [
    "COMPACT_THRESHOLD",
    "TAIL_SIZE",
    "compact_agent_history",
    "needs_compaction",
    "render_agent_history",
]
