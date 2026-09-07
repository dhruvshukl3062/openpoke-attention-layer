"""Interaction agent helpers for prompt construction."""

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List

from ...services.attention import get_agent_registry
from ...services.attention.ranking import shortlist

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()

#: How many agents the interaction agent sees per turn. Bounds the per-turn
#: context cost independently of how many agents exist.
SHORTLIST_SIZE = 8


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(_render_active_agents(latest_text))
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _humanize_age(moment: datetime, now: datetime) -> str:
    """Relative time reads better to a model than an ISO timestamp, and costs
    fewer tokens."""

    seconds = max((now - moment).total_seconds(), 0)
    if seconds < 3600:
        return "just now" if seconds < 300 else f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    days = int(seconds // 86400)
    return "yesterday" if days == 1 else f"{days}d ago"


# Render a ranked shortlist of execution agents relevant to this turn
def _render_active_agents(request_text: str = "") -> str:
    """Show the agents most likely to be relevant, not all of them.

    Upstream pasted every agent name into every turn, so prompt cost grew
    linearly with the roster and the model had to pick from bare strings. Here
    a bounded, ranked shortlist is rendered with the metadata that actually
    supports the decision -- purpose, recency, run count, what it last did.

    Agents named outright in the request are pinned in regardless of score, so
    an explicit reference can never be ranked away. The header carries the total
    count so the model knows the list is a selection rather than everything.
    """

    registry = get_agent_registry()
    candidates = registry.selectable()
    total = len(registry.all())

    if not candidates:
        return "<active_agents>\nNone\n</active_agents>"

    # Anything the request names by exact title is pinned, including agents that
    # have gone dormant and would otherwise be excluded entirely.
    named = [
        record.name
        for record in registry.all()
        if record.name and record.name.casefold() in (request_text or "").casefold()
    ]
    pool = candidates + [
        record for record in registry.all()
        if record.name in named and record not in candidates
    ]

    now = datetime.now(timezone.utc)
    picked = shortlist(
        pool, request_text or "", limit=SHORTLIST_SIZE, now=now, always_include=named
    )

    lines: List[str] = []
    for item in picked:
        record = item.record
        attrs = [f'name="{escape(record.name or "agent", quote=True)}"']
        if record.purpose:
            attrs.append(f'purpose="{escape(record.purpose, quote=True)}"')
        attrs.append(f'last_used="{_humanize_age(record.last_used_at, now)}"')
        if record.invocations:
            attrs.append(f'runs="{record.invocations}"')

        body = escape(record.summary.strip(), quote=False) if record.summary else ""
        if body:
            lines.append(f"<agent {' '.join(attrs)}>{body}</agent>")
        else:
            lines.append(f"<agent {' '.join(attrs)} />")

    header = f'<active_agents shown="{len(picked)}" total="{total}"'
    if len(picked) < total:
        header += ' note="most relevant first; name an agent directly to reach one not listed"'
    header += ">"

    return "\n".join([header, *lines, "</active_agents>"])


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
