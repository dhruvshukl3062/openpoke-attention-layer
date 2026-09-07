"""A structured registry of execution agents.

Upstream OpenPoke stores its agents as ``list[str]`` in ``roster.json`` and
renders every name into every interaction turn. That has two consequences as the
list grows: the prompt cost of each turn rises linearly and without bound, and
the model is asked to pick the right agent from bare strings with no purpose, no
history and no recency to go on.

This module replaces the bare string with a record -- what the agent is for,
which entities it touches, when it was last used, and a rolling summary of what
it has done -- so that a *shortlist* can be selected instead of the whole roster
being pasted in. Records are the input to ranking (see ``ranking.py``); nothing
here decides what the model sees.

Deliberately stdlib-only and synchronous: this is a small local file, the
callers are already async-heavy, and keeping it plain makes the unit tests
instant and the failure modes obvious.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional

from ...logging_config import logger

AgentStatus = Literal["active", "dormant", "archived"]

#: Agents untouched for this long stop appearing in shortlists by default. They
#: are not deleted -- an explicit reference by name still finds them.
DEFAULT_DORMANT_AFTER = timedelta(days=14)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _from_iso(value: Optional[str], fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AgentRecord:
    """What we know about one execution agent.

    ``purpose``, ``tags`` and ``entities`` are what make an agent findable
    without pasting its whole history into the prompt. ``entities`` is the
    highest-signal field -- the people, threads and subjects the agent deals
    with -- because a request that names Alice should find the Alice agent even
    when nothing else about the wording matches.
    """

    name: str
    purpose: str = ""
    tags: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    summary: str = ""
    status: AgentStatus = "active"
    invocations: int = 0
    created_at: datetime = field(default_factory=_utcnow)
    last_used_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "tags": list(self.tags),
            "entities": list(self.entities),
            "summary": self.summary,
            "status": self.status,
            "invocations": self.invocations,
            "created_at": _to_iso(self.created_at),
            "last_used_at": _to_iso(self.last_used_at),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, object], *, fallback_time: datetime) -> "AgentRecord":
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("agent record is missing a name")

        status = raw.get("status")
        if status not in ("active", "dormant", "archived"):
            status = "active"

        def _string_tuple(key: str) -> tuple[str, ...]:
            value = raw.get(key) or []
            if isinstance(value, str):
                value = [value]
            return tuple(str(item) for item in value if str(item).strip())

        try:
            invocations = int(raw.get("invocations") or 0)
        except (TypeError, ValueError):
            invocations = 0

        return cls(
            name=name,
            purpose=str(raw.get("purpose") or ""),
            tags=_string_tuple("tags"),
            entities=_string_tuple("entities"),
            summary=str(raw.get("summary") or ""),
            status=status,  # type: ignore[arg-type]
            invocations=invocations,
            created_at=_from_iso(raw.get("created_at"), fallback_time),  # type: ignore[arg-type]
            last_used_at=_from_iso(raw.get("last_used_at"), fallback_time),  # type: ignore[arg-type]
        )


class AgentRegistry:
    """Persisted collection of :class:`AgentRecord`, keyed by agent name.

    The clock is injected so that lifecycle behaviour (dormancy, recency) can be
    tested without sleeping or freezing global time.
    """

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], datetime] = _utcnow,
        dormant_after: timedelta = DEFAULT_DORMANT_AFTER,
    ) -> None:
        self._path = Path(path)
        self._now = now
        self._dormant_after = dormant_after
        self._lock = threading.RLock()
        self._records: Dict[str, AgentRecord] = {}
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        """Read from disk, migrating a legacy flat roster if that's what's there."""

        with self._lock:
            if not self._path.exists():
                self._records = {}
                return

            try:
                raw = json.loads(self._path.read_text(encoding="utf-8") or "null")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Failed to read agent registry, starting empty: {exc}")
                self._records = {}
                return

            fallback = self._now()

            # Legacy shape: ["Email to Alice", "Vercel Job Offer", ...]
            if isinstance(raw, list):
                self._records = {
                    str(name): AgentRecord(
                        name=str(name), created_at=fallback, last_used_at=fallback
                    )
                    for name in raw
                    if str(name).strip()
                }
                logger.info(f"Migrated {len(self._records)} agents from the legacy roster")
                return

            if not isinstance(raw, dict):
                self._records = {}
                return

            records: Dict[str, AgentRecord] = {}
            for entry in raw.get("agents", []):
                if not isinstance(entry, dict):
                    continue
                try:
                    record = AgentRecord.from_dict(entry, fallback_time=fallback)
                except ValueError as exc:
                    logger.warning(f"Skipping malformed agent record: {exc}")
                    continue
                records[record.name] = record
            self._records = records

    def save(self) -> None:
        """Write atomically, so a crash mid-write can't truncate the registry."""

        with self._lock:
            payload = {
                "version": 1,
                "agents": [record.to_dict() for record in self._records.values()],
            }
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                handle = tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=self._path.parent,
                    prefix=self._path.name,
                    suffix=".tmp",
                    delete=False,
                )
                with handle:
                    json.dump(payload, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(handle.name, self._path)
            except OSError as exc:
                logger.error(f"Failed to write agent registry: {exc}")

    # -- reads -------------------------------------------------------------

    def get(self, name: str) -> Optional[AgentRecord]:
        with self._lock:
            return self._records.get(name)

    def exists(self, name: str) -> bool:
        return self.get(name) is not None

    def all(self, *, include_archived: bool = False) -> List[AgentRecord]:
        with self._lock:
            records = list(self._records.values())
        if include_archived:
            return records
        return [record for record in records if record.status != "archived"]

    def selectable(self) -> List[AgentRecord]:
        """Records eligible for a shortlist: active only, most recent first.

        Dormant agents are excluded here rather than deleted -- they remain
        reachable by exact name, which is what makes dormancy safe.
        """

        active = [record for record in self.all() if record.status == "active"]
        return sorted(active, key=lambda record: record.last_used_at, reverse=True)

    def names(self) -> List[str]:
        """Compatibility with the flat-roster call sites."""

        return [record.name for record in self.all()]

    # -- writes ------------------------------------------------------------

    def upsert(
        self,
        name: str,
        *,
        purpose: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        entities: Optional[Iterable[str]] = None,
        summary: Optional[str] = None,
    ) -> AgentRecord:
        """Create an agent, or enrich an existing one. Never clobbers with None.

        Partial updates matter: the interaction agent names an agent up front but
        useful metadata (entities it turned out to touch, a summary of what it
        did) arrives later, and a later write must not erase an earlier one.
        """

        name = name.strip()
        if not name:
            raise ValueError("agent name cannot be empty")

        with self._lock:
            now = self._now()
            existing = self._records.get(name)

            if existing is None:
                record = AgentRecord(
                    name=name,
                    purpose=purpose or "",
                    tags=tuple(tags or ()),
                    entities=tuple(entities or ()),
                    summary=summary or "",
                    created_at=now,
                    last_used_at=now,
                )
            else:
                record = replace(
                    existing,
                    purpose=purpose if purpose is not None else existing.purpose,
                    tags=_merge(existing.tags, tags),
                    entities=_merge(existing.entities, entities),
                    summary=summary if summary is not None else existing.summary,
                )

            self._records[name] = record
            self.save()
            return record

    def touch(self, name: str) -> Optional[AgentRecord]:
        """Mark an agent as used now. Revives a dormant agent."""

        with self._lock:
            existing = self._records.get(name)
            if existing is None:
                return None
            record = replace(
                existing,
                last_used_at=self._now(),
                invocations=existing.invocations + 1,
                status="active" if existing.status == "dormant" else existing.status,
            )
            self._records[name] = record
            self.save()
            return record

    def set_status(self, name: str, status: AgentStatus) -> Optional[AgentRecord]:
        with self._lock:
            existing = self._records.get(name)
            if existing is None:
                return None
            record = replace(existing, status=status)
            self._records[name] = record
            self.save()
            return record

    def sweep_dormant(self) -> List[str]:
        """Move stale active agents to dormant. Returns the names moved."""

        cutoff = self._now() - self._dormant_after
        moved: List[str] = []
        with self._lock:
            for name, record in list(self._records.items()):
                if record.status == "active" and record.last_used_at < cutoff:
                    self._records[name] = replace(record, status="dormant")
                    moved.append(name)
            if moved:
                self.save()
        if moved:
            logger.info(f"Marked {len(moved)} agents dormant")
        return moved

    def clear(self) -> None:
        with self._lock:
            self._records = {}
            self.save()


def _merge(existing: tuple[str, ...], incoming: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Union preserving order, case-insensitively deduplicated."""

    if incoming is None:
        return existing
    seen = {item.casefold() for item in existing}
    merged = list(existing)
    for item in incoming:
        cleaned = str(item).strip()
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            merged.append(cleaned)
    return tuple(merged)


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_REGISTRY_PATH = _DATA_DIR / "execution_agents" / "registry.json"
_LEGACY_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_registry: Optional[AgentRegistry] = None
_factory_lock = threading.Lock()


def get_agent_registry() -> AgentRegistry:
    """Singleton accessor.

    On first use, seeds from the legacy ``roster.json`` when no registry exists
    yet, so an existing install keeps its agents.
    """

    global _registry
    if _registry is None:
        with _factory_lock:
            if _registry is None:
                source = _REGISTRY_PATH if _REGISTRY_PATH.exists() else _LEGACY_ROSTER_PATH
                registry = AgentRegistry(source)
                registry._path = _REGISTRY_PATH  # subsequent writes go to the new file
                _registry = registry
    return _registry


__all__ = [
    "AgentRecord",
    "AgentRegistry",
    "AgentStatus",
    "DEFAULT_DORMANT_AFTER",
    "get_agent_registry",
]
