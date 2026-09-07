"""The agent registry: records, lifecycle, persistence and migration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from server.services.attention.registry import AgentRecord, AgentRegistry


class Clock:
    """A hand-cranked clock, so lifecycle tests don't sleep or freeze the world."""

    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def registry(tmp_path, clock) -> AgentRegistry:
    return AgentRegistry(tmp_path / "registry.json", now=clock, dormant_after=timedelta(days=14))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def test_upsert_creates_a_record_with_timestamps(registry, clock):
    record = registry.upsert("Email to Alice", purpose="lunch thread", entities=["Alice"])

    assert record.name == "Email to Alice"
    assert record.purpose == "lunch thread"
    assert record.entities == ("Alice",)
    assert record.status == "active"
    assert record.invocations == 0
    assert record.created_at == clock.now


def test_upsert_enriches_without_clobbering(registry):
    """Metadata arrives in pieces, so a later write must not erase an earlier one."""

    registry.upsert("Email to Alice", purpose="lunch thread", entities=["Alice"])
    updated = registry.upsert("Email to Alice", entities=["alice@example.com"], summary="drafted")

    assert updated.purpose == "lunch thread", "purpose must survive an update that omits it"
    assert updated.entities == ("Alice", "alice@example.com"), "entities accumulate"
    assert updated.summary == "drafted"


def test_entity_merge_is_case_insensitive(registry):
    registry.upsert("Email to Alice", entities=["Alice"])
    updated = registry.upsert("Email to Alice", entities=["alice", "ALICE", "Bob"])

    assert updated.entities == ("Alice", "Bob")


def test_empty_name_is_rejected(registry):
    with pytest.raises(ValueError):
        registry.upsert("   ")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_touch_records_usage(registry, clock):
    registry.upsert("Email to Alice")
    clock.advance(hours=3)

    touched = registry.touch("Email to Alice")

    assert touched.invocations == 1
    assert touched.last_used_at == clock.now


def test_touch_on_a_missing_agent_returns_none(registry):
    assert registry.touch("Never Existed") is None


def test_sweep_marks_stale_agents_dormant(registry, clock):
    registry.upsert("Stale Agent")
    clock.advance(days=10)
    registry.upsert("Recent Agent")
    clock.advance(days=5)  # Stale is now 15 days old, Recent is 5

    moved = registry.sweep_dormant()

    assert moved == ["Stale Agent"]
    assert registry.get("Stale Agent").status == "dormant"
    assert registry.get("Recent Agent").status == "active"


def test_dormant_agents_leave_the_shortlist_but_not_the_registry(registry, clock):
    registry.upsert("Stale Agent")
    clock.advance(days=20)
    registry.sweep_dormant()

    assert "Stale Agent" not in [record.name for record in registry.selectable()]
    # Still addressable by name -- dormancy hides, it does not delete.
    assert registry.get("Stale Agent") is not None
    assert "Stale Agent" in registry.names()


def test_touching_a_dormant_agent_revives_it(registry, clock):
    registry.upsert("Stale Agent")
    clock.advance(days=20)
    registry.sweep_dormant()

    revived = registry.touch("Stale Agent")

    assert revived.status == "active"
    assert "Stale Agent" in [record.name for record in registry.selectable()]


def test_archived_agents_are_excluded_by_default(registry):
    registry.upsert("Done Agent")
    registry.set_status("Done Agent", "archived")

    assert registry.all() == []
    assert len(registry.all(include_archived=True)) == 1


def test_selectable_is_ordered_most_recent_first(registry, clock):
    registry.upsert("First")
    clock.advance(hours=1)
    registry.upsert("Second")
    clock.advance(hours=1)
    registry.touch("First")

    assert [record.name for record in registry.selectable()] == ["First", "Second"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_records_survive_a_reload(tmp_path, clock):
    path = tmp_path / "registry.json"
    original = AgentRegistry(path, now=clock)
    original.upsert("Email to Alice", purpose="lunch", entities=["Alice"], tags=["email"])
    original.touch("Email to Alice")

    reloaded = AgentRegistry(path, now=clock)
    record = reloaded.get("Email to Alice")

    assert record.purpose == "lunch"
    assert record.entities == ("Alice",)
    assert record.tags == ("email",)
    assert record.invocations == 1


def test_a_corrupt_file_starts_empty_rather_than_crashing(tmp_path, clock):
    path = tmp_path / "registry.json"
    path.write_text("{ not json at all", encoding="utf-8")

    registry = AgentRegistry(path, now=clock)

    assert registry.all() == []


def test_malformed_entries_are_skipped_not_fatal(tmp_path, clock):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "agents": [
                    {"name": "Good Agent", "purpose": "fine"},
                    {"purpose": "no name, should be skipped"},
                    "not even an object",
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = AgentRegistry(path, now=clock)

    assert [record.name for record in registry.all()] == ["Good Agent"]


def test_legacy_flat_roster_is_migrated(tmp_path, clock):
    """An existing install must keep its agents when the registry lands."""

    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Email to Alice", "Vercel Job Offer", ""]), encoding="utf-8")

    registry = AgentRegistry(path, now=clock)

    assert sorted(registry.names()) == ["Email to Alice", "Vercel Job Offer"]
    migrated = registry.get("Email to Alice")
    assert migrated.status == "active"
    assert migrated.created_at == clock.now
    assert migrated.purpose == "", "the legacy roster carries no metadata to migrate"


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path, clock):
    path = tmp_path / "registry.json"
    registry = AgentRegistry(path, now=clock)
    registry.upsert("Email to Alice")

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    assert json.loads(path.read_text())["version"] == 1


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_record_round_trips_through_dict(clock):
    record = AgentRecord(
        name="Email to Alice",
        purpose="lunch",
        tags=("email",),
        entities=("Alice",),
        summary="drafted and sent",
        status="dormant",
        invocations=4,
        created_at=clock.now,
        last_used_at=clock.now,
    )

    restored = AgentRecord.from_dict(record.to_dict(), fallback_time=clock.now)

    assert restored == record
