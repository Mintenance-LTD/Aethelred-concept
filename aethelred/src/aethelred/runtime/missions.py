"""Durable, revisioned registration for operational missions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from aethelred.core.models import Vec2
from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.operational import Mission, MissionCapability, OperatingArea


class MissionRegistryError(ValueError):
    """Raised when an operational mission is absent, malformed, or superseded."""


@dataclass(frozen=True)
class MissionRegistration:
    """A durable, operator-accountable registration of one mission revision."""

    mission: Mission
    operator: str
    rationale: str


class MissionRegistry:
    """Persist and recover the latest approved mission revision per mission ID."""

    def __init__(self, journal: JsonlAuditJournal) -> None:
        self._journal = journal
        self._missions: dict[UUID, MissionRegistration] = {}
        self.recover()

    def register(self, mission: Mission, operator: str, rationale: str) -> MissionRegistration:
        """Durably register a new mission or a strictly newer revision."""
        if not operator.strip() or not rationale.strip():
            raise MissionRegistryError("Mission operator identity and rationale are required")
        existing = self._missions.get(mission.mission_id)
        if existing is not None and mission.revision <= existing.mission.revision:
            raise MissionRegistryError("Mission revision must strictly increase")
        registration = MissionRegistration(mission, operator, rationale)
        self._missions[mission.mission_id] = registration
        self._journal.record(
            "mission_registered",
            correlation_id=str(mission.mission_id),
            payload={
                "mission": self._encode_mission(mission),
                "operator": operator,
                "rationale": rationale,
            },
        )
        return registration

    def require_registered(self, mission: Mission) -> Mission:
        """Return the current registered mission, rejecting unknown/stale objects."""
        try:
            registration = self._missions[mission.mission_id]
        except KeyError as error:
            raise MissionRegistryError("Mission is not registered") from error
        if registration.mission != mission:
            raise MissionRegistryError("Mission is not the current registered revision")
        return registration.mission

    def recover(self) -> None:
        """Reconstruct current mission revisions from the verified audit journal."""
        self._missions.clear()
        for event in self._journal.read_all():
            if event.get("event_type") != "mission_registered":
                continue
            try:
                mission_id = UUID(str(event["correlation_id"]))
                payload = event["payload"]
                if not isinstance(payload, dict):
                    raise TypeError("mission payload must be a mapping")
                mission = self._decode_mission(payload["mission"])
                operator = str(payload["operator"])
                rationale = str(payload["rationale"])
                if mission.mission_id != mission_id or not operator.strip() or not rationale.strip():
                    raise ValueError("mission registration does not match audit identity")
            except (KeyError, TypeError, ValueError) as error:
                raise MissionRegistryError("Invalid mission registration audit record") from error
            existing = self._missions.get(mission_id)
            if existing is not None and mission.revision <= existing.mission.revision:
                raise MissionRegistryError("Mission audit revisions are not strictly increasing")
            self._missions[mission_id] = MissionRegistration(mission, operator, rationale)

    @staticmethod
    def _encode_mission(mission: Mission) -> dict[str, Any]:
        return {
            "mission_id": str(mission.mission_id),
            "revision": mission.revision,
            "valid_from": mission.valid_from,
            "valid_until": mission.valid_until,
            "allowed_capabilities": sorted(capability.value for capability in mission.allowed_capabilities),
            "assigned_vehicle_ids": sorted(mission.assigned_vehicle_ids),
            "operating_area": {
                "minimum": {"x": mission.operating_area.minimum.x, "y": mission.operating_area.minimum.y},
                "maximum": {"x": mission.operating_area.maximum.x, "y": mission.operating_area.maximum.y},
            },
            "authorised_issuer_ids": sorted(mission.authorised_issuer_ids),
        }

    @staticmethod
    def _decode_mission(raw: object) -> Mission:
        if not isinstance(raw, dict):
            raise TypeError("mission must be a mapping")
        area = raw["operating_area"]
        if not isinstance(area, dict):
            raise TypeError("operating area must be a mapping")
        minimum = area["minimum"]
        maximum = area["maximum"]
        if not isinstance(minimum, dict) or not isinstance(maximum, dict):
            raise TypeError("operating-area bounds must be mappings")
        return Mission(
            mission_id=UUID(str(raw["mission_id"])),
            revision=int(raw["revision"]),
            valid_from=datetime.fromisoformat(str(raw["valid_from"])),
            valid_until=datetime.fromisoformat(str(raw["valid_until"])),
            allowed_capabilities=frozenset(
                MissionCapability(str(value)) for value in raw["allowed_capabilities"]
            ),
            assigned_vehicle_ids=frozenset(str(value) for value in raw["assigned_vehicle_ids"]),
            operating_area=OperatingArea(
                minimum=Vec2(x=float(minimum["x"]), y=float(minimum["y"])),
                maximum=Vec2(x=float(maximum["x"]), y=float(maximum["y"])),
            ),
            authorised_issuer_ids=frozenset(str(value) for value in raw["authorised_issuer_ids"]),
        )
