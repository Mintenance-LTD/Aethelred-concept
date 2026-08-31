"""Core data models, enumerations, and interfaces for Aethelred."""

from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.enums import (
    DroneRole,
    DroneStatus,
    EngagementOutcome,
    FormationType,
    TacticalActionType,
    ThreatCategory,
    ThreatType,
)
from aethelred.core.events import AdaptationEvent, EngagementEvent, LossEvent
from aethelred.core.models import (
    BattlefieldState,
    DroneState,
    ObjectiveState,
    TerrainCell,
    ThreatState,
    Vec2,
)

__all__ = [
    "AdaptationEvent",
    "BattlefieldState",
    "DroneRole",
    "DroneState",
    "DroneStatus",
    "EngagementEvent",
    "EngagementOutcome",
    "FormationType",
    "LossEvent",
    "ObjectiveState",
    "TacticalAction",
    "TacticalActionType",
    "TacticalDecision",
    "TerrainCell",
    "ThreatCategory",
    "ThreatState",
    "ThreatType",
    "Vec2",
]
