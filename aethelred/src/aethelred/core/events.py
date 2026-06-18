"""Event types for engagements, losses, and adaptations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from aethelred.core.enums import EngagementOutcome, ThreatType

if TYPE_CHECKING:
    pass


class EngagementEvent(BaseModel):
    """Record of an engagement between friendly and threat."""

    timestep: int
    friendly_id: UUID
    threat_id: UUID
    threat_type: Optional[ThreatType] = None
    outcome: EngagementOutcome
    friendly_health_before: float
    friendly_health_after: float
    threat_health_before: float
    threat_health_after: float
    distance: float

    model_config = {"arbitrary_types_allowed": True}


class LossEvent(BaseModel):
    """High-value training data: what happened when a unit was destroyed."""

    timestep: int
    lost_unit_id: UUID
    lost_unit_role: str
    lost_unit_position_x: float
    lost_unit_position_y: float
    cause_of_loss: ThreatType
    killing_threat_id: Optional[UUID] = None
    killing_threat_position_x: Optional[float] = None
    killing_threat_position_y: Optional[float] = None
    engagement_range: float = 0.0
    final_heading: float = 0.0
    final_velocity_x: float = 0.0
    final_velocity_y: float = 0.0
    num_friendlies_alive: int = 0
    num_threats_active: int = 0

    model_config = {"arbitrary_types_allowed": True}


class AdaptationEvent(BaseModel):
    """Record of a model adaptation triggered by the learning loop."""

    timestep: int
    trigger_loss_count: int
    threat_types_addressed: list[ThreatType]
    adaptation_method: str = "gradient_update"
    performance_before: dict[str, float] = Field(default_factory=dict)
    performance_after: dict[str, float] = Field(default_factory=dict)
