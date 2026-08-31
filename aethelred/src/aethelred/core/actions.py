"""Action space definitions for tactical decisions."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from aethelred.core.enums import FormationType, TacticalActionType
from aethelred.core.models import Vec2


class TacticalAction(BaseModel):
    """A single tactical decision for one unit or the swarm."""

    action_type: TacticalActionType
    target_unit_id: UUID | None = None
    target_position: Vec2 | None = None
    target_threat_id: UUID | None = None
    formation: FormationType | None = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    parameters: dict[str, float] = Field(default_factory=dict)


class TacticalDecision(BaseModel):
    """Complete set of tactical decisions output by the AI for one timestep."""

    timestep: int
    actions: list[TacticalAction] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_embedding: list[float] | None = None
