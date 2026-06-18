"""Action space definitions for tactical decisions."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from aethelred.core.enums import FormationType, TacticalActionType
from aethelred.core.models import Vec2


class TacticalAction(BaseModel):
    """A single tactical decision for one unit or the swarm."""

    action_type: TacticalActionType
    target_unit_id: Optional[UUID] = None
    target_position: Optional[Vec2] = None
    target_threat_id: Optional[UUID] = None
    formation: Optional[FormationType] = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    parameters: dict[str, float] = Field(default_factory=dict)


class TacticalDecision(BaseModel):
    """Complete set of tactical decisions output by the AI for one timestep."""

    timestep: int
    actions: list[TacticalAction] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_embedding: Optional[list[float]] = None
