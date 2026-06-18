"""Geometric formation patterns for swarm units."""

from __future__ import annotations

import math
from uuid import UUID

from aethelred.core.enums import FormationType
from aethelred.core.models import Vec2


class FormationManager:
    """Computes formation positions for swarm units."""

    def compute_positions(
        self,
        formation: FormationType,
        center: Vec2,
        unit_ids: list[UUID],
        spacing: float = 30.0,
        heading: float = 0.0,
    ) -> dict[UUID, Vec2]:
        """Compute target positions for each unit in the formation."""
        n = len(unit_ids)
        if n == 0:
            return {}

        offsets = self._get_offsets(formation, n, spacing)

        # Rotate offsets by heading
        positions = {}
        for uid, offset in zip(unit_ids, offsets):
            rotated = offset.rotated(heading)
            positions[uid] = center + rotated

        return positions

    def _get_offsets(
        self, formation: FormationType, n: int, spacing: float
    ) -> list[Vec2]:
        """Get relative offsets for a formation type."""
        if formation == FormationType.LINE:
            return self._line_offsets(n, spacing)
        elif formation == FormationType.WEDGE:
            return self._wedge_offsets(n, spacing)
        elif formation == FormationType.DIAMOND:
            return self._diamond_offsets(n, spacing)
        elif formation == FormationType.SCATTER:
            return self._scatter_offsets(n, spacing)
        elif formation == FormationType.COLUMN:
            return self._column_offsets(n, spacing)
        elif formation == FormationType.ORBIT:
            return self._orbit_offsets(n, spacing)
        else:
            return self._line_offsets(n, spacing)

    def _line_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """Units in a horizontal line."""
        offsets = []
        start = -(n - 1) / 2.0 * spacing
        for i in range(n):
            offsets.append(Vec2(x=0.0, y=start + i * spacing))
        return offsets

    def _wedge_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """V-shaped formation with leader at front."""
        offsets = [Vec2(x=0.0, y=0.0)]  # leader at center front
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets.append(Vec2(
                x=-row * spacing * 0.7,
                y=side * row * spacing * 0.5,
            ))
        return offsets

    def _diamond_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """Diamond/rhombus formation."""
        offsets = [Vec2(x=spacing, y=0.0)]  # point
        if n > 1:
            offsets.append(Vec2(x=-spacing, y=0.0))  # rear
        if n > 2:
            offsets.append(Vec2(x=0.0, y=spacing))  # right
        if n > 3:
            offsets.append(Vec2(x=0.0, y=-spacing))  # left

        # Additional units fill inner positions
        for i in range(4, n):
            angle = (2 * math.pi * i) / (n - 4 + 4)
            r = spacing * 0.5
            offsets.append(Vec2(x=r * math.cos(angle), y=r * math.sin(angle)))

        return offsets

    def _scatter_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """Dispersed positions for reducing concentrated vulnerability."""
        import random
        rng = random.Random(42)  # deterministic scatter
        offsets = []
        for _ in range(n):
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(spacing * 0.5, spacing * 2.0)
            offsets.append(Vec2(x=r * math.cos(angle), y=r * math.sin(angle)))
        return offsets

    def _column_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """Single-file column formation."""
        offsets = []
        for i in range(n):
            offsets.append(Vec2(x=-i * spacing, y=0.0))
        return offsets

    def _orbit_offsets(self, n: int, spacing: float) -> list[Vec2]:
        """Circular orbit pattern."""
        offsets = []
        for i in range(n):
            angle = (2 * math.pi * i) / n
            offsets.append(Vec2(
                x=spacing * math.cos(angle),
                y=spacing * math.sin(angle),
            ))
        return offsets
