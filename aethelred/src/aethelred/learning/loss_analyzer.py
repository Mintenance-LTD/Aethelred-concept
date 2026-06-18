"""Loss analysis: post-mortem reconstruction of unit destruction events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from aethelred.core.enums import ThreatCategory, ThreatType, THREAT_TYPE_TO_CATEGORY
from aethelred.core.events import LossEvent
from aethelred.core.models import Vec2
from aethelred.utils.geometry import compute_centroid


@dataclass
class LossAnalysis:
    """Result of analyzing unit losses."""

    threat_types: list[ThreatType]
    threat_categories: list[ThreatCategory]
    kill_zone_centers: list[Vec2]
    average_engagement_range: float
    total_losses: int
    losses_by_type: dict[str, int]
    recommended_evasion_directions: list[Vec2]
    patterns: list[str]


class LossAnalyzer:
    """
    Phase 2: Reconstructs what happened when units were destroyed.
    Extracts patterns and generates training data.
    """

    def __init__(self, cluster_radius: float = 100.0) -> None:
        self.cluster_radius = cluster_radius
        self._all_losses: list[LossEvent] = []

    def analyze(self, losses: list[LossEvent]) -> LossAnalysis:
        """Full post-mortem analysis of recent losses."""
        self._all_losses.extend(losses)

        if not losses:
            return LossAnalysis(
                threat_types=[],
                threat_categories=[],
                kill_zone_centers=[],
                average_engagement_range=0.0,
                total_losses=0,
                losses_by_type={},
                recommended_evasion_directions=[],
                patterns=[],
            )

        # 1. Identify threat types involved
        threat_types = list(set(loss.cause_of_loss for loss in losses))
        threat_categories = list(set(
            THREAT_TYPE_TO_CATEGORY.get(tt, ThreatCategory.KINETIC)
            for tt in threat_types
        ))

        # 2. Find kill zones (cluster loss positions)
        loss_positions = [
            Vec2(x=loss.lost_unit_position_x, y=loss.lost_unit_position_y)
            for loss in losses
        ]
        kill_zones = self._cluster_positions(loss_positions)

        # 3. Average engagement range
        ranges = [loss.engagement_range for loss in losses if loss.engagement_range > 0]
        avg_range = sum(ranges) / len(ranges) if ranges else 0.0

        # 4. Count by type
        type_counts = Counter(loss.cause_of_loss.value for loss in losses)

        # 5. Compute evasion directions (perpendicular to threat-to-unit vectors)
        evasion_dirs = []
        for loss in losses:
            if loss.killing_threat_position_x is not None:
                threat_pos = Vec2(
                    x=loss.killing_threat_position_x,
                    y=loss.killing_threat_position_y or 0.0,
                )
                unit_pos = Vec2(x=loss.lost_unit_position_x, y=loss.lost_unit_position_y)
                away = unit_pos - threat_pos
                if away.magnitude() > 0.1:
                    # Perpendicular directions (left and right of threat axis)
                    perp1 = away.rotated(1.5708)  # +90 degrees
                    perp2 = away.rotated(-1.5708)  # -90 degrees
                    evasion_dirs.extend([perp1.normalized(), perp2.normalized()])

        # 6. Detect patterns
        patterns = self._detect_patterns(losses)

        return LossAnalysis(
            threat_types=threat_types,
            threat_categories=threat_categories,
            kill_zone_centers=kill_zones,
            average_engagement_range=avg_range,
            total_losses=len(losses),
            losses_by_type=dict(type_counts),
            recommended_evasion_directions=evasion_dirs[:6],
            patterns=patterns,
        )

    def _cluster_positions(self, positions: list[Vec2]) -> list[Vec2]:
        """Simple clustering: merge nearby positions into kill zone centers."""
        if not positions:
            return []

        clusters: list[list[Vec2]] = []
        assigned = [False] * len(positions)

        for i, pos in enumerate(positions):
            if assigned[i]:
                continue
            cluster = [pos]
            assigned[i] = True

            for j in range(i + 1, len(positions)):
                if not assigned[j] and pos.distance_to(positions[j]) < self.cluster_radius:
                    cluster.append(positions[j])
                    assigned[j] = True

            clusters.append(cluster)

        return [compute_centroid(c) for c in clusters]

    def _detect_patterns(self, losses: list[LossEvent]) -> list[str]:
        """Detect tactical patterns in the losses."""
        patterns = []

        # Check for concentrated losses (kill zone)
        positions = [
            Vec2(x=loss.lost_unit_position_x, y=loss.lost_unit_position_y)
            for loss in losses
        ]
        if len(positions) >= 3:
            centroid = compute_centroid(positions)
            avg_dist = sum(p.distance_to(centroid) for p in positions) / len(positions)
            if avg_dist < 80.0:
                patterns.append("kill_zone_concentrated")

        # Check for consistent threat type
        types = [loss.cause_of_loss for loss in losses]
        if len(set(types)) == 1 and len(types) >= 2:
            patterns.append(f"single_threat_type_{types[0].value}")

        # Check for close-range kills
        close_kills = sum(1 for loss in losses if loss.engagement_range < 30.0)
        if close_kills > len(losses) * 0.5:
            patterns.append("close_range_ambush")

        # Check for long-range kills
        long_kills = sum(1 for loss in losses if loss.engagement_range > 80.0)
        if long_kills > len(losses) * 0.5:
            patterns.append("long_range_engagement")

        return patterns

    @property
    def total_losses_analyzed(self) -> int:
        return len(self._all_losses)
