"""Hierarchical threat classification for transfer learning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from aethelred.core.enums import THREAT_TYPE_TO_CATEGORY, ThreatCategory, ThreatType
from aethelred.core.events import LossEvent


@dataclass
class ThreatProfile:
    """Accumulated knowledge about a threat type."""

    threat_type: ThreatType
    category: ThreatCategory
    encounter_count: int = 0
    kill_count: int = 0  # how many of our units this threat has destroyed
    neutralized_count: int = 0  # how many of this threat we've destroyed
    average_engagement_range: float = 0.0
    average_kill_range: float = 0.0
    is_mastered: bool = False  # True if adaptation success rate > threshold

    @property
    def lethality(self) -> float:
        """Exchange ratio: fraction of engagements where this threat killed us.

        Defined over total resolved engagements (our losses + our kills of this
        threat type) so the value is informative rather than trivially 1.0.
        """
        resolved = self.kill_count + self.neutralized_count
        if resolved == 0:
            return 0.0
        return self.kill_count / resolved

    @property
    def counter_effectiveness(self) -> float:
        """Exchange ratio: fraction of engagements where we neutralized this threat."""
        resolved = self.kill_count + self.neutralized_count
        if resolved == 0:
            return 0.0
        return self.neutralized_count / resolved


class HierarchicalThreatClassifier:
    """
    Classifies threats hierarchically for transfer learning.
    Knowledge about one threat type can transfer to related types
    within the same category.
    """

    def __init__(self, mastery_threshold: float = 0.7) -> None:
        self.mastery_threshold = mastery_threshold
        self.profiles: dict[ThreatType, ThreatProfile] = {}
        self._category_knowledge: dict[ThreatCategory, float] = defaultdict(float)

    def update_from_loss(self, loss: LossEvent) -> ThreatProfile:
        """Update threat knowledge from a loss event."""
        tt = loss.cause_of_loss
        if tt not in self.profiles:
            self.profiles[tt] = ThreatProfile(
                threat_type=tt,
                category=THREAT_TYPE_TO_CATEGORY.get(tt, ThreatCategory.KINETIC),
            )

        profile = self.profiles[tt]
        profile.encounter_count += 1
        profile.kill_count += 1
        profile.average_kill_range = (
            (profile.average_kill_range * (profile.kill_count - 1) + loss.engagement_range)
            / profile.kill_count
        )

        return profile

    def record_neutralization(self, threat_type: ThreatType) -> None:
        """Record that we successfully neutralized a threat."""
        if threat_type not in self.profiles:
            self.profiles[threat_type] = ThreatProfile(
                threat_type=threat_type,
                category=THREAT_TYPE_TO_CATEGORY.get(threat_type, ThreatCategory.KINETIC),
            )
        self.profiles[threat_type].neutralized_count += 1

        # Check mastery
        profile = self.profiles[threat_type]
        if profile.counter_effectiveness >= self.mastery_threshold:
            profile.is_mastered = True
            self._category_knowledge[profile.category] = max(
                self._category_knowledge[profile.category],
                profile.counter_effectiveness,
            )

    def is_new_threat(self, threat_type: ThreatType) -> bool:
        """Check if this is a previously unseen threat type."""
        return threat_type not in self.profiles

    def get_related_knowledge(self, threat_type: ThreatType) -> float:
        """
        Get transfer learning knowledge from related threat types.
        Returns a score [0, 1] indicating how much prior knowledge
        is available from the same category.
        """
        category = THREAT_TYPE_TO_CATEGORY.get(threat_type, ThreatCategory.KINETIC)

        # Direct knowledge
        if threat_type in self.profiles:
            return self.profiles[threat_type].counter_effectiveness

        # Category-level transfer
        return self._category_knowledge.get(category, 0.0) * 0.5  # 50% transfer

    def get_priority_threats(self) -> list[ThreatType]:
        """Get threat types ordered by danger (high lethality, low counter-effectiveness)."""
        active = [
            (tt, p) for tt, p in self.profiles.items()
            if not p.is_mastered and p.encounter_count > 0
        ]
        active.sort(key=lambda x: x[1].lethality - x[1].counter_effectiveness, reverse=True)
        return [tt for tt, _ in active]

    def get_summary(self) -> dict[str, dict]:
        """Get a summary of all threat knowledge."""
        return {
            tt.value: {
                "encounters": p.encounter_count,
                "kills": p.kill_count,
                "neutralized": p.neutralized_count,
                "lethality": round(p.lethality, 3),
                "counter_eff": round(p.counter_effectiveness, 3),
                "mastered": p.is_mastered,
            }
            for tt, p in self.profiles.items()
        }
