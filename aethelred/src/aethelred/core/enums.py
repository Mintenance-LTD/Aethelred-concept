"""All enumerations for the Aethelred system."""

from enum import Enum


class DroneRole(str, Enum):
    MOTHER = "mother"
    RECON = "reconnaissance"
    ENGAGE = "engagement"
    EW = "electronic_warfare"
    RELAY = "relay"


class DroneStatus(str, Enum):
    ACTIVE = "active"
    DAMAGED = "damaged"
    DESTROYED = "destroyed"
    RTB = "returning_to_base"
    COMM_LOST = "communication_lost"


class ThreatType(str, Enum):
    SMALL_ARMS = "small_arms"
    AA_MISSILE = "anti_air_missile"
    EW_JAMMING = "ew_jamming"
    EW_SPOOFING = "ew_spoofing"
    LASER = "directed_energy"
    KINETIC_INTERCEPT = "kinetic_intercept"


class ThreatCategory(str, Enum):
    KINETIC = "kinetic"
    ELECTRONIC = "electronic"
    DIRECTED_ENERGY = "directed_energy"
    ENVIRONMENTAL = "environmental"


THREAT_TYPE_TO_CATEGORY: dict[ThreatType, ThreatCategory] = {
    ThreatType.SMALL_ARMS: ThreatCategory.KINETIC,
    ThreatType.AA_MISSILE: ThreatCategory.KINETIC,
    ThreatType.EW_JAMMING: ThreatCategory.ELECTRONIC,
    ThreatType.EW_SPOOFING: ThreatCategory.ELECTRONIC,
    ThreatType.LASER: ThreatCategory.DIRECTED_ENERGY,
    ThreatType.KINETIC_INTERCEPT: ThreatCategory.KINETIC,
}


class TacticalActionType(str, Enum):
    MOVE = "move"
    ENGAGE = "engage"
    EVADE = "evade"
    FORMATION_CHANGE = "formation_change"
    RECON = "recon"
    RELAY = "relay"
    RETREAT = "retreat"
    HOLD = "hold"


class FormationType(str, Enum):
    LINE = "line"
    WEDGE = "wedge"
    DIAMOND = "diamond"
    SCATTER = "scatter"
    COLUMN = "column"
    ORBIT = "orbit"


class EngagementOutcome(str, Enum):
    TARGET_DESTROYED = "target_destroyed"
    TARGET_DAMAGED = "target_damaged"
    MISS = "miss"
    UNIT_LOST = "unit_lost"
    MUTUAL_DESTRUCTION = "mutual_destruction"
    EVADED = "evaded"
