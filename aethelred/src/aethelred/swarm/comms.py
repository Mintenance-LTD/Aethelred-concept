"""Simulated communication system with degradation and jamming."""

from __future__ import annotations

import random
from dataclasses import dataclass
from uuid import UUID

from aethelred.core.models import DroneState, Vec2


@dataclass
class CommStatus:
    """Communication status for a single unit."""

    unit_id: UUID
    connected: bool
    signal_strength: float  # 0.0 to 1.0
    latency_ms: float
    bandwidth_kbps: float


class CommChannel:
    """Simulates communication between mother drone and swarm units."""

    def __init__(
        self,
        base_range: float = 500.0,
        base_bandwidth_kbps: float = 1000.0,
        base_latency_ms: float = 10.0,
        jam_degradation: float = 0.8,
    ) -> None:
        self.base_range = base_range
        self.base_bandwidth = base_bandwidth_kbps
        self.base_latency = base_latency_ms
        self.jam_degradation = jam_degradation
        self._jamming_zones: list[tuple[Vec2, float]] = []  # (center, radius)
        self._global_degradation: float = 0.0

    def set_global_degradation(self, level: float) -> None:
        """Set global comms degradation (0 = clear, 1 = total blackout)."""
        self._global_degradation = max(0.0, min(1.0, level))

    def add_jamming_zone(self, center: Vec2, radius: float) -> None:
        self._jamming_zones.append((center, radius))

    def clear_jamming(self) -> None:
        self._jamming_zones.clear()

    def get_status(
        self, mother_pos: Vec2, unit: DroneState
    ) -> CommStatus:
        """Compute communication status between mother and unit."""
        dist = mother_pos.distance_to(unit.position)
        effective_range = min(self.base_range, unit.comm_range)

        # Base signal strength (distance falloff)
        if dist > effective_range:
            signal = 0.0
        else:
            signal = 1.0 - (dist / effective_range) ** 2

        # Jamming reduction
        if unit.is_jammed:
            signal *= (1.0 - self.jam_degradation)

        for zone_center, zone_radius in self._jamming_zones:
            if unit.position.distance_to(zone_center) < zone_radius:
                signal *= (1.0 - self.jam_degradation)
                break

        # Global degradation
        signal *= (1.0 - self._global_degradation)

        # Random noise
        signal = max(0.0, signal + random.gauss(0, 0.02))

        connected = signal > 0.1
        latency = self.base_latency / max(signal, 0.01) if connected else float("inf")
        bandwidth = self.base_bandwidth * signal if connected else 0.0

        return CommStatus(
            unit_id=unit.id,
            connected=connected,
            signal_strength=min(1.0, max(0.0, signal)),
            latency_ms=min(5000.0, latency),
            bandwidth_kbps=bandwidth,
        )

    def can_reach(self, mother_pos: Vec2, unit: DroneState) -> bool:
        """Quick check if communication is possible."""
        return self.get_status(mother_pos, unit).connected

    def estimate_delta_transfer_time(
        self, mother_pos: Vec2, unit: DroneState, delta_size_bytes: int
    ) -> float:
        """Estimate time in ms to transfer a model delta to a unit."""
        status = self.get_status(mother_pos, unit)
        if not status.connected or status.bandwidth_kbps <= 0:
            return float("inf")
        size_kb = delta_size_bytes / 1024.0
        return (size_kb / status.bandwidth_kbps) * 1000.0 + status.latency_ms
