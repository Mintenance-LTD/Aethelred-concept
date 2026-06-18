"""Battlefield terrain, cover, and line-of-sight management."""

from __future__ import annotations

import numpy as np

from aethelred.config.settings import BattlefieldConfig
from aethelred.core.models import Vec2
from aethelred.utils.geometry import perlin_noise_2d


class Battlefield:
    """Manages the terrain grid and spatial properties."""

    def __init__(self, config: BattlefieldConfig) -> None:
        self.width = config.width
        self.height = config.height
        self.grid_resolution = config.grid_resolution
        self.terrain_type = config.terrain_type
        self.terrain_seed = config.terrain_seed

        self.elevation: np.ndarray = np.zeros(
            (config.grid_resolution, config.grid_resolution), dtype=np.float32
        )
        self.cover_map: np.ndarray = np.zeros(
            (config.grid_resolution, config.grid_resolution), dtype=np.float32
        )
        self.generate_terrain(config.terrain_seed, config.terrain_type)

    def generate_terrain(self, seed: int, terrain_type: str = "mixed") -> None:
        """Generate procedural terrain using value noise."""
        res = self.grid_resolution

        if terrain_type == "flat":
            self.elevation = np.zeros((res, res), dtype=np.float32)
            self.cover_map = np.zeros((res, res), dtype=np.float32)
        elif terrain_type == "urban":
            self.elevation = perlin_noise_2d((res, res), scale=5.0, seed=seed) * 20.0
            self.cover_map = (perlin_noise_2d((res, res), scale=3.0, seed=seed + 1) > 0.4).astype(
                np.float32
            ) * 0.8
        elif terrain_type == "forest":
            self.elevation = perlin_noise_2d((res, res), scale=8.0, seed=seed) * 10.0
            self.cover_map = perlin_noise_2d((res, res), scale=4.0, seed=seed + 1) * 0.6
        else:  # mixed
            self.elevation = perlin_noise_2d((res, res), scale=6.0, seed=seed) * 15.0
            cover_noise = perlin_noise_2d((res, res), scale=4.0, seed=seed + 1)
            self.cover_map = np.clip(cover_noise * 0.7, 0.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """Re-generate terrain (call if you need fresh terrain per episode)."""
        self.generate_terrain(self.terrain_seed, self.terrain_type)

    def _world_to_grid(self, pos: Vec2) -> tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(pos.x / self.width * (self.grid_resolution - 1))
        gy = int(pos.y / self.height * (self.grid_resolution - 1))
        gx = max(0, min(self.grid_resolution - 1, gx))
        gy = max(0, min(self.grid_resolution - 1, gy))
        return gy, gx

    def get_cover_at(self, position: Vec2) -> float:
        """Get cover value at a world position."""
        gy, gx = self._world_to_grid(position)
        return float(self.cover_map[gy, gx])

    def get_elevation_at(self, position: Vec2) -> float:
        """Get elevation at a world position."""
        gy, gx = self._world_to_grid(position)
        return float(self.elevation[gy, gx])

    def line_of_sight(self, from_pos: Vec2, to_pos: Vec2) -> bool:
        """Check if there is line-of-sight between two positions via raycasting."""
        from_elev = self.get_elevation_at(from_pos) + 5.0  # drone altitude offset
        to_elev = self.get_elevation_at(to_pos) + 5.0

        steps = max(
            int(from_pos.distance_to(to_pos) / (self.width / self.grid_resolution)), 2
        )

        for i in range(1, steps):
            t = i / steps
            check_pos = Vec2.lerp(from_pos, to_pos, t)
            terrain_elev = self.get_elevation_at(check_pos)
            ray_elev = from_elev + (to_elev - from_elev) * t
            if terrain_elev > ray_elev:
                return False
        return True

    def is_in_bounds(self, position: Vec2) -> bool:
        return 0 <= position.x <= self.width and 0 <= position.y <= self.height

    def clamp_to_bounds(self, position: Vec2) -> Vec2:
        return Vec2(
            x=max(0.0, min(self.width, position.x)),
            y=max(0.0, min(self.height, position.y)),
        )

    def get_terrain_as_array(self) -> np.ndarray:
        """Return combined terrain info as a single-channel array for the neural network."""
        return (self.elevation / 20.0 + self.cover_map) / 2.0
