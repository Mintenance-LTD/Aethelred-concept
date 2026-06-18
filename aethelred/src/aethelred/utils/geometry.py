"""Geometry and vector math utilities."""

from __future__ import annotations

import math

import numpy as np

from aethelred.core.models import Vec2


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(a: float, b: float) -> float:
    """Shortest signed angular difference from a to b."""
    return normalize_angle(b - a)


def point_in_circle(point: Vec2, center: Vec2, radius: float) -> bool:
    return point.distance_to(center) <= radius


def line_segment_intersection(
    p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2
) -> tuple[bool, float]:
    """Check if line segments (p1,p2) and (p3,p4) intersect. Returns (intersects, t)."""
    d1 = p2 - p1
    d2 = p4 - p3

    denom = d1.x * d2.y - d1.y * d2.x
    if abs(denom) < 1e-10:
        return False, 0.0

    t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / denom
    u = ((p3.x - p1.x) * d1.y - (p3.y - p1.y) * d1.x) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return True, t
    return False, 0.0


def compute_centroid(positions: list[Vec2]) -> Vec2:
    """Compute the centroid of a set of positions."""
    if not positions:
        return Vec2()
    avg_x = sum(p.x for p in positions) / len(positions)
    avg_y = sum(p.y for p in positions) / len(positions)
    return Vec2(x=avg_x, y=avg_y)


def generate_circle_positions(
    center: Vec2, radius: float, count: int, start_angle: float = 0.0
) -> list[Vec2]:
    """Generate evenly spaced positions on a circle."""
    positions = []
    for i in range(count):
        angle = start_angle + (2 * math.pi * i / count)
        positions.append(
            Vec2(
                x=center.x + radius * math.cos(angle),
                y=center.y + radius * math.sin(angle),
            )
        )
    return positions


def perlin_noise_2d(
    shape: tuple[int, int], scale: float = 10.0, seed: int = 42
) -> np.ndarray:
    """Generate 2D Perlin-like noise using numpy (simplified value noise)."""
    rng = np.random.RandomState(seed)
    h, w = shape

    # Generate random gradients at grid points
    grid_h = int(h / scale) + 2
    grid_w = int(w / scale) + 2
    gradients = rng.randn(grid_h, grid_w)

    result = np.zeros(shape, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            gy = y / scale
            gx = x / scale
            gy0 = int(gy)
            gx0 = int(gx)
            gy1 = gy0 + 1
            gx1 = gx0 + 1

            fy = gy - gy0
            fx = gx - gx0

            # Smoothstep interpolation
            fy = fy * fy * (3 - 2 * fy)
            fx = fx * fx * (3 - 2 * fx)

            n00 = gradients[gy0, gx0]
            n10 = gradients[gy1, gx0]
            n01 = gradients[gy0, gx1]
            n11 = gradients[gy1, gx1]

            nx0 = n00 * (1 - fx) + n01 * fx
            nx1 = n10 * (1 - fx) + n11 * fx
            result[y, x] = nx0 * (1 - fy) + nx1 * fy

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result
