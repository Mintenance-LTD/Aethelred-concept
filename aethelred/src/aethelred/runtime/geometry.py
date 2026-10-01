"""Immutable geometry for operational contracts, independent of simulation."""

from dataclasses import dataclass
from math import hypot
from typing import Protocol


class Coordinates(Protocol):
    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


@dataclass(frozen=True)
class Position:
    """A copied coordinate value; caller-owned vectors cannot change it."""

    x: float
    y: float

    @classmethod
    def copy_of(cls, value: Coordinates) -> "Position":
        return cls(float(value.x), float(value.y))

    def distance_to(self, other: Coordinates) -> float:
        return hypot(self.x - other.x, self.y - other.y)
