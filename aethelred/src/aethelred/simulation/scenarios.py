"""Scenario loading from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aethelred.config.settings import SimulationConfig


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load a scenario from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def apply_scenario_to_config(
    base_config: SimulationConfig, scenario: dict[str, Any]
) -> SimulationConfig:
    """Override simulation config with scenario-specific settings."""
    if "battlefield" in scenario:
        for key, value in scenario["battlefield"].items():
            if hasattr(base_config.battlefield, key):
                setattr(base_config.battlefield, key, value)

    if "physics" in scenario:
        for key, value in scenario["physics"].items():
            if hasattr(base_config.physics, key):
                setattr(base_config.physics, key, value)

    if "swarm" in scenario:
        for key, value in scenario["swarm"].items():
            if hasattr(base_config.swarm, key):
                setattr(base_config.swarm, key, value)

    if "threats" in scenario:
        for key, value in scenario["threats"].items():
            if hasattr(base_config.threats, key):
                setattr(base_config.threats, key, value)

    if "max_steps" in scenario:
        base_config.max_steps = scenario["max_steps"]

    return base_config
