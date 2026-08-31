"""Configuration system for Aethelred."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when a runtime configuration is malformed or unsafe to use."""



@dataclass
class BattlefieldConfig:
    width: float = 1000.0
    height: float = 1000.0
    grid_resolution: int = 50
    terrain_type: str = "mixed"
    terrain_seed: int = 42


@dataclass
class PhysicsConfig:
    dt: float = 0.1
    hit_probability_base: float = 0.3
    cover_damage_reduction: float = 0.5
    fuel_consumption_rate: float = 0.001
    damage_per_hit: float = 0.3


@dataclass
class SwarmConfig:
    num_recon: int = 4
    num_engage: int = 6
    num_ew: int = 2
    num_relay: int = 2
    initial_formation: str = "diamond"
    # When False, every unit obeys the central command (no comms-loss autonomy
    # fallback) — used to give the learned policy full authority in experiments.
    autonomy_enabled: bool = True

    @property
    def total_units(self) -> int:
        return self.num_recon + self.num_engage + self.num_ew + self.num_relay


@dataclass
class ThreatSpawnerConfig:
    initial_threats: int = 3
    max_threats: int = 10
    spawn_interval: int = 50
    default_profile: str = "patrol"


@dataclass
class RenderConfig:
    enabled: bool = True
    width_px: int = 800
    height_px: int = 800
    show_grid: bool = True
    show_ranges: bool = False
    show_los: bool = False
    fps: int = 30


@dataclass
class SimulationConfig:
    battlefield: BattlefieldConfig = field(default_factory=BattlefieldConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    threats: ThreatSpawnerConfig = field(default_factory=ThreatSpawnerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    max_steps: int = 1000
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "mission_progress": 1.0,
            "survival": 0.5,
            "efficiency": 0.3,
            "threat_neutralized": 0.8,
            "adaptation_bonus": 0.2,
            "loss_penalty": 0.3,
        }
    )


@dataclass
class StateEncoderConfig:
    friendly_feature_dim: int = 16
    threat_feature_dim: int = 18
    objective_feature_dim: int = 4
    entity_embed_dim: int = 64
    terrain_embed_dim: int = 32
    global_embed_dim: int = 16
    state_embed_dim: int = 256
    num_attention_heads: int = 4
    max_friendlies: int = 32
    max_threats: int = 16
    max_objectives: int = 8


@dataclass
class DecisionTransformerConfig:
    state_embed_dim: int = 256
    action_embed_dim: int = 64
    return_embed_dim: int = 32
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    context_length: int = 20
    dropout: float = 0.1
    max_units: int = 32
    max_threats: int = 16
    num_action_types: int = 8
    num_formations: int = 6
    max_episode_length: int = 1000


@dataclass
class MAMLConfig:
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_lr: float = 1e-4
    tasks_per_batch: int = 4


@dataclass
class EWCConfig:
    lambda_ewc: float = 1000.0
    fisher_sample_size: int = 200


@dataclass
class AdaptationConfig:
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    replay_buffer_capacity: int = 10000
    replay_alpha: float = 0.6
    enable_maml: bool = True
    enable_ewc: bool = True


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    gradient_clip: float = 1.0
    warmup_steps: int = 200
    total_training_steps: int = 100000  # horizon for cosine LR decay
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    target_return: float = 10.0  # return-to-go conditioning for the policy
    update_interval: int = 256  # steps collected before a PPO update
    normalize_rewards: bool = True  # scale rewards by running return std (stability)
    aux_coef: float = 0.0  # weight of the threat-composition auxiliary loss (0 = off)
    device: str = "cpu"


@dataclass
class LearningLoopConfig:
    loss_threshold: int = 3
    adaptation_interval: int = 100
    buffer_size: int = 10000
    enable_prediction: bool = True
    enable_maml: bool = True
    enable_ewc: bool = True


@dataclass
class AethelredConfig:
    """Root configuration."""

    schema_version: int = 1
    project_name: str = "aethelred"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    tactical_ai: DecisionTransformerConfig = field(default_factory=DecisionTransformerConfig)
    state_encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    training: TrainerConfig = field(default_factory=TrainerConfig)
    learning_loop: LearningLoopConfig = field(default_factory=LearningLoopConfig)
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AethelredConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration root must be a mapping")
        if "schema_version" not in data:
            raise ConfigurationError("Configuration must declare schema_version")
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AethelredConfig:
        config = cls()
        cls._apply_mapping(config, data, path="config")

        root_device_supplied = "device" in data
        training_data = data.get("training")
        training_device_supplied = isinstance(training_data, dict) and "device" in training_data
        if root_device_supplied and training_device_supplied and config.device != config.training.device:
            raise ConfigurationError(
                "config.device and config.training.device must match when both are supplied"
            )
        if root_device_supplied:
            config.training.device = config.device
        elif training_device_supplied:
            config.device = config.training.device

        cls._validate(config)
        return config

    @staticmethod
    def _apply_mapping(target: Any, values: Mapping[str, Any], path: str) -> None:
        """Apply a mapping recursively while rejecting unknown configuration keys."""
        known_fields = {item.name for item in fields(target)}
        unknown_keys = sorted(set(values) - known_fields)
        if unknown_keys:
            unknown = ", ".join(f"{path}.{key}" for key in unknown_keys)
            raise ConfigurationError(f"Unknown configuration key(s): {unknown}")

        for key, value in values.items():
            current = getattr(target, key)
            key_path = f"{path}.{key}"
            if is_dataclass(current):
                if not isinstance(value, Mapping):
                    raise ConfigurationError(f"{key_path} must be a mapping")
                AethelredConfig._apply_mapping(current, value, key_path)
            elif isinstance(current, dict):
                if not isinstance(value, Mapping):
                    raise ConfigurationError(f"{key_path} must be a mapping")
                setattr(target, key, dict(value))
            else:
                setattr(target, key, value)

    @staticmethod
    def _validate(config: AethelredConfig) -> None:
        """Validate safety-critical and simulation invariants before execution."""
        def require_positive(value: float, name: str) -> None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive finite number")

        def require_unit_interval(value: float, name: str) -> None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationError(f"{name} must be between 0 and 1")

        if type(config.schema_version) is not int or config.schema_version != 1:
            raise ConfigurationError("Unsupported configuration schema_version")
        for value, name in (
            (config.simulation.battlefield.width, "battlefield width"),
            (config.simulation.battlefield.height, "battlefield height"),
            (config.simulation.battlefield.grid_resolution, "battlefield grid_resolution"),
            (config.simulation.physics.dt, "physics dt"),
            (config.simulation.max_steps, "simulation max_steps"),
            (config.simulation.threats.max_threats, "threat max_threats"),
            (config.simulation.threats.spawn_interval, "threat spawn_interval"),
            (config.simulation.render.width_px, "render width_px"),
            (config.simulation.render.height_px, "render height_px"),
            (config.simulation.render.fps, "render fps"),
            (config.tactical_ai.hidden_dim, "tactical hidden_dim"),
            (config.tactical_ai.num_heads, "tactical num_heads"),
            (config.tactical_ai.num_layers, "tactical num_layers"),
            (config.tactical_ai.context_length, "tactical context_length"),
            (config.training.update_interval, "training update_interval"),
            (config.training.batch_size, "training batch_size"),
            (config.training.learning_rate, "training learning_rate"),
            (config.training.gradient_clip, "training gradient_clip"),
            (config.adaptation.replay_buffer_capacity, "adaptation replay_buffer_capacity"),
            (config.learning_loop.loss_threshold, "learning loss_threshold"),
            (config.learning_loop.adaptation_interval, "learning adaptation_interval"),
        ):
            require_positive(value, name)
        for value, name in (
            (config.simulation.physics.hit_probability_base, "physics hit_probability_base"),
            (config.simulation.physics.cover_damage_reduction, "physics cover_damage_reduction"),
            (config.training.gamma, "training gamma"),
            (config.training.gae_lambda, "training gae_lambda"),
            (config.adaptation.replay_alpha, "adaptation replay_alpha"),
        ):
            require_unit_interval(value, name)
        if config.simulation.threats.initial_threats < 0 or config.simulation.threats.initial_threats > config.simulation.threats.max_threats:
            raise ConfigurationError("threat initial_threats must be between 0 and max_threats")
        if config.tactical_ai.hidden_dim % config.tactical_ai.num_heads != 0:
            raise ConfigurationError("tactical hidden_dim must be divisible by num_heads")
        if not 0 <= config.tactical_ai.dropout < 1:
            raise ConfigurationError("tactical dropout must be at least 0 and less than 1")
        if config.tactical_ai.state_embed_dim != config.state_encoder.state_embed_dim:
            raise ConfigurationError("tactical and state_encoder state_embed_dim must match")
        if config.training.ppo_clip_ratio <= 0 or config.training.entropy_coef < 0 or config.training.value_coef < 0:
            raise ConfigurationError("training PPO coefficients must be non-negative and clip ratio positive")
        if any(count < 0 for count in (
            config.simulation.swarm.num_recon,
            config.simulation.swarm.num_engage,
            config.simulation.swarm.num_ew,
            config.simulation.swarm.num_relay,
        )):
            raise ConfigurationError("swarm unit counts must not be negative")
        for name, weight in config.simulation.reward_weights.items():
            if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ConfigurationError(f"reward weight {name!r} must be finite")
