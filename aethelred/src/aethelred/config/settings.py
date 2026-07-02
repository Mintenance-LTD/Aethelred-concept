"""Configuration system for Aethelred."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)



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
    # The env only consumes the discrete action_type (target_index/formation are
    # currently inert), so by default PPO scores the objective on action_type
    # alone — including the inert factors just injects importance-ratio and
    # entropy noise (audit C2). Set True to restore the old 3-factor objective.
    ppo_include_inert_factors: bool = False
    device: str = "cpu"


@dataclass
class LearningLoopConfig:
    loss_threshold: int = 3
    adaptation_interval: int = 100
    buffer_size: int = 10000
    enable_prediction: bool = True
    enable_maml: bool = True
    enable_ewc: bool = True
    # When False, the loop learns (classifier / counter bank / EWC / opponent
    # model) but does NOT overwrite the live policy weights. PPO training sets
    # this False so the optimizer is the sole writer of the policy (audit C5);
    # the standalone learning-loop demo leaves it True to show the weight update.
    apply_weight_updates: bool = True


@dataclass
class AethelredConfig:
    """Root configuration."""

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
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AethelredConfig:
        config = cls()
        for key, value in data.items():
            if not hasattr(config, key):
                logger.warning("Unknown config key ignored: %r", key)
                continue
            attr = getattr(config, key)
            if isinstance(attr, (SimulationConfig, DecisionTransformerConfig,
                                 StateEncoderConfig, AdaptationConfig,
                                 TrainerConfig, LearningLoopConfig)):
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if not hasattr(attr, sub_key):
                            logger.warning("Unknown config key ignored: %s.%s", key, sub_key)
                            continue
                        sub_attr = getattr(attr, sub_key)
                        if hasattr(sub_attr, '__dataclass_fields__') and isinstance(sub_value, dict):
                            for k, v in sub_value.items():
                                if hasattr(sub_attr, k):
                                    setattr(sub_attr, k, v)
                                else:
                                    logger.warning(
                                        "Unknown config key ignored: %s.%s.%s", key, sub_key, k
                                    )
                        else:
                            setattr(attr, sub_key, sub_value)
            else:
                setattr(config, key, value)
        return config
