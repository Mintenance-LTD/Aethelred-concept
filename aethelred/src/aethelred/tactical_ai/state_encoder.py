"""State encoding: converts BattlefieldState into neural network embeddings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aethelred.config.settings import StateEncoderConfig


def build_observation(
    state,
    width: float = 1000.0,
    height: float = 1000.0,
    max_f: int = 32,
    max_t: int = 16,
    max_o: int = 8,
    time_scale: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Convert a BattlefieldState into the model observation dict.

    Single source of truth for observation construction so the environment and
    the inference policy normalize positions identically (previously the policy
    hardcoded a /1000 scale while the env used the battlefield width — a drift
    risk whenever the battlefield is not 1000 wide).
    """
    from aethelred.core.models import DroneState, ObjectiveState, ThreatState

    f_features = np.zeros((max_f, DroneState.feature_dim()), dtype=np.float32)
    f_mask = np.zeros(max_f, dtype=np.float32)
    for i, drone in enumerate(state.friendly_units[:max_f]):
        f_features[i] = drone.to_feature_vector()
        f_mask[i] = 1.0 if drone.is_alive() else 0.0
    f_features[:, 0] /= width
    f_features[:, 1] /= height

    t_features = np.zeros((max_t, ThreatState.feature_dim()), dtype=np.float32)
    t_mask = np.zeros(max_t, dtype=np.float32)
    for i, threat in enumerate(state.active_threats[:max_t]):
        t_features[i] = threat.to_feature_vector()
        t_mask[i] = 1.0
    t_features[:, 0] /= width
    t_features[:, 1] /= height

    o_features = np.zeros((max_o, ObjectiveState.feature_dim()), dtype=np.float32)
    for i, obj in enumerate(state.objectives[:max_o]):
        o_features[i] = obj.to_feature_vector()
    o_features[:, 0] /= width
    o_features[:, 1] /= height

    if state.terrain_grid is not None:
        terrain = state.terrain_grid[np.newaxis, :, :].astype(np.float32)
    else:
        terrain = np.zeros((1, 50, 50), dtype=np.float32)

    total = max(len(state.friendly_units), 1)
    globals_vec = np.array([
        state.global_comms_degradation,
        state.weather_visibility,
        min(state.timestep / max(time_scale, 1.0), 1.0),
        len(state.active_friendlies) / total,
    ], dtype=np.float32)

    return {
        "friendlies": f_features,
        "friendly_mask": f_mask,
        "threats": t_features,
        "threat_mask": t_mask,
        "objectives": o_features,
        "terrain": terrain,
        "globals": globals_vec,
    }


class EntityEncoder(nn.Module):
    """
    Encodes a variable-length set of entities into a fixed-size embedding
    using a per-entity MLP followed by attention pooling.
    """

    def __init__(self, entity_feature_dim: int, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entities: (batch, max_entities, feature_dim)
            mask: (batch, max_entities) - 1 for valid, 0 for padding
        Returns:
            (batch, embed_dim) - pooled entity embedding
        """
        batch_size = entities.shape[0]

        # Per-entity encoding
        entity_embeds = self.entity_mlp(entities)  # (B, N, D)

        # Expand pool query for batch
        query = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, D)

        # Attention pooling - key_padding_mask expects True for positions to IGNORE
        key_padding_mask = ~mask.bool()  # invert: True = pad position

        # If a row has NO valid entities (e.g. all threats neutralized), masking
        # every key makes attention softmax over all -inf -> NaN. Let those rows
        # attend to one padding slot, then zero their output below.
        all_masked = key_padding_mask.all(dim=1)  # (B,)
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked, 0] = False

        pooled, _ = self.attention_pool(
            query=query,
            key=entity_embeds,
            value=entity_embeds,
            key_padding_mask=safe_mask,
        )
        pooled = pooled.squeeze(1)  # (B, D)
        # Empty groups contribute a zero embedding (not NaN).
        return pooled * (~all_masked).unsqueeze(-1).to(pooled.dtype)


class BattlefieldStateEncoder(nn.Module):
    """
    Encodes the full BattlefieldState into a single embedding vector.
    Combines: friendly units, threats, objectives, terrain, and global scalars.
    """

    def __init__(self, config: StateEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.friendly_encoder = EntityEncoder(
            entity_feature_dim=config.friendly_feature_dim,
            embed_dim=config.entity_embed_dim,
            num_heads=config.num_attention_heads,
        )
        self.threat_encoder = EntityEncoder(
            entity_feature_dim=config.threat_feature_dim,
            embed_dim=config.entity_embed_dim,
            num_heads=config.num_attention_heads,
        )
        self.objective_encoder = EntityEncoder(
            entity_feature_dim=config.objective_feature_dim,
            embed_dim=config.entity_embed_dim,
            num_heads=config.num_attention_heads,
        )

        # Small CNN for terrain grid
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, config.terrain_embed_dim),
        )

        # Global scalars
        self.global_mlp = nn.Sequential(
            nn.Linear(4, config.global_embed_dim),
            nn.GELU(),
            nn.Linear(config.global_embed_dim, config.global_embed_dim),
        )

        # Fusion: combine all embeddings into final state embedding
        fusion_input_dim = (
            config.entity_embed_dim * 3  # friendlies + threats + objectives
            + config.terrain_embed_dim
            + config.global_embed_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.state_embed_dim),
            nn.LayerNorm(config.state_embed_dim),
            nn.GELU(),
            nn.Linear(config.state_embed_dim, config.state_embed_dim),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: dict with keys from environment observation:
                'friendlies': (B, max_f, feature_dim)
                'friendly_mask': (B, max_f)
                'threats': (B, max_t, feature_dim)
                'threat_mask': (B, max_t)
                'objectives': (B, max_o, feature_dim)
                'terrain': (B, 1, H, W)
                'globals': (B, 4)
        Returns:
            (B, state_embed_dim) - complete state embedding
        """
        friendly_embed = self.friendly_encoder(
            obs["friendlies"], obs["friendly_mask"]
        )
        threat_embed = self.threat_encoder(
            obs["threats"], obs["threat_mask"]
        )

        # Objectives - create mask (non-zero positions)
        obj_mask = obs["objectives"].abs().sum(dim=-1) > 0
        objective_embed = self.objective_encoder(
            obs["objectives"], obj_mask.float()
        )

        terrain_embed = self.terrain_cnn(obs["terrain"])
        global_embed = self.global_mlp(obs["globals"])

        # Concatenate and fuse
        combined = torch.cat([
            friendly_embed,
            threat_embed,
            objective_embed,
            terrain_embed,
            global_embed,
        ], dim=-1)

        return self.fusion(combined)

    @staticmethod
    def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
        """Convert numpy observation dict to torch tensors with batch dim."""
        result = {}
        for key, value in obs.items():
            t = torch.from_numpy(np.array(value)).float().to(device)
            # Always add batch dimension as dim 0
            if key == "globals":
                # (4,) -> (1, 4)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
            elif key == "terrain":
                # (1, H, W) -> (1, 1, H, W)
                if t.dim() == 3:
                    t = t.unsqueeze(0)
            elif key in ("friendly_mask", "threat_mask"):
                # (N,) -> (1, N)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
            else:
                # (N, F) -> (1, N, F)
                if t.dim() == 2:
                    t = t.unsqueeze(0)
            result[key] = t
        return result
