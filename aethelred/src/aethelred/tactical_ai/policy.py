"""TacticalPolicy: high-level inference wrapper combining state encoder + decision transformer."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from aethelred.config.settings import DecisionTransformerConfig, StateEncoderConfig
from aethelred.core.actions import TacticalDecision
from aethelred.core.interfaces import TacticalAIInterface
from aethelred.core.models import BattlefieldState
from aethelred.tactical_ai.decision_transformer import DecisionTransformer
from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder, build_observation


class TacticalPolicy(TacticalAIInterface):
    """
    Full inference pipeline: BattlefieldState -> encode -> transformer -> sample action.
    Maintains a history buffer for the Decision Transformer's context window.
    """

    def __init__(
        self,
        state_encoder: BattlefieldStateEncoder,
        transformer: DecisionTransformer,
        config: DecisionTransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        self.state_encoder = state_encoder
        self.transformer = transformer
        self.config = config
        self.device = device or torch.device("cpu")

        self.state_encoder.to(self.device)
        self.transformer.to(self.device)

        # History buffer: (state_embed, action_embed, return_to_go)
        self._state_history: deque[torch.Tensor] = deque(maxlen=config.context_length)
        self._action_history: deque[torch.Tensor] = deque(maxlen=config.context_length)
        self._rtg_history: deque[float] = deque(maxlen=config.context_length)
        self._timestep_history: deque[int] = deque(maxlen=config.context_length)

        self._target_return = 10.0  # desired return-to-go (can be adjusted)
        self._current_return = self._target_return
        self._temperature = 1.0
        self._deterministic = False

        # World size for position normalization; keep in sync with the env's
        # battlefield via set_world_size so train/inference normalize identically.
        self._world_w = 1000.0
        self._world_h = 1000.0

    def set_world_size(self, width: float, height: float) -> None:
        """Align observation normalization with the simulation battlefield size."""
        self._world_w = float(width)
        self._world_h = float(height)

    @torch.no_grad()
    def decide(
        self, state: BattlefieldState, history: list[BattlefieldState] | None = None
    ) -> TacticalDecision:
        """Encode state -> single-step transformer -> sample -> TacticalDecision.

        Uses the same single-step forward (:meth:`forward_step`) that the PPO
        trainer differentiates through, so the deployed policy is exactly the
        policy that was optimized (train/inference parity).
        """
        self.state_encoder.eval()
        self.transformer.eval()

        obs = self._state_to_obs(state)
        obs_tensors = BattlefieldStateEncoder.obs_to_tensors(obs, self.device)
        state_embed = self.state_encoder(obs_tensors)  # (1, D)

        logits = self.forward_step(state_embed)
        sampled = self.transformer.action_head.sample_action(
            logits, temperature=self._temperature, deterministic=self._deterministic
        )
        gym_action = self.transformer.action_head.to_gym_action(sampled)
        return self._gym_action_to_decision(gym_action, state)

    def forward_step(self, state_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the decision transformer on a single-step context.

        Shared by inference (:meth:`decide`) and PPO training so both sample from
        / differentiate through the exact same function. Returns logits squeezed
        to ``(B, ...)``.
        """
        b = state_embed.shape[0]
        states = state_embed.unsqueeze(1)  # (B, 1, D)
        actions = torch.zeros(b, 1, self.config.action_embed_dim, device=state_embed.device)
        rtg = torch.full((b, 1, 1), float(self._target_return), device=state_embed.device)
        timesteps = torch.zeros(b, 1, dtype=torch.long, device=state_embed.device)
        logits = self.transformer(states, actions, rtg, timesteps)
        return {k: (v[:, -1] if v.dim() == 3 else v) for k, v in logits.items()}

    def update_return(self, reward: float) -> None:
        """Update return-to-go with actual reward received."""
        self._current_return = max(0.0, self._current_return - reward)

    def set_target_return(self, target: float) -> None:
        self._target_return = target
        self._current_return = target

    def set_temperature(self, temperature: float) -> None:
        self._temperature = temperature

    def set_deterministic(self, deterministic: bool) -> None:
        self._deterministic = deterministic

    def reset_history(self) -> None:
        """Clear context window between episodes."""
        self._state_history.clear()
        self._action_history.clear()
        self._rtg_history.clear()
        self._timestep_history.clear()
        self._current_return = self._target_return

    def encode_state(self, state: BattlefieldState) -> torch.Tensor:
        obs = self._state_to_obs(state)
        obs_tensors = BattlefieldStateEncoder.obs_to_tensors(obs, self.device)
        return self.state_encoder(obs_tensors)

    def get_policy_weights(self) -> dict[str, torch.Tensor]:
        weights = {}
        for name, param in self.state_encoder.named_parameters():
            weights[f"encoder.{name}"] = param.data.clone()
        for name, param in self.transformer.named_parameters():
            weights[f"transformer.{name}"] = param.data.clone()
        return weights

    def load_policy_weights(self, weights: dict[str, torch.Tensor]) -> None:
        encoder_weights = {
            k.replace("encoder.", ""): v for k, v in weights.items()
            if k.startswith("encoder.")
        }
        transformer_weights = {
            k.replace("transformer.", ""): v for k, v in weights.items()
            if k.startswith("transformer.")
        }
        if encoder_weights:
            self.state_encoder.load_state_dict(encoder_weights, strict=False)
        if transformer_weights:
            self.transformer.load_state_dict(transformer_weights, strict=False)

    def _state_to_obs(self, state: BattlefieldState) -> dict[str, np.ndarray]:
        """Convert BattlefieldState to observation dict (shared builder)."""
        return build_observation(state, width=self._world_w, height=self._world_h)

    def _gym_action_to_decision(
        self, gym_action: dict, state: BattlefieldState
    ) -> TacticalDecision:
        """Convert gym action dict back to a TacticalDecision."""
        from aethelred.core.actions import TacticalAction
        from aethelred.core.enums import FormationType, TacticalActionType
        from aethelred.core.models import Vec2

        action_types = list(TacticalActionType)
        action_type = action_types[gym_action["action_type"] % len(action_types)]

        target_pos = Vec2(
            x=float(gym_action["target_position"][0]) * self._world_w,
            y=float(gym_action["target_position"][1]) * self._world_h,
        )

        priority = float(gym_action["priority"][0])

        formations = list(FormationType)
        formation = formations[gym_action["formation"] % len(formations)]

        target_threat_id = None
        active_threats = state.active_threats
        target_idx = gym_action["target_index"]
        if active_threats and target_idx < len(active_threats):
            target_threat_id = active_threats[target_idx].id

        actions = []
        for drone in state.active_friendlies:
            actions.append(TacticalAction(
                action_type=action_type,
                target_unit_id=drone.id,
                target_position=target_pos,
                target_threat_id=target_threat_id,
                formation=formation,
                priority=priority,
            ))

        return TacticalDecision(
            timestep=state.timestep,
            actions=actions,
            confidence=0.5,
        )

    @staticmethod
    def build(
        encoder_config: StateEncoderConfig | None = None,
        transformer_config: DecisionTransformerConfig | None = None,
        device: str = "cpu",
    ) -> TacticalPolicy:
        """Factory method to build a fresh TacticalPolicy."""
        enc_cfg = encoder_config or StateEncoderConfig()
        tf_cfg = transformer_config or DecisionTransformerConfig()
        dev = torch.device(device)

        # The transformer consumes the encoder's output embedding, so the two
        # dims MUST agree. Treat tactical_ai.state_embed_dim as the single source
        # of truth and align the encoder to it (prevents a silent shape mismatch
        # when only one of the two is set in YAML).
        enc_cfg.state_embed_dim = tf_cfg.state_embed_dim

        encoder = BattlefieldStateEncoder(enc_cfg)
        transformer = DecisionTransformer(tf_cfg)
        return TacticalPolicy(encoder, transformer, tf_cfg, dev)
