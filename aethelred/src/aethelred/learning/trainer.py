"""Production training pipeline: PPO + Decision Transformer + curriculum learning.

The trainer owns the RL action loop: call :meth:`PPOTrainer.select_action` to get
an action (and stash the step), then :meth:`PPOTrainer.finish_step` once the reward
is known, then :meth:`PPOTrainer.train_on_rollout` periodically to run PPO.

Unlike the original implementation, the PPO update re-runs the *live* policy
(encoder + decision transformer) on the stored observations so gradients actually
flow back into the policy network — not just the value head.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from aethelred.config.settings import TrainerConfig
from aethelred.tactical_ai.policy import TacticalPolicy
from aethelred.tactical_ai.reward_model import RewardModel

logger = logging.getLogger(__name__)


@dataclass
class RolloutStep:
    """Single step of experience.

    We store the *raw observation tensors* (not the encoded embedding) so the
    PPO update can re-run the encoder and obtain gradients for it.
    """
    obs: dict[str, torch.Tensor]
    action_taken: dict[str, torch.Tensor]
    reward: float
    value: float
    log_prob: float
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float | None = None

    @property
    def episode_boundary(self) -> bool:
        """Whether the following observation belongs to a new episode."""
        return self.terminated or self.truncated


@dataclass
class RolloutBuffer:
    """Stores experience for PPO training."""
    steps: list[RolloutStep] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def compute_gae(
        self, gamma: float = 0.99, lam: float = 0.95, reward_scale: float = 1.0
    ) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation.

        ``reward_scale`` divides rewards into the value head's (normalized) units;
        since stored ``step.value`` is already in that scale, GAE stays consistent.
        """
        advantages: list[float] = []
        returns: list[float] = []
        gae = 0.0
        next_value = 0.0

        for step in reversed(self.steps):
            # A time limit is not an MDP terminal: retain a value bootstrap for
            # its final observation while preventing leakage into the next reset
            # episode.
            if step.terminated:
                successor_value = 0.0
            elif step.truncated:
                successor_value = step.bootstrap_value if step.bootstrap_value is not None else 0.0
            else:
                successor_value = next_value

            delta = step.reward * reward_scale + gamma * successor_value - step.value
            continuation = 0.0 if step.episode_boundary else 1.0
            gae = delta + gamma * lam * continuation * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + step.value)
            next_value = step.value

        return advantages, returns

    def discounted_return_std(self, gamma: float = 0.99) -> float:
        """Std of per-step discounted raw-reward sums (for reward normalization)."""
        rets = []
        running = 0.0
        for step in reversed(self.steps):
            if step.episode_boundary:
                running = 0.0
            running = step.reward + gamma * running
            rets.append(running)
        return float(np.std(rets)) if rets else 1.0


class ValueHead(nn.Module):
    """State value estimator for PPO."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_embed: torch.Tensor) -> torch.Tensor:
        return self.net(state_embed).squeeze(-1)


@dataclass
class CurriculumStage:
    """A single stage in the training curriculum."""
    name: str
    scenario: str
    episodes: int
    min_survival_rate: float = 0.0  # advance when this is met
    description: str = ""


class TrainingMetrics:
    """Tracks and logs training metrics."""

    def __init__(self, log_dir: str = "runs/aethelred") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_losses: deque[int] = deque(maxlen=100)
        self.episode_survivals: deque[float] = deque(maxlen=100)
        self.adaptation_counts: deque[int] = deque(maxlen=100)
        self.policy_losses: deque[float] = deque(maxlen=100)
        self.value_losses: deque[float] = deque(maxlen=100)
        self._global_step = 0
        self._episode = 0

    def log_episode(
        self,
        reward: float,
        losses: int,
        survival_rate: float,
        adaptations: int,
        alive: int,
        total_units: int,
        threats_active: int,
        steps: int,
    ) -> None:
        self._episode += 1
        self.episode_rewards.append(reward)
        self.episode_losses.append(losses)
        self.episode_survivals.append(survival_rate)
        self.adaptation_counts.append(adaptations)

        self.writer.add_scalar("episode/reward", reward, self._episode)
        self.writer.add_scalar("episode/losses", losses, self._episode)
        self.writer.add_scalar("episode/survival_rate", survival_rate, self._episode)
        self.writer.add_scalar("episode/adaptations", adaptations, self._episode)
        self.writer.add_scalar("episode/alive_units", alive, self._episode)
        self.writer.add_scalar("episode/threats_active", threats_active, self._episode)
        self.writer.add_scalar("episode/steps", steps, self._episode)

        # Running averages
        if len(self.episode_rewards) >= 10:
            self.writer.add_scalar(
                "avg/reward_100", np.mean(self.episode_rewards), self._episode
            )
            self.writer.add_scalar(
                "avg/survival_100", np.mean(self.episode_survivals), self._episode
            )
            self.writer.add_scalar(
                "avg/losses_100", np.mean(self.episode_losses), self._episode
            )

    def log_training(
        self, policy_loss: float, value_loss: float, entropy: float, grad_norm: float,
        lr: float = 0.0,
    ) -> None:
        self._global_step += 1
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

        self.writer.add_scalar("train/policy_loss", policy_loss, self._global_step)
        self.writer.add_scalar("train/value_loss", value_loss, self._global_step)
        self.writer.add_scalar("train/entropy", entropy, self._global_step)
        self.writer.add_scalar("train/grad_norm", grad_norm, self._global_step)
        self.writer.add_scalar("train/lr", lr, self._global_step)

    def log_curriculum(self, stage_name: str, stage_idx: int) -> None:
        self.writer.add_text("curriculum/stage", stage_name, self._episode)

    def log_adaptation(
        self, method: str, loss_before: float, loss_after: float, threats: list[str]
    ) -> None:
        self.writer.add_scalar("adapt/loss_before", loss_before, self._global_step)
        self.writer.add_scalar("adapt/loss_after", loss_after, self._global_step)
        self.writer.add_scalar(
            "adapt/improvement", loss_before - loss_after, self._global_step
        )

    def close(self) -> None:
        self.writer.close()

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def avg_survival(self) -> float:
        return float(np.mean(self.episode_survivals)) if self.episode_survivals else 0.0


class CheckpointManager:
    """Manages recoverable and selection-scored model checkpoints during training."""

    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._best_selection_score = float("-inf")
        self._checkpoints: list[Path] = []

    def save(
        self,
        policy: TacticalPolicy,
        episode: int,
        reward: float,
        metrics: dict,
        value_head: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        selection_score: float | None = None,
        selection_metric: str | None = None,
    ) -> Path:
        """Save a recoverable checkpoint and, when improved, the selected model.

        ``selection_score`` must come from an evaluation aggregate rather than
        the current episode's reward.  Periodic recovery checkpoints can omit
        it and can therefore never replace ``best_policy.pt`` by accident.
        """
        if selection_score is not None:
            if not math.isfinite(selection_score):
                raise ValueError("Checkpoint selection score must be finite")
            if not selection_metric or not selection_metric.strip():
                raise ValueError("Checkpoint selection metric is required")
        checkpoint = {
            "episode": episode,
            "reward": reward,
            "metrics": metrics,
            "policy_weights": policy.get_policy_weights(),
            "config": {
                "state_embed_dim": policy.config.state_embed_dim,
                "action_embed_dim": policy.config.action_embed_dim,
                "hidden_dim": policy.config.hidden_dim,
                "num_heads": policy.config.num_heads,
                "num_layers": policy.config.num_layers,
            },
        }
        if selection_score is not None:
            checkpoint["selection"] = {
                "metric": selection_metric,
                "score": selection_score,
            }
        if value_head is not None:
            checkpoint["value_head"] = value_head.state_dict()
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        path = self.checkpoint_dir / f"checkpoint_ep{episode:06d}.pt"
        torch.save(checkpoint, path)
        self._checkpoints.append(path)

        # Keep the promoted training checkpoint distinct from one-off rewards.
        if selection_score is not None and selection_score > self._best_selection_score:
            self._best_selection_score = selection_score
            best_path = self.checkpoint_dir / "best_policy.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                "New best policy: %s=%.4f (episode_reward=%.2f)",
                selection_metric,
                selection_score,
                reward,
            )

        # Prune old checkpoints
        while len(self._checkpoints) > self.keep_last:
            old = self._checkpoints.pop(0)
            if old.exists() and old.name != "best_policy.pt":
                old.unlink()

        return path

    def is_improved(self, selection_score: float) -> bool:
        """Return whether a finite aggregate evaluation would replace the best model."""
        if not math.isfinite(selection_score):
            raise ValueError("Checkpoint selection score must be finite")
        return selection_score > self._best_selection_score

    def load_best(self, policy: TacticalPolicy, device: str = "cpu") -> dict:
        """Load the best checkpoint."""
        path = self.checkpoint_dir / "best_policy.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best checkpoint at {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        policy.load_policy_weights(checkpoint["policy_weights"])
        return checkpoint

    def load_latest(self, policy: TacticalPolicy, device: str = "cpu") -> dict:
        """Load the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        checkpoint = torch.load(checkpoints[-1], map_location=device, weights_only=True)
        policy.load_policy_weights(checkpoint["policy_weights"])
        return checkpoint


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the tactical AI.
    Supports: GAE, gradient clipping, entropy bonus, value function clipping,
    curriculum learning, TensorBoard logging, checkpointing.

    The PPO update re-encodes stored observations and re-runs the decision
    transformer so gradients reach the encoder and transformer (not only the
    value head). Actions are generated by :meth:`select_action`, which uses the
    same single-step forward path that the update differentiates through, so the
    behaviour policy and the optimised policy are consistent.
    """

    def __init__(
        self,
        policy: TacticalPolicy,
        config: TrainerConfig,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs/aethelred",
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)

        # Value head for advantage estimation
        self.value_head = ValueHead(
            input_dim=policy.config.state_embed_dim
        ).to(self.device)

        # Auxiliary head: predicts the threat-type composition from the state
        # embedding. Its supervised loss forces the encoder to actually represent
        # WHICH threats are present — breaking the representational collapse where
        # winnable and deadly states map to identical embeddings (so PPO can never
        # learn a state-dependent response). 6 = len(ThreatType).
        self.aux_head = nn.Linear(policy.config.state_embed_dim, 6).to(self.device)

        # Optimizer covers policy + value head + aux head
        self.optimizer = torch.optim.AdamW(
            [
                {"params": policy.state_encoder.parameters(), "lr": config.learning_rate},
                {"params": policy.transformer.parameters(), "lr": config.learning_rate},
                {"params": self.value_head.parameters(), "lr": config.learning_rate * 3},
                {"params": self.aux_head.parameters(), "lr": config.learning_rate * 3},
            ],
            weight_decay=config.weight_decay,
        )
        # Base LRs captured up front so the schedule sets (not multiplies) the LR.
        self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        # Learning rate scheduler: cosine with warmup
        self._warmup_steps = config.warmup_steps
        self._total_train_steps = max(config.total_training_steps, config.warmup_steps + 1)
        self._total_opt_steps = 0

        self.rollout_buffer = RolloutBuffer()
        self.metrics = TrainingMetrics(log_dir=log_dir)
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.reward_model = RewardModel()

        self._target_return = config.target_return
        # Keep the policy's return-to-go conditioning in sync with the trainer so
        # forward_step uses a consistent RTG for both training and inference.
        self.policy.set_target_return(config.target_return)
        self._pending: tuple | None = None
        self._ret_std = 1.0  # running std of returns for reward normalization

        # Curriculum
        self._curriculum: list[CurriculumStage] = []
        self._current_stage = 0

    # --- Curriculum -------------------------------------------------------

    def set_curriculum(self, stages: list[CurriculumStage]) -> None:
        """Set training curriculum (progressive difficulty)."""
        self._curriculum = stages
        self._current_stage = 0
        if stages:
            logger.info(f"Curriculum set: {len(stages)} stages")
            for i, s in enumerate(stages):
                logger.info(f"  Stage {i}: {s.name} ({s.episodes} episodes, scenario={s.scenario})")

    def get_current_scenario(self) -> str | None:
        """Get the scenario for the current curriculum stage."""
        if self._curriculum and self._current_stage < len(self._curriculum):
            return self._curriculum[self._current_stage].scenario
        return None

    def check_curriculum_advance(self) -> bool:
        """Check if we should advance to the next curriculum stage."""
        if not self._curriculum or self._current_stage >= len(self._curriculum):
            return False

        stage = self._curriculum[self._current_stage]
        if self.metrics.avg_survival >= stage.min_survival_rate:
            self._current_stage += 1
            if self._current_stage < len(self._curriculum):
                next_stage = self._curriculum[self._current_stage]
                logger.info(
                    f"Curriculum advance: {stage.name} -> {next_stage.name} "
                    f"(survival={self.metrics.avg_survival:.2%})"
                )
                self.metrics.log_curriculum(next_stage.name, self._current_stage)
                return True
        return False

    # --- Action selection / experience collection -------------------------

    @staticmethod
    def _logp_entropy(
        logits: dict[str, torch.Tensor], actions: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint log-prob and entropy over the discrete action factors.

        Differentiable w.r.t. ``logits`` — this is what makes PPO train the
        policy network. (Continuous position/priority are treated as
        deterministic and excluded, as in the original design.)
        """
        log_prob = torch.zeros(logits["action_type_logits"].shape[0], device=logits["action_type_logits"].device)
        entropy = torch.zeros_like(log_prob)
        for logit_key, act_key in (
            ("action_type_logits", "action_type"),
            ("formation_logits", "formation"),
            ("target_logits", "target_index"),
        ):
            logp_all = F.log_softmax(logits[logit_key], dim=-1)
            idx = actions[act_key].long().view(-1, 1)
            log_prob = log_prob + logp_all.gather(-1, idx).squeeze(-1)
            entropy = entropy + (-(logp_all.exp() * logp_all).sum(dim=-1))
        return log_prob, entropy

    @torch.no_grad()
    def select_action(self, obs_tensors: dict[str, torch.Tensor]) -> dict:
        """Pick an action for the current observation and stash the step.

        Call :meth:`finish_step` with the resulting reward to commit it to the
        rollout buffer.
        """
        self.policy.state_encoder.eval()
        self.policy.transformer.eval()

        obs_dev = {k: v.to(self.device) for k, v in obs_tensors.items()}
        state_embed = self.policy.state_encoder(obs_dev)  # (1, D)
        value = self.value_head(state_embed).item()

        logits = self.policy.forward_step(state_embed)
        sampled = self.policy.transformer.action_head.sample_action(
            logits, deterministic=False
        )
        actions = {
            "action_type": sampled["action_type"].detach().cpu().view(1),
            "formation": sampled["formation"].detach().cpu().view(1),
            "target_index": sampled["target_index"].detach().cpu().view(1),
        }
        log_prob, _ = self._logp_entropy(logits, {k: v.to(self.device) for k, v in actions.items()})

        self._pending = (
            {k: v.detach().cpu() for k, v in obs_dev.items()},
            actions,
            value,
            float(log_prob.item()),
        )
        return self.policy.transformer.action_head.to_gym_action(sampled)

    def finish_step(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float | None = None,
    ) -> None:
        """Commit the action selected by the last :meth:`select_action` call."""
        if self._pending is None:
            raise RuntimeError("finish_step called without a preceding select_action")
        obs, actions, value, log_prob = self._pending
        self.rollout_buffer.steps.append(RolloutStep(
            obs=obs,
            action_taken=actions,
            reward=reward,
            value=value,
            log_prob=log_prob,
            terminated=terminated,
            truncated=truncated,
            bootstrap_value=bootstrap_value,
        ))
        self._pending = None

    @torch.no_grad()
    def estimate_value(self, obs_tensors: dict[str, torch.Tensor]) -> float:
        """Estimate the final-state bootstrap value for a truncated episode."""
        self.policy.state_encoder.eval()
        self.value_head.eval()
        obs_dev = {key: value.to(self.device) for key, value in obs_tensors.items()}
        return float(self.value_head(self.policy.state_encoder(obs_dev)).item())

    @property
    def ready_to_update(self) -> bool:
        return len(self.rollout_buffer) >= self.config.update_interval

    # --- PPO update -------------------------------------------------------

    def train_on_rollout(self) -> dict[str, float]:
        """Run PPO update on collected rollout data."""
        if len(self.rollout_buffer) < 16:
            self.rollout_buffer.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}

        self.policy.state_encoder.train()
        self.policy.transformer.train()
        self.value_head.train()

        # Reward normalization: keep value targets / advantages well-scaled so a
        # large-magnitude, high-variance reward can't swamp learning. Scale rewards
        # by a running std of discounted returns (EMA).
        reward_scale = 1.0
        if self.config.normalize_rewards:
            batch_std = self.rollout_buffer.discounted_return_std(self.config.gamma)
            if batch_std > 1e-6:
                # Snap on the first update, then track slowly.
                self._ret_std = batch_std if self._ret_std == 1.0 else (
                    0.99 * self._ret_std + 0.01 * batch_std
                )
            reward_scale = 1.0 / max(self._ret_std, 1e-6)

        # Compute advantages with GAE (rewards scaled into value-head units)
        advantages, returns = self.rollout_buffer.compute_gae(
            gamma=self.config.gamma, lam=self.config.gae_lambda, reward_scale=reward_scale
        )

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [s.log_prob for s in self.rollout_buffer.steps],
            dtype=torch.float32, device=self.device,
        )
        old_values = torch.tensor(
            [s.value for s in self.rollout_buffer.steps],
            dtype=torch.float32, device=self.device,
        )

        # Normalize advantages
        if advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        obs_keys = list(self.rollout_buffer.steps[0].obs.keys())

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        n = len(self.rollout_buffer)
        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(n)
            batch_size = min(self.config.batch_size, n)

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size].tolist()
                steps = [self.rollout_buffer.steps[i] for i in batch_idx]

                # Re-encode observations -> state embeddings (with gradients)
                obs_batch = {
                    k: torch.cat([s.obs[k] for s in steps], dim=0).to(self.device)
                    for k in obs_keys
                }
                state_embeds = self.policy.state_encoder(obs_batch)  # (B, D)

                # Fresh logits + values from the live policy
                logits = self.policy.forward_step(state_embeds)
                actions_batch = {
                    key: torch.cat([s.action_taken[key] for s in steps], dim=0).to(self.device)
                    for key in ("action_type", "formation", "target_index")
                }
                new_log_probs, entropy = self._logp_entropy(logits, actions_batch)
                entropy = entropy.mean()
                new_values = self.value_head(state_embeds)

                # PPO clipped objective. Clamp the log-ratio before exp() so a
                # large policy shift can't overflow to inf -> NaN gradients.
                idx_t = torch.tensor(batch_idx, device=self.device)
                batch_adv = advantages_t[idx_t]
                batch_old_lp = old_log_probs[idx_t]
                ratio = torch.exp((new_log_probs - batch_old_lp).clamp(-10.0, 10.0))
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.ppo_clip_ratio, 1.0 + self.config.ppo_clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                batch_returns = returns_t[idx_t]
                batch_old_values = old_values[idx_t]
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.config.ppo_clip_ratio, self.config.ppo_clip_ratio,
                )
                v_loss1 = F.mse_loss(new_values, batch_returns)
                v_loss2 = F.mse_loss(value_clipped, batch_returns)
                value_loss = torch.max(v_loss1, v_loss2)

                # Auxiliary representation loss: make the encoder predict the
                # threat-type composition (fraction of each type among active
                # threats) from the state embedding. Forces the embedding to carry
                # threat identity so value/policy CAN become state-dependent.
                aux_loss = torch.zeros((), device=self.device)
                if self.config.aux_coef > 0.0:
                    threats = obs_batch["threats"]          # (B, max_t, feat)
                    tmask = obs_batch["threat_mask"].float()  # (B, max_t)
                    type_oh = threats[:, :, 8:14]           # (B, max_t, 6) type one-hots
                    counts = (type_oh * tmask.unsqueeze(-1)).sum(dim=1)  # (B, 6)
                    denom = tmask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    aux_target = counts / denom             # (B, 6) type fractions
                    aux_pred = self.aux_head(state_embeds)
                    aux_loss = F.mse_loss(aux_pred, aux_target)

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.aux_coef * aux_loss
                )

                # Skip degenerate updates rather than poisoning the weights with
                # NaN/inf (which would make action sampling crash downstream).
                if not torch.isfinite(loss):
                    logger.warning("Non-finite PPO loss; skipping minibatch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.policy.state_encoder.parameters())
                    + list(self.policy.transformer.parameters())
                    + list(self.value_head.parameters())
                    + list(self.aux_head.parameters()),
                    self.config.gradient_clip,
                )
                if not torch.isfinite(grad_norm):
                    logger.warning("Non-finite grad norm; skipping optimizer step")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self._apply_lr_schedule()
                self.optimizer.step()
                self._total_opt_steps += 1

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += float(entropy.item())
                total_grad_norm += float(grad_norm)
                num_updates += 1

        if num_updates > 0:
            avg_pl = total_policy_loss / num_updates
            avg_vl = total_value_loss / num_updates
            avg_ent = total_entropy / num_updates
            avg_gn = total_grad_norm / num_updates
            self.metrics.log_training(
                avg_pl, avg_vl, avg_ent, avg_gn,
                lr=self.optimizer.param_groups[0]["lr"],
            )
        else:
            avg_pl = avg_vl = avg_ent = avg_gn = 0.0

        self.rollout_buffer.clear()
        return {
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "entropy": avg_ent,
            "grad_norm": avg_gn,
        }

    def _apply_lr_schedule(self) -> None:
        """Linear warmup followed by cosine decay to zero.

        Sets each param group's LR from its captured base LR (the original code
        multiplied in place, which collapsed the LR to 0 on the first step).
        """
        step = self._total_opt_steps
        if step < self._warmup_steps:
            factor = (step + 1) / max(self._warmup_steps, 1)
        else:
            progress = (step - self._warmup_steps) / max(
                self._total_train_steps - self._warmup_steps, 1
            )
            progress = min(1.0, progress)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = base * factor

    def close(self) -> None:
        self.metrics.close()
