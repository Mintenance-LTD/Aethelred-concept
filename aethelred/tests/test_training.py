"""Tests for the PPO training pipeline.

These pin the two critical bugs from the audit:
  - the LR schedule must not collapse to zero;
  - a PPO update must push gradients into the encoder AND transformer, not just
    the value head.
"""

from __future__ import annotations

import torch

from aethelred.config.settings import TrainerConfig
from aethelred.learning.trainer import PPOTrainer, RolloutStep
from aethelred.tactical_ai.policy import TacticalPolicy


def _obs() -> dict[str, torch.Tensor]:
    return {
        "friendlies": torch.randn(1, 32, 16),
        "friendly_mask": torch.ones(1, 32),
        "threats": torch.randn(1, 16, 18),
        "threat_mask": torch.ones(1, 16),
        "objectives": torch.randn(1, 8, 4),
        "terrain": torch.randn(1, 1, 50, 50),
        "globals": torch.randn(1, 4),
    }


def _step(done: bool = False) -> RolloutStep:
    return RolloutStep(
        obs={k: v.clone() for k, v in _obs().items()},
        action_taken={
            "action_type": torch.tensor([1]),
            "formation": torch.tensor([2]),
            "target_index": torch.tensor([3]),
        },
        reward=0.5,
        value=0.0,
        log_prob=-2.0,
        done=done,
    )


def test_lr_schedule_is_never_zero(tmp_path):
    cfg = TrainerConfig(warmup_steps=5, total_training_steps=1000, learning_rate=3e-4)
    policy = TacticalPolicy.build()
    trainer = PPOTrainer(policy, cfg, checkpoint_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "r"))
    lrs = []
    for _ in range(4):
        trainer._apply_lr_schedule()
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        trainer._total_opt_steps += 1
    assert all(lr > 0 for lr in lrs), f"LR collapsed to zero: {lrs}"
    assert lrs[-1] > lrs[0]  # warming up


def test_ppo_update_trains_the_policy(tmp_path):
    cfg = TrainerConfig(
        warmup_steps=0, total_training_steps=1000, learning_rate=1e-2,
        batch_size=16, ppo_epochs=2,
    )
    policy = TacticalPolicy.build()
    trainer = PPOTrainer(policy, cfg, checkpoint_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "r"))

    enc_before = {n: p.clone() for n, p in policy.state_encoder.named_parameters()}
    tf_before = {n: p.clone() for n, p in policy.transformer.named_parameters()}
    vh_before = {n: p.clone() for n, p in trainer.value_head.named_parameters()}

    trainer.rollout_buffer.steps = [_step(done=(i % 8 == 7)) for i in range(32)]
    trainer.train_on_rollout()

    enc_changed = sum(
        1 for n, p in policy.state_encoder.named_parameters()
        if not torch.equal(p, enc_before[n])
    )
    tf_changed = sum(
        1 for n, p in policy.transformer.named_parameters()
        if not torch.equal(p, tf_before[n])
    )
    vh_changed = sum(
        1 for n, p in trainer.value_head.named_parameters()
        if not torch.equal(p, vh_before[n])
    )

    assert enc_changed > 0, "state encoder received no gradient"
    assert tf_changed > 0, "decision transformer received no gradient"
    assert vh_changed > 0, "value head received no gradient"


def test_encoder_handles_empty_entity_groups():
    """No threats / objectives must not produce NaN embeddings (regression)."""
    from aethelred.core.enums import DroneRole
    from aethelred.core.models import BattlefieldState, DroneState, Vec2
    from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder

    policy = TacticalPolicy.build()
    state = BattlefieldState(
        timestep=0,
        friendly_units=[DroneState(role=DroneRole.RECON, position=Vec2(x=10.0, y=10.0))],
        threats=[],
        objectives=[],
    )
    obs = BattlefieldStateEncoder.obs_to_tensors(policy._state_to_obs(state), policy.device)
    with torch.no_grad():
        embed = policy.state_encoder(obs)
    assert torch.isfinite(embed).all(), "encoder produced NaN/inf on empty entity groups"
    assert policy.decide(state).actions  # full inference path also stays finite


def test_train_inference_parity():
    """decide() (inference) must use the same forward_step that PPO optimizes."""
    from aethelred.core.enums import DroneRole, TacticalActionType
    from aethelred.core.models import BattlefieldState, DroneState, Vec2
    from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder

    policy = TacticalPolicy.build()
    policy.set_deterministic(True)
    state = BattlefieldState(
        timestep=0,
        friendly_units=[DroneState(role=DroneRole.ENGAGE, position=Vec2(x=100.0, y=100.0))],
    )

    obs = BattlefieldStateEncoder.obs_to_tensors(policy._state_to_obs(state), policy.device)
    with torch.no_grad():
        logits = policy.forward_step(policy.state_encoder(obs))
        expected = int(logits["action_type_logits"].argmax(dim=-1).item())

    decision = policy.decide(state)
    chosen = decision.actions[0].action_type
    assert list(TacticalActionType)[expected] == chosen


def test_select_and_finish_collects_a_step(tmp_path):
    cfg = TrainerConfig(device="cpu")
    policy = TacticalPolicy.build()
    trainer = PPOTrainer(policy, cfg, checkpoint_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "r"))
    gym_action = trainer.select_action({k: v for k, v in _obs().items()})
    assert "action_type" in gym_action
    trainer.finish_step(reward=1.0, done=False)
    assert len(trainer.rollout_buffer) == 1
