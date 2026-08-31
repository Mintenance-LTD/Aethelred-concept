"""Gymnasium-compatible simulation environment for Aethelred."""

from __future__ import annotations

import random
from typing import Any, ClassVar
from uuid import UUID

import gymnasium as gym
import numpy as np

from aethelred.config.settings import SimulationConfig
from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.enums import (
    DroneRole,
    EngagementOutcome,
    FormationType,
    TacticalActionType,
    ThreatType,
)
from aethelred.core.events import LossEvent
from aethelred.core.models import BattlefieldState, DroneState, ObjectiveState, Vec2
from aethelred.simulation.battlefield import Battlefield
from aethelred.simulation.entity_manager import EntityManager
from aethelred.simulation.physics import SimplePhysics
from aethelred.simulation.renderer import BattlefieldRenderer
from aethelred.simulation.threat_spawner import ThreatSpawner
from aethelred.swarm.comms import CommChannel
from aethelred.swarm.swarm_unit import SwarmUnit
from aethelred.tactical_ai.state_encoder import build_observation


class AethelredEnv(gym.Env):
    """
    2D continuous-space tactical simulation environment.
    Gymnasium-compatible for standard RL training loops.
    """

    metadata: ClassVar[dict[str, object]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: SimulationConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        self.render_mode = render_mode

        self.battlefield = Battlefield(self.config.battlefield)
        self.entity_manager = EntityManager(self.config)
        self.threat_spawner = ThreatSpawner(self.config.threats)
        self.physics = SimplePhysics(self.config.physics)
        self._renderer: BattlefieldRenderer | None = None

        # Comms: units beyond range / under EW jamming fall back to autonomy.
        # The command node sits at the swarm rally point (matches spawn center),
        # so units are linked near base and go autonomous as they push forward.
        self.comm_channel = CommChannel(base_range=500.0)
        self._command_center = Vec2(
            x=self.config.battlefield.width * 0.3,
            y=self.config.battlefield.height * 0.5,
        )
        self._autonomous_units: dict[UUID, SwarmUnit] = {}
        # Anticipated threat positions (from the opponent model) for pre-positioning.
        self._anticipated_threats: list[Vec2] = []

        self._step_count = 0
        self._total_reward = 0.0
        self._current_state: BattlefieldState | None = None

        # Define spaces
        max_f = 32  # max friendlies
        max_t = 16  # max threats
        max_o = 8   # max objectives
        grid_res = self.config.battlefield.grid_resolution

        self.observation_space = gym.spaces.Dict({
            "friendlies": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_f, DroneState.feature_dim()), dtype=np.float32,
            ),
            "friendly_mask": gym.spaces.MultiBinary(max_f),
            "threats": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_t, 18), dtype=np.float32,
            ),
            "threat_mask": gym.spaces.MultiBinary(max_t),
            "objectives": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_o, 4), dtype=np.float32,
            ),
            "terrain": gym.spaces.Box(
                low=0, high=1, shape=(1, grid_res, grid_res), dtype=np.float32,
            ),
            "globals": gym.spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(len(TacticalActionType)),
            "target_position": gym.spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32,
            ),
            "priority": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "formation": gym.spaces.Discrete(len(FormationType)),
            "target_index": gym.spaces.Discrete(max_t),
        })

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        # Seed ALL randomness sources, not just NumPy: physics/comms/threat
        # spawning draw from Python's global `random`, so combat is otherwise
        # non-deterministic even when a seed is given.
        if seed is not None:
            random.seed(seed)
        rng = np.random.default_rng(seed)

        self.battlefield.reset()
        self.entity_manager.reset(rng=rng)
        self.threat_spawner.reset(rng=rng)

        # Spawn initial threats
        self.threat_spawner.spawn_initial_threats(
            self.entity_manager, self.battlefield, timestep=0
        )

        # Create initial objectives
        self._objectives = [
            ObjectiveState(
                position=Vec2(
                    x=float(rng.uniform(400, 900)),
                    y=float(rng.uniform(100, 900)),
                ),
                priority=float(rng.uniform(0.5, 1.0)),
            )
            for _ in range(3)
        ]

        self._autonomous_units.clear()
        self._anticipated_threats = []

        self._step_count = 0
        self._total_reward = 0.0
        self._current_state = self._build_state()

        obs = self._state_to_obs(self._current_state)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Decode a Gym action and execute its resulting decision.

        Training and operational-style callers should use ``step_decision`` so
        that a safety-authorised decision is executed without re-decoding.
        """
        return self.step_decision(self._decode_action(action))

    def step_decision(
        self, decision: TacticalDecision
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one already-decoded, safety-authorised decision exactly once."""
        state_before = self._current_state
        assert state_before is not None
        if decision.timestep != self._step_count:
            raise ValueError(
                "Decision timestep does not match the current simulation timestep: "
                f"{decision.timestep} != {self._step_count}"
            )

        # Apply movement
        self.entity_manager.apply_decision(decision, self.physics, self.battlefield)

        # Update threat behavior
        self.threat_spawner.update_threat_behavior(
            self.entity_manager, self.battlefield, self.physics
        )

        # Resolve engagements
        engagements, losses = self.entity_manager.resolve_engagements(
            decision, self.physics, self.battlefield, self._step_count
        )

        # Maybe spawn new threats
        self.threat_spawner.maybe_spawn(
            self._step_count, self.entity_manager, self.battlefield
        )

        # Update objective completion
        for obj in self._objectives:
            if obj.is_completed:
                continue
            for drone in self.entity_manager.get_active_drones():
                if drone.position.distance_to(obj.position) < 30.0:
                    obj.is_completed = True
                    break

        # Build new state
        self._step_count += 1
        self._current_state = self._build_state()

        # Compute reward
        reward = self._compute_reward(state_before, self._current_state, losses, engagements)
        self._total_reward += reward

        # Termination
        terminated = self._check_terminated()
        truncated = self._step_count >= self.config.max_steps

        if self.render_mode == "human":
            self.render()

        return (
            self._state_to_obs(self._current_state),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            self._renderer = BattlefieldRenderer(self.config.render)
        if self._current_state is None:
            return None
        terrain = self.battlefield.get_terrain_as_array()
        return self._renderer.render(
            self._current_state, terrain=terrain,
            mode=self.render_mode or "human",
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()

    def get_losses_since(self, timestep: int) -> list[LossEvent]:
        return self.entity_manager.get_losses_since(timestep)

    def get_all_engagements(self):
        return self.entity_manager.get_all_engagements()

    def get_current_state(self) -> BattlefieldState | None:
        return self._current_state

    def _build_state(self) -> BattlefieldState:
        return BattlefieldState(
            timestep=self._step_count,
            friendly_units=self.entity_manager.get_drone_list(),
            threats=self.entity_manager.get_threat_list(),
            objectives=self._objectives if hasattr(self, "_objectives") else [],
            terrain_grid=self.battlefield.get_terrain_as_array(),
            global_comms_degradation=0.0,
            weather_visibility=1.0,
        )

    def _state_to_obs(self, state: BattlefieldState) -> dict[str, np.ndarray]:
        """Convert BattlefieldState to observation dict (shared builder)."""
        obs = build_observation(
            state,
            width=self.config.battlefield.width,
            height=self.config.battlefield.height,
            time_scale=self.config.max_steps,
        )
        # Preserve this env's MultiBinary mask dtype contract.
        obs["friendly_mask"] = obs["friendly_mask"].astype(np.int8)
        obs["threat_mask"] = obs["threat_mask"].astype(np.int8)
        return obs

    def _decode_action(self, action: dict[str, Any]) -> TacticalDecision:
        """Decompose one high-level command into heterogeneous per-unit actions.

        The policy emits a single high-level intent (action type, target,
        formation). Rather than broadcasting it verbatim to every drone, each
        unit acts according to its role — so the swarm behaves heterogeneously:
        ENGAGE units close on and attack threats (distributed across them),
        RECON units observe/evade, EW and RELAY units hold the mesh. This is
        what makes the swarm more than N copies of one agent.
        """
        action_types = list(TacticalActionType)
        cmd = action_types[int(action.get("action_type", 0)) % len(action_types)]

        target_pos_norm = action.get("target_position", np.array([0.5, 0.5]))
        target_pos = Vec2(
            x=float(target_pos_norm[0]) * self.config.battlefield.width,
            y=float(target_pos_norm[1]) * self.config.battlefield.height,
        )

        priority = float(action.get("priority", [0.5])[0])
        formations = list(FormationType)
        formation = formations[int(action.get("formation", 0)) % len(formations)]
        active_threats = self.entity_manager.get_active_threats()

        autonomy = self.config.swarm.autonomy_enabled
        actions: list[TacticalAction] = []
        engage_i = 0
        for drone in self.entity_manager.get_active_drones():
            comm = self._comm_status_for(drone) if autonomy else "connected"
            if comm == "connected":
                # Commanded by the mother: role-based decomposition.
                unit = self._decode_unit_action(
                    drone, cmd, target_pos, formation, priority, active_threats, engage_i
                )
            else:
                # Out of comms / jammed: the unit decides locally (autonomy).
                unit = self._autonomous_action(drone, comm, active_threats)
            if drone.role == DroneRole.ENGAGE:
                engage_i += 1
            actions.append(unit)

        return TacticalDecision(timestep=self._step_count, actions=actions)

    def _comm_status_for(self, drone: DroneState) -> str:
        """Classify a drone's link to the command node as connected/degraded/lost.

        EW jamming threats within range set ``is_jammed`` (which the comm model
        degrades further); distance to the command node drives signal falloff.
        """
        drone.is_jammed = any(
            t.threat_type == ThreatType.EW_JAMMING
            and drone.position.distance_to(t.position) < t.estimated_range
            for t in self.entity_manager.get_active_threats()
        )
        signal = self.comm_channel.get_status(self._command_center, drone).signal_strength
        if signal > 0.5:
            return "connected"
        if signal > 0.1:
            return "degraded"
        return "lost"

    def _autonomous_action(
        self, drone: DroneState, comm: str, active_threats: list
    ) -> TacticalAction:
        """Let a (persistent) SwarmUnit decide locally when comms are degraded/lost."""
        unit = self._autonomous_units.get(drone.id)
        if unit is None:
            unit = SwarmUnit(drone)
            self._autonomous_units[drone.id] = unit
        unit.set_comm_status(comm)
        nearby = [
            t for t in active_threats
            if drone.position.distance_to(t.position) <= drone.sensor_range
        ]
        return unit.act(drone, nearby)

    def _nearest_anticipated(self, pos: Vec2) -> Vec2 | None:
        if not self._anticipated_threats:
            return None
        return min(self._anticipated_threats, key=lambda p: pos.distance_to(p))

    def set_anticipated_threats(self, positions: list[Vec2]) -> None:
        """Supply predicted threat positions (opponent model) for pre-positioning."""
        self._anticipated_threats = list(positions)

    def _decode_unit_action(
        self, drone, cmd, target_pos, formation, priority, active_threats, engage_i
    ) -> TacticalAction:
        """Role-specialized expansion of the high-level command for one drone.

        The policy is authoritative: ENGAGE units prosecute threats ONLY when the
        command is ENGAGE, and otherwise obey the commanded action (evade, hold,
        retreat, move). This makes the fight/withdraw decision a learned choice
        rather than a hardcoded reflex.
        """
        base = {"target_unit_id": drone.id, "formation": formation, "priority": priority}

        # Retreat is obeyed by everyone regardless of role.
        if cmd == TacticalActionType.RETREAT:
            return TacticalAction(action_type=TacticalActionType.RETREAT, target_position=target_pos, **base)

        # ENGAGE units prosecute threats only when explicitly commanded to.
        if drone.role == DroneRole.ENGAGE and active_threats and cmd == TacticalActionType.ENGAGE:
            threat = active_threats[engage_i % len(active_threats)]
            dist = drone.position.distance_to(threat.position)
            if dist <= drone.sensor_range and drone.ammo > 0.0:
                return TacticalAction(
                    action_type=TacticalActionType.ENGAGE,
                    target_position=threat.position,
                    target_threat_id=threat.id, **base,
                )
            # Out of range -> close the distance before firing
            return TacticalAction(
                action_type=TacticalActionType.MOVE,
                target_position=threat.position,
                target_threat_id=threat.id, **base,
            )

        # Recon scouts (toward anticipated threats for early warning) unless evading.
        if drone.role == DroneRole.RECON:
            if cmd == TacticalActionType.EVADE:
                return TacticalAction(
                    action_type=TacticalActionType.EVADE, target_position=target_pos, **base
                )
            recon_target = self._nearest_anticipated(drone.position) or target_pos
            return TacticalAction(
                action_type=TacticalActionType.RECON, target_position=recon_target, **base
            )

        if drone.role == DroneRole.EW:
            return TacticalAction(action_type=TacticalActionType.HOLD, target_position=target_pos, **base)

        if drone.role == DroneRole.RELAY:
            return TacticalAction(action_type=TacticalActionType.RELAY, target_position=target_pos, **base)

        # Engage units not told to engage, plus any other case: obey the command.
        return TacticalAction(action_type=cmd, target_position=target_pos, **base)

    def _compute_reward(
        self,
        state_before: BattlefieldState,
        state_after: BattlefieldState,
        losses: list[LossEvent],
        engagements: list | None = None,
    ) -> float:
        """Compute composite reward."""
        w = self.config.reward_weights
        reward = 0.0

        # Survival
        alive_after = len(state_after.active_friendlies)
        total = max(len(state_before.friendly_units), 1)
        survival_reward = alive_after / total
        reward += w.get("survival", 0.5) * survival_reward

        # Loss penalty (weight configurable via reward_weights["loss_penalty"])
        reward -= len(losses) * w.get("loss_penalty", 0.3)

        # Threat neutralization — credit GROSS kills this step (threats actually
        # destroyed), not the net threat-count change, which respawns cancel out.
        if engagements is not None:
            threats_killed = sum(
                1 for e in engagements if e.outcome == EngagementOutcome.TARGET_DESTROYED
            )
        else:
            threats_killed = max(0, len(state_before.active_threats) - len(state_after.active_threats))
        reward += w.get("threat_neutralized", 0.8) * threats_killed * 0.5

        # Mission progress
        completed_before = sum(1 for o in state_before.objectives if o.is_completed)
        completed_after = sum(1 for o in state_after.objectives if o.is_completed)
        new_completions = completed_after - completed_before
        reward += w.get("mission_progress", 1.0) * new_completions

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should end."""
        # All friendlies destroyed
        if not self.entity_manager.get_active_drones():
            return True
        # All objectives completed
        return hasattr(self, "_objectives") and all(o.is_completed for o in self._objectives)

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "total_reward": self._total_reward,
            "alive_drones": len(self.entity_manager.get_active_drones()),
            "active_threats": len(self.entity_manager.get_active_threats()),
            "total_losses": len(self.entity_manager.get_all_losses()),
            "objectives_completed": sum(
                1 for o in (self._objectives if hasattr(self, "_objectives") else [])
                if o.is_completed
            ),
        }
