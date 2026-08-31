"""Prioritized experience replay buffer for continual learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aethelred.core.enums import ThreatType
from aethelred.core.events import LossEvent


@dataclass
class Experience:
    """A single experience tuple."""

    state_features: np.ndarray  # encoded state features
    action_features: np.ndarray  # encoded action features
    reward: float
    next_state_features: np.ndarray
    done: bool
    loss_event: LossEvent | None = None
    threat_type: ThreatType | None = None
    timestep: int = 0


class PrioritizedReplayBuffer:
    """Prioritized experience replay. Loss events get higher priority."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._buffer: list[Experience] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self._position = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: Experience, priority: float | None = None) -> None:
        """Add an experience with optional priority."""
        if priority is None:
            # Loss events get a fixed 2x emphasis over the running max. This must
            # NOT feed back into _max_priority (below) or the bonus compounds every
            # add and overflows to inf -> NaN sampling probabilities.
            priority = self._max_priority * 2.0 if experience.loss_event else self._max_priority

        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
        else:
            self._buffer[self._position] = experience

        self._priorities[self._position] = priority ** self.alpha
        self._position = (self._position + 1) % self.capacity
        # NOTE: deliberately do NOT update _max_priority from the loss-event bonus
        # here — that compounded every add and overflowed. _max_priority is raised
        # only by update_priorities() from real (bounded) TD-error priorities.

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[list[Experience], np.ndarray, list[int]]:
        """Sample a batch with importance sampling weights."""
        n = len(self._buffer)
        if n == 0:
            return [], np.array([]), []

        batch_size = min(batch_size, n)
        priorities = self._priorities[:n]
        total = priorities.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(n, 1.0 / n)  # degenerate priorities -> uniform
        else:
            probs = priorities / total

        indices = np.random.choice(n, size=batch_size, p=probs, replace=False)
        experiences = [self._buffer[i] for i in indices]

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        return experiences, weights.astype(np.float32), indices.tolist()

    def sample_by_threat(self, threat_type: ThreatType, n: int) -> list[Experience]:
        """Sample experiences related to a specific threat type."""
        matching = [
            exp for exp in self._buffer
            if exp.threat_type == threat_type
        ]
        if not matching:
            return []
        indices = np.random.choice(len(matching), size=min(n, len(matching)), replace=False)
        return [matching[i] for i in indices]

    def update_priorities(self, indices: list[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences (after training)."""
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = (priority + 1e-6) ** self.alpha
            self._max_priority = max(self._max_priority, priority)


class TrajectoryBuffer:
    """Simple buffer for storing full trajectories (state, action, reward sequences)."""

    def __init__(self, maxlen: int = 10000) -> None:
        self.maxlen = maxlen
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

    def __len__(self) -> int:
        return len(self._rewards)

    def add(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
        reward: float,
        done: bool = False,
    ) -> None:
        if len(self._rewards) >= self.maxlen:
            self._states.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._dones.pop(0)

        self._states.append(state_features)
        self._actions.append(action_features)
        self._rewards.append(reward)
        self._dones.append(done)

    def get_recent(self, n: int) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Get the most recent n transitions."""
        return (
            self._states[-n:],
            self._actions[-n:],
            self._rewards[-n:],
        )

    def get_episode_rewards(self) -> list[list[float]]:
        """Split rewards into episodes based on done flags."""
        episodes = []
        current = []
        for r, d in zip(self._rewards, self._dones):
            current.append(r)
            if d:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)
        return episodes

    def clear(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
