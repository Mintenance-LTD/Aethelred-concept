"""Outcome-driven counter memory: a per-threat contextual bandit over actions.

This is the adaptive engine's *real* memory — the Mahoraga wheel. Unlike a
hardcoded rule table, it learns which high-level action actually works against
each threat phenomenon from observed outcomes, and because it is a plain table it
cannot catastrophically forget a counter it has already learned (the JJK rule:
once the general adapts, it stays adapted). A counter is only *trusted* once the
wheel has turned enough times (``min_turns``) and one action clearly dominates —
until then the engine reports zero mastery and the policy is on its own.

The bank is deliberately non-parametric and stands apart from the neural policy:
PPO owns the weights, the bank owns the memory, and the two never write to the
same place (this is the fix for the audit's "domain clash", C5). Callers consume
the bank through :meth:`recommend` / :meth:`mastery` / :meth:`action_prior`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aethelred.core.enums import TacticalActionType, ThreatType

_NUM_ACTIONS = len(TacticalActionType)


@dataclass
class CounterBank:
    """Per-``ThreatType`` running estimate of each action's mean outcome reward."""

    num_actions: int = _NUM_ACTIONS
    min_turns: int = 3            # turns of the wheel before a counter is trusted
    mastery_margin: float = 0.25  # value gap (best - mean) that earns full confidence
    _values: dict[ThreatType, np.ndarray] = field(default_factory=dict)
    _counts: dict[ThreatType, np.ndarray] = field(default_factory=dict)

    def _ensure(self, threat_type: ThreatType) -> None:
        if threat_type not in self._values:
            self._values[threat_type] = np.zeros(self.num_actions, dtype=np.float64)
            self._counts[threat_type] = np.zeros(self.num_actions, dtype=np.float64)

    def record_outcome(
        self, threat_type: ThreatType, action_index: int, reward: float
    ) -> None:
        """Register one outcome: ``action_index`` was taken against ``threat_type``
        and yielded ``reward``. Updated as an incremental (Welford) mean, so old
        turns are never discarded — the wheel only accumulates."""
        self._ensure(threat_type)
        i = int(action_index) % self.num_actions
        self._counts[threat_type][i] += 1.0
        c = self._counts[threat_type][i]
        v = self._values[threat_type][i]
        self._values[threat_type][i] = v + (reward - v) / c

    def turns(self, threat_type: ThreatType) -> int:
        """Total recorded outcomes for a threat (how far the wheel has turned)."""
        if threat_type not in self._counts:
            return 0
        return int(self._counts[threat_type].sum())

    def values(self, threat_type: ThreatType) -> np.ndarray:
        self._ensure(threat_type)
        return self._values[threat_type].copy()

    def recommend(self, threat_type: ThreatType) -> tuple[int | None, float]:
        """Best counter-action and a confidence in ``[0, 1]``.

        Returns ``(None, 0.0)`` until the wheel has turned ``min_turns`` times and
        at least one action has been tried. Confidence scales with how far the
        best action's value separates from the mean of the tried actions, and with
        how many times that best action has actually been exercised.
        """
        if self.turns(threat_type) < self.min_turns:
            return None, 0.0
        v = self._values[threat_type]
        c = self._counts[threat_type]
        tried = c > 0
        if not tried.any():
            return None, 0.0

        best = int(np.argmax(np.where(tried, v, -np.inf)))
        # Separate the best action from the strongest *alternative* that has been
        # tried; if nothing else has been tried, measure against a neutral 0
        # baseline (a consistently positive-outcome action is still evidence).
        others = tried.copy()
        others[best] = False
        baseline = float(v[others].max()) if others.any() else 0.0
        margin = float(v[best] - baseline)
        conf = float(np.clip(margin / self.mastery_margin, 0.0, 1.0))
        conf *= float(np.clip(c[best] / self.min_turns, 0.0, 1.0))
        return best, conf

    def mastery(self, threat_type: ThreatType) -> float:
        """Confidence that we have a trusted counter for this threat (``[0, 1]``)."""
        return self.recommend(threat_type)[1]

    def action_prior(self, threat_type: ThreatType, scale: float = 1.0) -> np.ndarray:
        """Additive logit bias toward the trusted counter (all-zero if untrusted).

        Intended to be *added* to the policy's action-type logits at selection
        time — a soft nudge, not an override, so PPO keeps authority over the
        final choice while benefiting from the general's memory.
        """
        prior = np.zeros(self.num_actions, dtype=np.float64)
        idx, conf = self.recommend(threat_type)
        if idx is not None and conf > 0.0:
            prior[idx] = scale * conf
        return prior

    def summary(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for tt in self._values:
            idx, conf = self.recommend(tt)
            out[tt.value] = {
                "turns": self.turns(tt),
                "best_action": (list(TacticalActionType)[idx].value if idx is not None else None),
                "confidence": round(conf, 3),
            }
        return out
