"""The 5-phase learning loop: the core innovation of Aethelred."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import torch

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.adaptation.opponent_model import OpponentBehaviorModel, PredictedThreatAction
from aethelred.config.settings import LearningLoopConfig
from aethelred.core.events import AdaptationEvent, EngagementEvent, LossEvent
from aethelred.core.models import BattlefieldState
from aethelred.learning.loss_analyzer import LossAnalysis, LossAnalyzer
from aethelred.tactical_ai.policy import TacticalPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineAdaptationCandidate:
    """An unapproved model update produced for offline evaluation only.

    Candidates deliberately have no method for applying their weights. A separate
    evaluation, approval, signing, and deployment workflow must handle promotion.
    """

    candidate_id: UUID
    created_at: datetime
    base_weights_digest: str
    updated_weights: dict[str, torch.Tensor]
    adaptation_event: AdaptationEvent


class LearningLoop:
    """
    Orchestrates the 5-phase learning cycle:
      Phase 1: Observation   - Collect state/action/reward trajectories
      Phase 2: Loss Analysis - Post-mortem when units die
      Phase 3: Adaptation    - Generate a candidate from loss analysis
      Phase 4: Evaluation    - Hand candidate to an offline evaluation workflow
      Phase 5: Prediction    - Model opponent behavior, pre-position

    This is a research-side loop. It must never mutate a policy used by an
    operational command path.
    """

    def __init__(
        self,
        tactical_policy: TacticalPolicy,
        adaptation_engine: AdaptationEngine,
        config: LearningLoopConfig,
        opponent_model: OpponentBehaviorModel | None = None,
        candidate_sink: Callable[[OfflineAdaptationCandidate], None] | None = None,
    ) -> None:
        self.tactical_policy = tactical_policy
        self.adaptation_engine = adaptation_engine
        self.config = config
        self.opponent_model = opponent_model or OpponentBehaviorModel()
        self.candidate_sink = candidate_sink
        self.loss_analyzer = LossAnalyzer()

        self.pending_losses: list[LossEvent] = []
        self.adaptation_history: list[AdaptationEvent] = []
        self.analysis_history: list[LossAnalysis] = []
        self.pending_candidates: list[OfflineAdaptationCandidate] = []

        self._steps_since_adaptation = 0
        self._total_steps = 0
        self._total_adaptations = 0

    # --- Phase 1: Observation ---

    def on_step(self, state: BattlefieldState, reward: float) -> None:
        """Called every simulation step. Phase 1: observation."""
        self._total_steps += 1
        self._steps_since_adaptation += 1

        # Feed opponent model with threat positions
        if self.config.enable_prediction:
            threat_positions = {t.id: t.position for t in state.active_threats}
            threat_velocities = {t.id: t.velocity for t in state.active_threats}
            self.opponent_model.observe_threat_positions(
                threat_positions, threat_velocities
            )

    # --- Phase 1.5: Loss Detection ---

    def on_loss(self, loss_event: LossEvent) -> None:
        """Called when a unit is destroyed. Queues for Phase 2."""
        self.pending_losses.append(loss_event)
        logger.info(
            f"[LOSS] Unit {loss_event.lost_unit_role} destroyed by "
            f"{loss_event.cause_of_loss.value} at range {loss_event.engagement_range:.0f}"
        )

    def on_engagements(self, engagements: list[EngagementEvent]) -> None:
        """Record threats neutralized by the swarm (updates threat mastery)."""
        if engagements:
            self.adaptation_engine.record_neutralizations(engagements)

    # --- Phases 2-5: Adaptation Cycle ---

    def maybe_adapt(self) -> AdaptationEvent | None:
        """
        Check if adaptation should trigger, then run Phases 2-5.
        Triggers when:
          - pending_losses reaches threshold
          - OR a new ThreatType is encountered
          - OR periodic interval reached
        """
        if not self._should_adapt():
            return None

        logger.info(
            f"[ADAPT] Triggering adaptation cycle "
            f"(losses={len(self.pending_losses)}, step={self._total_steps})"
        )

        # Phase 2: Loss Analysis
        analysis = self.loss_analyzer.analyze(self.pending_losses)
        self.analysis_history.append(analysis)

        logger.info(
            f"[PHASE 2] Analysis: {analysis.total_losses} losses, "
            f"threats={[t.value for t in analysis.threat_types]}, "
            f"patterns={analysis.patterns}"
        )

        # Phase 3: Adaptation. The result is an offline candidate only; policy
        # weights in the active runtime are intentionally never modified here.
        original_weights = self.tactical_policy.get_policy_weights()
        result = self.adaptation_engine.adapt(
            model=self.tactical_policy.transformer,
            losses=self.pending_losses,
        )

        adapted_weights = {}
        for name, param in result.updated_weights.items():
            adapted_weights[f"transformer.{name}"] = param
        # Keep encoder weights unchanged
        for name, param in self.tactical_policy.state_encoder.named_parameters():
            adapted_weights[f"encoder.{name}"] = param.data.clone()

        logger.info(
            f"[PHASE 3] Adaptation via {result.method}: "
            f"loss {result.loss_before:.4f} -> {result.loss_after:.4f}"
        )

        # Phase 5: Prediction — refresh the opponent model from the new losses.
        # (Consuming its predictions to pre-position the swarm is wired via
        # get_predictions(); see the learning loop's prediction hook.)
        if self.config.enable_prediction:
            self.opponent_model.update_from_losses(self.pending_losses)
            train_loss = self.opponent_model.train_step()
            logger.info(f"[PHASE 5] Opponent model updated (loss={train_loss:.4f})")

        # Record adaptation event
        event = AdaptationEvent(
            timestep=self._total_steps,
            trigger_loss_count=len(self.pending_losses),
            threat_types_addressed=analysis.threat_types,
            adaptation_method=result.method,
            performance_before={"loss": result.loss_before},
            performance_after={"loss": result.loss_after},
        )
        self.adaptation_history.append(event)
        candidate = OfflineAdaptationCandidate(
            candidate_id=uuid4(),
            created_at=datetime.now(UTC),
            base_weights_digest=self._weight_digest(original_weights),
            updated_weights={name: weight.detach().clone() for name, weight in adapted_weights.items()},
            adaptation_event=event,
        )
        self.pending_candidates.append(candidate)
        if self.candidate_sink is not None:
            self.candidate_sink(candidate)
        logger.info("[PHASE 4] Candidate %s queued for offline evaluation", candidate.candidate_id)
        self._total_adaptations += 1

        # Clear pending losses
        self.pending_losses.clear()
        self._steps_since_adaptation = 0

        return event

    @staticmethod
    def _weight_digest(weights: dict[str, torch.Tensor]) -> str:
        """Create a stable provenance digest for a candidate's base weights."""
        digest = sha256()
        for name in sorted(weights):
            digest.update(name.encode("utf-8"))
            digest.update(weights[name].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def get_predictions(
        self, state: BattlefieldState
    ) -> list[PredictedThreatAction]:
        """Get current opponent behavior predictions."""
        threat_positions = {t.id: t.position for t in state.active_threats}
        threat_velocities = {t.id: t.velocity for t in state.active_threats}
        return self.opponent_model.predict_next_actions(
            threat_positions, threat_velocities
        )

    def _should_adapt(self) -> bool:
        """Determine if adaptation threshold is met."""
        if not self.pending_losses:
            return False

        # New threat type seen
        for loss in self.pending_losses:
            if loss.cause_of_loss not in self.adaptation_engine.known_threat_types:
                return True

        # Loss threshold reached
        if len(self.pending_losses) >= self.config.loss_threshold:
            return True

        # Periodic interval
        return self._steps_since_adaptation >= self.config.adaptation_interval

    @property
    def stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "total_adaptations": self._total_adaptations,
            "pending_losses": len(self.pending_losses),
            "total_losses_analyzed": self.loss_analyzer.total_losses_analyzed,
            "known_threats": len(self.adaptation_engine.known_threat_types),
            "threat_summary": self.adaptation_engine.get_threat_summary(),
        }
