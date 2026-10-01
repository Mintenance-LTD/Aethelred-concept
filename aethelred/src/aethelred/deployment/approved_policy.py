"""Read-only policy boundary for verified release artefacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aethelred.deployment.release_verifier import ActiveReleaseVerifier, VerifiedReleaseArtifact
from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.operational import IntentProposal, Mission, WorldState


class ApprovedPolicyError(PermissionError):
    """Raised when a verified policy cannot make a bounded, current proposal."""


ProposalGenerator = Callable[[WorldState, Mission, datetime], IntentProposal]


@dataclass(frozen=True)
class ApprovedIntentPolicy:
    """A verified, read-only model facade with no command or weight-update surface."""

    artifact: VerifiedReleaseArtifact
    policy_id: str
    _propose: ProposalGenerator
    _journal: JsonlAuditJournal

    @classmethod
    def load(
        cls,
        verifier: ActiveReleaseVerifier,
        journal: JsonlAuditJournal,
        model_path: str | Path,
        loader: Callable[[Path], ProposalGenerator],
        policy_id: str,
        **runtime_provenance: object,
    ) -> ApprovedIntentPolicy:
        """Load a proposal generator only through active-release verification."""
        if not policy_id.strip():
            raise ApprovedPolicyError("Approved policy ID is required")
        artifact = verifier.verify(model_path, **runtime_provenance)  # type: ignore[arg-type]
        generator = loader(artifact.model_path)
        if not callable(generator):
            raise ApprovedPolicyError("Verified policy loader did not return a proposal generator")
        return cls(artifact, policy_id, generator, journal)

    def propose(self, state: WorldState, mission: Mission, now: datetime | None = None) -> IntentProposal:
        """Return a bound, unexpired proposal; this method cannot execute a command."""
        proposed_at = now or datetime.now(UTC)
        proposal = self._propose(state, mission, proposed_at)
        if not isinstance(proposal, IntentProposal):
            raise ApprovedPolicyError("Approved policy returned an invalid proposal")
        if (proposal.mission_id, proposal.mission_revision, proposal.state_revision, proposal.vehicle_id) != (
            mission.mission_id,
            mission.revision,
            state.revision,
            state.vehicle_id,
        ):
            raise ApprovedPolicyError("Approved policy proposal is not bound to current mission and state")
        if proposal.expires_at.tzinfo is None or proposal.expires_at <= proposed_at:
            raise ApprovedPolicyError("Approved policy proposal has expired")
        self._journal.record(
            "approved_policy_proposed",
            str(proposal.proposal_id),
            {"policy_id": self.policy_id, "release_id": self.artifact.registration.release_id},
        )
        return proposal
