"""Auditable release registration, activation, and rollback governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

from aethelred.deployment.model_manifest import ModelManifest
from aethelred.deployment.promotion import (
    ApprovedModelRelease,
    HeldOutEvaluation,
    HumanApproval,
    PromotionError,
)
from aethelred.runtime.audit import JsonlAuditJournal


@dataclass(frozen=True)
class ReleaseRegistration:
    """Stable identifier for an approved release record."""

    release_id: str
    approved_release: ApprovedModelRelease


@dataclass(frozen=True)
class RollbackRecord:
    """Accountable reversal from one approved release to another."""

    from_release_id: str
    to_release_id: str
    operator: str
    rationale: str
    occurred_at: datetime


class ReleaseLedger:
    """Maintain an approved-release set and a durable lifecycle audit trail.

    Activation only changes this governance ledger. It intentionally does not
    load a model, mutate runtime policy state, or dispatch a command.
    """

    def __init__(self, journal: JsonlAuditJournal) -> None:
        self.journal = journal
        self._registrations: dict[str, ReleaseRegistration] = {}
        self._active_release_id: str | None = None
        self.recover()

    @property
    def active_release_id(self) -> str | None:
        """Return the currently declared approved release, if any."""
        return self._active_release_id

    def active_registration(self) -> ReleaseRegistration:
        """Return the sole active approved release, or fail closed if absent."""
        if self._active_release_id is None:
            raise PromotionError("No approved release is active")
        return self._get_registered(self._active_release_id)

    def register(self, approved_release: ApprovedModelRelease) -> ReleaseRegistration:
        """Register a release that has already passed promotion approval."""
        release_id = self._release_id(approved_release)
        if release_id in self._registrations:
            raise PromotionError("Release is already registered")
        registration = ReleaseRegistration(release_id, approved_release)
        self._registrations[release_id] = registration
        self.journal.record(
            "release_registered",
            release_id,
            {
                "approved_release": {
                    "manifest": asdict(approved_release.manifest),
                    "evaluation": asdict(approved_release.evaluation),
                    "approval": asdict(approved_release.approval),
                },
            },
        )
        return registration

    def activate(self, release_id: str, operator: str, rationale: str) -> ReleaseRegistration:
        """Record activation of a registered release with accountable intent."""
        registration = self._get_registered(release_id)
        self._validate_actor(operator, rationale)
        previous_release_id = self._active_release_id
        self._active_release_id = release_id
        self.journal.record(
            "release_activated",
            release_id,
            {
                "previous_release_id": previous_release_id,
                "operator": operator,
                "rationale": rationale,
            },
        )
        return registration

    def rollback(self, target_release_id: str, operator: str, rationale: str) -> RollbackRecord:
        """Record an accountable rollback to an earlier registered release."""
        if self._active_release_id is None:
            raise PromotionError("Cannot roll back without an active release")
        if target_release_id == self._active_release_id:
            raise PromotionError("Rollback target is already active")
        self._get_registered(target_release_id)
        self._validate_actor(operator, rationale)
        record = RollbackRecord(
            from_release_id=self._active_release_id,
            to_release_id=target_release_id,
            operator=operator,
            rationale=rationale,
            occurred_at=datetime.now(UTC),
        )
        self._active_release_id = target_release_id
        self.journal.record(
            "release_rolled_back",
            target_release_id,
            {
                "from_release_id": record.from_release_id,
                "operator": operator,
                "rationale": rationale,
            },
        )
        return record

    def recover(self) -> str | None:
        """Replay durable journal records and reconstruct approved lifecycle state.

        Any malformed or contradictory release record fails closed rather than
        silently selecting a model after a restart.
        """
        self._registrations.clear()
        self._active_release_id = None
        for event in self.journal.read_all():
            event_type = event.get("event_type")
            release_id = event.get("correlation_id")
            payload = event.get("payload")
            if not isinstance(release_id, str) or not isinstance(payload, dict):
                if event_type and str(event_type).startswith("release_"):
                    raise PromotionError("Malformed release-audit event")
                continue
            if event_type == "release_registered":
                approved_release = self._decode_approved_release(payload)
                if self._release_id(approved_release) != release_id:
                    raise PromotionError("Release audit identifier does not match its manifest")
                if release_id in self._registrations:
                    raise PromotionError("Duplicate release registration in audit log")
                self._registrations[release_id] = ReleaseRegistration(release_id, approved_release)
            elif event_type == "release_activated":
                self._get_registered(release_id)
                self._active_release_id = release_id
            elif event_type == "release_rolled_back":
                from_release_id = payload.get("from_release_id")
                if from_release_id != self._active_release_id:
                    raise PromotionError("Rollback audit record does not match active release")
                self._get_registered(release_id)
                self._active_release_id = release_id
        return self._active_release_id

    def _get_registered(self, release_id: str) -> ReleaseRegistration:
        try:
            return self._registrations[release_id]
        except KeyError as error:
            raise PromotionError("Release has not been registered and approved") from error

    @staticmethod
    def _validate_actor(operator: str, rationale: str) -> None:
        if not operator.strip() or not rationale.strip():
            raise PromotionError("Operator identity and rationale are required")

    @staticmethod
    def _release_id(approved_release: ApprovedModelRelease) -> str:
        return sha256(approved_release.manifest.to_json().encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_approved_release(payload: dict[str, Any]) -> ApprovedModelRelease:
        try:
            raw = payload["approved_release"]
            if not isinstance(raw, dict):
                raise TypeError("approved_release must be a mapping")
            manifest_raw = raw["manifest"]
            evaluation_raw = raw["evaluation"]
            approval_raw = raw["approval"]
            if not all(isinstance(value, dict) for value in (manifest_raw, evaluation_raw, approval_raw)):
                raise TypeError("release payload sections must be mappings")
            manifest = ModelManifest(**manifest_raw)
            evaluation = HeldOutEvaluation(
                candidate_id=UUID(str(evaluation_raw["candidate_id"])),
                scenario_count=int(evaluation_raw["scenario_count"]),
                candidate_metrics=dict(evaluation_raw["candidate_metrics"]),
                baseline_metrics=dict(evaluation_raw["baseline_metrics"]),
                safety_checks=dict(evaluation_raw["safety_checks"]),
                report_sha256=str(evaluation_raw["report_sha256"]),
            )
            approval = HumanApproval(
                approver=str(approval_raw["approver"]),
                rationale=str(approval_raw["rationale"]),
                approved_at=datetime.fromisoformat(str(approval_raw["approved_at"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PromotionError("Invalid approved-release payload in audit log") from error
        return ApprovedModelRelease(manifest=manifest, evaluation=evaluation, approval=approval)
