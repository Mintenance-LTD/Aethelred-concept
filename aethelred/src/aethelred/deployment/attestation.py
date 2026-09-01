"""Verifiable release-attestation contracts for approved model artefacts."""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

from aethelred.deployment.model_manifest import ModelManifest

if TYPE_CHECKING:
    from aethelred.deployment.promotion import HeldOutEvaluation, HumanApproval


class ReleaseAttestationError(ValueError):
    """Raised when immutable release approval evidence cannot be verified."""


@dataclass(frozen=True)
class ReleaseAttestation:
    """A signed binding of manifest, evaluation evidence, and human approval."""

    issuer_id: str
    manifest_sha256: str
    evaluation_report_sha256: str
    approver: str
    approved_at: datetime
    signature: str

    def __post_init__(self) -> None:
        if not self.issuer_id.strip() or not self.approver.strip():
            raise ReleaseAttestationError("Attestation issuer and approver are required")
        if self.approved_at.tzinfo is None:
            raise ReleaseAttestationError("Attestation approval time must be timezone-aware")
        for value, name in ((self.manifest_sha256, "manifest"), (self.evaluation_report_sha256, "report"), (self.signature, "signature")):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ReleaseAttestationError(f"Attestation {name} must be a lowercase SHA-256 digest")


class ReleaseAttestationVerifier(Protocol):
    """A production integration can verify an attestation without exposing its signing key."""

    def verify(self, attestation: ReleaseAttestation, manifest: ModelManifest, evaluation: HeldOutEvaluation, approval: HumanApproval) -> None: ...


class HmacReleaseAttestor:
    """Local/SIL attestor; provide its key from a secret store, never source control."""

    def __init__(self, issuer_id: str, key: bytes) -> None:
        if not issuer_id.strip() or len(key) < 32:
            raise ValueError("Attestation issuer and a 32-byte key are required")
        self._issuer_id = issuer_id
        self._key = key

    def attest(self, manifest: ModelManifest, evaluation: HeldOutEvaluation, approval: HumanApproval) -> ReleaseAttestation:
        manifest_sha256 = sha256(manifest.to_json().encode("utf-8")).hexdigest()
        unsigned = self._unsigned(manifest_sha256, evaluation, approval)
        return ReleaseAttestation(self._issuer_id, manifest_sha256, evaluation.report_sha256, approval.approver, approval.approved_at, self._sign(unsigned))

    def verify(self, attestation: ReleaseAttestation, manifest: ModelManifest, evaluation: HeldOutEvaluation, approval: HumanApproval) -> None:
        expected_manifest = sha256(manifest.to_json().encode("utf-8")).hexdigest()
        if attestation.issuer_id != self._issuer_id or attestation.manifest_sha256 != expected_manifest or attestation.evaluation_report_sha256 != evaluation.report_sha256 or attestation.approver != approval.approver or attestation.approved_at != approval.approved_at:
            raise ReleaseAttestationError("Attestation does not bind the supplied release evidence")
        if not hmac.compare_digest(attestation.signature, self._sign(self._unsigned(expected_manifest, evaluation, approval))):
            raise ReleaseAttestationError("Release attestation signature is invalid")

    def _unsigned(self, manifest_sha256: str, evaluation: HeldOutEvaluation, approval: HumanApproval) -> bytes:
        return json.dumps({"issuer_id": self._issuer_id, "manifest_sha256": manifest_sha256, "evaluation_report_sha256": evaluation.report_sha256, "approver": approval.approver, "approved_at": approval.approved_at.astimezone(UTC).isoformat()}, sort_keys=True, separators=(",", ":")).encode()

    def _sign(self, unsigned: bytes) -> str:
        return hmac.new(self._key, unsigned, sha256).hexdigest()
