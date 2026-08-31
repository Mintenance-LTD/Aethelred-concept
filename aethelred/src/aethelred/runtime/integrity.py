"""Authenticated, expiring envelopes for non-offensive operational intents."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from aethelred.runtime.operational import IntentProposal


class IntegrityError(PermissionError):
    """Raised when an intent envelope fails authentication or freshness checks."""


@dataclass(frozen=True)
class AuthenticatedIntent:
    """A signed intent that still requires independent safety authorisation."""

    proposal: IntentProposal
    issuer_id: str
    issued_at: datetime
    nonce: str
    signature: str


class IntentAuthenticator:
    """Verify HMAC-protected intent envelopes and reject replayed nonces."""

    def __init__(self, secret: bytes, max_age: timedelta = timedelta(seconds=30)) -> None:
        if len(secret) < 32:
            raise ValueError("Intent-authentication secret must contain at least 32 bytes")
        if max_age <= timedelta():
            raise ValueError("Intent-envelope max_age must be positive")
        self._secret = secret
        self._max_age = max_age
        self._used_nonces: set[str] = set()

    def sign(
        self,
        proposal: IntentProposal,
        issuer_id: str,
        issued_at: datetime | None = None,
        nonce: str | None = None,
    ) -> AuthenticatedIntent:
        """Create an envelope; callers must protect the shared secret externally."""
        if not issuer_id.strip():
            raise IntegrityError("Intent issuer identity is required")
        timestamp = issued_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise IntegrityError("Intent issue time must be timezone-aware")
        envelope_nonce = nonce or str(uuid4())
        if not envelope_nonce:
            raise IntegrityError("Intent nonce is required")
        signature = self._sign_payload(proposal, issuer_id, timestamp, envelope_nonce)
        return AuthenticatedIntent(proposal, issuer_id, timestamp, envelope_nonce, signature)

    def verify(self, envelope: AuthenticatedIntent, now: datetime | None = None) -> IntentProposal:
        """Authenticate one envelope once, failing closed on tamper, age, or replay."""
        checked_at = now or datetime.now(UTC)
        if checked_at.tzinfo is None or envelope.issued_at.tzinfo is None:
            raise IntegrityError("Intent timestamps must be timezone-aware")
        if not envelope.issuer_id.strip() or not envelope.nonce:
            raise IntegrityError("Intent issuer identity and nonce are required")
        if envelope.issued_at > checked_at or checked_at - envelope.issued_at > self._max_age:
            raise IntegrityError("Intent envelope has expired or is not yet valid")
        expected = self._sign_payload(
            envelope.proposal, envelope.issuer_id, envelope.issued_at, envelope.nonce
        )
        if not hmac.compare_digest(envelope.signature, expected):
            raise IntegrityError("Intent envelope signature is invalid")
        if envelope.nonce in self._used_nonces:
            raise IntegrityError("Intent envelope nonce has already been used")
        self._used_nonces.add(envelope.nonce)
        return envelope.proposal

    def _sign_payload(
        self,
        proposal: IntentProposal,
        issuer_id: str,
        issued_at: datetime,
        nonce: str,
    ) -> str:
        target = proposal.target_position
        payload = {
            "proposal_id": str(proposal.proposal_id),
            "policy_id": proposal.policy_id,
            "mission_id": str(proposal.mission_id),
            "mission_revision": proposal.mission_revision,
            "state_revision": proposal.state_revision,
            "vehicle_id": proposal.vehicle_id,
            "capability": proposal.capability.value,
            "target_position": None if target is None else {"x": target.x, "y": target.y},
            "expires_at": proposal.expires_at.isoformat(),
            "issuer_id": issuer_id,
            "issued_at": issued_at.isoformat(),
            "nonce": nonce,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._secret, encoded, hashlib.sha256).hexdigest()
