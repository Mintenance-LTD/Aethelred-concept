"""Durable, approved configuration records for an operational runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID, uuid4

from aethelred.runtime.audit import JsonlAuditJournal


class RuntimeConfigurationError(ValueError):
    """Raised when a runtime configuration record is invalid or unavailable."""


@dataclass(frozen=True)
class RuntimeConfiguration:
    """An immutable, approved configuration revision with a canonical digest."""

    configuration_id: UUID
    revision: int
    sha256: str
    canonical_json: str
    registered_by: str
    rationale: str

    @property
    def values(self) -> Mapping[str, object]:
        """Return a fresh immutable view of the canonical configuration mapping."""
        decoded = json.loads(self.canonical_json)
        if not isinstance(decoded, dict):  # pragma: no cover - guarded at construction/recovery
            raise RuntimeConfigurationError("Stored runtime configuration is not a mapping")
        frozen = _freeze(decoded)
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by decoded shape
            raise RuntimeConfigurationError("Stored runtime configuration is not a mapping")
        return frozen


class RuntimeConfigurationRegistry:
    """Register, activate, recover, and roll back non-secret runtime configuration."""

    def __init__(self, journal: JsonlAuditJournal) -> None:
        self._journal = journal
        self._records: dict[UUID, RuntimeConfiguration] = {}
        self._active_id: UUID | None = None
        self.recover()

    @property
    def journal(self) -> JsonlAuditJournal:
        """Return the durable journal that establishes configuration authority."""
        return self._journal

    def register(
        self,
        values: Mapping[str, object],
        registered_by: str,
        rationale: str,
    ) -> RuntimeConfiguration:
        """Create a new immutable configuration revision from canonical JSON-safe values."""
        self._require_actor_and_rationale(registered_by, rationale)
        canonical_json = self._canonicalise(values)
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        if any(record.sha256 == digest for record in self._records.values()):
            raise RuntimeConfigurationError("Runtime configuration content is already registered")
        record = RuntimeConfiguration(
            configuration_id=uuid4(),
            revision=max((item.revision for item in self._records.values()), default=0) + 1,
            sha256=digest,
            canonical_json=canonical_json,
            registered_by=registered_by,
            rationale=rationale,
        )
        self._records[record.configuration_id] = record
        self._journal.record(
            "runtime_configuration_registered",
            correlation_id=str(record.configuration_id),
            payload=self._record_payload(record),
        )
        return record

    def activate(self, configuration_id: UUID, operator_id: str, rationale: str) -> RuntimeConfiguration:
        """Make a registered configuration the sole active revision."""
        self._require_actor_and_rationale(operator_id, rationale)
        record = self._require_record(configuration_id)
        self._active_id = configuration_id
        self._journal.record(
            "runtime_configuration_activated",
            correlation_id=str(configuration_id),
            payload={"operator_id": operator_id, "rationale": rationale, "revision": record.revision},
        )
        return record

    def rollback(self, configuration_id: UUID, operator_id: str, rationale: str) -> RuntimeConfiguration:
        """Restore a prior registered revision through an accountable audit event."""
        self._require_actor_and_rationale(operator_id, rationale)
        record = self._require_record(configuration_id)
        if configuration_id == self._active_id:
            raise RuntimeConfigurationError("Active runtime configuration cannot be rolled back to itself")
        previous_id = self._active_id
        self._active_id = configuration_id
        self._journal.record(
            "runtime_configuration_rolled_back",
            correlation_id=str(configuration_id),
            payload={
                "operator_id": operator_id,
                "rationale": rationale,
                "from_configuration_id": str(previous_id) if previous_id else None,
                "revision": record.revision,
            },
        )
        return record

    def active(self) -> RuntimeConfiguration:
        """Return the only configuration approved for an operational runtime."""
        if self._active_id is None:
            raise RuntimeConfigurationError("No runtime configuration is active")
        return self._require_record(self._active_id)

    def recover(self) -> None:
        """Recover immutable records and the latest activated/rolled-back revision."""
        self._records.clear()
        self._active_id = None
        for event in self._journal.read_all():
            event_type = event.get("event_type")
            if event_type == "runtime_configuration_registered":
                record = self._decode_record(event)
                if record.configuration_id in self._records or any(
                    item.revision >= record.revision for item in self._records.values()
                ):
                    raise RuntimeConfigurationError("Configuration audit revisions are invalid")
                self._records[record.configuration_id] = record
            elif event_type in {
                "runtime_configuration_activated",
                "runtime_configuration_rolled_back",
            }:
                try:
                    configuration_id = UUID(str(event["correlation_id"]))
                    payload = event["payload"]
                    if (
                        configuration_id not in self._records
                        or not isinstance(payload, dict)
                        or not str(payload.get("operator_id", "")).strip()
                        or not str(payload.get("rationale", "")).strip()
                    ):
                        raise ValueError("invalid activation payload")
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeConfigurationError("Invalid configuration activation audit record") from error
                self._active_id = configuration_id

    def _require_record(self, configuration_id: UUID) -> RuntimeConfiguration:
        try:
            return self._records[configuration_id]
        except KeyError as error:
            raise RuntimeConfigurationError("Runtime configuration is not registered") from error

    @staticmethod
    def _canonicalise(values: Mapping[str, object]) -> str:
        try:
            decoded = json.loads(json.dumps(dict(values), sort_keys=True, separators=(",", ":"), allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeConfigurationError("Runtime configuration must be finite JSON-compatible data") from error
        if not isinstance(decoded, dict):  # pragma: no cover - dict() guarantees this shape
            raise RuntimeConfigurationError("Runtime configuration must be a mapping")
        if _contains_secret_key(decoded):
            raise RuntimeConfigurationError("Runtime configuration must not contain credentials or secrets")
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _record_payload(record: RuntimeConfiguration) -> dict[str, object]:
        return {
            "revision": record.revision,
            "sha256": record.sha256,
            "canonical_json": record.canonical_json,
            "registered_by": record.registered_by,
            "rationale": record.rationale,
        }

    @classmethod
    def _decode_record(cls, event: dict[str, object]) -> RuntimeConfiguration:
        try:
            configuration_id = UUID(str(event["correlation_id"]))
            payload = event["payload"]
            if not isinstance(payload, dict):
                raise TypeError("configuration payload must be a mapping")
            revision = int(payload["revision"])
            canonical_json = str(payload["canonical_json"])
            values = json.loads(canonical_json)
            if not isinstance(values, dict) or cls._canonicalise(values) != canonical_json:
                raise ValueError("configuration is not canonical")
            digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            if digest != str(payload["sha256"]) or revision < 1:
                raise ValueError("configuration digest or revision is invalid")
            registered_by = str(payload["registered_by"])
            rationale = str(payload["rationale"])
            cls._require_actor_and_rationale(registered_by, rationale)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeConfigurationError("Invalid configuration registration audit record") from error
        return RuntimeConfiguration(
            configuration_id,
            revision,
            digest,
            canonical_json,
            registered_by,
            rationale,
        )

    @staticmethod
    def _require_actor_and_rationale(actor: str, rationale: str) -> None:
        if not actor.strip() or not rationale.strip():
            raise RuntimeConfigurationError("Operator identity and rationale are required")


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalised = str(key).lower().replace("-", "_")
            if normalised in {
                "secret",
                "password",
                "token",
                "api_key",
                "private_key",
                "credential",
                "credentials",
            }:
                return True
            if _contains_secret_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
