"""Append-only local audit journal for operational runtime decisions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock, RLock
from typing import Any, ClassVar
from uuid import UUID, uuid4


@dataclass(frozen=True)
class AuditEvent:
    """A durable record that links a decision to a mission and command."""

    event_id: UUID
    occurred_at: datetime
    event_type: str
    correlation_id: str
    payload: dict[str, Any]
    previous_hash: str | None
    event_hash: str


class AuditIntegrityError(ValueError):
    """Raised when a journal record is malformed, altered, or out of sequence."""


class JsonlAuditJournal:
    """A small, durable, append-only JSON Lines journal for local deployments."""

    _path_locks_guard: ClassVar[Any] = Lock()
    _path_locks: ClassVar[dict[Path, Any]] = {}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        lock_path = self.path.resolve()
        with self._path_locks_guard:
            self._lock = self._path_locks.setdefault(lock_path, RLock())

    def record(
        self,
        event_type: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        """Persist one event and fsync it before acknowledging the write."""
        with self._lock:
            existing_events = self.read_all()
            previous_hash = existing_events[-1]["event_hash"] if existing_events else None
            event_id = uuid4()
            occurred_at = datetime.now(UTC)
            unsigned_event: dict[str, Any] = {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "event_type": event_type,
                "correlation_id": correlation_id,
                "payload": payload,
                "previous_hash": previous_hash,
            }
            event = AuditEvent(
                event_id=event_id,
                occurred_at=occurred_at,
                event_type=event_type,
                correlation_id=correlation_id,
                payload=payload,
                previous_hash=previous_hash,
                event_hash=self._hash_event(unsigned_event),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = self._canonical_json(asdict(event))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{encoded}\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def read_all(self) -> list[dict[str, Any]]:
        """Return verified events in order, rejecting corruption or tampering."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous_hash: str | None = None
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise TypeError("event is not a mapping")
                    event_hash = event.pop("event_hash")
                    stored_previous_hash = event.get("previous_hash")
                    if stored_previous_hash != previous_hash:
                        raise AuditIntegrityError("previous hash does not match journal sequence")
                    calculated_hash = self._hash_event(event)
                    if event_hash != calculated_hash:
                        raise AuditIntegrityError("event hash does not match event content")
                    event["event_hash"] = event_hash
                except (json.JSONDecodeError, KeyError, TypeError, AuditIntegrityError) as error:
                    raise AuditIntegrityError(f"Invalid audit event at line {line_number}") from error
                events.append(event)
                previous_hash = event_hash
        return events

    @classmethod
    def _hash_event(cls, unsigned_event: dict[str, Any]) -> str:
        return sha256(cls._canonical_json(unsigned_event).encode("utf-8")).hexdigest()

    @classmethod
    def _canonical_json(cls, value: dict[str, Any]) -> str:
        return json.dumps(value, default=cls._json_default, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, (UUID, datetime)):
            return str(value)
        raise TypeError(f"Unsupported audit value: {type(value)!r}")
