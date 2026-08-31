"""Append-only local audit journal for operational runtime decisions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class AuditEvent:
    """A durable record that links a decision to a mission and command."""

    event_id: UUID
    occurred_at: datetime
    event_type: str
    correlation_id: str
    payload: dict[str, Any]


class JsonlAuditJournal:
    """A small, durable, append-only JSON Lines journal for local deployments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        event_type: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        """Persist one event and fsync it before acknowledging the write."""
        event = AuditEvent(
            event_id=uuid4(),
            occurred_at=datetime.now(UTC),
            event_type=event_type,
            correlation_id=correlation_id,
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(event), default=self._json_default, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{encoded}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def read_all(self) -> list[dict[str, Any]]:
        """Return recorded events in order, rejecting corrupt journal lines."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, (UUID, datetime)):
            return str(value)
        raise TypeError(f"Unsupported audit value: {type(value)!r}")
