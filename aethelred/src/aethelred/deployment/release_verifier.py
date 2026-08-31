"""Fail-closed verification of an active release before runtime model loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from aethelred.deployment.release_ledger import ReleaseLedger, ReleaseRegistration


class ReleaseVerificationError(ValueError):
    """Raised when an active release cannot be bound to its declared runtime."""


@dataclass(frozen=True)
class VerifiedReleaseArtifact:
    """The verified registration and exact local artefact safe to pass to a loader."""

    registration: ReleaseRegistration
    model_path: Path


Model = TypeVar("Model")


class ActiveReleaseVerifier:
    """Bind a ledger's active release to one exact artefact and runtime contract."""

    def __init__(self, ledger: ReleaseLedger) -> None:
        self._ledger = ledger

    def verify(
        self,
        model_path: str | Path,
        *,
        code_revision: str,
        configuration: dict[str, object],
        observation_schema: str,
        runtime_target: str,
    ) -> VerifiedReleaseArtifact:
        """Verify all active-release provenance before exposing a loader path."""
        registration = self._ledger.active_registration()
        try:
            verified_path = registration.approved_release.manifest.verify_artifact(
                model_path,
                code_revision=code_revision,
                configuration=configuration,
                observation_schema=observation_schema,
                runtime_target=runtime_target,
            )
        except (OSError, ValueError) as error:
            raise ReleaseVerificationError("Active release verification failed") from error
        return VerifiedReleaseArtifact(registration=registration, model_path=verified_path)

    def load(
        self,
        model_path: str | Path,
        loader: Callable[[Path], Model],
        *,
        code_revision: str,
        configuration: dict[str, object],
        observation_schema: str,
        runtime_target: str,
    ) -> Model:
        """Invoke a caller-supplied model loader only after full provenance verification."""
        artifact = self.verify(
            model_path,
            code_revision=code_revision,
            configuration=configuration,
            observation_schema=observation_schema,
            runtime_target=runtime_target,
        )
        return loader(artifact.model_path)
