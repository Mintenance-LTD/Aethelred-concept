"""Immutable provenance manifests for approved model artefacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelManifest:
    """The minimum provenance required to promote a model artefact."""

    schema_version: str
    model_name: str
    model_sha256: str
    code_revision: str
    configuration_sha256: str
    observation_schema: str
    evaluation_report_sha256: str
    runtime_target: str
    training_data_reference: str = "unrecorded"
    runtime_environment: str = "unrecorded"
    build_provenance: str = "unrecorded"

    def to_json(self) -> str:
        """Return a canonical serialisation suitable for review and signing."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write the immutable manifest next to the deployment artefact."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def verify_artifact(
        self,
        model_path: str | Path,
        *,
        code_revision: str,
        configuration: dict[str, object],
        observation_schema: str,
        runtime_target: str,
        training_data_reference: str | None = None,
        runtime_environment: str | None = None,
        build_provenance: str | None = None,
    ) -> Path:
        """Fail closed unless a runtime artefact matches this exact manifest."""
        model = Path(model_path)
        if not model.is_file() or model.name != self.model_name:
            raise ValueError("Model artefact path does not match the manifest")
        if _sha256_file(model) != self.model_sha256:
            raise ValueError("Model artefact digest does not match the manifest")
        configuration_bytes = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if hashlib.sha256(configuration_bytes).hexdigest() != self.configuration_sha256:
            raise ValueError("Runtime configuration does not match the manifest")
        if code_revision != self.code_revision:
            raise ValueError("Runtime code revision does not match the manifest")
        if observation_schema != self.observation_schema:
            raise ValueError("Runtime observation schema does not match the manifest")
        if runtime_target != self.runtime_target:
            raise ValueError("Runtime target does not match the manifest")
        if training_data_reference is not None and training_data_reference != self.training_data_reference:
            raise ValueError("Training-data reference does not match the manifest")
        if runtime_environment is not None and runtime_environment != self.runtime_environment:
            raise ValueError("Runtime environment does not match the manifest")
        if build_provenance is not None and build_provenance != self.build_provenance:
            raise ValueError("Build provenance does not match the manifest")
        return model

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        evaluation_report_path: str | Path,
        code_revision: str,
        configuration: dict[str, object],
        observation_schema: str,
        runtime_target: str,
        training_data_reference: str,
        runtime_environment: str,
        build_provenance: str,
    ) -> ModelManifest:
        """Build a manifest from immutable inputs and their SHA-256 digests."""
        model = Path(model_path)
        report = Path(evaluation_report_path)
        if not model.is_file() or not report.is_file():
            raise FileNotFoundError("Model artefact and evaluation report must both exist")
        configuration_bytes = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return cls(
            schema_version="1.1",
            model_name=model.name,
            model_sha256=_sha256_file(model),
            code_revision=code_revision,
            configuration_sha256=hashlib.sha256(configuration_bytes).hexdigest(),
            observation_schema=observation_schema,
            evaluation_report_sha256=_sha256_file(report),
            runtime_target=runtime_target,
            training_data_reference=_require_provenance(training_data_reference, "training data reference"),
            runtime_environment=_require_provenance(runtime_environment, "runtime environment"),
            build_provenance=_require_provenance(build_provenance, "build provenance"),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_provenance(value: str, name: str) -> str:
    if not value.strip() or value == "unrecorded":
        raise ValueError(f"Manifest {name} is required")
    return value
