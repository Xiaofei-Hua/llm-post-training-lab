"""Immutable source, sample-lineage, family-split, and manifest contracts.

The module intentionally operates on already available records.  It does not
download datasets and it never treats a mutable branch name as a revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SOURCE_REGISTRY_SCHEMA_VERSION = "d06-source-registry-v1"
DATA_RECORD_SCHEMA_VERSION = "d06-data-record-v1"
SPLIT_POLICY_SCHEMA_VERSION = "d06-family-split-policy-v1"
PARENT_LEDGER_SCHEMA_VERSION = "d06-parent-payload-ledger-v1"
TRANSFORM_REGISTRY_SCHEMA_VERSION = "d06-transform-registry-v1"
DATA_MANIFEST_SCHEMA_VERSION = "d06-data-manifest-v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}")
_LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+() -]{0,126}")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,126}")
_QUALITY_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,62}")
_MAX_TEXT_CHARS = 1_000_000
_MAX_RECORDS = 1_000_000
_MAX_JSONL_BYTES = 4 * 1024**3
_MAX_RECORD_LINE_BYTES = 8 * 1024**2
_MAX_MESSAGES = 1_024
_MAX_LINEAGE_PARENTS = 1_024
_MAX_METADATA_ENTRIES = 128
_MAX_SOURCES = 10_000
_FAMILY_DIMENSIONS = ("source", "problem", "template")


class DataContractError(ValueError):
    """Base error for malformed or unsafe data-contract inputs."""


class SourceRegistryError(DataContractError):
    """Raised when source revision or license evidence is incomplete."""


class RecordSchemaError(DataContractError):
    """Raised when a canonical sample violates the frozen record schema."""


class LineageError(DataContractError):
    """Raised when record ancestry is missing, cyclic, or crosses a split."""


class FamilyLeakageError(DataContractError):
    """Raised when a family component occurs in more than one split."""


class ManifestIntegrityError(DataContractError):
    """Raised when a manifest cannot be reproduced from its body."""


class RevisionKind(StrEnum):
    GIT_COMMIT = "git_commit"
    SHA256 = "sha256"


class DataUse(StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    REDISTRIBUTE = "redistribute"


class SplitName(StrEnum):
    UNASSIGNED = "UNASSIGNED"
    D_ANCHOR = "D_anchor"
    D_CORE = "D_core"
    D_SELECT = "D_select"
    D_TEACHER_GATE = "D_teacher_gate"
    D_DEV = "D_dev"
    EVALUATION = "E"


TRAINING_SPLITS = frozenset(
    {
        SplitName.D_ANCHOR,
        SplitName.D_CORE,
        SplitName.D_SELECT,
        SplitName.D_TEACHER_GATE,
        SplitName.D_DEV,
    }
)
SEALED_SPLITS = frozenset({SplitName.D_TEACHER_GATE, SplitName.EVALUATION})


def canonical_json_bytes(value: object) -> bytes:
    """Return the only JSON encoding used for semantic fingerprints."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise DataContractError(f"value is not canonical-JSON serializable: {error}") from error
    return encoded.encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise DataContractError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_loads(text: str) -> Any:
    """Parse RFC-style JSON while rejecting duplicate keys and NaN/Infinity."""

    if not isinstance(text, str):
        raise DataContractError("JSON input must be a string")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise DataContractError(f"invalid JSON: {error}") from error


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise DataContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise DataContractError(f"{field} must be a bounded portable identifier")
    if ".." in value or value.endswith(("/", ":")):
        raise DataContractError(f"{field} contains a forbidden path-like segment")
    return value


def _require_text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
    max_chars: int = _MAX_TEXT_CHARS,
) -> str:
    if not isinstance(value, str):
        raise DataContractError(f"{field} must be a string")
    if len(value) > max_chars:
        raise DataContractError(f"{field} exceeds {max_chars} characters")
    if not allow_empty and not value.strip():
        raise DataContractError(f"{field} must be non-empty")
    if "\x00" in value or "\r" in value:
        raise DataContractError(f"{field} contains NUL or non-canonical CR newlines")
    if any(unicodedata.category(character) == "Cs" for character in value):
        raise DataContractError(f"{field} contains an unpaired surrogate")
    if unicodedata.normalize("NFC", value) != value:
        raise DataContractError(f"{field} must already be NFC-normalized")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataContractError(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise DataContractError(f"{field} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str] | frozenset[str],
    field: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise DataContractError(
            f"{field} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _read_json_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise DataContractError(f"cannot stat {path}: {error}") from error
    if size > max_bytes:
        raise DataContractError(f"{path} exceeds {max_bytes} bytes")
    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise DataContractError(f"{path} must not contain a UTF-8 BOM")
        text = raw.decode("utf-8", errors="strict")
        parsed = strict_json_loads(text)
    except UnicodeDecodeError as error:
        raise DataContractError(f"{path} is not strict UTF-8") from error
    except DataContractError as error:
        raise DataContractError(f"{path} is not valid strict JSON: {error}") from error
    except OSError as error:
        raise DataContractError(f"cannot read {path}: {error}") from error
    return raw, parsed


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """One public source pinned to immutable revision and license evidence."""

    source_id: str
    uri: str
    revision_kind: RevisionKind
    revision: str
    license_expression: str
    license_url: str
    license_evidence_sha256: str
    allowed_uses: tuple[DataUse, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, field="source_id")
        parsed_uri = urlparse(self.uri)
        if parsed_uri.scheme not in {"https", "hf"} or not parsed_uri.netloc:
            raise SourceRegistryError("source uri must be an absolute public https:// or hf:// URI")
        if self.revision_kind is RevisionKind.GIT_COMMIT:
            if _GIT_COMMIT_PATTERN.fullmatch(self.revision) is None:
                raise SourceRegistryError("git_commit revision must be a full 40/64-hex commit")
        elif _SHA256_PATTERN.fullmatch(self.revision) is None:
            raise SourceRegistryError("sha256 revision must be a lowercase SHA-256 digest")
        if _LICENSE_PATTERN.fullmatch(self.license_expression) is None:
            raise SourceRegistryError("license_expression is not a bounded SPDX-like expression")
        parsed_license = urlparse(self.license_url)
        if parsed_license.scheme != "https" or not parsed_license.netloc:
            raise SourceRegistryError("license_url must be an absolute https:// URI")
        _require_sha256(self.license_evidence_sha256, field="license_evidence_sha256")
        if not self.allowed_uses:
            raise SourceRegistryError("allowed_uses must not be empty")
        if tuple(sorted(set(self.allowed_uses), key=str)) != self.allowed_uses:
            raise SourceRegistryError("allowed_uses must be unique and canonically sorted")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceDescriptor:
        required = {
            "source_id",
            "uri",
            "revision_kind",
            "revision",
            "license_expression",
            "license_url",
            "license_evidence_sha256",
            "allowed_uses",
        }
        _require_exact_keys(raw, required=required, field="source")
        uses_raw = raw["allowed_uses"]
        if not isinstance(uses_raw, list) or any(not isinstance(item, str) for item in uses_raw):
            raise SourceRegistryError("allowed_uses must be a JSON array of strings")
        try:
            revision_kind = RevisionKind(raw["revision_kind"])
            uses = tuple(sorted((DataUse(item) for item in uses_raw), key=str))
        except (TypeError, ValueError) as error:
            raise SourceRegistryError(f"invalid revision kind or data use: {error}") from error
        return cls(
            source_id=raw["source_id"],
            uri=raw["uri"],
            revision_kind=revision_kind,
            revision=raw["revision"],
            license_expression=raw["license_expression"],
            license_url=raw["license_url"],
            license_evidence_sha256=raw["license_evidence_sha256"],
            allowed_uses=uses,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "uri": self.uri,
            "revision_kind": self.revision_kind.value,
            "revision": self.revision,
            "license_expression": self.license_expression,
            "license_url": self.license_url,
            "license_evidence_sha256": self.license_evidence_sha256,
            "allowed_uses": [item.value for item in self.allowed_uses],
        }


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Closed-world registry: unknown licenses and unknown sources fail closed."""

    allowed_license_expressions: tuple[str, ...]
    sources: tuple[SourceDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.allowed_license_expressions:
            raise SourceRegistryError("allowed_license_expressions must not be empty")
        if tuple(sorted(set(self.allowed_license_expressions))) != self.allowed_license_expressions:
            raise SourceRegistryError(
                "allowed_license_expressions must be unique and canonically sorted"
            )
        for expression in self.allowed_license_expressions:
            if _LICENSE_PATTERN.fullmatch(expression) is None:
                raise SourceRegistryError(f"invalid allowed license expression: {expression!r}")
        source_ids = [source.source_id for source in self.sources]
        if not source_ids:
            raise SourceRegistryError("source registry must contain at least one source")
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise SourceRegistryError("sources must be unique and sorted by source_id")
        allowed = set(self.allowed_license_expressions)
        forbidden = [
            source.source_id for source in self.sources if source.license_expression not in allowed
        ]
        if forbidden:
            raise SourceRegistryError(
                f"sources use licenses outside the frozen allowlist: {sorted(forbidden)}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceRegistry:
        required = {"schema_version", "allowed_license_expressions", "sources"}
        _require_exact_keys(raw, required=required, field="source registry")
        if raw["schema_version"] != SOURCE_REGISTRY_SCHEMA_VERSION:
            raise SourceRegistryError(
                f"unsupported source registry schema: {raw['schema_version']!r}"
            )
        licenses_raw = raw["allowed_license_expressions"]
        sources_raw = raw["sources"]
        if not isinstance(licenses_raw, list) or any(
            not isinstance(item, str) for item in licenses_raw
        ):
            raise SourceRegistryError("allowed_license_expressions must be a string array")
        if not isinstance(sources_raw, list):
            raise SourceRegistryError("sources must be an array")
        if len(sources_raw) > _MAX_SOURCES:
            raise SourceRegistryError(f"source count exceeds {_MAX_SOURCES}")
        sources = tuple(
            sorted(
                (
                    SourceDescriptor.from_mapping(
                        _require_mapping(item, field=f"sources[{index}]")
                    )
                    for index, item in enumerate(sources_raw)
                ),
                key=lambda source: source.source_id,
            )
        )
        return cls(tuple(sorted(licenses_raw)), sources)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
            "allowed_license_expressions": list(self.allowed_license_expressions),
            "sources": [source.to_record() for source in self.sources],
        }

    def source(self, source_id: str) -> SourceDescriptor:
        for descriptor in self.sources:
            if descriptor.source_id == source_id:
                return descriptor
        raise SourceRegistryError(f"unknown source_id: {source_id}")

    def assert_usage(self, source_id: str, split: SplitName) -> SourceDescriptor:
        descriptor = self.source(source_id)
        if split is SplitName.UNASSIGNED:
            return descriptor
        required_use = (
            DataUse.EVALUATE if split is SplitName.EVALUATION else DataUse.TRAIN
        )
        if required_use not in descriptor.allowed_uses:
            raise SourceRegistryError(
                f"{source_id} does not permit {required_use.value} use for split {split.value}"
            )
        return descriptor


@dataclass(frozen=True, slots=True)
class LoadedSourceRegistry:
    path: str
    raw_sha256: str
    registry: SourceRegistry


def load_source_registry(path: str | Path) -> LoadedSourceRegistry:
    resolved = Path(path)
    raw, parsed = _read_json_bytes(resolved, max_bytes=16 * 1024**2)
    registry = SourceRegistry.from_mapping(_require_mapping(parsed, field="source registry"))
    return LoadedSourceRegistry(
        path=resolved.as_posix(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        registry=registry,
    )


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise RecordSchemaError(f"unsupported message role: {self.role!r}")
        _require_text(self.content, field="message.content")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> Message:
        _require_exact_keys(raw, required={"role", "content"}, field=field)
        return cls(role=raw["role"], content=raw["content"])

    def to_record(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class LineageParent:
    sample_id: str
    payload_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.sample_id, field="lineage.parent.sample_id")
        _require_sha256(self.payload_sha256, field="lineage.parent.payload_sha256")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field: str) -> LineageParent:
        _require_exact_keys(raw, required={"sample_id", "payload_sha256"}, field=field)
        return cls(sample_id=raw["sample_id"], payload_sha256=raw["payload_sha256"])

    def to_record(self) -> dict[str, str]:
        return {"sample_id": self.sample_id, "payload_sha256": self.payload_sha256}


@dataclass(frozen=True, slots=True)
class ParentPayloadEntry:
    sample_id: str
    split: SplitName
    payload_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.sample_id, field="parent ledger sample_id")
        if self.split is SplitName.UNASSIGNED:
            raise LineageError("parent ledger entries must have an assigned split")
        _require_sha256(self.payload_sha256, field="parent ledger payload_sha256")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> ParentPayloadEntry:
        _require_exact_keys(
            raw,
            required={"sample_id", "split", "payload_sha256"},
            field=f"parent ledger entries[{index}]",
        )
        try:
            split = SplitName(raw["split"])
        except (TypeError, ValueError) as error:
            raise LineageError(f"parent ledger entries[{index}] has invalid split") from error
        return cls(
            sample_id=raw["sample_id"],
            split=split,
            payload_sha256=raw["payload_sha256"],
        )

    def to_record(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split": self.split.value,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParentPayloadLedger:
    entries: tuple[ParentPayloadEntry, ...] = ()

    def __post_init__(self) -> None:
        sample_ids = [entry.sample_id for entry in self.entries]
        if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise LineageError("parent ledger entries must be unique and sorted by sample_id")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ParentPayloadLedger:
        _require_exact_keys(
            raw,
            required={"schema_version", "entries"},
            field="parent payload ledger",
        )
        if raw["schema_version"] != PARENT_LEDGER_SCHEMA_VERSION:
            raise LineageError(
                f"unsupported parent payload ledger schema: {raw['schema_version']!r}"
            )
        entries_raw = raw["entries"]
        if not isinstance(entries_raw, list):
            raise LineageError("parent payload ledger entries must be an array")
        if len(entries_raw) > _MAX_RECORDS:
            raise LineageError(f"parent payload ledger exceeds {_MAX_RECORDS} entries")
        entries = tuple(
            sorted(
                (
                    ParentPayloadEntry.from_mapping(
                        _require_mapping(item, field=f"parent ledger entries[{index}]"),
                        index=index,
                    )
                    for index, item in enumerate(entries_raw)
                ),
                key=lambda entry: entry.sample_id,
            )
        )
        return cls(entries=entries)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PARENT_LEDGER_SCHEMA_VERSION,
            "entries": [entry.to_record() for entry in self.entries],
        }

    def entry(self, sample_id: str) -> ParentPayloadEntry:
        for entry in self.entries:
            if entry.sample_id == sample_id:
                return entry
        raise LineageError(f"unresolved external lineage parent: {sample_id}")


EMPTY_PARENT_PAYLOAD_LEDGER = ParentPayloadLedger()


@dataclass(frozen=True, slots=True)
class LoadedParentPayloadLedger:
    path: str
    raw_sha256: str
    ledger: ParentPayloadLedger


def load_parent_payload_ledger(path: str | Path) -> LoadedParentPayloadLedger:
    resolved = Path(path)
    raw, parsed = _read_json_bytes(resolved, max_bytes=128 * 1024**2)
    return LoadedParentPayloadLedger(
        path=resolved.as_posix(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        ledger=ParentPayloadLedger.from_mapping(
            _require_mapping(parsed, field="parent payload ledger")
        ),
    )


@dataclass(frozen=True, slots=True)
class TransformLineage:
    transform_name: str
    transform_version: str
    code_sha256: str
    config_sha256: str
    parents: tuple[LineageParent, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.transform_name, field="lineage.transform_name")
        if _VERSION_PATTERN.fullmatch(self.transform_version) is None:
            raise LineageError("lineage.transform_version must be immutable and bounded")
        _require_sha256(self.code_sha256, field="lineage.code_sha256")
        _require_sha256(self.config_sha256, field="lineage.config_sha256")
        parent_ids = [parent.sample_id for parent in self.parents]
        if parent_ids != sorted(parent_ids) or len(parent_ids) != len(set(parent_ids)):
            raise LineageError("lineage parents must be unique and sorted by sample_id")
        if self.transform_name == "ingest" and self.parents:
            raise LineageError("ingest lineage must not declare parents")
        if self.transform_name != "ingest" and not self.parents:
            raise LineageError("non-ingest lineage must declare at least one parent")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TransformLineage:
        required = {
            "transform_name",
            "transform_version",
            "code_sha256",
            "config_sha256",
            "parents",
        }
        _require_exact_keys(raw, required=required, field="lineage")
        parents_raw = raw["parents"]
        if not isinstance(parents_raw, list):
            raise LineageError("lineage.parents must be an array")
        if len(parents_raw) > _MAX_LINEAGE_PARENTS:
            raise LineageError(f"lineage parent count exceeds {_MAX_LINEAGE_PARENTS}")
        parents = tuple(
            sorted(
                (
                    LineageParent.from_mapping(
                        _require_mapping(item, field=f"lineage.parents[{index}]"),
                        field=f"lineage.parents[{index}]",
                    )
                    for index, item in enumerate(parents_raw)
                ),
                key=lambda parent: parent.sample_id,
            )
        )
        return cls(
            transform_name=raw["transform_name"],
            transform_version=raw["transform_version"],
            code_sha256=raw["code_sha256"],
            config_sha256=raw["config_sha256"],
            parents=parents,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "transform_name": self.transform_name,
            "transform_version": self.transform_version,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "parents": [parent.to_record() for parent in self.parents],
        }


def _require_repository_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LineageError(f"{field} must be a non-empty POSIX repository-relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LineageError(f"{field} must stay inside the repository")
    canonical = path.as_posix()
    if canonical != value:
        raise LineageError(f"{field} must use canonical POSIX spelling")
    return canonical


@dataclass(frozen=True, slots=True)
class TransformArtifact:
    transform_name: str
    transform_version: str
    code_path: str
    code_sha256: str
    config_path: str
    config_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.transform_name, field="transform artifact name")
        if _VERSION_PATTERN.fullmatch(self.transform_version) is None:
            raise LineageError("transform artifact version must be immutable and bounded")
        _require_repository_relative_path(self.code_path, field="transform artifact code_path")
        _require_repository_relative_path(
            self.config_path, field="transform artifact config_path"
        )
        _require_sha256(self.code_sha256, field="transform artifact code_sha256")
        _require_sha256(self.config_sha256, field="transform artifact config_sha256")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.transform_name,
            self.transform_version,
            self.code_sha256,
            self.config_sha256,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> TransformArtifact:
        required = {
            "transform_name",
            "transform_version",
            "code_path",
            "code_sha256",
            "config_path",
            "config_sha256",
        }
        _require_exact_keys(raw, required=required, field=f"transform artifacts[{index}]")
        return cls(**{field: raw[field] for field in required})

    def to_record(self) -> dict[str, str]:
        return {
            "transform_name": self.transform_name,
            "transform_version": self.transform_version,
            "code_path": self.code_path,
            "code_sha256": self.code_sha256,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class TransformRegistry:
    artifacts: tuple[TransformArtifact, ...]

    def __post_init__(self) -> None:
        keys = [artifact.key for artifact in self.artifacts]
        if not keys:
            raise LineageError("transform registry must contain at least one artifact")
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise LineageError("transform artifacts must be unique and canonically sorted")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TransformRegistry:
        _require_exact_keys(
            raw,
            required={"schema_version", "artifacts"},
            field="transform registry",
        )
        if raw["schema_version"] != TRANSFORM_REGISTRY_SCHEMA_VERSION:
            raise LineageError(
                f"unsupported transform registry schema: {raw['schema_version']!r}"
            )
        artifacts_raw = raw["artifacts"]
        if not isinstance(artifacts_raw, list):
            raise LineageError("transform registry artifacts must be an array")
        if len(artifacts_raw) > _MAX_SOURCES:
            raise LineageError(f"transform artifact count exceeds {_MAX_SOURCES}")
        artifacts = tuple(
            sorted(
                (
                    TransformArtifact.from_mapping(
                        _require_mapping(item, field=f"transform artifacts[{index}]"),
                        index=index,
                    )
                    for index, item in enumerate(artifacts_raw)
                ),
                key=lambda artifact: artifact.key,
            )
        )
        return cls(artifacts=artifacts)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TRANSFORM_REGISTRY_SCHEMA_VERSION,
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
        }

    def assert_lineage(self, lineage: TransformLineage) -> TransformArtifact:
        key = (
            lineage.transform_name,
            lineage.transform_version,
            lineage.code_sha256,
            lineage.config_sha256,
        )
        for artifact in self.artifacts:
            if artifact.key == key:
                return artifact
        raise LineageError(
            "lineage transform is absent from the frozen transform registry: "
            f"{lineage.transform_name}@{lineage.transform_version}"
        )


@dataclass(frozen=True, slots=True)
class LoadedTransformRegistry:
    path: str
    raw_sha256: str
    registry: TransformRegistry


def load_transform_registry(path: str | Path) -> LoadedTransformRegistry:
    resolved = Path(path)
    raw, parsed = _read_json_bytes(resolved, max_bytes=16 * 1024**2)
    return LoadedTransformRegistry(
        path=resolved.as_posix(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        registry=TransformRegistry.from_mapping(
            _require_mapping(parsed, field="transform registry")
        ),
    )


@dataclass(frozen=True, slots=True)
class DataRecord:
    """Canonical data record with bounded text and content-addressed lineage."""

    sample_id: str
    source_id: str
    source_revision: str
    split: SplitName
    source_family: str
    problem_family: str
    template_family: str
    problem: str
    messages: tuple[Message, ...]
    reference_answer: str | None
    response: str | None
    quality: tuple[tuple[str, bool | None], ...]
    strata: tuple[tuple[str, str], ...]
    lineage: TransformLineage

    def __post_init__(self) -> None:
        _require_identifier(self.sample_id, field="sample_id")
        _require_identifier(self.source_id, field="source_id")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise RecordSchemaError("source_revision must be a non-empty string")
        for field_name, value in self.families.items():
            _require_identifier(value, field=f"families.{field_name}")
        _require_text(self.problem, field="problem")
        if not self.messages or not any(message.role == "user" for message in self.messages):
            raise RecordSchemaError("messages must contain at least one user message")
        if self.reference_answer is not None:
            _require_text(self.reference_answer, field="reference_answer")
        if self.response is not None:
            _require_text(self.response, field="response")
        quality_keys = [key for key, _ in self.quality]
        if quality_keys != sorted(quality_keys) or len(quality_keys) != len(set(quality_keys)):
            raise RecordSchemaError("quality entries must be unique and sorted")
        for key, value in self.quality:
            if _QUALITY_KEY_PATTERN.fullmatch(key) is None:
                raise RecordSchemaError(f"invalid quality key: {key!r}")
            if value is not None and not isinstance(value, bool):
                raise RecordSchemaError(f"quality.{key} must be bool or null")
        strata_keys = [key for key, _ in self.strata]
        if strata_keys != sorted(strata_keys) or len(strata_keys) != len(set(strata_keys)):
            raise RecordSchemaError("strata entries must be unique and sorted")
        for key, value in self.strata:
            if _QUALITY_KEY_PATTERN.fullmatch(key) is None:
                raise RecordSchemaError(f"invalid strata key: {key!r}")
            _require_identifier(value, field=f"strata.{key}")

    @property
    def families(self) -> dict[str, str]:
        return {
            "source": self.source_family,
            "problem": self.problem_family,
            "template": self.template_family,
        }

    @property
    def content_record(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "messages": [message.to_record() for message in self.messages],
            "reference_answer": self.reference_answer,
            "response": self.response,
            "quality": dict(self.quality),
        }

    @property
    def content_sha256(self) -> str:
        return sha256_json(self.content_record)

    @property
    def payload_sha256(self) -> str:
        return sha256_json(self.to_record())

    @property
    def lineage_sha256(self) -> str:
        return sha256_json(self.lineage.to_record())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_number: int | None = None) -> DataRecord:
        prefix = f"line {line_number}: " if line_number is not None else ""
        required = {
            "schema_version",
            "sample_id",
            "source_id",
            "source_revision",
            "split",
            "families",
            "problem",
            "messages",
            "reference_answer",
            "response",
            "quality",
            "strata",
            "lineage",
        }
        try:
            _require_exact_keys(raw, required=required, field="data record")
            if raw["schema_version"] != DATA_RECORD_SCHEMA_VERSION:
                raise RecordSchemaError(
                    f"unsupported data record schema: {raw['schema_version']!r}"
                )
            families = _require_mapping(raw["families"], field="families")
            _require_exact_keys(
                families, required=set(_FAMILY_DIMENSIONS), field="families"
            )
            messages_raw = raw["messages"]
            if not isinstance(messages_raw, list):
                raise RecordSchemaError("messages must be an array")
            if len(messages_raw) > _MAX_MESSAGES:
                raise RecordSchemaError(f"message count exceeds {_MAX_MESSAGES}")
            messages = tuple(
                Message.from_mapping(
                    _require_mapping(item, field=f"messages[{index}]"),
                    field=f"messages[{index}]",
                )
                for index, item in enumerate(messages_raw)
            )
            quality_raw = _require_mapping(raw["quality"], field="quality")
            strata_raw = _require_mapping(raw["strata"], field="strata")
            if len(quality_raw) > _MAX_METADATA_ENTRIES:
                raise RecordSchemaError(
                    f"quality entry count exceeds {_MAX_METADATA_ENTRIES}"
                )
            if len(strata_raw) > _MAX_METADATA_ENTRIES:
                raise RecordSchemaError(
                    f"strata entry count exceeds {_MAX_METADATA_ENTRIES}"
                )
            if not all(isinstance(key, str) for key in quality_raw):
                raise RecordSchemaError("quality keys must be strings")
            if not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in strata_raw.items()
            ):
                raise RecordSchemaError("strata must map string keys to string values")
            try:
                split = SplitName(raw["split"])
            except (TypeError, ValueError) as error:
                raise RecordSchemaError(f"invalid split: {raw['split']!r}") from error
            reference_answer = raw["reference_answer"]
            response = raw["response"]
            if reference_answer is not None and not isinstance(reference_answer, str):
                raise RecordSchemaError("reference_answer must be string or null")
            if response is not None and not isinstance(response, str):
                raise RecordSchemaError("response must be string or null")
            return cls(
                sample_id=raw["sample_id"],
                source_id=raw["source_id"],
                source_revision=raw["source_revision"],
                split=split,
                source_family=families["source"],
                problem_family=families["problem"],
                template_family=families["template"],
                problem=raw["problem"],
                messages=messages,
                reference_answer=reference_answer,
                response=response,
                quality=tuple(sorted(quality_raw.items())),
                strata=tuple(sorted(strata_raw.items())),
                lineage=TransformLineage.from_mapping(
                    _require_mapping(raw["lineage"], field="lineage")
                ),
            )
        except DataContractError as error:
            if prefix and not str(error).startswith(prefix):
                raise type(error)(prefix + str(error)) from error
            raise

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": DATA_RECORD_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "split": self.split.value,
            "families": self.families,
            "problem": self.problem,
            "messages": [message.to_record() for message in self.messages],
            "reference_answer": self.reference_answer,
            "response": self.response,
            "quality": dict(self.quality),
            "strata": dict(self.strata),
            "lineage": self.lineage.to_record(),
        }

    def contamination_fields(self) -> tuple[tuple[str, str], ...]:
        fields: list[tuple[str, str]] = [("problem", self.problem)]
        fields.extend(
            (f"messages.{index}.{message.role}", message.content)
            for index, message in enumerate(self.messages)
        )
        if self.reference_answer is not None:
            fields.append(("reference_answer", self.reference_answer))
        if self.response is not None:
            fields.append(("response", self.response))
        return tuple(fields)


@dataclass(frozen=True, slots=True)
class LoadedDataRecords:
    path: str
    raw_sha256: str
    records: tuple[DataRecord, ...]


def load_data_records(
    path: str | Path,
    *,
    maximum_records: int = _MAX_RECORDS,
) -> LoadedDataRecords:
    if (
        not isinstance(maximum_records, int)
        or isinstance(maximum_records, bool)
        or not 1 <= maximum_records <= _MAX_RECORDS
    ):
        raise DataContractError("maximum_records must be a positive integer")
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError as error:
        raise DataContractError(f"cannot stat {resolved}: {error}") from error
    if size > _MAX_JSONL_BYTES:
        raise DataContractError(f"{resolved} exceeds {_MAX_JSONL_BYTES} bytes")
    records: list[DataRecord] = []
    seen_ids: set[str] = set()
    raw_hasher = hashlib.sha256()
    total_bytes = 0
    try:
        with resolved.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                total_bytes += len(raw_line)
                if total_bytes > _MAX_JSONL_BYTES:
                    raise RecordSchemaError(f"{resolved} exceeds {_MAX_JSONL_BYTES} bytes")
                raw_hasher.update(raw_line)
                if line_number == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                    raise RecordSchemaError(f"{resolved} must not contain a UTF-8 BOM")
                if b"\r" in raw_line:
                    raise RecordSchemaError(f"{resolved} must use canonical LF newlines")
                encoded_line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                if len(encoded_line) > _MAX_RECORD_LINE_BYTES:
                    raise RecordSchemaError(
                        f"line {line_number}: record exceeds {_MAX_RECORD_LINE_BYTES} bytes"
                    )
                try:
                    line = encoded_line.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise RecordSchemaError(
                        f"line {line_number}: input is not strict UTF-8"
                    ) from error
                if not line.strip():
                    raise RecordSchemaError(
                        f"line {line_number}: blank lines are forbidden"
                    )
                if len(records) >= maximum_records:
                    raise RecordSchemaError(f"record count exceeds {maximum_records}")
                try:
                    parsed = strict_json_loads(line)
                except DataContractError as error:
                    raise RecordSchemaError(f"line {line_number}: {error}") from error
                record = DataRecord.from_mapping(
                    _require_mapping(parsed, field=f"line {line_number}"),
                    line_number=line_number,
                )
                if record.sample_id in seen_ids:
                    raise RecordSchemaError(
                        f"line {line_number}: duplicate sample_id {record.sample_id!r}"
                    )
                seen_ids.add(record.sample_id)
                records.append(record)
    except OSError as error:
        raise DataContractError(f"cannot read {resolved}: {error}") from error
    if not records:
        raise RecordSchemaError("data JSONL must contain at least one record")
    return LoadedDataRecords(
        path=resolved.as_posix(),
        raw_sha256=raw_hasher.hexdigest(),
        records=tuple(records),
    )


@dataclass(frozen=True, slots=True)
class FamilyLeak:
    dimension: str
    family_id: str
    splits: tuple[SplitName, ...]
    sample_ids: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "family_id_sha256": hashlib.sha256(self.family_id.encode()).hexdigest(),
            "splits": [split.value for split in self.splits],
            "sample_ids": list(self.sample_ids),
        }


def find_family_leaks(records: Sequence[DataRecord]) -> tuple[FamilyLeak, ...]:
    records = tuple(records)
    members: dict[tuple[str, str], list[DataRecord]] = defaultdict(list)
    for record in records:
        if record.split is SplitName.UNASSIGNED:
            raise FamilyLeakageError("cannot audit family leakage with UNASSIGNED records")
        for dimension, family_id in record.families.items():
            members[(dimension, family_id)].append(record)
    leaks: list[FamilyLeak] = []
    for (dimension, family_id), family_records in members.items():
        splits = tuple(sorted({record.split for record in family_records}, key=str))
        if len(splits) > 1:
            leaks.append(
                FamilyLeak(
                    dimension=dimension,
                    family_id=family_id,
                    splits=splits,
                    sample_ids=tuple(sorted(record.sample_id for record in family_records)),
                )
            )
    return tuple(sorted(leaks, key=lambda item: (item.dimension, item.family_id)))


def assert_family_disjoint(records: Sequence[DataRecord]) -> None:
    leaks = find_family_leaks(records)
    if leaks:
        first = leaks[0]
        raise FamilyLeakageError(
            f"family crosses splits: {first.dimension}/{first.family_id} -> "
            f"{[split.value for split in first.splits]}"
        )


def _validate_lineage(
    records: Sequence[DataRecord],
    parent_ledger: ParentPayloadLedger,
) -> None:
    by_id = {record.sample_id: record for record in records}
    ledger_ids = {entry.sample_id for entry in parent_ledger.entries}
    overlap = ledger_ids & set(by_id)
    if overlap:
        raise LineageError(
            f"parent ledger duplicates in-set records: {sorted(overlap)[:3]}"
        )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(record: DataRecord) -> None:
        if record.sample_id in visited:
            return
        if record.sample_id in visiting:
            raise LineageError(f"lineage cycle detected at {record.sample_id}")
        visiting.add(record.sample_id)
        for parent_link in record.lineage.parents:
            if parent_link.sample_id == record.sample_id:
                raise LineageError(f"{record.sample_id} is its own lineage parent")
            parent = by_id.get(parent_link.sample_id)
            if parent is not None:
                visit(parent)
        visiting.remove(record.sample_id)
        visited.add(record.sample_id)

    for current in records:
        visit(current)
    for record in records:
        for parent_link in record.lineage.parents:
            parent = by_id.get(parent_link.sample_id)
            if parent is None:
                external_parent = parent_ledger.entry(parent_link.sample_id)
                parent_payload_sha256 = external_parent.payload_sha256
                parent_split = external_parent.split
            else:
                parent_payload_sha256 = parent.payload_sha256
                parent_split = parent.split
            if parent_payload_sha256 != parent_link.payload_sha256:
                raise LineageError(
                    f"{record.sample_id} parent digest mismatch for {parent_link.sample_id}"
                )
            if parent_split is not record.split:
                raise LineageError(
                    f"lineage crosses split boundary: {parent_link.sample_id} "
                    f"({parent_split.value}) -> {record.sample_id} ({record.split.value})"
                )


@dataclass(frozen=True, slots=True)
class RecordSetSummary:
    record_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    semantic_sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "semantic_sha256": self.semantic_sha256,
        }


def validate_record_set(
    records: Sequence[DataRecord],
    registry: SourceRegistry,
    *,
    transform_registry: TransformRegistry,
    require_assigned: bool = True,
    require_family_disjoint: bool = True,
    parent_ledger: ParentPayloadLedger = EMPTY_PARENT_PAYLOAD_LEDGER,
) -> RecordSetSummary:
    records = tuple(records)
    if not records:
        raise RecordSchemaError("record set must not be empty")
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise RecordSchemaError("record set contains duplicate sample_id values")
    if require_assigned and any(record.split is SplitName.UNASSIGNED for record in records):
        raise RecordSchemaError("final record set contains UNASSIGNED records")
    content_hashes: dict[str, str] = {}
    for record in records:
        transform_registry.assert_lineage(record.lineage)
        source = registry.assert_usage(record.source_id, record.split)
        if record.source_revision != source.revision:
            raise SourceRegistryError(
                f"{record.sample_id} revision does not match registry source {record.source_id}"
            )
        duplicate = content_hashes.get(record.content_sha256)
        if duplicate is not None:
            raise RecordSchemaError(
                f"exact duplicate content: {duplicate} and {record.sample_id}"
            )
        content_hashes[record.content_sha256] = record.sample_id
    _validate_lineage(records, parent_ledger)
    if require_family_disjoint:
        assert_family_disjoint(records)
    ordered = sorted(records, key=lambda item: item.sample_id)
    semantic_sha = sha256_json(
        [[record.sample_id, record.payload_sha256] for record in ordered]
    )
    return RecordSetSummary(
        record_count=len(records),
        split_counts=dict(Counter(record.split.value for record in records)),
        source_counts=dict(Counter(record.source_id for record in records)),
        semantic_sha256=semantic_sha,
    )


@dataclass(frozen=True, slots=True)
class SplitAllocation:
    split: SplitName
    weight: int

    def __post_init__(self) -> None:
        if self.split in {SplitName.UNASSIGNED, SplitName.EVALUATION}:
            raise DataContractError("automatic allocations are limited to training/dev splits")
        if not isinstance(self.weight, int) or isinstance(self.weight, bool) or self.weight <= 0:
            raise DataContractError("split allocation weight must be a positive integer")

    def to_record(self) -> dict[str, object]:
        return {"split": self.split.value, "weight": self.weight}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> SplitAllocation:
        _require_exact_keys(
            raw, required={"split", "weight"}, field=f"allocations[{index}]"
        )
        try:
            split = SplitName(raw["split"])
        except (TypeError, ValueError) as error:
            raise DataContractError(f"invalid allocation split: {raw['split']!r}") from error
        return cls(split=split, weight=raw["weight"])


@dataclass(frozen=True, slots=True)
class FamilySplitPolicy:
    """Order-invariant component assignment using a frozen weighted hash bucket."""

    namespace: str
    allocations: tuple[SplitAllocation, ...]
    family_dimensions: tuple[str, ...] = _FAMILY_DIMENSIONS

    def __post_init__(self) -> None:
        _require_identifier(self.namespace, field="split policy namespace")
        if self.family_dimensions != _FAMILY_DIMENSIONS:
            raise DataContractError(
                f"family_dimensions must be exactly {_FAMILY_DIMENSIONS!r}"
            )
        split_names = [allocation.split for allocation in self.allocations]
        if not split_names:
            raise DataContractError("split policy must have at least one allocation")
        if split_names != sorted(split_names, key=str) or len(split_names) != len(set(split_names)):
            raise DataContractError("split allocations must be unique and sorted by split name")

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SPLIT_POLICY_SCHEMA_VERSION,
            "namespace": self.namespace,
            "family_dimensions": list(self.family_dimensions),
            "allocations": [allocation.to_record() for allocation in self.allocations],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FamilySplitPolicy:
        required = {
            "schema_version",
            "namespace",
            "family_dimensions",
            "allocations",
        }
        _require_exact_keys(raw, required=required, field="family split policy")
        if raw["schema_version"] != SPLIT_POLICY_SCHEMA_VERSION:
            raise DataContractError(
                f"unsupported family split policy schema: {raw['schema_version']!r}"
            )
        dimensions = raw["family_dimensions"]
        allocations = raw["allocations"]
        if not isinstance(dimensions, list) or any(
            not isinstance(item, str) for item in dimensions
        ):
            raise DataContractError("family_dimensions must be a string array")
        if not isinstance(allocations, list):
            raise DataContractError("allocations must be an array")
        parsed_allocations = tuple(
            sorted(
                (
                    SplitAllocation.from_mapping(
                        _require_mapping(item, field=f"allocations[{index}]"),
                        index=index,
                    )
                    for index, item in enumerate(allocations)
                ),
                key=lambda item: str(item.split),
            )
        )
        return cls(
            namespace=raw["namespace"],
            allocations=parsed_allocations,
            family_dimensions=tuple(dimensions),
        )


@dataclass(frozen=True, slots=True)
class LoadedFamilySplitPolicy:
    path: str
    raw_sha256: str
    policy: FamilySplitPolicy


def load_family_split_policy(path: str | Path) -> LoadedFamilySplitPolicy:
    resolved = Path(path)
    raw, parsed = _read_json_bytes(resolved, max_bytes=16 * 1024**2)
    return LoadedFamilySplitPolicy(
        path=resolved.as_posix(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        policy=FamilySplitPolicy.from_mapping(
            _require_mapping(parsed, field="family split policy")
        ),
    )


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    policy_sha256: str
    component_count: int
    assignment_sha256: str
    split_counts: dict[str, int]
    assignments: dict[str, str]

    def to_record(self) -> dict[str, object]:
        return {
            "policy_sha256": self.policy_sha256,
            "component_count": self.component_count,
            "assignment_sha256": self.assignment_sha256,
            "split_counts": dict(sorted(self.split_counts.items())),
            "assignments": dict(sorted(self.assignments.items())),
        }


def assign_family_disjoint_splits(
    records: Sequence[DataRecord],
    policy: FamilySplitPolicy,
) -> tuple[tuple[DataRecord, ...], SplitAssignment]:
    """Assign transitive family components, independent of input row order.

    Allocation weights are deterministic proportions, not an exact-count
    promise.  D15 must select a source pool whose resulting frozen counts meet
    the preregistered materialization targets.
    """

    records = tuple(records)
    if not records:
        raise DataContractError("cannot split an empty record set")
    if any(record.split is not SplitName.UNASSIGNED for record in records):
        raise DataContractError("automatic split input must be UNASSIGNED")
    if any(record.lineage.parents for record in records):
        raise DataContractError(
            "automatic split assignment is limited to root records; derive children after split"
        )
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise DataContractError("automatic split input has duplicate sample IDs")
    dsu = _DisjointSet(ids)
    family_owner: dict[tuple[str, str], str] = {}
    for record in sorted(records, key=lambda item: item.sample_id):
        for dimension in policy.family_dimensions:
            key = (dimension, record.families[dimension])
            previous = family_owner.setdefault(key, record.sample_id)
            dsu.union(previous, record.sample_id)
    components: dict[str, list[DataRecord]] = defaultdict(list)
    for record in records:
        components[dsu.find(record.sample_id)].append(record)

    total_weight = sum(allocation.weight for allocation in policy.allocations)
    cumulative: list[tuple[int, SplitName]] = []
    running = 0
    for allocation in policy.allocations:
        running += allocation.weight
        cumulative.append((running, allocation.split))

    assignments: dict[str, str] = {}
    assigned_records: list[DataRecord] = []
    for component in sorted(
        components.values(), key=lambda values: min(record.sample_id for record in values)
    ):
        component_payload = [
            [record.sample_id, record.families]
            for record in sorted(component, key=lambda item: item.sample_id)
        ]
        digest = hashlib.sha256(
            policy.namespace.encode("utf-8") + b"\0" + canonical_json_bytes(component_payload)
        ).digest()
        bucket = int.from_bytes(digest, "big") % total_weight
        selected = next(split for boundary, split in cumulative if bucket < boundary)
        for record in component:
            assignments[record.sample_id] = selected.value
            assigned_records.append(replace(record, split=selected))
    ordered = tuple(sorted(assigned_records, key=lambda item: item.sample_id))
    assert_family_disjoint(ordered)
    assignment_sha = sha256_json(dict(sorted(assignments.items())))
    return ordered, SplitAssignment(
        policy_sha256=policy.sha256,
        component_count=len(components),
        assignment_sha256=assignment_sha,
        split_counts=dict(Counter(assignments.values())),
        assignments=dict(sorted(assignments.items())),
    )


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    sample_id: str
    split: str
    source_id: str
    source_revision: str
    payload_sha256: str
    content_sha256: str
    lineage_sha256: str
    family_sha256: dict[str, str]

    @classmethod
    def from_data_record(cls, record: DataRecord) -> ManifestRecord:
        return cls(
            sample_id=record.sample_id,
            split=record.split.value,
            source_id=record.source_id,
            source_revision=record.source_revision,
            payload_sha256=record.payload_sha256,
            content_sha256=record.content_sha256,
            lineage_sha256=record.lineage_sha256,
            family_sha256={
                dimension: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for dimension, value in sorted(record.families.items())
            },
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> ManifestRecord:
        required = {
            "sample_id",
            "split",
            "source_id",
            "source_revision",
            "payload_sha256",
            "content_sha256",
            "lineage_sha256",
            "family_sha256",
        }
        _require_exact_keys(raw, required=required, field=f"manifest.records[{index}]")
        family = _require_mapping(raw["family_sha256"], field="family_sha256")
        _require_exact_keys(family, required=set(_FAMILY_DIMENSIONS), field="family_sha256")
        for dimension, digest in family.items():
            _require_sha256(digest, field=f"family_sha256.{dimension}")
        for field in ("payload_sha256", "content_sha256", "lineage_sha256"):
            _require_sha256(raw[field], field=field)
        _require_identifier(raw["sample_id"], field="sample_id")
        _require_identifier(raw["source_id"], field="source_id")
        try:
            split = SplitName(raw["split"])
        except (TypeError, ValueError) as error:
            raise ManifestIntegrityError(f"invalid manifest split: {raw['split']!r}") from error
        if split is SplitName.UNASSIGNED:
            raise ManifestIntegrityError("manifest records cannot be UNASSIGNED")
        if not isinstance(raw["source_revision"], str) or not raw["source_revision"]:
            raise ManifestIntegrityError("source_revision must be non-empty")
        return cls(
            sample_id=raw["sample_id"],
            split=split.value,
            source_id=raw["source_id"],
            source_revision=raw["source_revision"],
            payload_sha256=raw["payload_sha256"],
            content_sha256=raw["content_sha256"],
            lineage_sha256=raw["lineage_sha256"],
            family_sha256=dict(sorted(family.items())),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "payload_sha256": self.payload_sha256,
            "content_sha256": self.content_sha256,
            "lineage_sha256": self.lineage_sha256,
            "family_sha256": dict(sorted(self.family_sha256.items())),
        }


@dataclass(frozen=True, slots=True)
class DataManifest:
    source_registry_sha256: str
    transform_registry_sha256: str
    split_policy_sha256: str
    parent_ledger_sha256: str
    contamination_policy_sha256: str
    contamination_report_sha256: str
    record_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    record_set_sha256: str
    records: tuple[ManifestRecord, ...]
    manifest_sha256: str

    @property
    def body_record(self) -> dict[str, object]:
        return {
            "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
            "source_registry_sha256": self.source_registry_sha256,
            "transform_registry_sha256": self.transform_registry_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "contamination_policy_sha256": self.contamination_policy_sha256,
            "contamination_report_sha256": self.contamination_report_sha256,
            "record_count": self.record_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "record_set_sha256": self.record_set_sha256,
            "records": [record.to_record() for record in self.records],
        }

    def to_record(self) -> dict[str, object]:
        return {**self.body_record, "manifest_sha256": self.manifest_sha256}

    def verify(self) -> None:
        for field, value in {
            "source_registry_sha256": self.source_registry_sha256,
            "transform_registry_sha256": self.transform_registry_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "contamination_policy_sha256": self.contamination_policy_sha256,
            "contamination_report_sha256": self.contamination_report_sha256,
            "record_set_sha256": self.record_set_sha256,
            "manifest_sha256": self.manifest_sha256,
        }.items():
            _require_sha256(value, field=field)
        if self.record_count != len(self.records):
            raise ManifestIntegrityError("manifest record_count does not match records")
        sample_ids = [record.sample_id for record in self.records]
        if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ManifestIntegrityError("manifest records must be unique and sorted")
        if dict(Counter(record.split for record in self.records)) != self.split_counts:
            raise ManifestIntegrityError("manifest split_counts do not match records")
        if dict(Counter(record.source_id for record in self.records)) != self.source_counts:
            raise ManifestIntegrityError("manifest source_counts do not match records")
        if sha256_json(
            [[record.sample_id, record.payload_sha256] for record in self.records]
        ) != self.record_set_sha256:
            raise ManifestIntegrityError("manifest record_set_sha256 mismatch")
        if sha256_json(self.body_record) != self.manifest_sha256:
            raise ManifestIntegrityError("manifest_sha256 mismatch")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DataManifest:
        required = {
            "schema_version",
            "source_registry_sha256",
            "transform_registry_sha256",
            "split_policy_sha256",
            "parent_ledger_sha256",
            "contamination_policy_sha256",
            "contamination_report_sha256",
            "record_count",
            "split_counts",
            "source_counts",
            "record_set_sha256",
            "records",
            "manifest_sha256",
        }
        _require_exact_keys(raw, required=required, field="data manifest")
        if raw["schema_version"] != DATA_MANIFEST_SCHEMA_VERSION:
            raise ManifestIntegrityError(
                f"unsupported data manifest schema: {raw['schema_version']!r}"
            )
        records_raw = raw["records"]
        if not isinstance(records_raw, list):
            raise ManifestIntegrityError("manifest records must be an array")
        records = tuple(
            ManifestRecord.from_mapping(
                _require_mapping(item, field=f"manifest.records[{index}]"), index=index
            )
            for index, item in enumerate(records_raw)
        )
        split_counts = _require_count_mapping(raw["split_counts"], field="split_counts")
        source_counts = _require_count_mapping(raw["source_counts"], field="source_counts")
        record_count = raw["record_count"]
        if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 0:
            raise ManifestIntegrityError("record_count must be a non-negative integer")
        manifest = cls(
            source_registry_sha256=raw["source_registry_sha256"],
            transform_registry_sha256=raw["transform_registry_sha256"],
            split_policy_sha256=raw["split_policy_sha256"],
            parent_ledger_sha256=raw["parent_ledger_sha256"],
            contamination_policy_sha256=raw["contamination_policy_sha256"],
            contamination_report_sha256=raw["contamination_report_sha256"],
            record_count=record_count,
            split_counts=split_counts,
            source_counts=source_counts,
            record_set_sha256=raw["record_set_sha256"],
            records=records,
            manifest_sha256=raw["manifest_sha256"],
        )
        manifest.verify()
        return manifest


def _require_count_mapping(value: object, *, field: str) -> dict[str, int]:
    mapping = _require_mapping(value, field=field)
    result: dict[str, int] = {}
    for key, count in mapping.items():
        if not key or not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ManifestIntegrityError(f"{field} must map non-empty strings to integer counts")
        result[key] = count
    return dict(sorted(result.items()))


def _build_data_manifest_from_verified_scan(
    records: Sequence[DataRecord],
    registry: SourceRegistry,
    *,
    transform_registry: TransformRegistry,
    split_policy_sha256: str,
    parent_ledger: ParentPayloadLedger,
    contamination_policy_sha256: str,
    contamination_report_sha256: str,
) -> DataManifest:
    records = tuple(records)
    _require_sha256(split_policy_sha256, field="split_policy_sha256")
    _require_sha256(contamination_policy_sha256, field="contamination_policy_sha256")
    _require_sha256(contamination_report_sha256, field="contamination_report_sha256")
    summary = validate_record_set(
        records,
        registry,
        transform_registry=transform_registry,
        parent_ledger=parent_ledger,
    )
    manifest_records = tuple(
        ManifestRecord.from_data_record(record)
        for record in sorted(records, key=lambda item: item.sample_id)
    )
    provisional = DataManifest(
        source_registry_sha256=registry.sha256,
        transform_registry_sha256=transform_registry.sha256,
        split_policy_sha256=split_policy_sha256,
        parent_ledger_sha256=parent_ledger.sha256,
        contamination_policy_sha256=contamination_policy_sha256,
        contamination_report_sha256=contamination_report_sha256,
        record_count=summary.record_count,
        split_counts=dict(sorted(summary.split_counts.items())),
        source_counts=dict(sorted(summary.source_counts.items())),
        record_set_sha256=summary.semantic_sha256,
        records=manifest_records,
        manifest_sha256="0" * 64,
    )
    manifest = replace(provisional, manifest_sha256=sha256_json(provisional.body_record))
    manifest.verify()
    return manifest


def write_json_atomic(record: Mapping[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    descriptor: int | None = None
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)


def write_data_manifest(manifest: DataManifest, output_path: str | Path) -> None:
    manifest.verify()
    write_json_atomic(manifest.to_record(), output_path)


def load_data_manifest(path: str | Path) -> DataManifest:
    _, parsed = _read_json_bytes(Path(path), max_bytes=2 * 1024**3)
    return DataManifest.from_mapping(_require_mapping(parsed, field="data manifest"))


def split_distribution_error(
    assignment: SplitAssignment,
    policy: FamilySplitPolicy,
) -> dict[str, float]:
    """Diagnostic only: actual-minus-target proportion for each split."""

    total_records = sum(assignment.split_counts.values())
    total_weight = sum(allocation.weight for allocation in policy.allocations)
    if total_records == 0 or total_weight == 0:
        raise DataContractError("cannot measure an empty split distribution")
    return {
        allocation.split.value: (
            assignment.split_counts.get(allocation.split.value, 0) / total_records
            - allocation.weight / total_weight
        )
        for allocation in policy.allocations
    }


def max_absolute_split_error(
    assignment: SplitAssignment,
    policy: FamilySplitPolicy,
) -> float:
    return max(math.fabs(value) for value in split_distribution_error(assignment, policy).values())
