"""Versioned D07 benchmark, generation, and sealed-reference contracts.

The generator-facing objects in this module deliberately contain no reference
answers.  References are loaded into :class:`SealedAnswerVault`, which is a
scoring capability rather than a mapping.  This is an API/process boundary,
not a Python sandbox: production runs must additionally keep the vault outside
the model-serving process and enforce filesystem permissions.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from posttrain_lab.data import canonical_json_bytes, strict_json_loads, write_json_atomic

BENCHMARK_DESCRIPTOR_SCHEMA_VERSION = "d07-benchmark-descriptor-v1"
PUBLIC_BENCHMARK_ITEM_SCHEMA_VERSION = "d07-public-benchmark-item-v1"
SEALED_REFERENCE_SCHEMA_VERSION = "d07-sealed-reference-v1"
GENERATION_PROTOCOL_SCHEMA_VERSION = "d07-generation-protocol-v1"
CHECKPOINT_IDENTITY_SCHEMA_VERSION = "d07-checkpoint-identity-v1"
GENERATION_RECORD_SCHEMA_VERSION = "d07-generation-record-v1"
GENERATION_MANIFEST_SCHEMA_VERSION = "d07-generation-manifest-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+@:/-]{0,255}")
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_MAX_JSON_BYTES = 64 * 1024**2
_MAX_JSONL_BYTES = 16 * 1024**3
_MAX_LINE_BYTES = 16 * 1024**2
_MAX_ITEMS = 1_000_000
_MAX_SAMPLES_PER_ITEM = 1_024
_MAX_PROMPT_CHARS = 1_000_000
_MAX_GENERATED_CHARS = 4_000_000
_MAX_TOKEN_ID = 2**31 - 1
_MAX_SEED = 2**63 - 1


class EvaluationContractError(ValueError):
    """Raised when a D07 artifact violates a frozen contract."""


class SealedReferenceError(EvaluationContractError):
    """Raised when the protected reference store is invalid or incompatible."""


class GenerationContractError(EvaluationContractError):
    """Raised when generation requests or outputs are incomplete or inconsistent."""


class BenchmarkTask(StrEnum):
    EXACT_MATH = "exact_math"
    STRICT_LABEL = "strict_label"


class RevisionKind(StrEnum):
    GIT_COMMIT = "git_commit"
    SHA256 = "sha256"


class DecodingMode(StrEnum):
    GREEDY = "greedy"
    SAMPLING = "sampling"


class GenerationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class FinishReason(StrEnum):
    EOS = "eos"
    STOP_TOKEN = "stop_token"
    STOP_SEQUENCE = "stop_sequence"
    LENGTH = "length"


def _require_exact_keys(
    raw: Mapping[str, Any], *, required: set[str] | frozenset[str], field: str
) -> None:
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise EvaluationContractError(
            f"{field} has invalid keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EvaluationContractError(f"{field} must be a string-keyed mapping")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise EvaluationContractError(f"{field} must be a bounded portable identifier")
    if ".." in value or value.endswith(("/", ":")):
        raise EvaluationContractError(f"{field} contains a forbidden path-like segment")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvaluationContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, *, kind: RevisionKind, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluationContractError(f"{field} must be a string")
    pattern = _GIT_REVISION_RE if kind is RevisionKind.GIT_COMMIT else _SHA256_RE
    if pattern.fullmatch(value) is None:
        expectation = "full 40/64-hex Git commit" if kind is RevisionKind.GIT_COMMIT else "SHA-256"
        raise EvaluationContractError(f"{field} must be a {expectation}")
    return value


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationContractError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise EvaluationContractError(f"{field} must be in {interval}")
    return value


def _require_text(
    value: object,
    *,
    field: str,
    maximum_chars: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise EvaluationContractError(f"{field} must be a string")
    if len(value) > maximum_chars:
        raise EvaluationContractError(f"{field} exceeds {maximum_chars} characters")
    if not allow_empty and not value.strip():
        raise EvaluationContractError(f"{field} must be non-empty")
    if "\x00" in value or "\r" in value:
        raise EvaluationContractError(f"{field} contains NUL or non-canonical CR")
    if any(unicodedata.category(character) == "Cs" for character in value):
        raise EvaluationContractError(f"{field} contains an unpaired surrogate")
    if unicodedata.normalize("NFC", value) != value:
        raise EvaluationContractError(f"{field} must already be NFC-normalized")
    return value


def _read_json(path: str | Path, *, maximum_bytes: int = _MAX_JSON_BYTES) -> tuple[bytes, Any]:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError as error:
        raise EvaluationContractError(f"cannot stat {resolved}: {error}") from error
    if size > maximum_bytes:
        raise EvaluationContractError(f"{resolved} exceeds {maximum_bytes} bytes")
    try:
        with resolved.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as error:
        raise EvaluationContractError(f"cannot read {resolved}: {error}") from error
    if len(raw) > maximum_bytes:
        raise EvaluationContractError(f"{resolved} exceeds {maximum_bytes} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise EvaluationContractError(f"{resolved} must not contain a UTF-8 BOM")
    if b"\r" in raw:
        raise EvaluationContractError(f"{resolved} must use canonical LF newlines")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EvaluationContractError(f"{resolved} is not strict UTF-8") from error
    try:
        parsed = strict_json_loads(text)
    except ValueError as error:
        raise EvaluationContractError(f"{resolved} is not valid strict JSON: {error}") from error
    return raw, parsed


def _iter_jsonl(
    path: str | Path,
    *,
    maximum_records: int,
) -> tuple[str, tuple[tuple[int, Mapping[str, Any]], ...]]:
    if not 1 <= maximum_records <= _MAX_ITEMS * _MAX_SAMPLES_PER_ITEM:
        raise EvaluationContractError("maximum_records is outside the supported bound")
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError as error:
        raise EvaluationContractError(f"cannot stat {resolved}: {error}") from error
    if size > _MAX_JSONL_BYTES:
        raise EvaluationContractError(f"{resolved} exceeds {_MAX_JSONL_BYTES} bytes")
    hasher = hashlib.sha256()
    rows: list[tuple[int, Mapping[str, Any]]] = []
    total_bytes = 0
    try:
        with resolved.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                total_bytes += len(raw_line)
                if total_bytes > _MAX_JSONL_BYTES:
                    raise EvaluationContractError(f"{resolved} exceeds {_MAX_JSONL_BYTES} bytes")
                hasher.update(raw_line)
                if line_number == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                    raise EvaluationContractError(f"{resolved} must not contain a UTF-8 BOM")
                if b"\r" in raw_line:
                    raise EvaluationContractError(f"{resolved} must use canonical LF newlines")
                encoded = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                if len(encoded) > _MAX_LINE_BYTES:
                    raise EvaluationContractError(
                        f"line {line_number}: record exceeds {_MAX_LINE_BYTES} bytes"
                    )
                if not encoded.strip():
                    raise EvaluationContractError(f"line {line_number}: blank lines are forbidden")
                if len(rows) >= maximum_records:
                    raise EvaluationContractError(f"record count exceeds {maximum_records}")
                try:
                    text = encoded.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise EvaluationContractError(
                        f"line {line_number}: input is not strict UTF-8"
                    ) from error
                try:
                    parsed = strict_json_loads(text)
                except ValueError as error:
                    raise EvaluationContractError(f"line {line_number}: {error}") from error
                rows.append((line_number, _require_mapping(parsed, field=f"line {line_number}")))
    except OSError as error:
        raise EvaluationContractError(f"cannot read {resolved}: {error}") from error
    if not rows:
        raise EvaluationContractError("JSONL input must contain at least one record")
    return hasher.hexdigest(), tuple(rows)


def _write_jsonl_atomic(records: Sequence[Mapping[str, object]], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temporary_path: str | None = None
    hasher = hashlib.sha256()
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            for record in records:
                line = canonical_json_bytes(record) + b"\n"
                hasher.update(line)
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkDescriptor:
    benchmark_id: str
    revision_kind: RevisionKind
    benchmark_revision: str
    split_name: str
    task: BenchmarkTask
    item_count: int
    public_items_sha256: str
    sealed_references_sha256: str
    source_registry_sha256: str
    data_manifest_sha256: str
    schema_version: str = BENCHMARK_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_DESCRIPTOR_SCHEMA_VERSION:
            raise EvaluationContractError("unsupported benchmark descriptor schema")
        _require_identifier(self.benchmark_id, field="benchmark_id")
        _require_revision(
            self.benchmark_revision,
            kind=self.revision_kind,
            field="benchmark_revision",
        )
        _require_identifier(self.split_name, field="split_name")
        _require_int(self.item_count, field="item_count", minimum=1, maximum=_MAX_ITEMS)
        for field in (
            "public_items_sha256",
            "sealed_references_sha256",
            "source_registry_sha256",
            "data_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkDescriptor:
        required = {
            "schema_version",
            "benchmark_id",
            "revision_kind",
            "benchmark_revision",
            "split_name",
            "task",
            "item_count",
            "public_items_sha256",
            "sealed_references_sha256",
            "source_registry_sha256",
            "data_manifest_sha256",
        }
        _require_exact_keys(raw, required=required, field="benchmark descriptor")
        try:
            revision_kind = RevisionKind(raw["revision_kind"])
            task = BenchmarkTask(raw["task"])
        except (TypeError, ValueError) as error:
            raise EvaluationContractError("invalid benchmark revision kind or task") from error
        return cls(
            schema_version=raw["schema_version"],
            benchmark_id=raw["benchmark_id"],
            revision_kind=revision_kind,
            benchmark_revision=raw["benchmark_revision"],
            split_name=raw["split_name"],
            task=task,
            item_count=raw["item_count"],
            public_items_sha256=raw["public_items_sha256"],
            sealed_references_sha256=raw["sealed_references_sha256"],
            source_registry_sha256=raw["source_registry_sha256"],
            data_manifest_sha256=raw["data_manifest_sha256"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "revision_kind": self.revision_kind.value,
            "benchmark_revision": self.benchmark_revision,
            "split_name": self.split_name,
            "task": self.task.value,
            "item_count": self.item_count,
            "public_items_sha256": self.public_items_sha256,
            "sealed_references_sha256": self.sealed_references_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest()


def _load_benchmark_descriptor_with_raw_sha256(
    path: str | Path,
) -> tuple[str, BenchmarkDescriptor]:
    raw, parsed = _read_json(path)
    descriptor = BenchmarkDescriptor.from_mapping(_require_mapping(parsed, field="descriptor"))
    return hashlib.sha256(raw).hexdigest(), descriptor


def load_benchmark_descriptor(path: str | Path) -> BenchmarkDescriptor:
    _, descriptor = _load_benchmark_descriptor_with_raw_sha256(path)
    return descriptor


@dataclass(frozen=True, slots=True)
class PublicBenchmarkItem:
    benchmark_id: str
    benchmark_revision: str
    item_id: str
    item_index: int
    prompt: str
    strata: tuple[tuple[str, str], ...]
    schema_version: str = PUBLIC_BENCHMARK_ITEM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PUBLIC_BENCHMARK_ITEM_SCHEMA_VERSION:
            raise EvaluationContractError("unsupported public benchmark item schema")
        _require_identifier(self.benchmark_id, field="benchmark_id")
        if _GIT_REVISION_RE.fullmatch(self.benchmark_revision) is None:
            raise EvaluationContractError("benchmark_revision must be a full immutable revision")
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", minimum=0, maximum=_MAX_ITEMS - 1)
        _require_text(
            self.prompt,
            field="prompt",
            maximum_chars=_MAX_PROMPT_CHARS,
        )
        if tuple(sorted(self.strata)) != self.strata:
            raise EvaluationContractError("strata must be canonically sorted")
        if len({key for key, _ in self.strata}) != len(self.strata):
            raise EvaluationContractError("strata keys must be unique")
        for key, value in self.strata:
            _require_identifier(key, field="strata key")
            _require_identifier(value, field=f"strata.{key}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_number: int) -> PublicBenchmarkItem:
        required = {
            "schema_version",
            "benchmark_id",
            "benchmark_revision",
            "item_id",
            "item_index",
            "prompt",
            "strata",
        }
        _require_exact_keys(raw, required=required, field=f"public item line {line_number}")
        strata = _require_mapping(raw["strata"], field=f"line {line_number}.strata")
        if any(not isinstance(value, str) for value in strata.values()):
            raise EvaluationContractError(f"line {line_number}.strata values must be strings")
        return cls(
            schema_version=raw["schema_version"],
            benchmark_id=raw["benchmark_id"],
            benchmark_revision=raw["benchmark_revision"],
            item_id=raw["item_id"],
            item_index=raw["item_index"],
            prompt=raw["prompt"],
            strata=tuple(sorted(strata.items())),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "item_id": self.item_id,
            "item_index": self.item_index,
            "prompt": self.prompt,
            "strata": dict(self.strata),
        }

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedPublicBenchmark:
    descriptor: BenchmarkDescriptor
    path: str
    raw_sha256: str
    items: tuple[PublicBenchmarkItem, ...]

    @property
    def item_set_sha256(self) -> str:
        payload = [
            {
                "item_id": item.item_id,
                "item_index": item.item_index,
                "prompt_sha256": item.prompt_sha256,
            }
            for item in self.items
        ]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_public_benchmark(
    descriptor: BenchmarkDescriptor,
    path: str | Path,
) -> LoadedPublicBenchmark:
    raw_sha256, rows = _iter_jsonl(path, maximum_records=descriptor.item_count)
    if raw_sha256 != descriptor.public_items_sha256:
        raise EvaluationContractError("public benchmark bytes do not match descriptor")
    items = tuple(
        PublicBenchmarkItem.from_mapping(raw, line_number=line_number) for line_number, raw in rows
    )
    if len(items) != descriptor.item_count:
        raise EvaluationContractError("public benchmark item_count does not match descriptor")
    if tuple(item.item_index for item in items) != tuple(range(len(items))):
        raise EvaluationContractError("public items must be ordered by consecutive item_index")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise EvaluationContractError("public benchmark contains duplicate item_id")
    for item in items:
        if (
            item.benchmark_id != descriptor.benchmark_id
            or item.benchmark_revision != descriptor.benchmark_revision
        ):
            raise EvaluationContractError(
                f"public item {item.item_id} does not match descriptor identity"
            )
    return LoadedPublicBenchmark(
        descriptor=descriptor,
        path=Path(path).as_posix(),
        raw_sha256=raw_sha256,
        items=items,
    )


@dataclass(frozen=True, slots=True)
class _SealedReference:
    benchmark_id: str
    benchmark_revision: str
    item_id: str
    reference: str
    schema_version: str = SEALED_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_REFERENCE_SCHEMA_VERSION:
            raise SealedReferenceError("unsupported sealed reference schema")
        _require_identifier(self.benchmark_id, field="sealed benchmark_id")
        if _GIT_REVISION_RE.fullmatch(self.benchmark_revision) is None:
            raise SealedReferenceError("sealed benchmark_revision must be immutable")
        _require_identifier(self.item_id, field="sealed item_id")
        _require_text(
            self.reference,
            field="sealed reference",
            maximum_chars=_MAX_PROMPT_CHARS,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_number: int) -> _SealedReference:
        required = {
            "schema_version",
            "benchmark_id",
            "benchmark_revision",
            "item_id",
            "reference",
        }
        _require_exact_keys(raw, required=required, field=f"sealed line {line_number}")
        try:
            return cls(
                schema_version=raw["schema_version"],
                benchmark_id=raw["benchmark_id"],
                benchmark_revision=raw["benchmark_revision"],
                item_id=raw["item_id"],
                reference=raw["reference"],
            )
        except EvaluationContractError as error:
            raise SealedReferenceError(str(error)) from error


class SealedAnswerVault:
    """Opaque reference-scoring capability with redacted representation.

    There is intentionally no mapping protocol, reference getter, serializer,
    or pickle support.  The runner receives only :class:`LoadedPublicBenchmark`;
    this object belongs exclusively to the evaluator process.
    """

    __slots__ = (
        "__benchmark_id",
        "__benchmark_revision",
        "__item_set_sha256",
        "__raw_sha256",
        "__references",
        "__task",
    )

    def __init__(
        self,
        *,
        benchmark_id: str,
        benchmark_revision: str,
        task: BenchmarkTask,
        references: Mapping[str, str],
        raw_sha256: str,
        item_set_sha256: str,
    ) -> None:
        self.__benchmark_id = benchmark_id
        self.__benchmark_revision = benchmark_revision
        self.__task = task
        self.__references = dict(references)
        self.__raw_sha256 = raw_sha256
        self.__item_set_sha256 = item_set_sha256

    def __repr__(self) -> str:
        return (
            f"SealedAnswerVault(benchmark_id={self.__benchmark_id!r}, "
            f"item_count={len(self.__references)}, references=<redacted>)"
        )

    def __reduce__(self) -> object:
        raise TypeError("SealedAnswerVault cannot be serialized or sent to a generator process")

    @property
    def benchmark_id(self) -> str:
        return self.__benchmark_id

    @property
    def benchmark_revision(self) -> str:
        return self.__benchmark_revision

    @property
    def task(self) -> BenchmarkTask:
        return self.__task

    @property
    def raw_sha256(self) -> str:
        return self.__raw_sha256

    @property
    def item_count(self) -> int:
        return len(self.__references)

    @property
    def item_set_sha256(self) -> str:
        return self.__item_set_sha256

    def _score_with(
        self,
        item_id: str,
        prediction: str,
        scorer: Callable[[BenchmarkTask, str, str], object],
    ) -> object:
        try:
            reference = self.__references[item_id]
        except KeyError as error:
            raise SealedReferenceError(f"unknown sealed item_id: {item_id}") from error
        return scorer(self.__task, reference, prediction)


def load_sealed_answer_vault(
    public: LoadedPublicBenchmark,
    path: str | Path,
) -> SealedAnswerVault:
    descriptor = public.descriptor
    try:
        raw_sha256, rows = _iter_jsonl(path, maximum_records=descriptor.item_count)
    except EvaluationContractError as error:
        raise SealedReferenceError(str(error)) from error
    if raw_sha256 != descriptor.sealed_references_sha256:
        raise SealedReferenceError("sealed reference bytes do not match descriptor")
    references = tuple(
        _SealedReference.from_mapping(raw, line_number=line_number) for line_number, raw in rows
    )
    if len(references) != descriptor.item_count:
        raise SealedReferenceError("sealed reference count does not match descriptor")
    by_id: dict[str, str] = {}
    for reference in references:
        if reference.item_id in by_id:
            raise SealedReferenceError(f"duplicate sealed item_id: {reference.item_id}")
        if (
            reference.benchmark_id != descriptor.benchmark_id
            or reference.benchmark_revision != descriptor.benchmark_revision
        ):
            raise SealedReferenceError(
                f"sealed item {reference.item_id} does not match descriptor identity"
            )
        by_id[reference.item_id] = reference.reference
    public_ids = {item.item_id for item in public.items}
    if set(by_id) != public_ids:
        missing = sorted(public_ids - set(by_id))[:3]
        extra = sorted(set(by_id) - public_ids)[:3]
        raise SealedReferenceError(
            f"sealed/public item sets differ; missing={missing}, extra={extra}"
        )
    item_set_sha256 = hashlib.sha256(canonical_json_bytes(sorted(by_id))).hexdigest()
    return SealedAnswerVault(
        benchmark_id=descriptor.benchmark_id,
        benchmark_revision=descriptor.benchmark_revision,
        task=descriptor.task,
        references=by_id,
        raw_sha256=raw_sha256,
        item_set_sha256=item_set_sha256,
    )


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    model_id: str
    model_revision: str
    checkpoint_sha256: str
    schema_version: str = CHECKPOINT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_IDENTITY_SCHEMA_VERSION:
            raise GenerationContractError("unsupported checkpoint identity schema")
        _require_identifier(self.model_id, field="model_id")
        if _GIT_REVISION_RE.fullmatch(self.model_revision) is None:
            raise GenerationContractError("model_revision must be a full immutable revision")
        _require_sha256(self.checkpoint_sha256, field="checkpoint_sha256")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CheckpointIdentity:
        required = {
            "schema_version",
            "model_id",
            "model_revision",
            "checkpoint_sha256",
        }
        _require_exact_keys(raw, required=required, field="checkpoint identity")
        return cls(
            schema_version=raw["schema_version"],
            model_id=raw["model_id"],
            model_revision=raw["model_revision"],
            checkpoint_sha256=raw["checkpoint_sha256"],
        )

    def to_record(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class GenerationProtocol:
    mode: DecodingMode
    samples_per_item: int
    max_new_tokens: int
    system_prompt: str
    chat_template_sha256: str
    eos_token_id: int
    stop_token_ids: tuple[int, ...]
    stop_sequences: tuple[str, ...]
    temperature_ppm: int | None
    top_p_ppm: int | None
    top_k: int | None
    seed_namespace: str | None
    base_seed: int | None
    schema_version: str = GENERATION_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_PROTOCOL_SCHEMA_VERSION:
            raise GenerationContractError("unsupported generation protocol schema")
        _require_int(
            self.samples_per_item,
            field="samples_per_item",
            minimum=1,
            maximum=_MAX_SAMPLES_PER_ITEM,
        )
        _require_int(self.max_new_tokens, field="max_new_tokens", minimum=1, maximum=65_536)
        _require_text(
            self.system_prompt,
            field="system_prompt",
            maximum_chars=65_536,
            allow_empty=True,
        )
        _require_sha256(self.chat_template_sha256, field="chat_template_sha256")
        _require_int(self.eos_token_id, field="eos_token_id", maximum=_MAX_TOKEN_ID)
        if tuple(sorted(set(self.stop_token_ids))) != self.stop_token_ids:
            raise GenerationContractError("stop_token_ids must be unique and sorted")
        if self.eos_token_id not in self.stop_token_ids:
            raise GenerationContractError("stop_token_ids must include eos_token_id")
        for token_id in self.stop_token_ids:
            _require_int(token_id, field="stop_token_id", maximum=_MAX_TOKEN_ID)
        if tuple(sorted(set(self.stop_sequences))) != self.stop_sequences:
            raise GenerationContractError("stop_sequences must be unique and sorted")
        for index, sequence in enumerate(self.stop_sequences):
            _require_text(
                sequence,
                field=f"stop_sequences[{index}]",
                maximum_chars=4_096,
            )
        if self.mode is DecodingMode.GREEDY:
            if self.samples_per_item != 1:
                raise GenerationContractError("greedy mode requires samples_per_item=1")
            if any(
                value is not None
                for value in (
                    self.temperature_ppm,
                    self.top_p_ppm,
                    self.top_k,
                    self.seed_namespace,
                    self.base_seed,
                )
            ):
                raise GenerationContractError("greedy mode forbids sampling parameters and seed")
        else:
            if self.samples_per_item < 2:
                raise GenerationContractError("sampling mode requires at least two samples")
            _require_int(
                self.temperature_ppm,
                field="temperature_ppm",
                minimum=1,
                maximum=2_000_000,
            )
            _require_int(self.top_p_ppm, field="top_p_ppm", minimum=1, maximum=1_000_000)
            _require_int(self.top_k, field="top_k", minimum=0, maximum=1_000_000)
            _require_identifier(self.seed_namespace, field="seed_namespace")
            _require_int(self.base_seed, field="base_seed", minimum=0, maximum=_MAX_SEED)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GenerationProtocol:
        required = {
            "schema_version",
            "mode",
            "samples_per_item",
            "max_new_tokens",
            "system_prompt",
            "chat_template_sha256",
            "eos_token_id",
            "stop_token_ids",
            "stop_sequences",
            "temperature_ppm",
            "top_p_ppm",
            "top_k",
            "seed_namespace",
            "base_seed",
        }
        _require_exact_keys(raw, required=required, field="generation protocol")
        stop_ids = raw["stop_token_ids"]
        stop_sequences = raw["stop_sequences"]
        if not isinstance(stop_ids, list) or not isinstance(stop_sequences, list):
            raise GenerationContractError("stop_token_ids and stop_sequences must be arrays")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in stop_ids):
            raise GenerationContractError("stop_token_ids must contain integers")
        if any(not isinstance(value, str) for value in stop_sequences):
            raise GenerationContractError("stop_sequences must contain strings")
        try:
            mode = DecodingMode(raw["mode"])
        except (TypeError, ValueError) as error:
            raise GenerationContractError("invalid generation mode") from error
        return cls(
            schema_version=raw["schema_version"],
            mode=mode,
            samples_per_item=raw["samples_per_item"],
            max_new_tokens=raw["max_new_tokens"],
            system_prompt=raw["system_prompt"],
            chat_template_sha256=raw["chat_template_sha256"],
            eos_token_id=raw["eos_token_id"],
            stop_token_ids=tuple(stop_ids),
            stop_sequences=tuple(stop_sequences),
            temperature_ppm=raw["temperature_ppm"],
            top_p_ppm=raw["top_p_ppm"],
            top_k=raw["top_k"],
            seed_namespace=raw["seed_namespace"],
            base_seed=raw["base_seed"],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "samples_per_item": self.samples_per_item,
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": self.system_prompt,
            "chat_template_sha256": self.chat_template_sha256,
            "eos_token_id": self.eos_token_id,
            "stop_token_ids": list(self.stop_token_ids),
            "stop_sequences": list(self.stop_sequences),
            "temperature_ppm": self.temperature_ppm,
            "top_p_ppm": self.top_p_ppm,
            "top_k": self.top_k,
            "seed_namespace": self.seed_namespace,
            "base_seed": self.base_seed,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest()

    def seed_for(
        self,
        *,
        benchmark_id: str,
        benchmark_revision: str,
        item_id: str,
        sample_index: int,
    ) -> int | None:
        if self.mode is DecodingMode.GREEDY:
            return None
        _require_int(
            sample_index,
            field="sample_index",
            minimum=0,
            maximum=self.samples_per_item - 1,
        )
        payload = {
            "schema_version": "d07-paired-generation-seed-v1",
            "protocol_sha256": self.digest,
            "benchmark_id": benchmark_id,
            "benchmark_revision": benchmark_revision,
            "item_id": item_id,
            "sample_index": sample_index,
        }
        seed_bytes = hashlib.sha256(canonical_json_bytes(payload)).digest()[:8]
        return int.from_bytes(seed_bytes, "big") & _MAX_SEED


def _load_generation_protocol_with_raw_sha256(
    path: str | Path,
) -> tuple[str, GenerationProtocol]:
    raw, parsed = _read_json(path)
    protocol = GenerationProtocol.from_mapping(_require_mapping(parsed, field="protocol"))
    return hashlib.sha256(raw).hexdigest(), protocol


def load_generation_protocol(path: str | Path) -> GenerationProtocol:
    _, protocol = _load_generation_protocol_with_raw_sha256(path)
    return protocol


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    benchmark_id: str
    benchmark_revision: str
    item_id: str
    item_index: int
    sample_index: int
    prompt: str
    prompt_sha256: str
    checkpoint: CheckpointIdentity
    protocol: GenerationProtocol
    seed: int | None

    def __post_init__(self) -> None:
        _require_sha256(self.request_id, field="request_id")
        _require_identifier(self.benchmark_id, field="benchmark_id")
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", maximum=_MAX_ITEMS - 1)
        _require_int(
            self.sample_index,
            field="sample_index",
            maximum=self.protocol.samples_per_item - 1,
        )
        _require_text(self.prompt, field="prompt", maximum_chars=_MAX_PROMPT_CHARS)
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != self.prompt_sha256:
            raise GenerationContractError("prompt_sha256 does not match prompt")
        expected_seed = self.protocol.seed_for(
            benchmark_id=self.benchmark_id,
            benchmark_revision=self.benchmark_revision,
            item_id=self.item_id,
            sample_index=self.sample_index,
        )
        if self.seed != expected_seed:
            raise GenerationContractError("request seed does not match frozen protocol")

    def to_record(self) -> dict[str, object]:
        """Return the complete generator-facing payload; no reference can appear here."""

        return {
            "request_id": self.request_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "item_id": self.item_id,
            "item_index": self.item_index,
            "sample_index": self.sample_index,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "checkpoint": self.checkpoint.to_record(),
            "protocol": self.protocol.to_record(),
            "seed": self.seed,
        }


def build_generation_request(
    item: PublicBenchmarkItem,
    *,
    checkpoint: CheckpointIdentity,
    protocol: GenerationProtocol,
    sample_index: int,
) -> GenerationRequest:
    seed = protocol.seed_for(
        benchmark_id=item.benchmark_id,
        benchmark_revision=item.benchmark_revision,
        item_id=item.item_id,
        sample_index=sample_index,
    )
    request_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "d07-generation-request-id-v1",
                "benchmark_id": item.benchmark_id,
                "benchmark_revision": item.benchmark_revision,
                "item_id": item.item_id,
                "item_index": item.item_index,
                "sample_index": sample_index,
                "prompt_sha256": item.prompt_sha256,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "protocol_sha256": protocol.digest,
                "seed": seed,
            }
        )
    ).hexdigest()
    return GenerationRequest(
        request_id=request_id,
        benchmark_id=item.benchmark_id,
        benchmark_revision=item.benchmark_revision,
        item_id=item.item_id,
        item_index=item.item_index,
        sample_index=sample_index,
        prompt=item.prompt,
        prompt_sha256=item.prompt_sha256,
        checkpoint=checkpoint,
        protocol=protocol,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    request_id: str
    status: GenerationStatus
    generated_text: str | None
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason | None
    error_code: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.request_id, field="response.request_id")
        if not isinstance(self.output_token_ids, tuple):
            raise GenerationContractError("response.output_token_ids must be an immutable tuple")
        if self.status is GenerationStatus.COMPLETED:
            if self.generated_text is None or self.finish_reason is None:
                raise GenerationContractError(
                    "completed response requires generated_text and finish_reason"
                )
            _require_text(
                self.generated_text,
                field="generated_text",
                maximum_chars=_MAX_GENERATED_CHARS,
                allow_empty=True,
            )
            if self.error_code is not None:
                raise GenerationContractError("completed response cannot contain error_code")
            if not self.output_token_ids:
                raise GenerationContractError(
                    "completed response requires complete non-empty output_token_ids"
                )
            for token_id in self.output_token_ids:
                _require_int(token_id, field="output_token_id", maximum=_MAX_TOKEN_ID)
        else:
            if (
                self.generated_text is not None
                or self.output_token_ids
                or self.finish_reason is not None
            ):
                raise GenerationContractError(
                    "failed response cannot contain generated output or finish_reason"
                )
            if (
                not isinstance(self.error_code, str)
                or _ERROR_CODE_RE.fullmatch(self.error_code) is None
            ):
                raise GenerationContractError("failed response requires a portable error_code")


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    request_id: str
    benchmark_id: str
    benchmark_revision: str
    item_id: str
    item_index: int
    sample_index: int
    checkpoint_sha256: str
    protocol_sha256: str
    prompt_sha256: str
    seed: int | None
    status: GenerationStatus
    generated_text: str | None
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason | None
    error_code: str | None
    record_sha256: str
    schema_version: str = GENERATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RECORD_SCHEMA_VERSION:
            raise GenerationContractError("unsupported generation record schema")
        for field in (
            "request_id",
            "checkpoint_sha256",
            "protocol_sha256",
            "prompt_sha256",
            "record_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _require_identifier(self.benchmark_id, field="benchmark_id")
        _require_identifier(self.item_id, field="item_id")
        _require_int(self.item_index, field="item_index", maximum=_MAX_ITEMS - 1)
        _require_int(self.sample_index, field="sample_index", maximum=_MAX_SAMPLES_PER_ITEM - 1)
        if self.seed is not None:
            _require_int(self.seed, field="seed", maximum=_MAX_SEED)
        if not isinstance(self.output_token_ids, tuple):
            raise GenerationContractError("output_token_ids must be an immutable tuple")
        response = GenerationResponse(
            request_id=self.request_id,
            status=self.status,
            generated_text=self.generated_text,
            output_token_ids=self.output_token_ids,
            finish_reason=self.finish_reason,
            error_code=self.error_code,
        )
        del response
        expected_digest = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if expected_digest != self.record_sha256:
            raise GenerationContractError("generation record_sha256 mismatch")

    @classmethod
    def from_request_response(
        cls,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> GenerationRecord:
        if request.request_id != response.request_id:
            raise GenerationContractError("response request_id does not match request")
        unsigned = {
            "schema_version": GENERATION_RECORD_SCHEMA_VERSION,
            "request_id": request.request_id,
            "benchmark_id": request.benchmark_id,
            "benchmark_revision": request.benchmark_revision,
            "item_id": request.item_id,
            "item_index": request.item_index,
            "sample_index": request.sample_index,
            "checkpoint_sha256": request.checkpoint.checkpoint_sha256,
            "protocol_sha256": request.protocol.digest,
            "prompt_sha256": request.prompt_sha256,
            "seed": request.seed,
            "status": response.status.value,
            "generated_text": response.generated_text,
            "output_token_ids": list(response.output_token_ids),
            "finish_reason": (
                response.finish_reason.value if response.finish_reason is not None else None
            ),
            "error_code": response.error_code,
        }
        record = cls(
            request_id=request.request_id,
            benchmark_id=request.benchmark_id,
            benchmark_revision=request.benchmark_revision,
            item_id=request.item_id,
            item_index=request.item_index,
            sample_index=request.sample_index,
            checkpoint_sha256=request.checkpoint.checkpoint_sha256,
            protocol_sha256=request.protocol.digest,
            prompt_sha256=request.prompt_sha256,
            seed=request.seed,
            status=response.status,
            generated_text=response.generated_text,
            output_token_ids=response.output_token_ids,
            finish_reason=response.finish_reason,
            error_code=response.error_code,
            record_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )
        record.assert_protocol_semantics(request.protocol)
        return record

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, line_number: int) -> GenerationRecord:
        required = {
            "schema_version",
            "request_id",
            "benchmark_id",
            "benchmark_revision",
            "item_id",
            "item_index",
            "sample_index",
            "checkpoint_sha256",
            "protocol_sha256",
            "prompt_sha256",
            "seed",
            "status",
            "generated_text",
            "output_token_ids",
            "finish_reason",
            "error_code",
            "record_sha256",
        }
        _require_exact_keys(raw, required=required, field=f"generation line {line_number}")
        token_ids = raw["output_token_ids"]
        if not isinstance(token_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        ):
            raise GenerationContractError(f"line {line_number}: output_token_ids must be integers")
        try:
            status = GenerationStatus(raw["status"])
            finish_reason = (
                FinishReason(raw["finish_reason"]) if raw["finish_reason"] is not None else None
            )
        except (TypeError, ValueError) as error:
            raise GenerationContractError(
                f"line {line_number}: invalid status/finish_reason"
            ) from error
        return cls(
            schema_version=raw["schema_version"],
            request_id=raw["request_id"],
            benchmark_id=raw["benchmark_id"],
            benchmark_revision=raw["benchmark_revision"],
            item_id=raw["item_id"],
            item_index=raw["item_index"],
            sample_index=raw["sample_index"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            protocol_sha256=raw["protocol_sha256"],
            prompt_sha256=raw["prompt_sha256"],
            seed=raw["seed"],
            status=status,
            generated_text=raw["generated_text"],
            output_token_ids=tuple(token_ids),
            finish_reason=finish_reason,
            error_code=raw["error_code"],
            record_sha256=raw["record_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "item_id": self.item_id,
            "item_index": self.item_index,
            "sample_index": self.sample_index,
            "checkpoint_sha256": self.checkpoint_sha256,
            "protocol_sha256": self.protocol_sha256,
            "prompt_sha256": self.prompt_sha256,
            "seed": self.seed,
            "status": self.status.value,
            "generated_text": self.generated_text,
            "output_token_ids": list(self.output_token_ids),
            "finish_reason": self.finish_reason.value if self.finish_reason is not None else None,
            "error_code": self.error_code,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "record_sha256": self.record_sha256}

    def assert_protocol_semantics(self, protocol: GenerationProtocol) -> None:
        """Revalidate persisted output semantics against its frozen protocol."""

        if self.protocol_sha256 != protocol.digest:
            raise GenerationContractError("generation record protocol_sha256 mismatch")
        expected_seed = protocol.seed_for(
            benchmark_id=self.benchmark_id,
            benchmark_revision=self.benchmark_revision,
            item_id=self.item_id,
            sample_index=self.sample_index,
        )
        if self.seed != expected_seed:
            raise GenerationContractError("generation record seed does not match protocol")
        if self.status is GenerationStatus.FAILED:
            return
        if len(self.output_token_ids) > protocol.max_new_tokens:
            raise GenerationContractError("output token count exceeds max_new_tokens")
        if self.finish_reason is FinishReason.LENGTH and (
            len(self.output_token_ids) != protocol.max_new_tokens
        ):
            raise GenerationContractError(
                "length finish requires exactly max_new_tokens output tokens"
            )
        if self.finish_reason is FinishReason.EOS and (
            not self.output_token_ids or self.output_token_ids[-1] != protocol.eos_token_id
        ):
            raise GenerationContractError("eos finish must end with eos_token_id")
        if self.finish_reason is FinishReason.STOP_TOKEN and (
            not self.output_token_ids or self.output_token_ids[-1] not in protocol.stop_token_ids
        ):
            raise GenerationContractError("stop_token finish must end with an allowed token")
        if self.finish_reason is FinishReason.STOP_SEQUENCE and not any(
            (self.generated_text or "").endswith(sequence) for sequence in protocol.stop_sequences
        ):
            raise GenerationContractError(
                "stop_sequence finish must end with a frozen stop sequence"
            )


@dataclass(frozen=True, slots=True)
class GenerationBatch:
    run_id: str
    descriptor: BenchmarkDescriptor
    public_items_sha256: str
    public_item_set_sha256: str
    checkpoint: CheckpointIdentity
    protocol: GenerationProtocol
    records: tuple[GenerationRecord, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, field="run_id")
        if not isinstance(self.records, tuple):
            raise GenerationContractError("generation batch records must be an immutable tuple")
        if self.public_items_sha256 != self.descriptor.public_items_sha256:
            raise GenerationContractError("batch public_items_sha256 mismatch")
        _require_sha256(self.public_item_set_sha256, field="public_item_set_sha256")
        expected_count = self.descriptor.item_count * self.protocol.samples_per_item
        if len(self.records) != expected_count:
            raise GenerationContractError(
                f"generation batch has {len(self.records)} records; expected {expected_count}"
            )
        expected_order = tuple(
            (item_index, sample_index)
            for item_index in range(self.descriptor.item_count)
            for sample_index in range(self.protocol.samples_per_item)
        )
        actual_order = tuple((record.item_index, record.sample_index) for record in self.records)
        if actual_order != expected_order:
            raise GenerationContractError("generation records are not a complete canonical grid")
        if len({record.request_id for record in self.records}) != len(self.records):
            raise GenerationContractError("generation batch contains duplicate request_id")
        for record in self.records:
            if (
                record.benchmark_id != self.descriptor.benchmark_id
                or record.benchmark_revision != self.descriptor.benchmark_revision
                or record.checkpoint_sha256 != self.checkpoint.checkpoint_sha256
                or record.protocol_sha256 != self.protocol.digest
            ):
                raise GenerationContractError(
                    f"generation record {record.request_id} is incompatible with batch"
                )
            record.assert_protocol_semantics(self.protocol)

    @property
    def record_set_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "request_id": record.request_id,
                        "record_sha256": record.record_sha256,
                    }
                    for record in self.records
                ]
            )
        ).hexdigest()

    @property
    def failed_count(self) -> int:
        return sum(record.status is GenerationStatus.FAILED for record in self.records)


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    run_id: str
    benchmark_descriptor_sha256: str
    public_items_sha256: str
    public_item_set_sha256: str
    checkpoint: CheckpointIdentity
    protocol: GenerationProtocol
    records_file: str
    records_file_sha256: str
    record_count: int
    record_set_sha256: str
    failed_count: int
    manifest_sha256: str
    schema_version: str = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_MANIFEST_SCHEMA_VERSION:
            raise GenerationContractError("unsupported generation manifest schema")
        _require_identifier(self.run_id, field="run_id")
        for field in (
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "public_item_set_sha256",
            "records_file_sha256",
            "record_set_sha256",
            "manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if Path(self.records_file).name != self.records_file or self.records_file in {".", ".."}:
            raise GenerationContractError("records_file must be a sibling basename")
        _require_int(self.record_count, field="record_count", minimum=1)
        _require_int(self.failed_count, field="failed_count", maximum=self.record_count)
        expected = hashlib.sha256(canonical_json_bytes(self.unsigned_record())).hexdigest()
        if expected != self.manifest_sha256:
            raise GenerationContractError("generation manifest_sha256 mismatch")

    @classmethod
    def from_batch(
        cls,
        batch: GenerationBatch,
        *,
        records_file: str,
        records_file_sha256: str,
    ) -> GenerationManifest:
        unsigned = {
            "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
            "run_id": batch.run_id,
            "benchmark_descriptor_sha256": batch.descriptor.digest,
            "public_items_sha256": batch.public_items_sha256,
            "public_item_set_sha256": batch.public_item_set_sha256,
            "checkpoint": batch.checkpoint.to_record(),
            "protocol": batch.protocol.to_record(),
            "protocol_sha256": batch.protocol.digest,
            "records_file": records_file,
            "records_file_sha256": records_file_sha256,
            "record_count": len(batch.records),
            "record_set_sha256": batch.record_set_sha256,
            "failed_count": batch.failed_count,
        }
        return cls(
            run_id=batch.run_id,
            benchmark_descriptor_sha256=batch.descriptor.digest,
            public_items_sha256=batch.public_items_sha256,
            public_item_set_sha256=batch.public_item_set_sha256,
            checkpoint=batch.checkpoint,
            protocol=batch.protocol,
            records_file=records_file,
            records_file_sha256=records_file_sha256,
            record_count=len(batch.records),
            record_set_sha256=batch.record_set_sha256,
            failed_count=batch.failed_count,
            manifest_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GenerationManifest:
        required = {
            "schema_version",
            "run_id",
            "benchmark_descriptor_sha256",
            "public_items_sha256",
            "public_item_set_sha256",
            "checkpoint",
            "protocol",
            "protocol_sha256",
            "records_file",
            "records_file_sha256",
            "record_count",
            "record_set_sha256",
            "failed_count",
            "manifest_sha256",
        }
        _require_exact_keys(raw, required=required, field="generation manifest")
        checkpoint = CheckpointIdentity.from_mapping(
            _require_mapping(raw["checkpoint"], field="checkpoint")
        )
        protocol = GenerationProtocol.from_mapping(
            _require_mapping(raw["protocol"], field="protocol")
        )
        if raw["protocol_sha256"] != protocol.digest:
            raise GenerationContractError("generation manifest protocol_sha256 mismatch")
        return cls(
            schema_version=raw["schema_version"],
            run_id=raw["run_id"],
            benchmark_descriptor_sha256=raw["benchmark_descriptor_sha256"],
            public_items_sha256=raw["public_items_sha256"],
            public_item_set_sha256=raw["public_item_set_sha256"],
            checkpoint=checkpoint,
            protocol=protocol,
            records_file=raw["records_file"],
            records_file_sha256=raw["records_file_sha256"],
            record_count=raw["record_count"],
            record_set_sha256=raw["record_set_sha256"],
            failed_count=raw["failed_count"],
            manifest_sha256=raw["manifest_sha256"],
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "benchmark_descriptor_sha256": self.benchmark_descriptor_sha256,
            "public_items_sha256": self.public_items_sha256,
            "public_item_set_sha256": self.public_item_set_sha256,
            "checkpoint": self.checkpoint.to_record(),
            "protocol": self.protocol.to_record(),
            "protocol_sha256": self.protocol.digest,
            "records_file": self.records_file,
            "records_file_sha256": self.records_file_sha256,
            "record_count": self.record_count,
            "record_set_sha256": self.record_set_sha256,
            "failed_count": self.failed_count,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True, slots=True)
class LoadedGenerationBundle:
    manifest_path: str
    records_path: str
    manifest: GenerationManifest
    batch: GenerationBatch


def write_generation_bundle(
    batch: GenerationBatch,
    *,
    records_path: str | Path,
    manifest_path: str | Path,
) -> GenerationManifest:
    records_destination = Path(records_path)
    manifest_destination = Path(manifest_path)
    resolved_records = records_destination.resolve(strict=False)
    resolved_manifest = manifest_destination.resolve(strict=False)
    if resolved_records.parent != resolved_manifest.parent:
        raise GenerationContractError("generation records and manifest must be siblings")
    if resolved_records == resolved_manifest:
        raise GenerationContractError("generation records and manifest paths must differ")
    raw_sha256 = _write_jsonl_atomic(
        tuple(record.to_record() for record in batch.records), records_destination
    )
    manifest = GenerationManifest.from_batch(
        batch,
        records_file=records_destination.name,
        records_file_sha256=raw_sha256,
    )
    write_json_atomic(manifest.to_record(), manifest_destination)
    return manifest


def load_generation_bundle(
    descriptor: BenchmarkDescriptor,
    public: LoadedPublicBenchmark,
    manifest_path: str | Path,
) -> LoadedGenerationBundle:
    _, parsed = _read_json(manifest_path)
    manifest = GenerationManifest.from_mapping(
        _require_mapping(parsed, field="generation manifest")
    )
    if (
        manifest.benchmark_descriptor_sha256 != descriptor.digest
        or manifest.public_items_sha256 != public.raw_sha256
        or manifest.public_item_set_sha256 != public.item_set_sha256
    ):
        raise GenerationContractError("generation manifest does not match benchmark snapshot")
    records_path = Path(manifest_path).parent / manifest.records_file
    raw_sha256, rows = _iter_jsonl(records_path, maximum_records=manifest.record_count)
    if raw_sha256 != manifest.records_file_sha256:
        raise GenerationContractError("generation records bytes do not match manifest")
    records = tuple(
        GenerationRecord.from_mapping(raw, line_number=line_number) for line_number, raw in rows
    )
    if len(records) != manifest.record_count:
        raise GenerationContractError("generation record_count does not match manifest")
    batch = GenerationBatch(
        run_id=manifest.run_id,
        descriptor=descriptor,
        public_items_sha256=manifest.public_items_sha256,
        public_item_set_sha256=manifest.public_item_set_sha256,
        checkpoint=manifest.checkpoint,
        protocol=manifest.protocol,
        records=records,
    )
    if (
        batch.record_set_sha256 != manifest.record_set_sha256
        or batch.failed_count != manifest.failed_count
    ):
        raise GenerationContractError("generation records semantic summary mismatch")
    for record, request in zip(
        records,
        (
            build_generation_request(
                item,
                checkpoint=manifest.checkpoint,
                protocol=manifest.protocol,
                sample_index=sample_index,
            )
            for item in public.items
            for sample_index in range(manifest.protocol.samples_per_item)
        ),
        strict=True,
    ):
        if (
            record.request_id != request.request_id
            or record.prompt_sha256 != request.prompt_sha256
            or record.seed != request.seed
            or record.item_id != request.item_id
        ):
            raise GenerationContractError(
                f"generation record does not match public request: {record.request_id}"
            )
    return LoadedGenerationBundle(
        manifest_path=Path(manifest_path).as_posix(),
        records_path=records_path.as_posix(),
        manifest=manifest,
        batch=batch,
    )
