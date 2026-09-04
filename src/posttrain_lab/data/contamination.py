"""Deterministic exact/fuzzy contamination detection without raw-text reports."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .registry import (
    EMPTY_PARENT_PAYLOAD_LEDGER,
    TRAINING_SPLITS,
    DataContractError,
    DataManifest,
    DataRecord,
    ParentPayloadLedger,
    SourceRegistry,
    SplitName,
    TransformRegistry,
    _build_data_manifest_from_verified_scan,
    canonical_json_bytes,
    sha256_json,
    write_json_atomic,
)

CONTAMINATION_POLICY_SCHEMA_VERSION = "d06-contamination-policy-v1"
CONTAMINATION_REPORT_SCHEMA_VERSION = "d06-contamination-report-v1"

_SPACE_PATTERN = re.compile(r"\s+")
_NUMBER_COMMA_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_LATEX_SPACING_PATTERN = re.compile(
    r"\\(?:[,;:!]|(?:quad|qquad|enspace|thinspace)\b)"
)
_LEXEME_PATTERN = re.compile(r"\\[A-Za-z]+|[\w]+(?:\.[0-9]+)?|[^\s]", re.UNICODE)
_FIELD_PRIORITY = {
    "problem": 0,
    "prompt": 1,
    "solution": 2,
    "context": 3,
}
_REQUIRED_FIELD_GROUPS = ("context", "problem", "prompt", "solution")


class ContaminationError(DataContractError):
    """Raised when contamination inputs or a cleanliness assertion fail."""


class MatchKind(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ContaminationPolicy:
    """Frozen normalization and decision thresholds.

    Scores are represented as integer parts-per-million to avoid platform
    dependent policy hashes and ambiguous decimal threshold serialization.
    """

    normalization_version: str = "nfkc-lexeme-v1"
    character_ngram_size: int = 5
    token_ngram_size: int = 3
    minimum_exact_chars: int = 16
    minimum_fuzzy_chars: int = 32
    fuzzy_jaccard_ppm: int = 820_000
    fuzzy_containment_ppm: int = 920_000
    review_margin_ppm: int = 50_000
    maximum_text_chars: int = 1_000_000
    included_field_groups: tuple[str, ...] = _REQUIRED_FIELD_GROUPS
    exempt_context_normalized_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.normalization_version != "nfkc-lexeme-v1":
            raise ContaminationError("unsupported normalization_version")
        for field, value, low, high in (
            ("character_ngram_size", self.character_ngram_size, 2, 16),
            ("token_ngram_size", self.token_ngram_size, 1, 8),
            ("minimum_exact_chars", self.minimum_exact_chars, 1, 10_000),
            ("minimum_fuzzy_chars", self.minimum_fuzzy_chars, 1, 100_000),
            ("fuzzy_jaccard_ppm", self.fuzzy_jaccard_ppm, 1, 1_000_000),
            ("fuzzy_containment_ppm", self.fuzzy_containment_ppm, 1, 1_000_000),
            ("review_margin_ppm", self.review_margin_ppm, 0, 999_999),
            ("maximum_text_chars", self.maximum_text_chars, 1, 10_000_000),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ContaminationError(f"{field} must be an integer in [{low}, {high}]")
        if self.minimum_fuzzy_chars < self.minimum_exact_chars:
            raise ContaminationError("minimum_fuzzy_chars cannot be below minimum_exact_chars")
        if self.review_margin_ppm >= min(
            self.fuzzy_jaccard_ppm, self.fuzzy_containment_ppm
        ):
            raise ContaminationError("review margin would make a threshold non-positive")
        if self.included_field_groups != _REQUIRED_FIELD_GROUPS:
            raise ContaminationError(
                f"included_field_groups must be exactly {_REQUIRED_FIELD_GROUPS!r}; "
                "context exclusions require an exact normalized-hash exemption"
            )
        exemptions = self.exempt_context_normalized_sha256
        if tuple(sorted(set(exemptions))) != exemptions:
            raise ContaminationError(
                "exempt_context_normalized_sha256 must be unique and sorted"
            )
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in exemptions):
            raise ContaminationError(
                "exempt_context_normalized_sha256 must contain full lowercase SHA-256 values"
            )

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CONTAMINATION_POLICY_SCHEMA_VERSION,
            "normalization_version": self.normalization_version,
            "character_ngram_size": self.character_ngram_size,
            "token_ngram_size": self.token_ngram_size,
            "minimum_exact_chars": self.minimum_exact_chars,
            "minimum_fuzzy_chars": self.minimum_fuzzy_chars,
            "fuzzy_jaccard_ppm": self.fuzzy_jaccard_ppm,
            "fuzzy_containment_ppm": self.fuzzy_containment_ppm,
            "review_margin_ppm": self.review_margin_ppm,
            "maximum_text_chars": self.maximum_text_chars,
            "included_field_groups": list(self.included_field_groups),
            "exempt_context_normalized_sha256": list(
                self.exempt_context_normalized_sha256
            ),
        }


DEFAULT_CONTAMINATION_POLICY = ContaminationPolicy()


def normalize_for_contamination(
    text: str,
    *,
    policy: ContaminationPolicy = DEFAULT_CONTAMINATION_POLICY,
) -> str:
    """Conservatively normalize text while retaining semantic punctuation."""

    if not isinstance(text, str):
        raise ContaminationError("contamination text must be a string")
    if len(text) > policy.maximum_text_chars:
        raise ContaminationError(
            f"contamination text exceeds {policy.maximum_text_chars} characters"
        )
    if "\x00" in text or any(unicodedata.category(character) == "Cs" for character in text):
        raise ContaminationError("contamination text contains NUL or surrogate code points")
    normalized = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", text).casefold()
    )
    normalized = (
        normalized.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    normalized = normalized.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = _LATEX_SPACING_PATTERN.sub(" ", normalized)
    normalized = _NUMBER_COMMA_PATTERN.sub("", normalized)
    lexemes = _LEXEME_PATTERN.findall(normalized)
    return " ".join(lexemes).strip()


def _field_group(field: str) -> str:
    if field == "problem":
        return "problem"
    if field in {"reference_answer", "response"} or field.endswith(".assistant"):
        return "solution"
    if field.endswith(".user"):
        return "prompt"
    return "context"


def _ngrams(values: Sequence[str], size: int) -> frozenset[str]:
    if not values:
        return frozenset()
    if len(values) < size:
        return frozenset({"\x1f".join(values)})
    return frozenset(
        "\x1f".join(values[index : index + size])
        for index in range(len(values) - size + 1)
    )


@dataclass(frozen=True, slots=True)
class ContentFingerprint:
    record_id: str
    split: str
    source_id: str
    field: str
    field_group: str
    normalized_sha256: str
    normalized_chars: int
    character_ngrams: frozenset[str]
    token_ngrams: frozenset[str]

    @property
    def fragment_id(self) -> str:
        return sha256_json(
            {
                "record_id": self.record_id,
                "split": self.split,
                "source_id": self.source_id,
                "field": self.field,
                "normalized_sha256": self.normalized_sha256,
            }
        )


def _fingerprints(
    records: Sequence[DataRecord],
    policy: ContaminationPolicy,
) -> tuple[ContentFingerprint, ...]:
    fingerprints: list[ContentFingerprint] = []
    for record in sorted(records, key=lambda item: item.sample_id):
        by_normalized_hash: dict[str, ContentFingerprint] = {}
        normalized_fragments: list[tuple[str, str, str]] = []
        for field, text in record.contamination_fields():
            field_group = _field_group(field)
            if field_group not in policy.included_field_groups:
                continue
            normalized = normalize_for_contamination(text, policy=policy)
            if not normalized:
                continue
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if (
                field_group == "context"
                and normalized_hash in policy.exempt_context_normalized_sha256
            ):
                continue
            normalized_fragments.append((field, field_group, normalized))
        aggregate_specs = (
            (
                "aggregate.prompt",
                "prompt",
                tuple(
                    normalized
                    for field, _, normalized in normalized_fragments
                    if field == "problem" or field.startswith("messages.")
                ),
            ),
            (
                "aggregate.solution_trace",
                "solution",
                tuple(
                    normalized
                    for field, _, normalized in normalized_fragments
                    if field in {"reference_answer", "response"}
                    or field.endswith(".assistant")
                    or field.endswith(".tool")
                ),
            ),
            (
                "aggregate.full_record",
                "context",
                tuple(normalized for _, _, normalized in normalized_fragments),
            ),
        )
        for field, field_group, parts in aggregate_specs:
            if len(parts) >= 2:
                normalized_fragments.append((field, field_group, " ".join(parts)))

        for field, field_group, normalized in normalized_fragments:
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            compact_characters = tuple(
                character for character in normalized if not character.isspace()
            )
            tokens = tuple(normalized.split())
            current = ContentFingerprint(
                record_id=record.sample_id,
                split=record.split.value,
                source_id=record.source_id,
                field=field,
                field_group=field_group,
                normalized_sha256=normalized_hash,
                normalized_chars=len(normalized),
                character_ngrams=_ngrams(compact_characters, policy.character_ngram_size),
                token_ngrams=_ngrams(tokens, policy.token_ngram_size),
            )
            previous = by_normalized_hash.get(normalized_hash)
            if previous is None or (
                _FIELD_PRIORITY[current.field_group], current.field
            ) < (_FIELD_PRIORITY[previous.field_group], previous.field):
                by_normalized_hash[normalized_hash] = current
        fingerprints.extend(by_normalized_hash.values())
    return tuple(
        sorted(
            fingerprints,
            key=lambda item: (item.record_id, item.field, item.normalized_sha256),
        )
    )


def _preferred_match(match: ContaminationMatch) -> tuple[object, ...]:
    kind_priority = {
        MatchKind.EXACT: 0,
        MatchKind.FUZZY: 1,
        MatchKind.REVIEW: 2,
    }
    aggregate_penalty = int(
        match.train_field.startswith("aggregate.")
        or match.eval_field.startswith("aggregate.")
    )
    strongest_score = max(
        match.character_jaccard_ppm,
        match.character_containment_ppm,
        match.token_jaccard_ppm,
        match.token_containment_ppm,
    )
    return (
        kind_priority[match.kind],
        aggregate_penalty,
        -strongest_score,
        match.train_field,
        match.eval_field,
        match.match_id,
    )


def _intersection_size(left: frozenset[str], right: frozenset[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(item in right for item in left)


def _scores(left: frozenset[str], right: frozenset[str]) -> tuple[int, int]:
    if not left or not right:
        return 0, 0
    intersection = _intersection_size(left, right)
    union = len(left) + len(right) - intersection
    jaccard = round(intersection * 1_000_000 / union)
    containment = round(intersection * 1_000_000 / min(len(left), len(right)))
    return jaccard, containment


@dataclass(frozen=True, slots=True)
class ContaminationMatch:
    match_id: str
    kind: MatchKind
    train_record_id: str
    train_source_id: str
    train_field: str
    train_normalized_sha256: str
    eval_record_id: str
    eval_source_id: str
    eval_field: str
    eval_normalized_sha256: str
    character_jaccard_ppm: int
    character_containment_ppm: int
    token_jaccard_ppm: int
    token_containment_ppm: int

    def to_record(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "kind": self.kind.value,
            "train_record_id": self.train_record_id,
            "train_source_id": self.train_source_id,
            "train_field": self.train_field,
            "train_normalized_sha256": self.train_normalized_sha256,
            "eval_record_id": self.eval_record_id,
            "eval_source_id": self.eval_source_id,
            "eval_field": self.eval_field,
            "eval_normalized_sha256": self.eval_normalized_sha256,
            "character_jaccard_ppm": self.character_jaccard_ppm,
            "character_containment_ppm": self.character_containment_ppm,
            "token_jaccard_ppm": self.token_jaccard_ppm,
            "token_containment_ppm": self.token_containment_ppm,
        }


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    policy_sha256: str
    training_fingerprint_sha256: str
    evaluation_fingerprint_sha256: str
    training_record_count: int
    evaluation_record_count: int
    training_fragment_count: int
    evaluation_fragment_count: int
    candidate_pair_count: int
    match_counts: dict[str, int]
    matches: tuple[ContaminationMatch, ...]
    passed: bool

    @property
    def body_record(self) -> dict[str, object]:
        return {
            "schema_version": CONTAMINATION_REPORT_SCHEMA_VERSION,
            "policy_sha256": self.policy_sha256,
            "training_fingerprint_sha256": self.training_fingerprint_sha256,
            "evaluation_fingerprint_sha256": self.evaluation_fingerprint_sha256,
            "training_record_count": self.training_record_count,
            "evaluation_record_count": self.evaluation_record_count,
            "training_fragment_count": self.training_fragment_count,
            "evaluation_fragment_count": self.evaluation_fragment_count,
            "candidate_pair_count": self.candidate_pair_count,
            "match_counts": dict(sorted(self.match_counts.items())),
            "matches": [match.to_record() for match in self.matches],
            "passed": self.passed,
        }

    @property
    def report_sha256(self) -> str:
        return sha256_json(self.body_record)

    def to_record(self) -> dict[str, object]:
        return {**self.body_record, "report_sha256": self.report_sha256}

    def assert_clean(self) -> None:
        if not self.passed:
            raise ContaminationError(
                f"contamination gate failed with {len(self.matches)} exact/fuzzy/review matches"
            )


def _fingerprint_set_sha256(fingerprints: Sequence[ContentFingerprint]) -> str:
    return sha256_json(
        [
            [
                fingerprint.fragment_id,
                fingerprint.record_id,
                fingerprint.field,
                fingerprint.normalized_sha256,
            ]
            for fingerprint in fingerprints
        ]
    )


def _classify_pair(
    train: ContentFingerprint,
    evaluation: ContentFingerprint,
    policy: ContaminationPolicy,
) -> tuple[MatchKind | None, tuple[int, int, int, int]]:
    same = train.normalized_sha256 == evaluation.normalized_sha256
    if (
        same
        and min(train.normalized_chars, evaluation.normalized_chars)
        >= policy.minimum_exact_chars
    ):
        return MatchKind.EXACT, (1_000_000, 1_000_000, 1_000_000, 1_000_000)
    if min(train.normalized_chars, evaluation.normalized_chars) < policy.minimum_fuzzy_chars:
        return None, (0, 0, 0, 0)
    char_jaccard, char_containment = _scores(
        train.character_ngrams, evaluation.character_ngrams
    )
    token_jaccard, token_containment = _scores(train.token_ngrams, evaluation.token_ngrams)
    scores = (char_jaccard, char_containment, token_jaccard, token_containment)
    if (
        max(char_jaccard, token_jaccard) >= policy.fuzzy_jaccard_ppm
        or max(char_containment, token_containment) >= policy.fuzzy_containment_ppm
    ):
        return MatchKind.FUZZY, scores
    if (
        max(char_jaccard, token_jaccard)
        >= policy.fuzzy_jaccard_ppm - policy.review_margin_ppm
        or max(char_containment, token_containment)
        >= policy.fuzzy_containment_ppm - policy.review_margin_ppm
    ):
        return MatchKind.REVIEW, scores
    return None, scores


def scan_contamination(
    training_records: Sequence[DataRecord],
    evaluation_records: Sequence[DataRecord],
    *,
    policy: ContaminationPolicy = DEFAULT_CONTAMINATION_POLICY,
) -> ContaminationReport:
    """Find every candidate sharing at least one configured n-gram.

    The inverted indices are a complete retrieval route for all non-zero
    n-gram intersections; unlike MinHash/LSH they do not probabilistically
    omit a pair above the frozen threshold.
    """

    training_records = tuple(training_records)
    evaluation_records = tuple(evaluation_records)
    if not training_records or not evaluation_records:
        raise ContaminationError("training and evaluation record sets must both be non-empty")
    if any(record.split not in {
        SplitName.D_ANCHOR,
        SplitName.D_CORE,
        SplitName.D_SELECT,
        SplitName.D_TEACHER_GATE,
        SplitName.D_DEV,
    } for record in training_records):
        raise ContaminationError("training_records contains non-training or unassigned splits")
    if any(record.split is not SplitName.EVALUATION for record in evaluation_records):
        raise ContaminationError("evaluation_records must all belong to split E")
    train_ids = {record.sample_id for record in training_records}
    eval_ids = {record.sample_id for record in evaluation_records}
    if len(train_ids) != len(training_records) or len(eval_ids) != len(evaluation_records):
        raise ContaminationError("contamination inputs contain duplicate record IDs")
    overlap = train_ids & eval_ids
    if overlap:
        raise ContaminationError(f"training/evaluation sample IDs overlap: {sorted(overlap)[:3]}")

    training = _fingerprints(training_records, policy)
    evaluation = _fingerprints(evaluation_records, policy)
    exact_index: dict[str, set[int]] = defaultdict(set)
    character_index: dict[str, set[int]] = defaultdict(set)
    token_index: dict[str, set[int]] = defaultdict(set)
    for index, fingerprint in enumerate(training):
        exact_index[fingerprint.normalized_sha256].add(index)
        if fingerprint.normalized_chars >= policy.minimum_fuzzy_chars:
            for ngram in fingerprint.character_ngrams:
                character_index[ngram].add(index)
            for ngram in fingerprint.token_ngrams:
                token_index[ngram].add(index)

    matches: list[ContaminationMatch] = []
    candidate_pair_count = 0
    for eval_fingerprint in evaluation:
        candidates = set(exact_index.get(eval_fingerprint.normalized_sha256, ()))
        if eval_fingerprint.normalized_chars >= policy.minimum_fuzzy_chars:
            for ngram in eval_fingerprint.character_ngrams:
                candidates.update(character_index.get(ngram, ()))
            for ngram in eval_fingerprint.token_ngrams:
                candidates.update(token_index.get(ngram, ()))
        for train_index in sorted(candidates):
            candidate_pair_count += 1
            train_fingerprint = training[train_index]
            kind, scores = _classify_pair(train_fingerprint, eval_fingerprint, policy)
            if kind is None:
                continue
            char_jaccard, char_containment, token_jaccard, token_containment = scores
            match_body = {
                "train_fragment_id": train_fingerprint.fragment_id,
                "eval_fragment_id": eval_fingerprint.fragment_id,
            }
            matches.append(
                ContaminationMatch(
                    match_id=hashlib.sha256(canonical_json_bytes(match_body)).hexdigest(),
                    kind=kind,
                    train_record_id=train_fingerprint.record_id,
                    train_source_id=train_fingerprint.source_id,
                    train_field=train_fingerprint.field,
                    train_normalized_sha256=train_fingerprint.normalized_sha256,
                    eval_record_id=eval_fingerprint.record_id,
                    eval_source_id=eval_fingerprint.source_id,
                    eval_field=eval_fingerprint.field,
                    eval_normalized_sha256=eval_fingerprint.normalized_sha256,
                    character_jaccard_ppm=char_jaccard,
                    character_containment_ppm=char_containment,
                    token_jaccard_ppm=token_jaccard,
                    token_containment_ppm=token_containment,
                )
            )
    best_by_record_pair: dict[tuple[str, str], ContaminationMatch] = {}
    for match in matches:
        pair = (match.train_record_id, match.eval_record_id)
        previous = best_by_record_pair.get(pair)
        if previous is None or _preferred_match(match) < _preferred_match(previous):
            best_by_record_pair[pair] = match
    ordered_matches = tuple(best_by_record_pair[pair] for pair in sorted(best_by_record_pair))
    counts = Counter(match.kind.value for match in ordered_matches)
    for kind in MatchKind:
        counts.setdefault(kind.value, 0)
    return ContaminationReport(
        policy_sha256=policy.sha256,
        training_fingerprint_sha256=_fingerprint_set_sha256(training),
        evaluation_fingerprint_sha256=_fingerprint_set_sha256(evaluation),
        training_record_count=len(training_records),
        evaluation_record_count=len(evaluation_records),
        training_fragment_count=len(training),
        evaluation_fragment_count=len(evaluation),
        candidate_pair_count=candidate_pair_count,
        match_counts=dict(sorted(counts.items())),
        matches=ordered_matches,
        passed=not ordered_matches,
    )


def build_data_manifest(
    records: Sequence[DataRecord],
    registry: SourceRegistry,
    *,
    transform_registry: TransformRegistry,
    split_policy_sha256: str,
    policy: ContaminationPolicy = DEFAULT_CONTAMINATION_POLICY,
    parent_ledger: ParentPayloadLedger = EMPTY_PARENT_PAYLOAD_LEDGER,
) -> DataManifest:
    """Freeze a manifest only after recomputing a clean scan over its exact records.

    A caller-supplied boolean or report hash is not a cleanliness attestation.  The
    builder derives the train/evaluation partitions from ``records``, reruns the
    frozen policy, and binds the resulting policy/report hashes into the manifest.
    """

    records = tuple(records)
    training_records = tuple(record for record in records if record.split in TRAINING_SPLITS)
    evaluation_records = tuple(
        record for record in records if record.split is SplitName.EVALUATION
    )
    unsupported = tuple(
        record.sample_id
        for record in records
        if record.split not in {*TRAINING_SPLITS, SplitName.EVALUATION}
    )
    if unsupported:
        raise ContaminationError(
            f"manifest records contain unsupported splits: {list(unsupported[:3])}"
        )
    report = scan_contamination(training_records, evaluation_records, policy=policy)
    report.assert_clean()
    return _build_data_manifest_from_verified_scan(
        records,
        registry,
        transform_registry=transform_registry,
        split_policy_sha256=split_policy_sha256,
        parent_ledger=parent_ledger,
        contamination_policy_sha256=policy.sha256,
        contamination_report_sha256=report.report_sha256,
    )


def quarantined_family_ids(
    training_records: Sequence[DataRecord],
    matches: Iterable[ContaminationMatch],
) -> tuple[str, ...]:
    """Expand matched records through all transitive family connections."""

    training_records = tuple(training_records)
    matches = tuple(matches)
    by_id = {record.sample_id: record for record in training_records}
    if len(by_id) != len(training_records):
        raise ContaminationError("training records contain duplicate IDs")
    family_members: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in training_records:
        for dimension, family_id in record.families.items():
            family_members[(dimension, family_id)].add(record.sample_id)
    seeds = {match.train_record_id for match in matches}
    unknown = seeds - set(by_id)
    if unknown:
        raise ContaminationError(f"matches reference unknown training records: {sorted(unknown)}")
    quarantined = set(seeds)
    queue = deque(sorted(seeds))
    while queue:
        sample_id = queue.popleft()
        record = by_id[sample_id]
        for dimension, family_id in record.families.items():
            for relative in sorted(family_members[(dimension, family_id)]):
                if relative not in quarantined:
                    quarantined.add(relative)
                    queue.append(relative)
    return tuple(sorted(quarantined))


def quarantine_contaminated_families(
    training_records: Sequence[DataRecord],
    report: ContaminationReport,
) -> tuple[tuple[DataRecord, ...], tuple[DataRecord, ...]]:
    training_records = tuple(training_records)
    quarantine_ids = set(quarantined_family_ids(training_records, report.matches))
    kept = tuple(
        sorted(
            (record for record in training_records if record.sample_id not in quarantine_ids),
            key=lambda item: item.sample_id,
        )
    )
    quarantined = tuple(
        sorted(
            (record for record in training_records if record.sample_id in quarantine_ids),
            key=lambda item: item.sample_id,
        )
    )
    if not kept:
        raise ContaminationError("quarantine removed the entire training registry")
    return kept, quarantined


def write_contamination_report(
    report: ContaminationReport,
    output_path: str | Path,
) -> None:
    write_json_atomic(report.to_record(), output_path)
