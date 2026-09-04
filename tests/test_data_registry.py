from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.data.contamination import ContaminationError, build_data_manifest
from posttrain_lab.data.registry import (
    DATA_MANIFEST_SCHEMA_VERSION,
    DATA_RECORD_SCHEMA_VERSION,
    SOURCE_REGISTRY_SCHEMA_VERSION,
    DataContractError,
    DataManifest,
    DataRecord,
    DataUse,
    FamilyLeakageError,
    FamilySplitPolicy,
    LineageError,
    LineageParent,
    ManifestIntegrityError,
    Message,
    ParentPayloadEntry,
    ParentPayloadLedger,
    RecordSchemaError,
    RevisionKind,
    SourceDescriptor,
    SourceRegistry,
    SourceRegistryError,
    SplitAllocation,
    SplitName,
    TransformArtifact,
    TransformLineage,
    TransformRegistry,
    assert_family_disjoint,
    assign_family_disjoint_splits,
    canonical_json_bytes,
    find_family_leaks,
    load_data_manifest,
    load_data_records,
    load_family_split_policy,
    load_source_registry,
    max_absolute_split_error,
    sha256_json,
    split_distribution_error,
    strict_json_loads,
    validate_record_set,
    write_data_manifest,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def make_source(
    source_id: str = "public/train",
    *,
    revision: str = "1" * 40,
    revision_kind: RevisionKind = RevisionKind.GIT_COMMIT,
    license_expression: str = "Apache-2.0",
    uses: tuple[DataUse, ...] = (DataUse.REDISTRIBUTE, DataUse.TRAIN),
) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=source_id,
        uri=f"https://example.invalid/{source_id}",
        revision_kind=revision_kind,
        revision=revision,
        license_expression=license_expression,
        license_url="https://example.invalid/license",
        license_evidence_sha256=digest(f"license:{source_id}"),
        allowed_uses=uses,
    )


def make_registry(*sources: SourceDescriptor) -> SourceRegistry:
    if not sources:
        sources = (make_source(),)
    return SourceRegistry(
        allowed_license_expressions=("Apache-2.0", "MIT"),
        sources=tuple(sorted(sources, key=lambda source: source.source_id)),
    )


def make_lineage(
    *,
    transform_name: str = "ingest",
    parents: tuple[LineageParent, ...] = (),
) -> TransformLineage:
    return TransformLineage(
        transform_name=transform_name,
        transform_version="1.0.0",
        code_sha256=digest("transform-code"),
        config_sha256=digest("transform-config"),
        parents=parents,
    )


def make_transform_registry() -> TransformRegistry:
    return TransformRegistry(
        artifacts=tuple(
            TransformArtifact(
                transform_name=transform_name,
                transform_version="1.0.0",
                code_path=f"tests/fixtures/{transform_name}.py",
                code_sha256=digest("transform-code"),
                config_path=f"tests/fixtures/{transform_name}.json",
                config_sha256=digest("transform-config"),
            )
            for transform_name in ("filter", "ingest", "merge", "normalize")
        )
    )


def make_record(
    sample_id: str,
    *,
    source_id: str = "public/train",
    source_revision: str = "1" * 40,
    split: SplitName = SplitName.D_CORE,
    source_family: str | None = None,
    problem_family: str | None = None,
    template_family: str | None = None,
    problem: str | None = None,
    response: str | None = None,
    lineage: TransformLineage | None = None,
) -> DataRecord:
    problem_text = problem or f"Compute the exact value for synthetic item {sample_id}."
    return DataRecord(
        sample_id=sample_id,
        source_id=source_id,
        source_revision=source_revision,
        split=split,
        source_family=source_family or f"sf:{sample_id}",
        problem_family=problem_family or f"pf:{sample_id}",
        template_family=template_family or f"tf:{sample_id}",
        problem=problem_text,
        messages=(Message("user", problem_text),),
        reference_answer=f"answer-{sample_id}",
        response=response or f"Verified derivation for {sample_id}.",
        quality=(("answer_verified", True), ("format_valid", True)),
        strata=(("answer_type", "integer"), ("difficulty", "easy")),
        lineage=lineage or make_lineage(),
    )


def make_policy() -> FamilySplitPolicy:
    return FamilySplitPolicy(
        namespace="d06-test-split-v1",
        allocations=tuple(
            sorted(
                (
                    SplitAllocation(SplitName.D_ANCHOR, 10),
                    SplitAllocation(SplitName.D_CORE, 2),
                    SplitAllocation(SplitName.D_DEV, 1),
                    SplitAllocation(SplitName.D_SELECT, 1),
                    SplitAllocation(SplitName.D_TEACHER_GATE, 1),
                ),
                key=lambda item: str(item.split),
            )
        ),
    )


def manifest_for(records: tuple[DataRecord, ...]) -> DataManifest:
    evaluation = make_record(
        "eval:manifest",
        source_id="public/eval",
        source_revision="2" * 40,
        split=SplitName.EVALUATION,
        problem=(
            "Classify a synthetic graph invariant that is deliberately unrelated "
            "to every training fixture in this manifest."
        ),
        response="A distinct held-out derivation.",
    )
    return build_data_manifest(
        (*records, evaluation),
        make_registry(
            make_source(),
            make_source(
                "public/eval",
                revision="2" * 40,
                uses=(DataUse.EVALUATE,),
            ),
        ),
        transform_registry=make_transform_registry(),
        split_policy_sha256=digest("split-policy"),
    )


def test_source_registry_is_content_addressed_and_order_canonical() -> None:
    first = make_source("public/a")
    second = make_source("public/b", license_expression="MIT")
    registry = make_registry(second, first)
    assert [source.source_id for source in registry.sources] == ["public/a", "public/b"]
    assert registry.sha256 == sha256_json(registry.to_record())
    assert registry.source("public/b") == second


@pytest.mark.parametrize("revision", ["main", "deadbeef", "A" * 40, "1" * 39])
def test_source_rejects_mutable_or_noncanonical_git_revision(revision: str) -> None:
    with pytest.raises(SourceRegistryError, match="full 40/64-hex"):
        make_source(revision=revision)


def test_sha256_source_revision_is_supported() -> None:
    source = make_source(revision=digest("snapshot"), revision_kind=RevisionKind.SHA256)
    assert source.revision == digest("snapshot")


@pytest.mark.parametrize(
    ("uri", "license_url"),
    [
        ("file:///tmp/data", "https://example.invalid/license"),
        ("https://example.invalid/data", "http://example.invalid/license"),
        ("relative/path", "https://example.invalid/license"),
    ],
)
def test_source_requires_public_uri_and_https_license_evidence(
    uri: str, license_url: str
) -> None:
    source = make_source()
    with pytest.raises(SourceRegistryError):
        replace(source, uri=uri, license_url=license_url)


def test_source_registry_rejects_unapproved_license() -> None:
    source = make_source(license_expression="CC-BY-NC-4.0")
    with pytest.raises(SourceRegistryError, match="outside the frozen allowlist"):
        make_registry(source)


def test_source_usage_fails_closed() -> None:
    evaluation = make_source(
        "public/eval",
        uses=(DataUse.EVALUATE,),
    )
    registry = make_registry(evaluation)
    assert registry.assert_usage("public/eval", SplitName.EVALUATION) is evaluation
    with pytest.raises(SourceRegistryError, match="does not permit train"):
        registry.assert_usage("public/eval", SplitName.D_CORE)
    with pytest.raises(SourceRegistryError, match="unknown source_id"):
        registry.assert_usage("public/missing", SplitName.D_CORE)


def test_source_registry_loader_is_strict_and_tracks_raw_hash(tmp_path: Path) -> None:
    registry = make_registry()
    path = tmp_path / "sources.json"
    payload = json.dumps(registry.to_record(), sort_keys=True).encode()
    path.write_bytes(payload)
    loaded = load_source_registry(path)
    assert loaded.registry == registry
    assert loaded.raw_sha256 == hashlib.sha256(payload).hexdigest()

    path.write_bytes(b"\xef\xbb\xbf" + payload)
    with pytest.raises(DataContractError, match="BOM"):
        load_source_registry(path)


def test_source_registry_mapping_rejects_unknown_fields() -> None:
    raw = make_registry().to_record()
    raw["unexpected"] = True
    with pytest.raises(DataContractError, match="invalid keys"):
        SourceRegistry.from_mapping(raw)


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": 1, "value": 2}',
        '{"value": NaN}',
        '{"value": Infinity}',
    ],
)
def test_strict_json_parser_rejects_duplicate_keys_and_nonfinite_values(
    payload: str,
) -> None:
    with pytest.raises(DataContractError):
        strict_json_loads(payload)


def test_data_record_round_trip_and_hash_domains() -> None:
    record = make_record("train:001")
    restored = DataRecord.from_mapping(record.to_record())
    assert restored == record
    assert restored.payload_sha256 == record.payload_sha256
    assert restored.content_sha256 == record.content_sha256
    assert restored.lineage_sha256 == record.lineage_sha256
    changed_split = replace(record, split=SplitName.D_DEV)
    assert changed_split.content_sha256 == record.content_sha256
    assert changed_split.payload_sha256 != record.payload_sha256


def test_record_schema_rejects_missing_extra_and_invalid_schema() -> None:
    raw = make_record("train:002").to_record()
    raw.pop("problem")
    with pytest.raises(DataContractError, match="missing"):
        DataRecord.from_mapping(raw)

    raw = make_record("train:002").to_record()
    raw["extra"] = 1
    with pytest.raises(DataContractError, match="extra"):
        DataRecord.from_mapping(raw)

    raw = make_record("train:002").to_record()
    raw["schema_version"] = "future"
    with pytest.raises(RecordSchemaError, match="unsupported"):
        DataRecord.from_mapping(raw)


@pytest.mark.parametrize("bad_text", ["bad\x00text", "bad\r\ntext", "e\u0301"])
def test_record_rejects_noncanonical_text(bad_text: str) -> None:
    assert (
        bad_text != unicodedata.normalize("NFC", bad_text)
        or "\x00" in bad_text
        or "\r" in bad_text
    )
    with pytest.raises(DataContractError):
        make_record("train:003", problem=bad_text)


def test_record_requires_user_message() -> None:
    record = make_record("train:004")
    with pytest.raises(RecordSchemaError, match="user message"):
        replace(record, messages=(Message("system", "Policy text."),))


def test_jsonl_loader_rejects_blank_duplicate_and_bom(tmp_path: Path) -> None:
    record = make_record("train:005")
    line = json.dumps(record.to_record(), sort_keys=True)
    path = tmp_path / "records.jsonl"

    path.write_text(line + "\n\n", encoding="utf-8")
    with pytest.raises(RecordSchemaError, match="blank"):
        load_data_records(path)

    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(RecordSchemaError, match="duplicate"):
        load_data_records(path)

    path.write_bytes(b"\xef\xbb\xbf" + line.encode())
    with pytest.raises(RecordSchemaError, match="BOM"):
        load_data_records(path)

    path.write_bytes((line + "\r\n").encode())
    with pytest.raises(RecordSchemaError, match="LF newlines"):
        load_data_records(path)

    path.write_text(line.replace('"problem":', '"problem":"shadow","problem":'))
    with pytest.raises(RecordSchemaError, match="duplicate JSON object key"):
        load_data_records(path)


def test_jsonl_loader_has_explicit_record_bound(tmp_path: Path) -> None:
    records = [make_record(f"train:{index:03d}") for index in range(2)]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(record.to_record()) for record in records) + "\n")
    with pytest.raises(RecordSchemaError, match="exceeds 1"):
        load_data_records(path, maximum_records=1)


def test_record_set_checks_revision_license_and_duplicates() -> None:
    registry = make_registry()
    valid = make_record("train:006")
    summary = validate_record_set(
        (valid,), registry, transform_registry=make_transform_registry()
    )
    assert summary.record_count == 1
    assert summary.split_counts == {"D_core": 1}

    with pytest.raises(SourceRegistryError, match="revision"):
        validate_record_set(
            (replace(valid, source_revision="2" * 40),),
            registry,
            transform_registry=make_transform_registry(),
        )

    duplicate = replace(valid, sample_id="train:007")
    with pytest.raises(RecordSchemaError, match="exact duplicate content"):
        validate_record_set(
            (valid, duplicate), registry, transform_registry=make_transform_registry()
        )


@pytest.mark.parametrize("dimension", ["source_family", "problem_family", "template_family"])
def test_family_leakage_is_checked_per_dimension(dimension: str) -> None:
    first = make_record("train:008", split=SplitName.D_ANCHOR)
    second = make_record("train:009", split=SplitName.D_DEV)
    second = replace(second, **{dimension: getattr(first, dimension)})
    leaks = find_family_leaks((first, second))
    assert len(leaks) == 1
    assert leaks[0].sample_ids == ("train:008", "train:009")
    with pytest.raises(FamilyLeakageError, match="family crosses splits"):
        assert_family_disjoint((first, second))


def test_external_lineage_parent_requires_payload_ledger_resolution() -> None:
    lineage = make_lineage(
        transform_name="normalize",
        parents=(LineageParent("raw:external", digest("raw-external")),),
    )
    record = make_record("train:010", lineage=lineage)
    with pytest.raises(LineageError, match="unresolved external"):
        validate_record_set(
            (record,),
            make_registry(),
            transform_registry=make_transform_registry(),
        )
    ledger = ParentPayloadLedger(
        entries=(
            ParentPayloadEntry(
                sample_id="raw:external",
                split=SplitName.D_CORE,
                payload_sha256=digest("raw-external"),
            ),
        )
    )
    validate_record_set(
        (record,),
        make_registry(),
        transform_registry=make_transform_registry(),
        parent_ledger=ledger,
    )


def test_internal_lineage_parent_hash_and_split_are_enforced() -> None:
    parent = make_record("train:011")
    child = make_record(
        "train:012",
        lineage=make_lineage(
            transform_name="filter",
            parents=(LineageParent(parent.sample_id, parent.payload_sha256),),
        ),
    )
    validate_record_set(
        (parent, child),
        make_registry(),
        transform_registry=make_transform_registry(),
    )

    bad_hash = replace(
        child,
        lineage=make_lineage(
            transform_name="filter",
            parents=(LineageParent(parent.sample_id, digest("wrong")),),
        ),
    )
    with pytest.raises(LineageError, match="digest mismatch"):
        validate_record_set(
            (parent, bad_hash),
            make_registry(),
            transform_registry=make_transform_registry(),
        )

    cross_split = replace(child, split=SplitName.D_DEV)
    with pytest.raises(LineageError, match="crosses split"):
        validate_record_set(
            (parent, cross_split),
            make_registry(),
            transform_registry=make_transform_registry(),
            require_family_disjoint=False,
        )

    metadata_changed_parent = replace(parent, problem_family="pf:changed-after-link")
    assert metadata_changed_parent.content_sha256 == parent.content_sha256
    assert metadata_changed_parent.payload_sha256 != parent.payload_sha256
    with pytest.raises(LineageError, match="digest mismatch"):
        validate_record_set(
            (metadata_changed_parent, child),
            make_registry(),
            transform_registry=make_transform_registry(),
        )


def test_record_set_rejects_unregistered_transform_artifact() -> None:
    unknown = replace(
        make_record("train:transform-mismatch"),
        lineage=TransformLineage(
            transform_name="ingest",
            transform_version="1.0.0",
            code_sha256=digest("unknown-code"),
            config_sha256=digest("transform-config"),
            parents=(),
        ),
    )
    with pytest.raises(LineageError, match="transform registry"):
        validate_record_set(
            (unknown,),
            make_registry(),
            transform_registry=make_transform_registry(),
        )


def test_lineage_cycle_is_detected() -> None:
    left_base = make_record("train:013")
    right_base = make_record("train:014")
    left = replace(
        left_base,
        lineage=make_lineage(
            transform_name="merge",
            parents=(LineageParent(right_base.sample_id, right_base.payload_sha256),),
        ),
    )
    right = replace(
        right_base,
        lineage=make_lineage(
            transform_name="merge",
            parents=(LineageParent(left_base.sample_id, left_base.payload_sha256),),
        ),
    )
    with pytest.raises(LineageError, match="cycle"):
        validate_record_set(
            (left, right),
            make_registry(),
            transform_registry=make_transform_registry(),
        )


def test_family_split_is_order_invariant_and_transitive() -> None:
    records = (
        make_record(
            "pool:001",
            split=SplitName.UNASSIGNED,
            source_family="sf:shared",
        ),
        make_record(
            "pool:002",
            split=SplitName.UNASSIGNED,
            source_family="sf:shared",
            problem_family="pf:bridge",
        ),
        make_record(
            "pool:003",
            split=SplitName.UNASSIGNED,
            problem_family="pf:bridge",
        ),
        make_record("pool:004", split=SplitName.UNASSIGNED),
    )
    assigned, report = assign_family_disjoint_splits(records, make_policy())
    reversed_assigned, reversed_report = assign_family_disjoint_splits(
        tuple(reversed(records)), make_policy()
    )
    assert assigned == reversed_assigned
    assert report == reversed_report
    assert report.component_count == 2
    assert report.assignments["pool:001"] == report.assignments["pool:003"]
    assert_family_disjoint(assigned)


def test_split_policy_change_changes_policy_hash() -> None:
    policy = make_policy()
    assert replace(policy, namespace="d06-test-split-v2").sha256 != policy.sha256
    assert policy.sha256 == sha256_json(policy.to_record())


def test_split_policy_loader_and_distribution_diagnostic(tmp_path: Path) -> None:
    policy = make_policy()
    path = tmp_path / "split-policy.json"
    path.write_text(json.dumps(policy.to_record()), encoding="utf-8")
    loaded = load_family_split_policy(path)
    assert loaded.policy == policy
    assert loaded.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    records = tuple(
        make_record(f"pool:{index:03d}", split=SplitName.UNASSIGNED)
        for index in range(100)
    )
    _, assignment = assign_family_disjoint_splits(records, policy)
    errors = split_distribution_error(assignment, policy)
    assert set(errors) == {allocation.split.value for allocation in policy.allocations}
    assert sum(errors.values()) == pytest.approx(0.0)
    assert max_absolute_split_error(assignment, policy) == max(abs(x) for x in errors.values())


def test_auto_split_refuses_preassigned_or_duplicate_records() -> None:
    record = make_record("pool:005")
    with pytest.raises(DataContractError, match="UNASSIGNED"):
        assign_family_disjoint_splits((record,), make_policy())
    unassigned = replace(record, split=SplitName.UNASSIGNED)
    with pytest.raises(DataContractError, match="duplicate"):
        assign_family_disjoint_splits((unassigned, unassigned), make_policy())
    derived = replace(
        unassigned,
        lineage=make_lineage(
            transform_name="normalize",
            parents=(LineageParent("raw:parent", digest("raw-parent")),),
        ),
    )
    with pytest.raises(DataContractError, match="root records"):
        assign_family_disjoint_splits((derived,), make_policy())


def test_manifest_is_deterministic_raw_text_free_and_round_trips(tmp_path: Path) -> None:
    records = (
        make_record("train:015", split=SplitName.D_CORE),
        make_record("train:016", split=SplitName.D_CORE),
    )
    first = manifest_for(records)
    second = manifest_for(tuple(reversed(records)))
    assert first == second
    encoded = canonical_json_bytes(first.to_record())
    assert b"Compute the exact value" not in encoded
    assert b"Verified derivation" not in encoded

    output = tmp_path / "manifest.json"
    write_data_manifest(first, output)
    assert load_data_manifest(output) == first


def test_manifest_recomputes_contamination_and_rejects_dirty_records() -> None:
    leaked_problem = "A benchmark sentence copied exactly into the training registry."
    training = make_record("train:017", problem=leaked_problem)
    evaluation = make_record(
        "eval:dirty",
        source_id="public/eval",
        source_revision="2" * 40,
        split=SplitName.EVALUATION,
        problem=leaked_problem,
    )
    with pytest.raises(ContaminationError, match="contamination gate failed"):
        build_data_manifest(
            (training, evaluation),
            make_registry(
                make_source(),
                make_source(
                    "public/eval",
                    revision="2" * 40,
                    uses=(DataUse.EVALUATE,),
                ),
            ),
            transform_registry=make_transform_registry(),
            split_policy_sha256=digest("split-policy"),
        )


def test_manifest_snapshots_adversarial_sequence_before_any_validation() -> None:
    clean_records = (
        make_record("train:snapshot-clean"),
        make_record(
            "eval:snapshot-clean",
            source_id="public/eval",
            source_revision="2" * 40,
            split=SplitName.EVALUATION,
            problem="A held-out topology statement with no overlap to the training item.",
        ),
    )
    leaked = "This exact benchmark payload must never enter a frozen training manifest."
    dirty_records = (
        make_record("train:snapshot-dirty", problem=leaked),
        make_record(
            "eval:snapshot-dirty",
            source_id="public/eval",
            source_revision="2" * 40,
            split=SplitName.EVALUATION,
            problem=leaked,
        ),
    )

    class SwitchingSequence:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            selected = clean_records if self.iterations <= 3 else dirty_records
            return iter(selected)

        def __len__(self) -> int:
            return 2

    records = SwitchingSequence()
    manifest = build_data_manifest(
        records,  # type: ignore[arg-type]
        make_registry(
            make_source(),
            make_source(
                "public/eval",
                revision="2" * 40,
                uses=(DataUse.EVALUATE,),
            ),
        ),
        transform_registry=make_transform_registry(),
        split_policy_sha256=digest("split-policy"),
    )
    assert records.iterations == 1
    assert {record.sample_id for record in manifest.records} == {
        "train:snapshot-clean",
        "eval:snapshot-clean",
    }


@pytest.mark.parametrize(
    "field",
    [
        "manifest_sha256",
        "record_set_sha256",
        "source_registry_sha256",
        "transform_registry_sha256",
        "parent_ledger_sha256",
        "contamination_report_sha256",
    ],
)
def test_manifest_detects_top_level_hash_tampering(field: str) -> None:
    raw = manifest_for((make_record("train:018"),)).to_record()
    raw[field] = digest(f"tampered:{field}")
    with pytest.raises((ManifestIntegrityError, DataContractError)):
        DataManifest.from_mapping(raw)


def test_manifest_detects_record_tampering() -> None:
    raw = manifest_for((make_record("train:019"),)).to_record()
    raw["records"][0]["payload_sha256"] = digest("tampered-record")
    with pytest.raises(ManifestIntegrityError):
        DataManifest.from_mapping(raw)


def test_manifest_schema_and_record_schema_are_explicit() -> None:
    assert manifest_for((make_record("train:020"),)).to_record()["schema_version"] == (
        DATA_MANIFEST_SCHEMA_VERSION
    )
    assert make_record("train:020").to_record()["schema_version"] == DATA_RECORD_SCHEMA_VERSION
    assert make_registry().to_record()["schema_version"] == SOURCE_REGISTRY_SCHEMA_VERSION


@settings(max_examples=100, deadline=None)
@given(st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=20))
def test_canonical_json_hash_is_mapping_order_invariant(values: dict[str, int]) -> None:
    reversed_items = dict(reversed(list(values.items())))
    assert canonical_json_bytes(values) == canonical_json_bytes(reversed_items)
    assert sha256_json(values) == sha256_json(reversed_items)


@settings(max_examples=100, deadline=None)
@given(st.permutations(tuple(f"pool:{index:03d}" for index in range(8))))
def test_family_split_is_permutation_invariant(order: list[str]) -> None:
    records = tuple(
        make_record(sample_id, split=SplitName.UNASSIGNED)
        for sample_id in order
    )
    _, assignment = assign_family_disjoint_splits(records, make_policy())
    canonical_records = tuple(sorted(records, key=lambda record: record.sample_id))
    _, canonical_assignment = assign_family_disjoint_splits(canonical_records, make_policy())
    assert assignment == canonical_assignment
