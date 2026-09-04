from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.data import canonical_json_bytes
from posttrain_lab.evaluation import (
    CheckpointIdentity,
    DecodingMode,
    EvaluationContractError,
    FinishReason,
    GenerationContractError,
    GenerationResponse,
    GenerationStatus,
    SealedReferenceError,
    load_benchmark_descriptor,
    load_generation_bundle,
    load_generation_protocol,
    load_public_benchmark,
    load_sealed_answer_vault,
    prepare_generation_requests,
    run_generation,
    write_generation_bundle,
)

FIXTURE_ROOT = Path("tests/fixtures/evaluation_contract")


def checkpoint(*, digest: str = "b" * 64) -> CheckpointIdentity:
    return CheckpointIdentity(
        model_id="synthetic/student",
        model_revision="5" * 40,
        checkpoint_sha256=digest,
    )


def loaded_public():
    descriptor = load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json")
    return load_public_benchmark(descriptor, FIXTURE_ROOT / "public_items.jsonl")


def completed_response(request, text: str = "Final answer: 42") -> GenerationResponse:
    return GenerationResponse(
        request_id=request.request_id,
        status=GenerationStatus.COMPLETED,
        generated_text=text,
        output_token_ids=(*text.encode("utf-8"), request.protocol.eos_token_id),
        finish_reason=FinishReason.EOS,
        error_code=None,
    )


def complete_backend(requests):
    return tuple(completed_response(request) for request in reversed(requests))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_public_and_sealed_inputs_load_as_separate_capabilities() -> None:
    public = loaded_public()
    vault = load_sealed_answer_vault(public, FIXTURE_ROOT / "sealed_references.jsonl")
    assert len(public.items) == 6
    assert vault.item_count == 6
    assert vault.raw_sha256 == public.descriptor.sealed_references_sha256
    assert "references=<redacted>" in repr(vault)
    assert not hasattr(vault, "keys")
    assert not hasattr(vault, "items")
    with pytest.raises(TypeError):
        vars(vault)
    with pytest.raises(TypeError):
        asdict(vault)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(vault)


def test_generator_request_surface_contains_no_reference_field() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "sampling_protocol.json")
    requests = prepare_generation_requests(
        public,
        checkpoint=checkpoint(),
        protocol=protocol,
    )
    assert len(requests) == 48

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {nested for item in value.values() for nested in keys(item)}
        if isinstance(value, list):
            return {nested for item in value for nested in keys(item)}
        return set()

    forbidden = {"reference", "reference_answer", "gold", "target", "label"}
    assert all(keys(request.to_record()).isdisjoint(forbidden) for request in requests)


def test_descriptor_rejects_unknown_or_mutable_identity(tmp_path: Path) -> None:
    payload = json.loads((FIXTURE_ROOT / "benchmark_descriptor.json").read_text(encoding="utf-8"))
    payload["surprise"] = True
    path = tmp_path / "descriptor.json"
    write_json(path, payload)
    with pytest.raises(EvaluationContractError, match="invalid keys"):
        load_benchmark_descriptor(path)

    payload.pop("surprise")
    payload["benchmark_revision"] = "main"
    write_json(path, payload)
    with pytest.raises(EvaluationContractError, match="full 40/64-hex"):
        load_benchmark_descriptor(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"reference": "42"}), "invalid keys"),
        (lambda row: row.update({"item_index": 2}), "consecutive"),
        (lambda row: row.update({"benchmark_revision": "6" * 40}), "identity"),
        (lambda row: row.update({"prompt": "e\u0301"}), "NFC"),
    ],
)
def test_public_loader_rejects_schema_identity_and_canonicalization(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    source_lines = (FIXTURE_ROOT / "public_items.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(source_lines[0])
    mutation(first)
    source_lines[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":"))
    path = tmp_path / "public.jsonl"
    path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    descriptor = replace(
        load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json"),
        public_items_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(EvaluationContractError, match=message):
        load_public_benchmark(descriptor, path)


def test_public_loader_checks_raw_hash_before_parsing() -> None:
    descriptor = replace(
        load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json"),
        public_items_sha256="0" * 64,
    )
    with pytest.raises(EvaluationContractError, match="bytes do not match"):
        load_public_benchmark(descriptor, FIXTURE_ROOT / "public_items.jsonl")


@pytest.mark.parametrize("suffix", [b"\n", b"\r\n", b"\xef\xbb\xbf"])
def test_public_loader_rejects_blank_crlf_and_bom(tmp_path: Path, suffix: bytes) -> None:
    original = (FIXTURE_ROOT / "public_items.jsonl").read_bytes()
    if suffix == b"\xef\xbb\xbf":
        raw = suffix + original
    elif suffix == b"\r\n":
        raw = original.replace(b"\n", suffix)
    else:
        raw = original + suffix
    path = tmp_path / "public.jsonl"
    path.write_bytes(raw)
    descriptor = replace(
        load_benchmark_descriptor(FIXTURE_ROOT / "benchmark_descriptor.json"),
        public_items_sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(EvaluationContractError, match=r"blank|LF|BOM"):
        load_public_benchmark(descriptor, path)


def test_sealed_loader_rejects_wrong_hash_duplicate_and_item_set(tmp_path: Path) -> None:
    public = loaded_public()
    with pytest.raises(SealedReferenceError, match="bytes do not match"):
        load_sealed_answer_vault(
            public,
            FIXTURE_ROOT / "public_items.jsonl",
        )

    rows = (FIXTURE_ROOT / "sealed_references.jsonl").read_text(encoding="utf-8").splitlines()
    rows[-1] = rows[0]
    path = tmp_path / "sealed.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    descriptor = replace(
        public.descriptor,
        sealed_references_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    changed_public = replace(public, descriptor=descriptor)
    with pytest.raises(SealedReferenceError, match="duplicate"):
        load_sealed_answer_vault(changed_public, path)


@pytest.mark.parametrize(
    ("filename", "mode", "samples", "temperature", "top_p", "top_k", "seed"),
    [
        ("greedy_protocol.json", DecodingMode.GREEDY, 1, None, None, None, None),
        ("sampling_protocol.json", DecodingMode.SAMPLING, 8, 700000, 950000, 0, 20260904),
    ],
)
def test_frozen_protocols_load_with_exact_semantics(
    filename: str,
    mode: DecodingMode,
    samples: int,
    temperature: int | None,
    top_p: int | None,
    top_k: int | None,
    seed: int | None,
) -> None:
    protocol = load_generation_protocol(FIXTURE_ROOT / filename)
    assert protocol.mode is mode
    assert protocol.samples_per_item == samples
    assert protocol.temperature_ppm == temperature
    assert protocol.top_p_ppm == top_p
    assert protocol.top_k == top_k
    assert protocol.base_seed == seed
    assert len(protocol.digest) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"samples_per_item": 2}, "samples_per_item=1"),
        ({"temperature_ppm": 700000}, "forbids sampling"),
        ({"stop_token_ids": (3,)}, "include eos"),
        ({"stop_token_ids": (2, 2)}, "unique and sorted"),
    ],
)
def test_greedy_protocol_rejects_ambiguous_controls(
    changes: dict[str, object], message: str
) -> None:
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    with pytest.raises(EvaluationContractError, match=message):
        replace(protocol, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"samples_per_item": 1}, "at least two"),
        ({"temperature_ppm": None}, "integer"),
        ({"top_p_ppm": 1_000_001}, r"\[1, 1000000\]"),
        ({"seed_namespace": None}, "identifier"),
        ({"base_seed": -1}, r"\[0,"),
    ],
)
def test_sampling_protocol_rejects_incomplete_controls(
    changes: dict[str, object], message: str
) -> None:
    protocol = load_generation_protocol(FIXTURE_ROOT / "sampling_protocol.json")
    with pytest.raises(EvaluationContractError, match=message):
        replace(protocol, **changes)


@settings(max_examples=100, deadline=None)
@given(
    item_id=st.from_regex(r"item:[a-z0-9]{1,16}", fullmatch=True),
    sample_index=st.integers(min_value=0, max_value=7),
)
def test_sampling_seed_is_deterministic_bounded_and_checkpoint_independent(
    item_id: str,
    sample_index: int,
) -> None:
    protocol = load_generation_protocol(FIXTURE_ROOT / "sampling_protocol.json")
    kwargs = {
        "benchmark_id": "synthetic/math-eval",
        "benchmark_revision": "4" * 40,
        "item_id": item_id,
        "sample_index": sample_index,
    }
    first = protocol.seed_for(**kwargs)
    second = protocol.seed_for(**kwargs)
    assert first == second
    assert first is not None and 0 <= first < 2**63


def test_paired_seeds_match_across_checkpoints_but_request_ids_do_not() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "sampling_protocol.json")
    first = prepare_generation_requests(
        public,
        checkpoint=checkpoint(digest="b" * 64),
        protocol=protocol,
    )
    second = prepare_generation_requests(
        public,
        checkpoint=checkpoint(digest="c" * 64),
        protocol=protocol,
    )
    assert [request.seed for request in first] == [request.seed for request in second]
    assert [request.request_id for request in first] != [request.request_id for request in second]


def test_backend_response_order_is_canonicalized() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-order",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    assert [(record.item_index, record.sample_index) for record in batch.records] == [
        (index, 0) for index in range(6)
    ]
    assert batch.failed_count == 0


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "nonresponse"])
def test_backend_response_bijection_fails_closed(failure: str) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        responses = [completed_response(request) for request in requests]
        if failure == "missing":
            return responses[:-1]
        if failure == "duplicate":
            return [*responses, responses[0]]
        if failure == "unexpected":
            return [*responses[:-1], replace(responses[-1], request_id="0" * 64)]
        return [*responses[:-1], object()]

    with pytest.raises(GenerationContractError, match=r"differs|duplicate|non-response"):
        run_generation(
            public,
            run_id="d07-bad-backend",
            checkpoint=checkpoint(),
            protocol=protocol,
            backend=backend,
        )


def test_failed_response_is_preserved_not_silently_dropped() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        responses = [completed_response(request) for request in requests]
        responses[2] = GenerationResponse(
            request_id=requests[2].request_id,
            status=GenerationStatus.FAILED,
            generated_text=None,
            output_token_ids=(),
            finish_reason=None,
            error_code="backend.timeout",
        )
        return responses

    batch = run_generation(
        public,
        run_id="d07-failed-output",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=backend,
    )
    assert batch.failed_count == 1
    assert batch.records[2].error_code == "backend.timeout"


@pytest.mark.parametrize(
    ("reason", "tokens", "text", "message"),
    [
        (FinishReason.EOS, (7,), "x", "eos finish"),
        (FinishReason.STOP_TOKEN, (7,), "x", "allowed token"),
        (FinishReason.STOP_SEQUENCE, (7,), "x", "frozen stop sequence"),
        (FinishReason.LENGTH, (7,), "x", "exactly max_new_tokens"),
    ],
)
def test_finish_reason_is_verified_against_frozen_protocol(
    reason: FinishReason,
    tokens: tuple[int, ...],
    text: str,
    message: str,
) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")

    def backend(requests):
        return [
            GenerationResponse(
                request.request_id,
                GenerationStatus.COMPLETED,
                text,
                tokens,
                reason,
                None,
            )
            for request in requests
        ]

    with pytest.raises(GenerationContractError, match=message):
        run_generation(
            public,
            run_id="d07-invalid-finish",
            checkpoint=checkpoint(),
            protocol=protocol,
            backend=backend,
        )


def test_generation_bundle_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-round-trip",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    records_path = tmp_path / "generations.jsonl"
    manifest_path = tmp_path / "generation_manifest.json"
    manifest = write_generation_bundle(
        batch,
        records_path=records_path,
        manifest_path=manifest_path,
    )
    loaded = load_generation_bundle(public.descriptor, public, manifest_path)
    assert loaded.batch == batch
    assert loaded.manifest == manifest
    assert records_path.read_bytes().endswith(b"\n")

    records_path.write_bytes(records_path.read_bytes() + b" ")
    with pytest.raises(EvaluationContractError, match=r"blank|bytes do not match"):
        load_generation_bundle(public.descriptor, public, manifest_path)


def test_batch_revalidates_persisted_finish_semantics() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-protocol-revalidation",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    original = batch.records[0]
    unsigned = {**original.unsigned_record(), "output_token_ids": [7]}
    forged = replace(
        original,
        output_token_ids=(7,),
        record_sha256=hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    )
    with pytest.raises(GenerationContractError, match="eos finish"):
        replace(batch, records=(forged, *batch.records[1:]))


def test_generation_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-path",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    records_path = tmp_path / "generations.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_generation_bundle(batch, records_path=records_path, manifest_path=manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records_file"] = "../generations.jsonl"
    write_json(manifest_path, payload)
    with pytest.raises(
        GenerationContractError,
        match=r"sibling basename|manifest_sha256",
    ):
        load_generation_bundle(public.descriptor, public, manifest_path)


def test_generation_writer_requires_sibling_distinct_paths(tmp_path: Path) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-writer-path",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    with pytest.raises(GenerationContractError, match="siblings"):
        write_generation_bundle(
            batch,
            records_path=tmp_path / "one" / "records.jsonl",
            manifest_path=tmp_path / "two" / "manifest.json",
        )
    with pytest.raises(GenerationContractError, match="must differ"):
        write_generation_bundle(
            batch,
            records_path=tmp_path / "same.json",
            manifest_path=tmp_path / "same.json",
        )
    with pytest.raises(GenerationContractError, match="must differ"):
        write_generation_bundle(
            batch,
            records_path=tmp_path / "alias" / ".." / "same.json",
            manifest_path=tmp_path / "same.json",
        )


def test_completed_response_requires_complete_output_token_ids() -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    request = prepare_generation_requests(
        public,
        checkpoint=checkpoint(),
        protocol=protocol,
    )[0]
    with pytest.raises(GenerationContractError, match="non-empty output_token_ids"):
        GenerationResponse(
            request_id=request.request_id,
            status=GenerationStatus.COMPLETED,
            generated_text="Final answer: 42<END>",
            output_token_ids=(),
            finish_reason=FinishReason.STOP_SEQUENCE,
            error_code=None,
        )


def test_generation_writer_replaces_distinct_leaf_symlinks_without_touching_targets(
    tmp_path: Path,
) -> None:
    public = loaded_public()
    protocol = load_generation_protocol(FIXTURE_ROOT / "greedy_protocol.json")
    batch = run_generation(
        public,
        run_id="d07-symlink-writer",
        checkpoint=checkpoint(),
        protocol=protocol,
        backend=complete_backend,
    )
    target_dir = tmp_path / "targets"
    link_dir = tmp_path / "links"
    target_dir.mkdir()
    link_dir.mkdir()
    records_target = target_dir / "actual-records.jsonl"
    manifest_target = target_dir / "actual-manifest.json"
    records_target.write_text("records target sentinel\n", encoding="utf-8")
    manifest_target.write_text("manifest target sentinel\n", encoding="utf-8")
    records_path = link_dir / "records.jsonl"
    manifest_path = link_dir / "manifest.json"
    records_path.symlink_to(records_target)
    manifest_path.symlink_to(manifest_target)

    write_generation_bundle(
        batch,
        records_path=records_path,
        manifest_path=manifest_path,
    )

    assert not records_path.is_symlink()
    assert not manifest_path.is_symlink()
    assert records_target.read_text(encoding="utf-8") == "records target sentinel\n"
    assert manifest_target.read_text(encoding="utf-8") == "manifest target sentinel\n"
    assert load_generation_bundle(public.descriptor, public, manifest_path).batch == batch
