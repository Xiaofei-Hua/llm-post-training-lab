from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.data.contamination import (
    CONTAMINATION_POLICY_SCHEMA_VERSION,
    DEFAULT_CONTAMINATION_POLICY,
    ContaminationError,
    MatchKind,
    normalize_for_contamination,
    quarantine_contaminated_families,
    quarantined_family_ids,
    scan_contamination,
)
from posttrain_lab.data.registry import (
    DataRecord,
    Message,
    SplitName,
    TransformLineage,
    canonical_json_bytes,
    sha256_json,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def record(
    sample_id: str,
    problem: str,
    *,
    split: SplitName,
    source_id: str | None = None,
    source_family: str | None = None,
    problem_family: str | None = None,
    template_family: str | None = None,
    messages: tuple[Message, ...] | None = None,
    reference_answer: str | None = None,
    response: str | None = None,
) -> DataRecord:
    return DataRecord(
        sample_id=sample_id,
        source_id=source_id or ("public/eval" if split is SplitName.EVALUATION else "public/train"),
        source_revision="2" * 40 if split is SplitName.EVALUATION else "1" * 40,
        split=split,
        source_family=source_family or f"sf:{sample_id}",
        problem_family=problem_family or f"pf:{sample_id}",
        template_family=template_family or f"tf:{sample_id}",
        problem=problem,
        messages=messages or (Message("user", problem),),
        reference_answer=reference_answer,
        response=response,
        quality=(("answer_verified", True),),
        strata=(("difficulty", "synthetic"),),
        lineage=TransformLineage(
            transform_name="ingest",
            transform_version="1.0.0",
            code_sha256=digest("ingest-code"),
            config_sha256=digest("ingest-config"),
            parents=(),
        ),
    )


def unrelated_pair() -> tuple[DataRecord, DataRecord]:
    training = record(
        "train:clean",
        "Determine the number of blue marbles after three independent exchanges.",
        split=SplitName.D_CORE,
        response="The resulting synthetic count is eleven.",
    )
    evaluation = record(
        "eval:clean",
        "Prove that every finite tree with two vertices has a leaf endpoint.",
        split=SplitName.EVALUATION,
        source_id="public/eval",
        reference_answer="A degree-one vertex exists.",
    )
    return training, evaluation


def test_normalization_handles_unicode_case_spacing_latex_and_digit_groups() -> None:
    left = "  FINAL\u00a0Value: \\dfrac{1,000}{2} \u2212 x  "
    right = "final value:\\frac { 1000 } { 2 } - x"
    assert normalize_for_contamination(left) == normalize_for_contamination(right)


def test_normalization_retains_semantic_punctuation() -> None:
    assert normalize_for_contamination("x + y") != normalize_for_contamination("x - y")
    assert normalize_for_contamination("f(x)") != normalize_for_contamination("f[x]")


def test_policy_is_versioned_and_content_addressed() -> None:
    policy = DEFAULT_CONTAMINATION_POLICY
    assert policy.to_record()["schema_version"] == CONTAMINATION_POLICY_SCHEMA_VERSION
    assert policy.sha256 == sha256_json(policy.to_record())
    assert replace(policy, fuzzy_jaccard_ppm=830_000).sha256 != policy.sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"character_ngram_size": 1},
        {"token_ngram_size": 0},
        {"minimum_exact_chars": 100, "minimum_fuzzy_chars": 50},
        {"fuzzy_jaccard_ppm": 40_000, "review_margin_ppm": 50_000},
        {"included_field_groups": ("problem", "unknown")},
        {"included_field_groups": ("problem", "problem")},
    ],
)
def test_policy_rejects_unsafe_or_noncanonical_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContaminationError):
        replace(DEFAULT_CONTAMINATION_POLICY, **kwargs)


def test_exact_match_is_blocking_and_report_contains_no_raw_text() -> None:
    copied = "Find all integers n such that n squared plus three equals nineteen."
    training = record("train:exact", copied, split=SplitName.D_ANCHOR)
    evaluation = record("eval:exact", copied.upper(), split=SplitName.EVALUATION)
    report = scan_contamination((training,), (evaluation,))
    assert not report.passed
    assert report.match_counts == {"exact": 1, "fuzzy": 0, "review": 0}
    assert report.matches[0].kind is MatchKind.EXACT
    assert report.matches[0].character_jaccard_ppm == 1_000_000
    serialized = canonical_json_bytes(report.to_record())
    assert copied.encode() not in serialized
    assert report.report_sha256 == sha256_json(report.body_record)
    with pytest.raises(ContaminationError, match="gate failed"):
        report.assert_clean()


def test_fuzzy_match_detects_small_prompt_rewrite() -> None:
    training = record(
        "train:fuzzy",
        "A merchant has forty red boxes and twenty blue boxes; compute the total number of boxes.",
        split=SplitName.D_CORE,
    )
    evaluation = record(
        "eval:fuzzy",
        "A merchant has forty red boxes and twenty blue boxes. Compute the total number of boxes.",
        split=SplitName.EVALUATION,
    )
    report = scan_contamination((training,), (evaluation,))
    assert not report.passed
    assert {match.kind for match in report.matches} == {MatchKind.FUZZY}
    assert any(
        max(match.character_containment_ppm, match.token_containment_ppm) >= 920_000
        for match in report.matches
    )


def test_solution_containment_detects_benchmark_problem_inside_trace() -> None:
    benchmark = (
        "Let positive integers a and b satisfy a plus b equals thirty; "
        "determine their maximum product."
    )
    training = record(
        "train:trace-copy",
        "Explain a generic inequality technique using a fresh synthetic setup.",
        split=SplitName.D_ANCHOR,
        response=f"Archived exercise: {benchmark} The copied solution follows after this sentence.",
    )
    evaluation = record("eval:contained", benchmark, split=SplitName.EVALUATION)
    report = scan_contamination((training,), (evaluation,))
    matches = [match for match in report.matches if match.train_field == "response"]
    assert matches
    assert matches[0].kind is MatchKind.FUZZY
    assert max(
        matches[0].character_containment_ppm,
        matches[0].token_containment_ppm,
    ) == 1_000_000


def test_borderline_pair_is_fail_closed_as_review() -> None:
    policy = replace(
        DEFAULT_CONTAMINATION_POLICY,
        fuzzy_jaccard_ppm=1_000_000,
        fuzzy_containment_ppm=1_000_000,
        review_margin_ppm=600_000,
    )
    training = record(
        "train:review",
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        split=SplitName.D_DEV,
    )
    evaluation = record(
        "eval:review",
        "alpha bravo charlie delta echo kilo lima mike november oscar",
        split=SplitName.EVALUATION,
    )
    report = scan_contamination((training,), (evaluation,), policy=policy)
    assert report.match_counts["review"] >= 1
    assert not report.passed


def test_short_shared_answer_is_not_treated_as_contamination() -> None:
    training, evaluation = unrelated_pair()
    training = replace(training, reference_answer="42")
    evaluation = replace(evaluation, reference_answer="42")
    report = scan_contamination((training,), (evaluation,))
    assert report.passed
    assert report.matches == ()


def test_system_context_is_scanned_by_default() -> None:
    shared_system = Message(
        "system",
        "You are a careful mathematical assistant that always checks every derivation.",
    )
    training, evaluation = unrelated_pair()
    training = replace(
        training,
        messages=(shared_system, Message("user", training.problem)),
    )
    evaluation = replace(
        evaluation,
        messages=(shared_system, Message("user", evaluation.problem)),
    )
    report = scan_contamination((training,), (evaluation,))
    assert not report.passed
    assert report.matches[0].kind is MatchKind.EXACT


def test_context_exemption_requires_exact_frozen_normalized_hash() -> None:
    shared_system = Message(
        "system",
        "This deliberately shared context is long enough for deterministic exact matching.",
    )
    training, evaluation = unrelated_pair()
    training = replace(training, messages=(shared_system, Message("user", training.problem)))
    evaluation = replace(evaluation, messages=(shared_system, Message("user", evaluation.problem)))
    normalized = normalize_for_contamination(shared_system.content)
    exemption = hashlib.sha256(normalized.encode()).hexdigest()
    policy = replace(
        DEFAULT_CONTAMINATION_POLICY,
        exempt_context_normalized_sha256=(exemption,),
    )
    report = scan_contamination((training,), (evaluation,), policy=policy)
    assert report.passed


def test_prompt_aggregate_catches_text_split_across_short_messages() -> None:
    shared = "This benchmark fragment is split across message boundaries"
    training, evaluation = unrelated_pair()
    training = replace(
        training,
        messages=(Message("user", shared[:28]), Message("user", shared[28:])),
    )
    evaluation = replace(evaluation, messages=(Message("user", shared),))
    report = scan_contamination((training,), (evaluation,))
    assert not report.passed
    assert report.matches[0].train_field.startswith("aggregate.")


def test_tool_trace_is_scanned_and_aggregated() -> None:
    first = "A benchmark solution hidden inside a tool"
    second = " trace must still be detected by the audit."
    training, evaluation = unrelated_pair()
    training = replace(
        training,
        messages=(
            Message("user", training.problem),
            Message("tool", first),
            Message("tool", second),
        ),
    )
    evaluation = replace(
        evaluation,
        messages=(Message("user", evaluation.problem), Message("assistant", first + second)),
    )
    assert not scan_contamination((training,), (evaluation,)).passed


def test_clean_report_is_deterministic_under_record_reordering() -> None:
    train_a, eval_a = unrelated_pair()
    train_b = record(
        "train:second",
        "Integrate a polynomial whose coefficients form an alternating sequence.",
        split=SplitName.D_SELECT,
    )
    eval_b = record(
        "eval:second",
        "Classify the connected components of a graph after removing one bridge.",
        split=SplitName.EVALUATION,
    )
    first = scan_contamination((train_a, train_b), (eval_a, eval_b))
    second = scan_contamination((train_b, train_a), (eval_b, eval_a))
    assert first == second
    assert first.passed
    first.assert_clean()


@pytest.mark.parametrize(
    ("train_split", "eval_split"),
    [
        (SplitName.UNASSIGNED, SplitName.EVALUATION),
        (SplitName.EVALUATION, SplitName.EVALUATION),
        (SplitName.D_CORE, SplitName.D_DEV),
    ],
)
def test_scan_rejects_wrong_split_domains(
    train_split: SplitName, eval_split: SplitName
) -> None:
    training, evaluation = unrelated_pair()
    with pytest.raises(ContaminationError):
        scan_contamination(
            (replace(training, split=train_split),),
            (replace(evaluation, split=eval_split),),
        )


def test_scan_rejects_empty_duplicate_and_overlapping_ids() -> None:
    training, evaluation = unrelated_pair()
    with pytest.raises(ContaminationError, match="non-empty"):
        scan_contamination((), (evaluation,))
    with pytest.raises(ContaminationError, match="duplicate"):
        scan_contamination((training, training), (evaluation,))
    evaluation = replace(evaluation, sample_id=training.sample_id)
    with pytest.raises(ContaminationError, match="overlap"):
        scan_contamination((training,), (evaluation,))


def test_normalizer_enforces_input_type_and_size() -> None:
    policy = replace(DEFAULT_CONTAMINATION_POLICY, maximum_text_chars=10)
    with pytest.raises(ContaminationError, match="exceeds"):
        normalize_for_contamination("x" * 11, policy=policy)
    with pytest.raises(ContaminationError, match="string"):
        normalize_for_contamination(42)  # type: ignore[arg-type]


def test_quarantine_expands_across_transitive_family_graph() -> None:
    copied = "Evaluate the exact synthetic recurrence after twelve deterministic iterations."
    first = record(
        "train:q1",
        copied,
        split=SplitName.D_CORE,
        source_family="sf:linked",
    )
    second = record(
        "train:q2",
        "A distinct second training problem with no copied benchmark wording.",
        split=SplitName.D_CORE,
        source_family="sf:linked",
        problem_family="pf:bridge",
    )
    third = record(
        "train:q3",
        "A distinct third training problem linked only through the bridge family.",
        split=SplitName.D_CORE,
        problem_family="pf:bridge",
    )
    safe = record(
        "train:safe",
        "A safe isolated training problem about a pentagon and its diagonal count.",
        split=SplitName.D_CORE,
    )
    evaluation = record("eval:q", copied, split=SplitName.EVALUATION)
    report = scan_contamination((first, second, third, safe), (evaluation,))
    assert quarantined_family_ids((first, second, third, safe), report.matches) == (
        "train:q1",
        "train:q2",
        "train:q3",
    )
    kept, quarantined = quarantine_contaminated_families(
        (first, second, third, safe), report
    )
    assert [item.sample_id for item in kept] == ["train:safe"]
    assert [item.sample_id for item in quarantined] == [
        "train:q1",
        "train:q2",
        "train:q3",
    ]
    assert scan_contamination(kept, (evaluation,)).passed


def test_quarantine_rejects_unknown_match_record() -> None:
    copied = "A copied benchmark statement that is deliberately long enough for matching."
    training = record("train:known", copied, split=SplitName.D_CORE)
    evaluation = record("eval:known", copied, split=SplitName.EVALUATION)
    report = scan_contamination((training,), (evaluation,))
    forged = replace(report.matches[0], train_record_id="train:unknown")
    with pytest.raises(ContaminationError, match="unknown"):
        quarantined_family_ids((training,), (forged,))


def test_quarantine_cannot_silently_empty_registry() -> None:
    copied = "A copied benchmark problem that is long enough to trigger the exact gate."
    training = record("train:only", copied, split=SplitName.D_CORE)
    evaluation = record("eval:only", copied, split=SplitName.EVALUATION)
    report = scan_contamination((training,), (evaluation,))
    with pytest.raises(ContaminationError, match="entire training registry"):
        quarantine_contaminated_families((training,), report)


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.text(
            alphabet=st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyz")),
            min_size=2,
            max_size=8,
        ),
        min_size=5,
        max_size=20,
        unique=True,
    ),
    st.sampled_from([" ", "  ", "\n", "\t"]),
)
def test_case_and_whitespace_variants_have_identical_normalization(
    tokens: list[str], separator: str
) -> None:
    base = " ".join(tokens)
    variant = separator.join(token.upper() for token in tokens)
    assert normalize_for_contamination(base) == normalize_for_contamination(variant)
