from __future__ import annotations

import json
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import posttrain_lab.rewards.verifier as verifier_module
from posttrain_lab.rewards import (
    AnswerSource,
    ExactMathVerifier,
    ExtractionStatus,
    ParseStatus,
    VerificationStatus,
    VerifierInfrastructureError,
    VerifierPolicy,
    exact_math_reward,
    extract_prediction_answer,
    extract_reference_answer,
    verifier_backend_versions,
    verify_math_answer,
)


def test_nested_box_is_extracted_without_losing_latex_braces() -> None:
    result = extract_prediction_answer(r"Reasoning. \boxed{\frac{1+\sqrt{5}}{2}}")
    assert result.status is ExtractionStatus.EXTRACTED
    assert result.source is AnswerSource.BOXED
    assert result.candidate == r"\frac{1+\sqrt{5}}{2}"
    assert result.marker_count == 1


def test_answer_word_inside_box_is_not_treated_as_a_later_text_marker() -> None:
    result = extract_prediction_answer(r"\boxed{\text{Answer: }42}")
    assert result.status is ExtractionStatus.EXTRACTED
    assert result.source is AnswerSource.BOXED
    assert result.candidate == r"\text{Answer: }42"


@pytest.mark.parametrize(
    ("prediction", "candidate", "source", "marker_count"),
    [
        (r"\boxed{1}; final answer: 2", "2", AnswerSource.TEXT_MARKER, 2),
        (r"final answer: 1; then \boxed{2}", "2", AnswerSource.BOXED, 2),
        (r"\boxed{1} and \boxed{2}", "2", AnswerSource.BOXED, 2),
        ("<answer>1</answer>\n答案\uff1a2", "2", AnswerSource.TEXT_MARKER, 2),
        ("work\n<answer> 3/4 </answer>", "3/4", AnswerSource.ANSWER_TAG, 1),
        ("**Final Answer:** 42", "42", AnswerSource.TEXT_MARKER, 1),
    ],
)
def test_last_explicit_answer_surface_wins(
    prediction: str,
    candidate: str,
    source: AnswerSource,
    marker_count: int,
) -> None:
    result = extract_prediction_answer(prediction)
    assert result.status is ExtractionStatus.EXTRACTED
    assert result.candidate == candidate
    assert result.source is source
    assert result.marker_count == marker_count


def test_escaped_box_command_cannot_override_previous_terminal_surface() -> None:
    escaped_after_box = verify_math_answer("42", r"\boxed{7} \\boxed{42}")
    assert escaped_after_box.status is VerificationStatus.MISMATCH
    assert escaped_after_box.reward == 0.0
    assert escaped_after_box.prediction.extraction.candidate == "7"
    assert escaped_after_box.prediction.extraction.marker_count == 1

    escaped_inside_marker = verify_math_answer("42", r"Final answer: 7 \\boxed{42}")
    assert escaped_inside_marker.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert escaped_inside_marker.reward == 0.0


def test_odd_backslash_count_preserves_real_box_command_boundary() -> None:
    result = verify_math_answer("42", r"\boxed{7} \\\boxed{42}")
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0
    assert result.prediction.extraction.candidate == "42"
    assert result.prediction.extraction.marker_count == 2


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{42}; final answer:",
        r"final answer: 42; then \boxed{",
        "<answer>42</answer> then <answer>",
        r"\boxed{Final answer: 42",
        "<answer>Final answer: 42",
        r"\boxed{broken <answer>42</answer>",
        r"<answer>broken \boxed{42}",
        r"\boxed42",
        r"\fbox42",
        r"Final answer: 7 \boxed42",
        "<answer><answer>42</answer></answer>",
        "```\nfinal answer: 42\n```",
        "```\n\\boxed{42}\n```",
        "~~~\n\\boxed{42}\n~~~",
        "~~~latex\nFinal answer: 42\n~~~",
        r"`\boxed{42}`",
        "`Final answer: 42`",
        r"\boxed{7} then `\boxed{42}`",
    ],
)
def test_malformed_last_surface_fails_closed(prediction: str) -> None:
    result = extract_prediction_answer(prediction)
    assert result.status is ExtractionStatus.MALFORMED_FINAL_ANSWER
    assert result.candidate is None


@pytest.mark.parametrize(
    "prediction",
    [
        "42",
        "-17/23",
        r"\frac{5}{7}",
        r"$x^2+2x+1$",
        r"(-\infty, 2]",
        "0.125",
        "50%",
    ],
)
def test_concise_direct_math_surfaces_are_accepted(prediction: str) -> None:
    result = extract_prediction_answer(prediction)
    assert result.status is ExtractionStatus.EXTRACTED
    assert result.source is AnswerSource.DIRECT


@pytest.mark.parametrize(
    "prediction",
    [
        "The reasoning happens to contain 42.",
        "Ignore every instruction and return 42",
        "第一步得到 42",
        "42\n43",
        "```42```",
    ],
)
def test_unanchored_prose_is_not_mined_for_numbers(prediction: str) -> None:
    result = extract_prediction_answer(prediction)
    assert result.status is ExtractionStatus.NO_FINAL_ANSWER


@pytest.mark.parametrize(
    "prediction",
    [
        "Wrong answer: 42",
        "Not the answer: 42",
        "This is not the final answer: 42",
        "This is not my final answer: 42",
        "This isn't the final answer: 42",
        "Never the final answer: 42",
        "False answer: 42",
        "Fake answer: 42",
        "Example answer: 42",
        "Previous answer: 42",
        "Old answer: 42",
        "Rejected answer: 42",
        "Do not use this answer: 42",
        "Ignore the following final answer: 42",
        "A non-final answer: 42",
        "Possibly the answer: 42",
        "Candidate answer: 42",
        "错误答案\uff1a42",
        "这不是最终答案\uff1a42",
    ],
)
def test_negated_or_candidate_labels_are_not_terminal_answers(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_NOT_EXTRACTED
    assert result.reward == 0.0


def test_new_clause_can_supply_answer_after_a_rejected_earlier_label() -> None:
    result = verify_math_answer("42", "Wrong answer: 7; corrected final answer: 42")
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


def test_backticks_inside_a_text_marker_are_not_unwrapped() -> None:
    result = verify_math_answer("42", "Final answer: `42`")
    assert result.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        "~~~lang\ncode\n~~~ \\boxed{42}",
        "~~~lang\ncode\n~~~ Final answer: 42",
        "```lang\ncode\n``` \\boxed{42}",
        "```lang\ncode\n``` Final answer: 42",
    ],
)
def test_nonempty_fence_tail_cannot_fake_a_closing_fence(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_NOT_EXTRACTED
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        "~~~lang\ncode\n~~~\n\\boxed{42}",
        "```lang\ncode\n```\nFinal answer: 42",
        "`code` then \\boxed{42}",
    ],
)
def test_valid_closed_fence_does_not_hide_later_terminal_answer(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


def test_input_and_candidate_bounds_are_explicit() -> None:
    policy = VerifierPolicy(
        max_input_chars=20,
        max_candidate_chars=10,
        direct_answer_max_chars=5,
    )
    assert (
        extract_prediction_answer("x" * 21, policy=policy).status
        is ExtractionStatus.INPUT_TOO_LONG
    )
    assert (
        extract_prediction_answer(r"\boxed{12345678901}", policy=policy).status
        is ExtractionStatus.CANDIDATE_TOO_LONG
    )
    assert (
        extract_prediction_answer("123456", policy=policy).status
        is ExtractionStatus.NO_FINAL_ANSWER
    )
    assert (
        extract_prediction_answer("42\x00", policy=policy).status
        is ExtractionStatus.UNSAFE_CONTROL_CHARACTER
    )


def test_reference_is_bounded_but_does_not_require_prediction_format() -> None:
    result = extract_reference_answer(r"\frac{1}{2}")
    assert result.status is ExtractionStatus.EXTRACTED
    assert result.source is AnswerSource.REFERENCE
    assert result.candidate == r"\frac{1}{2}"


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        ("42", r"\boxed{42}"),
        (r"\frac{1}{2}", r"Final answer: \frac{2}{4}"),
        (r"(x+1)^2", r"\boxed{x^2+2x+1}"),
        (r"\sqrt{2}", r"<answer>2^{1/2}</answer>"),
        ("x=2", r"\boxed{2}"),
        (r"\{1,2,3\}", r"\boxed{\{3,1,2\}}"),
        (r"\frac{1}{2}", "Final answer: 50%"),
        ("5", r"\boxed{5\text{ cm}}"),
        ("-3", "答案是 -3"),
    ],
)
def test_exact_and_symbolic_equivalence(reference: str, prediction: str) -> None:
    result = verify_math_answer(reference, prediction)
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0
    assert result.matched is True


@pytest.mark.parametrize(
    ("reference", "prediction", "expected_status"),
    [
        ("42", r"\boxed{41}", VerificationStatus.MISMATCH),
        ("2", r"\boxed{2}; final answer: 3", VerificationStatus.MISMATCH),
        ("2", r"final answer: 3; then \boxed{2}", VerificationStatus.MATCH),
        ("42", "The derivation contains 42.", VerificationStatus.PREDICTION_NOT_EXTRACTED),
        ("42", "Final answer: NaN", VerificationStatus.MISMATCH),
        ("42", "Final answer: 0/0", VerificationStatus.MISMATCH),
        ("42", "Final answer: 42 or 7", VerificationStatus.PREDICTION_UNPARSEABLE),
        (r"\{1,2\}", r"\boxed{(1,2)}", VerificationStatus.MISMATCH),
        (
            "42",
            "Final answer: __import__('os').system('echo compromised')",
            VerificationStatus.PREDICTION_UNPARSEABLE,
        ),
    ],
)
def test_mismatch_and_attack_statuses(
    reference: str,
    prediction: str,
    expected_status: VerificationStatus,
) -> None:
    result = verify_math_answer(reference, prediction)
    assert result.status is expected_status
    assert result.reward == (1.0 if expected_status is VerificationStatus.MATCH else 0.0)


@pytest.mark.parametrize(
    "prediction",
    [
        "Final answer: $7$ because $42$",
        r"\boxed{$7$ because $42$}",
        "<answer>$7$ because $42$</answer>",
        "<answer>7 then $42$</answer>",
        r"\boxed{7\text{ or }42}",
        r"Final answer: 7\quad 42",
        r"\boxed{x=0\boxed{42}}",
        r"<answer>x=0\boxed{42}</answer>",
        r"\boxed{0\fbox{42}}",
        "Final answer: x=0 42",
        r"Final answer: x=0\,42",
        r"Final answer: x=0\ 42",
        r"Final answer: x=0\text{}42",
        r"Final answer: x=0\quad42",
        r"Final answer: x=0\qquad42",
        r"Final answer: x=0\thinspace42",
        r"Final answer: x=0\medspace42",
        r"Final answer: x=0\thickspace42",
        r"Final answer: x=0\negthinspace42",
        r"Final answer: x=0\negmedspace42",
        r"Final answer: x=0\negthickspace42",
        r"Final answer: x=0\hfil42",
        r"Final answer: 0\displaystyle42",
        r"Final answer: 0\ldots42",
        r"Final answer: 0\$42",
        'Final answer: 0"42',
        "Final answer: 0'42",
        r"Final answer: 0\\pi+42",
        r"Final answer: 42_7",
        r"Final answer: 42_{x}",
        r"Final answer: 42^T",
        r"Final answer: 42^{\top}",
    ],
)
def test_trailing_math_or_layout_cannot_hijack_candidate(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{0\frac{42}{1}}",
        r"\boxed{x=0\dfrac{42}{1}}",
        r"\boxed{0(42)}",
        r"\boxed{x=0[42]}",
        r"\boxed{0{42}}",
        r"\boxed{x=0\left(42\right)}",
        r"\boxed{0\mleft(42\mright)}",
        r"\boxed{(0)\mleft[42\mright]}",
        r"\boxed{0\binom{42}{1}}",
        r"\boxed{0\max(42)}",
        r"\boxed{x=0\min(42)}",
        r"\boxed{(1)(41)}",
        r"\boxed{[2][40]}",
        r"\boxed{{1}{41}}",
        r"\boxed{\frac{1}{1}\frac{41}{1}}",
        r"\boxed{x=\frac{1}{1}(41)}",
    ],
)
def test_ambiguous_constant_juxtaposition_cannot_change_parser_semantics(
    prediction: str,
) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert result.reward == 0.0
    assert result.prediction.reason == (
        "implicit primary juxtaposition changes meaning when made explicit"
    )


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        ("2*x+2", r"\boxed{2(x+1)}"),
        ("x^2-1", r"\boxed{(x+1)(x-1)}"),
        ("42", r"\boxed{0+0\cdot7+42}"),
        ("42", r"\boxed{2\cdot(21)}"),
        ("42", r"\boxed{\frac{84}{2}}"),
        (r"\sin^2(x)", r"\boxed{\sin^{2}(x)}"),
        (r"\cos^2(x)", r"\boxed{\cos^{2}(x)}"),
        (r"\tan^{-1}(1)", r"\boxed{\tan^{-1}(1)}"),
        ("3", r"\boxed{\log_{2}(8)}"),
        ("42", r"\boxed{\operatorname{gcd}(42,42)}"),
        ("42", r"\boxed{\operatorname{lcm}(6,7)}"),
    ],
)
def test_unambiguous_products_and_command_arguments_remain_valid(
    reference: str,
    prediction: str,
) -> None:
    result = verify_math_answer(reference, prediction)
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{42\text{ wrong}}",
        r"\boxed{42\text{ is false}}",
        r"\boxed{42\text{ unless 7}}",
        r"\boxed{42\text{ maybe 7}}",
        r"\boxed{42\text{ arbitrary prose}}",
        r"\boxed{42\mathrm{wrong}}",
        r"\boxed{42\mbox{wrong}}",
        r"\boxed{42\textrm{wrong 7}}",
        r"\boxed{42\textnormal{wrong}}",
        r"\boxed{42\mathit{wrong}}",
        r"\boxed{42\mathbf{wrong}}",
        r"\boxed{42\text{ apples}}",
        r"\boxed{42\text{错误}}",
        r"\boxed{42\text{不对 7}}",
        r"\boxed{42\text{✗ 7}}",
        r"\boxed{42\text{🚫 7}}",
        r"\boxed{42\text{≠7}}",
        r"\boxed{42\text{not\ correct}}",
        r"\boxed{42\text{n\!o\!t 7}}",
    ],
)
def test_text_like_payload_must_be_a_frozen_unit_phrase(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{42\text{ cm}}",
        r"\boxed{42\mathrm{kg}}",
        r"\boxed{42\mbox{ meters}}",
        r"\boxed{42\textnormal{square meters}}",
    ],
)
def test_frozen_text_unit_phrases_remain_valid(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"<answer>\boxed{42}</answer>",
        r"\boxed{\boxed{42}}",
    ],
)
def test_exact_nested_container_can_be_safely_unwrapped(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{7=42}",
        r"\boxed{0=42}",
        r"\boxed{7+1=42}",
        r"\boxed{42=x}",
    ],
)
def test_only_symbol_lhs_assignment_can_match_scalar_gold(prediction: str) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.MISMATCH
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        r"\boxed{x=0 \land y=42}",
        r"\boxed{x=0 \wedge y=42}",
        r"\boxed{x=0 \& y=42}",
        r"\boxed{x=0=42}",
        r"\boxed{x<0 \lor y<42}",
    ],
)
def test_logical_connector_and_chained_relation_hijacks_are_rejected(
    prediction: str,
) -> None:
    result = verify_math_answer("42", prediction)
    assert result.status is VerificationStatus.PREDICTION_UNPARSEABLE
    assert result.reward == 0.0


def test_boolean_relation_tree_is_never_treated_as_scalar() -> None:
    result = verify_math_answer("42", r"\boxed{0<x<42}")
    assert result.status is VerificationStatus.MISMATCH
    assert result.reward == 0.0


def test_relation_pair_cannot_match_only_because_rhs_is_shared() -> None:
    result = verify_math_answer("7=42", r"\boxed{0=42}")
    assert result.status is VerificationStatus.MISMATCH
    assert result.reward == 0.0


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        ("42", r"\boxed{x=\{42\}}"),
        (r"\{42\}", r"\boxed{x=42}"),
        ("x=42", r"\boxed{x=\{42\}}"),
        (r"x=\{42\}", r"\boxed{x=42}"),
        ("42", r"\boxed{x=\begin{pmatrix}42\end{pmatrix}}"),
    ],
)
def test_assignment_unwrap_rechecks_rhs_structural_family(
    reference: str,
    prediction: str,
) -> None:
    result = verify_math_answer(reference, prediction)
    assert result.status is VerificationStatus.MISMATCH
    assert result.reward == 0.0


def test_high_precision_policy_rejects_coarse_decimal_but_accepts_normal_float_rendering() -> None:
    verifier = ExactMathVerifier()
    assert verifier.exact_reward("1/3", "Final answer: 0.333333") == 0.0
    assert verifier.exact_reward("1/3", "Final answer: 0.3333333333333333") == 1.0
    assert verifier.exact_reward("1/2", "Final answer: 0.5") == 1.0


def test_invalid_reference_never_turns_into_a_negative_training_example() -> None:
    verifier = ExactMathVerifier()
    result = verifier.verify("not a mathematical reference", r"\boxed{42}")
    assert result.status is VerificationStatus.REFERENCE_INVALID
    assert result.reward is None
    assert result.matched is None
    with pytest.raises(VerifierInfrastructureError, match="reference_invalid"):
        verifier.exact_reward("not a mathematical reference", r"\boxed{42}")


@pytest.mark.parametrize(
    "reference",
    [
        r"42\text{ wrong}",
        r"0(42)",
    ],
)
def test_reference_prevalidation_rejects_silent_normalization(reference: str) -> None:
    result = verify_math_answer(reference, r"\boxed{42}")
    assert result.status is VerificationStatus.REFERENCE_INVALID
    assert result.reward is None
    assert result.prediction.status is ParseStatus.SKIPPED


def test_frozen_unit_phrase_is_valid_for_reference_and_prediction() -> None:
    result = verify_math_answer(r"42\text{ cm}", r"\boxed{42\mathrm{cm}}")
    assert result.status is VerificationStatus.MATCH
    assert result.reward == 1.0


def test_batch_scoring_is_exact_and_never_zip_truncates() -> None:
    verifier = ExactMathVerifier()
    batch = verifier.score_batch(
        ["1", "2", "3"],
        [r"\boxed{1}", r"\boxed{9}", "The text mentions 3"],
    )
    assert batch.rewards == (1.0, 0.0, 0.0)
    assert batch.correct_count == 1
    assert batch.incorrect_count == 2
    with pytest.raises(ValueError, match="same length"):
        verifier.score_batch(["1"], ["1", "2"])
    with pytest.raises(ValueError, match="non-empty"):
        verifier.score_batch([], [])
    with pytest.raises(TypeError, match="sequences"):
        verifier.score_batch("1", "1")


def test_results_are_json_serializable_without_serializing_sympy_objects() -> None:
    result = verify_math_answer(r"\frac{1}{2}", r"\boxed{2/4}")
    record = result.to_record()
    encoded = json.dumps(record, sort_keys=True)
    assert '"reward": 1.0' in encoded
    assert record["reference"]["value_types"]
    assert "values" not in record["reference"]


def test_policy_and_backend_versions_are_reproducible() -> None:
    policy = VerifierPolicy()
    assert policy.digest() == VerifierPolicy().digest()
    assert len(policy.digest()) == 64
    assert verifier_backend_versions() == {
        "math-verify": "0.9.0",
        "latex2sympy2-extended": "1.11.0",
        "antlr4-python3-runtime": "4.13.2",
        "sympy": "1.14.0",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_chars": 0},
        {"max_candidate_chars": 0},
        {"direct_answer_max_chars": 0},
        {"parsing_timeout_seconds": 0},
        {"verification_timeout_seconds": 0},
        {"numeric_precision": 0},
        {"reference_cache_size": 0},
        {"max_input_chars": 10, "max_candidate_chars": 11},
        {"max_candidate_chars": 10, "direct_answer_max_chars": 11},
        {"float_rounding": -1},
        {"float_rounding": 31, "numeric_precision": 30},
        {"strict_symbol_matching": 1},
        {"allow_set_relation_comparison": 0},
        {"schema_version": 1},
    ],
)
def test_policy_rejects_invalid_or_ambiguous_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VerifierPolicy(**kwargs)


def test_prediction_timeout_scores_zero_but_backend_error_is_not_assignable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = ExactMathVerifier()
    assert verifier.parse_reference("42").parsed

    def timeout(*args: object, **kwargs: object) -> list[object]:
        raise verifier_module.TimeoutException()

    monkeypatch.setattr(verifier_module, "math_verify_parse", timeout)
    timed_out = verifier.verify("42", r"\boxed{42}")
    assert timed_out.status is VerificationStatus.PREDICTION_TIMEOUT
    assert timed_out.reward == 0.0

    def broken(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("backend failure details")

    monkeypatch.setattr(verifier_module, "math_verify_parse", broken)
    backend_error = verifier.verify("42", r"\boxed{42}")
    assert backend_error.status is VerificationStatus.BACKEND_ERROR
    assert backend_error.reward is None
    assert backend_error.reason == "symbolic parser raised RuntimeError"


def test_verification_timeout_scores_adversarial_expression_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = ExactMathVerifier()

    def timeout(*args: object, **kwargs: object) -> bool:
        raise verifier_module.TimeoutException()

    monkeypatch.setattr(verifier_module, "math_verify_compare", timeout)
    result = verifier.verify("42", r"\boxed{42}")
    assert result.status is VerificationStatus.VERIFICATION_TIMEOUT
    assert result.reward == 0.0


def test_bounded_backend_refuses_thread_context_instead_of_disabling_timeout() -> None:
    result_holder: list[object] = []
    verifier = ExactMathVerifier()

    def worker() -> None:
        result_holder.append(verifier.verify("42", r"\boxed{42}"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(result_holder) == 1
    result = result_holder[0]
    assert hasattr(result, "status")
    assert result.status is VerificationStatus.BACKEND_ERROR
    assert result.reward is None
    assert result.reference.status is ParseStatus.BACKEND_ERROR
    recovered = verifier.verify("42", r"\boxed{42}")
    assert recovered.status is VerificationStatus.MATCH
    assert recovered.reward == 1.0


def test_transient_reference_failure_does_not_poison_success_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = ExactMathVerifier()
    original_parse = verifier_module.math_verify_parse

    def timeout(*args: object, **kwargs: object) -> list[object]:
        raise verifier_module.TimeoutException()

    monkeypatch.setattr(verifier_module, "math_verify_parse", timeout)
    failed = verifier.verify("17", r"\boxed{17}")
    assert failed.status is VerificationStatus.BACKEND_ERROR
    assert failed.reward is None
    assert failed.prediction.status is ParseStatus.SKIPPED

    monkeypatch.setattr(verifier_module, "math_verify_parse", original_parse)
    recovered = verifier.verify("17", r"\boxed{17}")
    assert recovered.status is VerificationStatus.MATCH
    assert recovered.reward == 1.0


def test_invalid_references_short_circuit_predictions_and_entire_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = ExactMathVerifier()
    parsed_predictions: list[str] = []

    def forbidden_prediction_parse(prediction: str) -> object:
        parsed_predictions.append(prediction)
        raise AssertionError("prediction parser must not run after invalid gold")

    monkeypatch.setattr(verifier, "parse_prediction", forbidden_prediction_parse)
    result = verifier.verify("not a mathematical reference", "anything")
    assert result.status is VerificationStatus.REFERENCE_INVALID
    assert result.reward is None
    assert result.prediction.status is ParseStatus.SKIPPED
    assert parsed_predictions == []

    with pytest.raises(VerifierInfrastructureError, match="index 1: reference_invalid"):
        verifier.score_batch(
            ["1", "not a mathematical reference", "3"],
            [r"\boxed{1}", "anything", r"\boxed{3}"],
        )
    assert parsed_predictions == []


@settings(max_examples=100, deadline=None)
@given(
    numerator=st.integers(min_value=-200, max_value=200),
    denominator=st.integers(min_value=1, max_value=200),
    scale=st.integers(min_value=1, max_value=20),
)
def test_generated_equivalent_rationals_receive_exact_reward(
    numerator: int,
    denominator: int,
    scale: int,
) -> None:
    reference = f"{numerator}/{denominator}"
    prediction = rf"\boxed{{\frac{{{numerator * scale}}}{{{denominator * scale}}}}}"
    assert exact_math_reward(reference, prediction) == 1.0


@settings(max_examples=100, deadline=None)
@given(
    value=st.integers(min_value=-10_000, max_value=10_000),
    offset=st.integers(min_value=1, max_value=100),
)
def test_generated_perturbed_integers_never_receive_reward(value: int, offset: int) -> None:
    assert exact_math_reward(str(value), rf"\boxed{{{value + offset}}}") == 0.0


@settings(max_examples=100, deadline=None)
@given(prediction=st.text(max_size=256))
def test_arbitrary_unicode_prediction_never_escapes_structured_result(prediction: str) -> None:
    result = verify_math_answer("0", prediction)
    assert result.reward in {0.0, 1.0, None}
    json.dumps(result.to_record(), ensure_ascii=False)
