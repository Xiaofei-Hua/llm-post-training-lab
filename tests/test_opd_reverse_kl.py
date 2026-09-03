from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.train import (
    LossTokenBudget,
    MaskValidationError,
    chunked_masked_causal_linear_reverse_kl,
    masked_causal_reverse_kl,
    plan_torch_loss_budget,
    torch_completion_loss_mask,
)


def transform_logits(
    logits: torch.Tensor,
    *,
    scale: float = 1.0,
    softcap: float | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    transformed = logits * scale
    if softcap is not None:
        transformed = softcap * torch.tanh(transformed / softcap)
    return transformed / temperature


def independent_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    student_scale: float = 1.0,
    teacher_scale: float = 1.0,
    student_softcap: float | None = None,
    teacher_softcap: float | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = loss_mask[:, 1:].nonzero(as_tuple=False)
    batch_indices = selected[:, 0]
    prediction_positions = selected[:, 1]
    student = transform_logits(
        student_logits[batch_indices, prediction_positions],
        scale=student_scale,
        softcap=student_softcap,
        temperature=temperature,
    )
    teacher = transform_logits(
        teacher_logits[batch_indices, prediction_positions],
        scale=teacher_scale,
        softcap=teacher_softcap,
        temperature=temperature,
    )
    student_log_probs = student.log_softmax(dim=-1)
    teacher_log_probs = teacher.log_softmax(dim=-1)
    student_probs = student_log_probs.exp()
    per_token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    entropy = -(student_probs * student_log_probs).sum(dim=-1)
    return per_token_kl.mean(), entropy.sum()


def test_causal_shift_and_global_token_mean_match_independent_formula() -> None:
    student = torch.tensor(
        [
            [
                [3.0, 0.0, -1.0],
                [-8.0, 8.0, 0.0],
                [0.0, 2.0, -2.0],
                [9.0, -9.0, 0.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [
                [1.0, 2.0, -1.0],
                [8.0, -8.0, 0.0],
                [1.0, -1.0, 2.0],
                [-9.0, 9.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor([[False, True, False, True]])
    output = masked_causal_reverse_kl(
        student,
        teacher,
        mask,
        max_tokens_per_chunk=1,
    )
    expected_loss, expected_entropy = independent_reverse_kl(student, teacher, mask)

    assert output.local_token_count == 2
    assert output.normalization_token_count == 2
    torch.testing.assert_close(output.loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output.loss_sum, expected_loss.detach() * 2)
    torch.testing.assert_close(output.student_entropy_sum, expected_entropy.detach())
    assert not output.student_entropy_sum.requires_grad
    assert output.local_student_entropy_mean == pytest.approx(float(expected_entropy.detach() / 2))

    output.loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[0, 1]).item() == 0
    assert torch.count_nonzero(student.grad[0, 3]).item() == 0


def test_reduction_is_token_mean_not_equal_weight_sequence_mean() -> None:
    student = torch.tensor(
        [
            [[3.0, -3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[-2.0, 2.0], [-1.0, 1.0], [-3.0, 3.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    teacher = torch.zeros_like(student)
    mask = torch.tensor(
        [[False, True, False, False], [False, True, True, True]],
        dtype=torch.bool,
    )
    output = masked_causal_reverse_kl(student, teacher, mask)
    expected, _ = independent_reverse_kl(student, teacher, mask)
    row_zero, _ = independent_reverse_kl(student[:1], teacher[:1], mask[:1])
    row_one, _ = independent_reverse_kl(student[1:], teacher[1:], mask[1:])

    torch.testing.assert_close(output.loss, expected)
    assert not torch.isclose(output.loss, (row_zero + row_one) / 2)


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    batch_size=st.integers(min_value=1, max_value=4),
    sequence_length=st.integers(min_value=2, max_value=9),
    vocabulary_size=st.integers(min_value=2, max_value=19),
    chunk_size=st.integers(min_value=1, max_value=7),
)
def test_generated_batches_match_independent_full_vocabulary_formula(
    seed: int,
    batch_size: int,
    sequence_length: int,
    vocabulary_size: int,
    chunk_size: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(
        (batch_size, sequence_length, vocabulary_size),
        dtype=torch.float64,
        generator=generator,
    )
    teacher = torch.randn(
        (batch_size, sequence_length, vocabulary_size),
        dtype=torch.float64,
        generator=generator,
    )
    mask = torch.rand((batch_size, sequence_length), generator=generator).ge(0.5)
    mask[:, 0] = False
    if not bool(mask.any().item()):
        mask[0, 1] = True

    output = masked_causal_reverse_kl(
        student,
        teacher,
        mask,
        max_tokens_per_chunk=chunk_size,
    )
    expected_loss, expected_entropy = independent_reverse_kl(student, teacher, mask)
    torch.testing.assert_close(output.loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        output.student_entropy_sum,
        expected_entropy,
        rtol=1e-12,
        atol=1e-12,
    )


def test_student_logit_value_and_gradient_match_independent_formula() -> None:
    generator = torch.Generator().manual_seed(20260903)
    student = torch.randn((2, 5, 7), dtype=torch.float64, generator=generator)
    teacher = torch.randn((2, 5, 7), dtype=torch.float64, generator=generator)
    mask = torch.tensor([[False, True, False, True, True], [False, False, True, True, False]])

    actual_student = student.clone().requires_grad_(True)
    actual_loss = masked_causal_reverse_kl(
        actual_student,
        teacher,
        mask,
        max_tokens_per_chunk=2,
    ).loss
    actual_gradient = torch.autograd.grad(actual_loss, actual_student)[0]

    expected_student = student.clone().requires_grad_(True)
    expected_loss, _ = independent_reverse_kl(expected_student, teacher, mask)
    expected_gradient = torch.autograd.grad(expected_loss, expected_student)[0]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_materialized_path_passes_autograd_gradcheck() -> None:
    generator = torch.Generator().manual_seed(41)
    student = torch.randn((1, 4, 5), dtype=torch.float64, generator=generator, requires_grad=True)
    teacher = torch.randn((1, 4, 5), dtype=torch.float64, generator=generator)
    mask = torch.tensor([[False, True, False, True]])

    def loss_function(candidate: torch.Tensor) -> torch.Tensor:
        return masked_causal_reverse_kl(
            candidate,
            teacher,
            mask,
            max_tokens_per_chunk=1,
            loss_compute_dtype=torch.float64,
        ).loss

    assert torch.autograd.gradcheck(loss_function, (student,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_identical_distributions_are_zero_and_additive_constants_are_invariant() -> None:
    generator = torch.Generator().manual_seed(53)
    logits = torch.randn((2, 4, 6), dtype=torch.float64, generator=generator)
    mask = torch.tensor([[False, True, True, False], [False, False, True, True]])
    identical = masked_causal_reverse_kl(logits, logits.clone(), mask).loss
    shifted = masked_causal_reverse_kl(
        logits + torch.tensor([[[7.0]], [[-11.0]]]),
        logits + torch.tensor([[[-3.0]], [[5.0]]]),
        mask,
    ).loss
    assert identical.item() == pytest.approx(0.0, abs=1e-15)
    assert shifted.item() == pytest.approx(0.0, abs=1e-15)


def test_reverse_kl_orientation_is_not_forward_kl() -> None:
    student_probability = torch.tensor([0.8, 0.2], dtype=torch.float64)
    teacher_probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    student_logits = student_probability.log().reshape(1, 1, 2).repeat(1, 2, 1)
    teacher_logits = teacher_probability.log().reshape(1, 1, 2).repeat(1, 2, 1)
    mask = torch.tensor([[False, True]])
    output = masked_causal_reverse_kl(student_logits, teacher_logits, mask)
    reverse_kl = (
        student_probability * (student_probability.log() - teacher_probability.log())
    ).sum()
    forward_kl = (
        teacher_probability * (teacher_probability.log() - student_probability.log())
    ).sum()
    torch.testing.assert_close(output.loss, reverse_kl)
    assert not torch.isclose(output.loss, forward_kl)


def test_materialized_path_processes_only_bounded_position_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import posttrain_lab.train.opd_reverse_kl as opd_module

    student = torch.randn((2, 5, 7))
    teacher = torch.randn((2, 5, 7))
    mask = torch.tensor([[False, True, True, True, True], [False, True, True, False, True]])
    chunk_rows = []
    original = opd_module._reverse_kl_sums_from_logits

    def recording_reverse_kl(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = args[0]
        assert isinstance(candidate, torch.Tensor)
        chunk_rows.append(candidate.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(opd_module, "_reverse_kl_sums_from_logits", recording_reverse_kl)
    masked_causal_reverse_kl(student, teacher, mask, max_tokens_per_chunk=3)
    assert chunk_rows == [3, 3, 1]


def test_chunked_linear_path_matches_materialized_value_and_student_gradients() -> None:
    generator = torch.Generator().manual_seed(67)
    student_hidden = torch.randn((2, 6, 3), dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn((2, 6, 5), dtype=torch.float64, generator=generator)
    student_weight = torch.randn((9, 3), dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn((9, 5), dtype=torch.float64, generator=generator)
    student_bias = torch.randn((9,), dtype=torch.float64, generator=generator)
    teacher_bias = torch.randn((9,), dtype=torch.float64, generator=generator)
    mask = torch.tensor(
        [
            [False, True, True, False, True, False],
            [False, False, True, True, True, True],
        ]
    )

    actual_hidden = student_hidden.clone().requires_grad_(True)
    actual_weight = student_weight.clone().requires_grad_(True)
    actual_bias = student_bias.clone().requires_grad_(True)
    actual = chunked_masked_causal_linear_reverse_kl(
        actual_hidden,
        actual_weight,
        teacher_hidden,
        teacher_weight,
        mask,
        student_lm_head_bias=actual_bias,
        teacher_lm_head_bias=teacher_bias,
        max_tokens_per_chunk=2,
        loss_compute_dtype=torch.float64,
    )
    actual_gradients = torch.autograd.grad(
        actual.loss,
        (actual_hidden, actual_weight, actual_bias),
    )

    expected_hidden = student_hidden.clone().requires_grad_(True)
    expected_weight = student_weight.clone().requires_grad_(True)
    expected_bias = student_bias.clone().requires_grad_(True)
    expected_student_logits = F.linear(expected_hidden, expected_weight, expected_bias)
    expected_teacher_logits = F.linear(teacher_hidden, teacher_weight, teacher_bias)
    expected_loss, expected_entropy = independent_reverse_kl(
        expected_student_logits,
        expected_teacher_logits,
        mask,
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (expected_hidden, expected_weight, expected_bias),
    )

    torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual.student_entropy_sum,
        expected_entropy,
        rtol=1e-12,
        atol=1e-12,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_model_logit_transforms_and_temperature_match_reference_order() -> None:
    generator = torch.Generator().manual_seed(79)
    student_hidden = torch.randn((1, 5, 3), dtype=torch.float64, generator=generator)
    teacher_hidden = torch.randn((1, 5, 4), dtype=torch.float64, generator=generator)
    student_weight = torch.randn((7, 3), dtype=torch.float64, generator=generator)
    teacher_weight = torch.randn((7, 4), dtype=torch.float64, generator=generator)
    mask = torch.tensor([[False, True, True, False, True]])
    kwargs = {
        "student_logit_scale": 1.25,
        "teacher_logit_scale": 0.75,
        "student_final_logit_softcapping": 2.5,
        "teacher_final_logit_softcapping": 3.5,
        "temperature": 1.7,
    }
    output = chunked_masked_causal_linear_reverse_kl(
        student_hidden,
        student_weight,
        teacher_hidden,
        teacher_weight,
        mask,
        max_tokens_per_chunk=1,
        loss_compute_dtype=torch.float64,
        **kwargs,
    )
    expected, _ = independent_reverse_kl(
        F.linear(student_hidden, student_weight),
        F.linear(teacher_hidden, teacher_weight),
        mask,
        student_scale=1.25,
        teacher_scale=0.75,
        student_softcap=2.5,
        teacher_softcap=3.5,
        temperature=1.7,
    )
    torch.testing.assert_close(output.loss, expected, rtol=1e-12, atol=1e-12)
    assert output.temperature == 1.7


def test_linear_projection_never_exceeds_configured_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import posttrain_lab.train.opd_reverse_kl as opd_module

    student_hidden = torch.randn((2, 5, 3))
    teacher_hidden = torch.randn((2, 5, 4))
    student_weight = torch.randn((7, 3))
    teacher_weight = torch.randn((7, 4))
    mask = torch.tensor([[False, True, True, True, True], [False, True, True, False, True]])
    projected_rows = []
    original_linear = F.linear

    def recording_linear(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_rows.append(input_tensor.shape[0])
        return original_linear(input_tensor, weight, bias)

    monkeypatch.setattr(opd_module.F, "linear", recording_linear)
    chunked_masked_causal_linear_reverse_kl(
        student_hidden,
        student_weight,
        teacher_hidden,
        teacher_weight,
        mask,
        max_tokens_per_chunk=3,
    )
    assert projected_rows == [3, 3, 3, 3, 1, 1]


def test_linear_path_does_not_save_vocabulary_logits_for_backward() -> None:
    student_hidden = torch.randn((2, 5, 4), requires_grad=True)
    teacher_hidden = torch.randn((2, 5, 6))
    student_weight = torch.randn((13, 4), requires_grad=True)
    teacher_weight = torch.randn((13, 6))
    student_bias = torch.randn((13,), requires_grad=True)
    mask = torch.tensor([[False, True, True, True, True], [False, True, True, False, True]])
    saved_shapes: list[tuple[int, ...]] = []

    def record_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
        output = chunked_masked_causal_linear_reverse_kl(
            student_hidden,
            student_weight,
            teacher_hidden,
            teacher_weight,
            mask,
            student_lm_head_bias=student_bias,
            max_tokens_per_chunk=3,
        )

    assert not any(len(shape) == 2 and shape[-1] == 13 for shape in saved_shapes)
    output.loss.backward()
    assert student_hidden.grad is not None
    assert student_weight.grad is not None
    assert student_bias.grad is not None


def test_bfloat16_uses_float32_and_extreme_finite_logits_remain_finite() -> None:
    student = torch.tensor(
        [
            [
                [10_000.0, 0.0, -10_000.0],
                [-10_000.0, 0.0, 10_000.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [
                [-10_000.0, 0.0, 10_000.0],
                [10_000.0, 0.0, -10_000.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    mask = torch.tensor([[False, True, True]])
    output = masked_causal_reverse_kl(student, teacher, mask, max_tokens_per_chunk=1)
    assert output.loss.dtype == torch.float32
    assert output.loss_sum.dtype == torch.float32
    assert output.student_entropy_sum.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert student.grad is not None and student.grad.dtype == torch.bfloat16
    assert torch.isfinite(student.grad).all()


def test_mixed_low_precision_linear_inputs_use_float32_projection_and_loss() -> None:
    student_hidden = torch.randn((1, 4, 3), dtype=torch.bfloat16, requires_grad=True)
    student_weight = torch.randn((7, 3), dtype=torch.float32, requires_grad=True)
    teacher_hidden = torch.randn((1, 4, 5), dtype=torch.bfloat16)
    teacher_weight = torch.randn((7, 5), dtype=torch.float32)
    output = chunked_masked_causal_linear_reverse_kl(
        student_hidden,
        student_weight,
        teacher_hidden,
        teacher_weight,
        torch.tensor([[False, True, True, False]]),
        max_tokens_per_chunk=1,
    )
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert student_hidden.grad is not None and student_hidden.grad.dtype == torch.bfloat16
    assert student_weight.grad is not None and student_weight.grad.dtype == torch.float32
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(student_weight.grad).all()


def test_accumulation_and_ddp_scaling_reconstruct_global_token_mean_gradient() -> None:
    generator = torch.Generator().manual_seed(83)
    base = torch.randn((4, 4, 5), dtype=torch.float64, generator=generator) - 2.0
    feature = torch.randn((4, 4, 5), dtype=torch.float64, generator=generator) * 0.1
    teacher = torch.randn((4, 4, 5), dtype=torch.float64, generator=generator)
    mask = torch.tensor(
        [
            [False, True, False, False],
            [False, True, True, False],
            [False, True, True, True],
            [False, False, True, False],
        ]
    )
    global_count = int(mask.sum().item())

    full_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    full_loss = masked_causal_reverse_kl(
        base + full_parameter * feature,
        teacher,
        mask,
    ).loss
    expected_gradient = torch.autograd.grad(full_loss, full_parameter)[0]

    accumulated_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    accumulated_losses = []
    for start, end in ((0, 1), (1, 4)):
        accumulated_losses.append(
            masked_causal_reverse_kl(
                base[start:end] + accumulated_parameter * feature[start:end],
                teacher[start:end],
                mask[start:end],
                global_token_count=global_count,
                ddp_world_size=1,
            ).loss
        )
    accumulated_gradient = torch.autograd.grad(
        sum(accumulated_losses),
        accumulated_parameter,
    )[0]
    torch.testing.assert_close(accumulated_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    rank_gradients = []
    for start, end in ((0, 2), (2, 4)):
        rank_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        rank_loss = masked_causal_reverse_kl(
            base[start:end] + rank_parameter * feature[start:end],
            teacher[start:end],
            mask[start:end],
            global_token_count=global_count,
            ddp_world_size=2,
        ).loss
        rank_gradients.append(torch.autograd.grad(rank_loss, rank_parameter)[0])
    ddp_averaged_gradient = torch.stack(rank_gradients).mean()
    torch.testing.assert_close(ddp_averaged_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_empty_linear_shard_is_finite_and_connects_every_student_tensor() -> None:
    student_hidden = torch.tensor([[[torch.nan, torch.inf], [1.0, 2.0]]], requires_grad=True)
    student_weight = torch.tensor(
        [[torch.nan, 1.0], [torch.inf, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    student_bias = torch.tensor([torch.inf, 1.0, 2.0], requires_grad=True)
    output = chunked_masked_causal_linear_reverse_kl(
        student_hidden,
        student_weight,
        torch.randn((1, 2, 4)),
        torch.randn((3, 4)),
        torch.zeros((1, 2), dtype=torch.bool),
        student_lm_head_bias=student_bias,
        global_token_count=3,
        ddp_world_size=2,
    )
    assert output.loss.item() == 0.0
    assert output.local_token_count == 0
    assert math.isnan(output.local_student_entropy_mean)
    output.loss.backward()
    for value in (student_hidden, student_weight, student_bias):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad).item() == 0


def test_d01_completion_mask_and_exact_budget_feed_d04() -> None:
    input_ids = torch.tensor(
        [
            [9, 4, 2, 0, 0],
            [8, 3, 5, 6, 2],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    candidate_mask = torch_completion_loss_mask(
        input_ids,
        completion_start=1,
        attention_mask=attention,
        eos_token_id=2,
    )
    selection = plan_torch_loss_budget(
        LossTokenBudget(5),
        candidate_mask,
        sample_ids=["prompt-a", "prompt-b"],
        generation_indices=[0, 0],
    )
    student = torch.randn((2, 5, 11), requires_grad=True)
    teacher = torch.randn((2, 5, 11))
    output = masked_causal_reverse_kl(student, teacher, selection.loss_mask)
    assert output.local_token_count == selection.selected_tokens == 5
    output.loss.backward()
    assert student.grad is not None


def test_unselected_nonfinite_logits_are_ignored_but_selected_values_fail() -> None:
    student = torch.tensor([[[-1.0, 1.0], [torch.nan, torch.inf], [torch.nan, torch.inf]]])
    teacher = torch.tensor([[[1.0, -1.0], [torch.inf, torch.nan], [-torch.inf, torch.nan]]])
    mask = torch.tensor([[False, True, False]])
    output = masked_causal_reverse_kl(student, teacher, mask)
    assert torch.isfinite(output.loss)

    with pytest.raises(FloatingPointError, match="selected Student"):
        masked_causal_reverse_kl(
            student,
            teacher,
            torch.tensor([[False, False, True]]),
        )


@pytest.mark.parametrize(
    ("student", "teacher", "mask", "message"),
    [
        (
            torch.randn((1, 3, 4)),
            torch.randn((1, 3, 5)),
            torch.tensor([[False, True, False]]),
            "identical shapes",
        ),
        (
            torch.randn((1, 3, 4)),
            torch.randn((1, 3, 4)),
            torch.tensor([[True, False, False]]),
            "position 0",
        ),
        (
            torch.randn((1, 3, 4)),
            torch.randn((1, 3, 4)),
            torch.tensor([[0, 1, 0]]),
            "Boolean",
        ),
        (
            torch.ones((1, 3, 4), dtype=torch.int64),
            torch.randn((1, 3, 4)),
            torch.tensor([[False, True, False]]),
            "student_logits must use",
        ),
        (
            torch.randn((1, 3, 4)),
            torch.empty((1, 3, 4), device="meta"),
            torch.tensor([[False, True, False]]),
            "share a device",
        ),
        (
            torch.randn((1, 3, 4)),
            torch.randn((1, 3, 4)),
            torch.tensor([[False, True]]),
            "loss_mask shape",
        ),
    ],
)
def test_materialized_tensor_contract_rejects_invalid_inputs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(MaskValidationError, match=message):
        masked_causal_reverse_kl(student, teacher, mask)


def test_teacher_inputs_must_be_stop_gradient() -> None:
    mask = torch.tensor([[False, True]])
    with pytest.raises(MaskValidationError, match="Teacher logits"):
        masked_causal_reverse_kl(
            torch.randn((1, 2, 3)),
            torch.randn((1, 2, 3), requires_grad=True),
            mask,
        )

    common = {
        "student_hidden_states": torch.randn((1, 2, 2)),
        "student_lm_head_weight": torch.randn((3, 2)),
        "teacher_hidden_states": torch.randn((1, 2, 4)),
        "teacher_lm_head_weight": torch.randn((3, 4)),
        "loss_mask": mask,
    }
    for name in ("teacher_hidden_states", "teacher_lm_head_weight"):
        invalid = dict(common)
        invalid[name] = invalid[name].clone().requires_grad_(True)
        with pytest.raises(MaskValidationError, match=name):
            chunked_masked_causal_linear_reverse_kl(**invalid)

    with pytest.raises(MaskValidationError, match="teacher_lm_head_bias"):
        chunked_masked_causal_linear_reverse_kl(
            **common,
            teacher_lm_head_bias=torch.randn((3,), requires_grad=True),
        )


@pytest.mark.parametrize(
    ("student_weight", "teacher_weight", "message"),
    [
        (torch.randn((4, 2)), torch.randn((5, 3)), "vocabulary sizes"),
        (torch.randn((4, 4)), torch.randn((4, 3)), "Student LM-head input"),
        (torch.randn((4, 2)), torch.randn((4, 5)), "Teacher LM-head input"),
    ],
)
def test_linear_shape_contract_rejects_incompatible_heads(
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(MaskValidationError, match=message):
        chunked_masked_causal_linear_reverse_kl(
            torch.randn((1, 3, 2)),
            student_weight,
            torch.randn((1, 3, 3)),
            teacher_weight,
            torch.tensor([[False, True, False]]),
        )


def test_normalization_metadata_must_be_complete_and_consistent() -> None:
    student = torch.randn((1, 3, 4))
    teacher = torch.randn((1, 3, 4))
    mask = torch.tensor([[False, True, True]])
    with pytest.raises(ValueError, match="provided together"):
        masked_causal_reverse_kl(
            student,
            teacher,
            mask,
            global_token_count=2,
        )
    with pytest.raises(ValueError, match="smaller"):
        masked_causal_reverse_kl(
            student,
            teacher,
            mask,
            global_token_count=1,
            ddp_world_size=1,
        )
    with pytest.raises(MaskValidationError, match="selects no"):
        masked_causal_reverse_kl(
            student,
            teacher,
            torch.zeros_like(mask),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": math.inf},
        {"student_logit_scale": -1.0},
        {"teacher_logit_scale": math.nan},
        {"student_final_logit_softcapping": 0.0},
        {"teacher_final_logit_softcapping": -2.0},
        {"max_tokens_per_chunk": 0},
        {"loss_compute_dtype": torch.bfloat16},
    ],
)
def test_invalid_numeric_contracts_fail_loudly(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        masked_causal_reverse_kl(
            torch.randn((1, 2, 3)),
            torch.randn((1, 2, 3)),
            torch.tensor([[False, True]]),
            **kwargs,
        )
