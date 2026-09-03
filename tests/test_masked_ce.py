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
    chunked_masked_causal_linear_cross_entropy,
    masked_causal_cross_entropy,
    plan_torch_loss_budget,
    torch_assistant_target_loss_mask,
)


def independent_masked_ce(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    selected = mask[:, 1:].nonzero(as_tuple=False)
    batch_indices = selected[:, 0]
    prediction_positions = selected[:, 1]
    targets = labels[batch_indices, prediction_positions + 1]
    selected_logits = logits[batch_indices, prediction_positions]
    selected_log_probabilities = selected_logits.log_softmax(dim=-1)
    return -selected_log_probabilities.gather(1, targets.unsqueeze(1)).mean()


def test_causal_shift_and_token_mean_are_exact() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, -2.0],
                [0.0, 0.0, 0.0],
                [-1.0, 3.0, 0.0],
                [9.0, -9.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, -100, 1]])
    mask = torch.tensor([[False, True, False, True]])
    output = masked_causal_cross_entropy(logits, labels, mask, max_tokens_per_chunk=1)
    expected = independent_masked_ce(logits, labels, mask)

    assert output.local_token_count == 2
    assert output.normalization_token_count == 2
    torch.testing.assert_close(output.loss_sum, expected.detach() * 2)
    torch.testing.assert_close(output.loss, expected)

    output.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 1]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 3]).item() == 0


def test_reduction_is_global_token_mean_not_sequence_mean() -> None:
    logits = torch.zeros((2, 4, 2), dtype=torch.float64)
    logits[0, 0] = torch.tensor([8.0, -8.0])
    logits[1, 0:3] = torch.tensor([-2.0, 2.0])
    labels = torch.tensor([[-100, 0, -100, -100], [-100, 0, 0, 0]])
    mask = torch.tensor([[False, True, False, False], [False, True, True, True]])
    output = masked_causal_cross_entropy(logits, labels, mask)
    expected = independent_masked_ce(logits, labels, mask)

    row_zero = independent_masked_ce(logits[:1], labels[:1], mask[:1])
    row_one = independent_masked_ce(logits[1:], labels[1:], mask[1:])
    sequence_mean = (row_zero + row_one) / 2
    torch.testing.assert_close(output.loss, expected)
    assert not torch.isclose(output.loss, sequence_mean)


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    batch_size=st.integers(min_value=1, max_value=4),
    sequence_length=st.integers(min_value=2, max_value=10),
    vocabulary_size=st.integers(min_value=2, max_value=17),
    chunk_size=st.integers(min_value=1, max_value=8),
)
def test_value_matches_independent_formula_across_generated_batches(
    seed: int,
    batch_size: int,
    sequence_length: int,
    vocabulary_size: int,
    chunk_size: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(
        (batch_size, sequence_length, vocabulary_size),
        dtype=torch.float64,
        generator=generator,
    )
    labels = torch.randint(
        vocabulary_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    mask = torch.rand((batch_size, sequence_length), generator=generator).ge(0.45)
    mask[:, 0] = False
    if not bool(mask.any().item()):
        mask[0, 1] = True

    actual = masked_causal_cross_entropy(
        logits,
        labels,
        mask,
        max_tokens_per_chunk=chunk_size,
    ).loss
    expected = independent_masked_ce(logits, labels, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_logits_gradient_matches_independent_formula() -> None:
    generator = torch.Generator().manual_seed(20260903)
    logits = torch.randn((2, 5, 7), dtype=torch.float64, generator=generator)
    labels = torch.randint(7, (2, 5), generator=generator)
    mask = torch.tensor([[False, True, False, True, True], [False, False, True, True, False]])

    actual_logits = logits.clone().requires_grad_(True)
    actual_loss = masked_causal_cross_entropy(
        actual_logits, labels, mask, max_tokens_per_chunk=2
    ).loss
    actual_gradient = torch.autograd.grad(actual_loss, actual_logits)[0]

    expected_logits = logits.clone().requires_grad_(True)
    expected_loss = independent_masked_ce(expected_logits, labels, mask)
    expected_gradient = torch.autograd.grad(expected_loss, expected_logits)[0]

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_logits_path_passes_autograd_gradcheck() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((1, 4, 5), dtype=torch.float64, generator=generator, requires_grad=True)
    labels = torch.tensor([[-100, 2, 1, 4]])
    mask = torch.tensor([[False, True, False, True]])

    def loss_function(candidate_logits: torch.Tensor) -> torch.Tensor:
        return masked_causal_cross_entropy(
            candidate_logits,
            labels,
            mask,
            max_tokens_per_chunk=1,
            loss_compute_dtype=torch.float64,
        ).loss

    assert torch.autograd.gradcheck(
        loss_function,
        (logits,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_chunked_linear_path_matches_materialized_logits_value_and_gradients() -> None:
    generator = torch.Generator().manual_seed(101)
    hidden = torch.randn((2, 6, 4), dtype=torch.float64, generator=generator)
    weight = torch.randn((9, 4), dtype=torch.float64, generator=generator)
    bias = torch.randn((9,), dtype=torch.float64, generator=generator)
    labels = torch.randint(9, (2, 6), generator=generator)
    mask = torch.tensor(
        [
            [False, True, True, False, True, False],
            [False, False, True, True, True, True],
        ]
    )

    actual_hidden = hidden.clone().requires_grad_(True)
    actual_weight = weight.clone().requires_grad_(True)
    actual_bias = bias.clone().requires_grad_(True)
    actual = chunked_masked_causal_linear_cross_entropy(
        actual_hidden,
        actual_weight,
        labels,
        mask,
        lm_head_bias=actual_bias,
        max_tokens_per_chunk=2,
        loss_compute_dtype=torch.float64,
    )
    actual_gradients = torch.autograd.grad(actual.loss, (actual_hidden, actual_weight, actual_bias))

    expected_hidden = hidden.clone().requires_grad_(True)
    expected_weight = weight.clone().requires_grad_(True)
    expected_bias = bias.clone().requires_grad_(True)
    materialized_logits = F.linear(expected_hidden, expected_weight, expected_bias)
    expected_loss = independent_masked_ce(materialized_logits, labels, mask)
    expected_gradients = torch.autograd.grad(
        expected_loss, (expected_hidden, expected_weight, expected_bias)
    )

    torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_linear_path_never_projects_more_than_configured_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import posttrain_lab.train.masked_ce as masked_ce_module

    hidden = torch.randn((2, 5, 3))
    weight = torch.randn((7, 3))
    labels = torch.randint(7, (2, 5))
    mask = torch.tensor([[False, True, True, True, True], [False, True, True, True, False]])
    projected_token_counts = []
    original_linear = F.linear

    def recording_linear(
        input_tensor: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_token_counts.append(input_tensor.shape[0])
        return original_linear(input_tensor, linear_weight, linear_bias)

    monkeypatch.setattr(masked_ce_module.F, "linear", recording_linear)
    chunked_masked_causal_linear_cross_entropy(
        hidden,
        weight,
        labels,
        mask,
        max_tokens_per_chunk=3,
    )
    assert projected_token_counts == [3, 3, 1]


def test_linear_path_does_not_save_vocabulary_logits_for_backward() -> None:
    hidden = torch.randn((2, 5, 4), requires_grad=True)
    weight = torch.randn((13, 4), requires_grad=True)
    bias = torch.randn((13,), requires_grad=True)
    labels = torch.randint(13, (2, 5))
    mask = torch.tensor([[False, True, True, True, True], [False, True, True, True, False]])
    saved_shapes: list[tuple[int, ...]] = []

    def record_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
        output = chunked_masked_causal_linear_cross_entropy(
            hidden,
            weight,
            labels,
            mask,
            lm_head_bias=bias,
            max_tokens_per_chunk=3,
        )

    assert not any(len(shape) == 2 and shape[-1] == 13 for shape in saved_shapes)
    output.loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_bfloat16_logits_use_float32_loss_accumulation() -> None:
    logits = torch.randn((2, 4, 11), dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(11, (2, 4))
    mask = torch.tensor([[False, True, True, False], [False, True, False, True]])
    output = masked_causal_cross_entropy(logits, labels, mask)
    assert output.loss.dtype == torch.float32
    assert output.loss_sum.dtype == torch.float32
    assert math.isfinite(float(output.loss.detach()))
    output.loss.backward()
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16


def test_extreme_bfloat16_logits_keep_cross_entropy_finite() -> None:
    logits = torch.tensor(
        [[[10_000.0, 0.0, -10_000.0], [-10_000.0, 0.0, 10_000.0], [0.0, 0.0, 0.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 2, 0]])
    mask = torch.tensor([[False, True, True]])
    output = masked_causal_cross_entropy(logits, labels, mask)
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cpu_autocast_supports_bfloat16_hidden_with_float32_lm_head() -> None:
    hidden = torch.randn((1, 4, 3), dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((5, 3), dtype=torch.float32, requires_grad=True)
    bias = torch.randn((5,), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3]])
    mask = torch.tensor([[False, True, True, False]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = chunked_masked_causal_linear_cross_entropy(
            hidden,
            weight,
            labels,
            mask,
            lm_head_bias=bias,
            max_tokens_per_chunk=1,
        )
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert hidden.grad is not None and hidden.grad.dtype == torch.bfloat16
    assert weight.grad is not None and weight.grad.dtype == torch.float32
    assert bias.grad is not None and bias.grad.dtype == torch.float32


def test_mixed_linear_dtypes_require_active_autocast() -> None:
    with pytest.raises(MaskValidationError, match="autocast"):
        chunked_masked_causal_linear_cross_entropy(
            torch.randn((1, 3, 2), dtype=torch.bfloat16),
            torch.randn((4, 2), dtype=torch.float32),
            torch.tensor([[-100, 1, 2]]),
            torch.tensor([[False, True, False]]),
        )


def test_linear_path_rejects_zero_hidden_dimension() -> None:
    with pytest.raises(MaskValidationError, match="hidden dimension"):
        chunked_masked_causal_linear_cross_entropy(
            torch.empty((1, 3, 0)),
            torch.empty((4, 0)),
            torch.tensor([[-100, 1, 2]]),
            torch.tensor([[False, True, False]]),
        )


def test_single_process_gradient_accumulation_reconstructs_global_token_mean() -> None:
    generator = torch.Generator().manual_seed(211)
    hidden = torch.randn((2, 5, 3), dtype=torch.float64, generator=generator)
    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, 2, 1, 0, 3]])
    mask = torch.tensor([[False, True, False, False, False], [False, True, True, True, False]])
    initial_weight = torch.randn((5, 3), dtype=torch.float64, generator=generator)
    global_token_count = int(mask.sum().item())

    full_weight = initial_weight.clone().requires_grad_(True)
    full_loss = masked_causal_cross_entropy(F.linear(hidden, full_weight), labels, mask).loss
    expected_gradient = torch.autograd.grad(full_loss, full_weight)[0]

    accumulated_weight = initial_weight.clone().requires_grad_(True)
    microbatch_losses = []
    for row in range(hidden.shape[0]):
        microbatch_losses.append(
            chunked_masked_causal_linear_cross_entropy(
                hidden[row : row + 1],
                accumulated_weight,
                labels[row : row + 1],
                mask[row : row + 1],
                global_token_count=global_token_count,
                ddp_world_size=1,
            ).loss
        )
    actual_gradient = torch.autograd.grad(sum(microbatch_losses), accumulated_weight)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_ddp_scaling_reconstructs_global_token_mean_gradient() -> None:
    generator = torch.Generator().manual_seed(303)
    rank_hidden = [
        torch.randn((1, 5, 3), dtype=torch.float64, generator=generator),
        torch.randn((1, 5, 3), dtype=torch.float64, generator=generator),
    ]
    rank_labels = [torch.tensor([[-100, 1, 2, 3, 4]]), torch.tensor([[-100, 2, 1, 0, 3]])]
    rank_masks = [
        torch.tensor([[False, True, False, False, False]]),
        torch.tensor([[False, True, True, True, False]]),
    ]
    initial_weight = torch.randn((5, 3), dtype=torch.float64, generator=generator)
    global_token_count = sum(int(mask.sum().item()) for mask in rank_masks)

    full_weight = initial_weight.clone().requires_grad_(True)
    full_logits = F.linear(torch.cat(rank_hidden), full_weight)
    full_loss = masked_causal_cross_entropy(
        full_logits,
        torch.cat(rank_labels),
        torch.cat(rank_masks),
        loss_compute_dtype=torch.float64,
    ).loss
    expected_gradient = torch.autograd.grad(full_loss, full_weight)[0]

    rank_gradients = []
    for hidden, labels, mask in zip(rank_hidden, rank_labels, rank_masks, strict=True):
        rank_weight = initial_weight.clone().requires_grad_(True)
        rank_loss = chunked_masked_causal_linear_cross_entropy(
            hidden,
            rank_weight,
            labels,
            mask,
            max_tokens_per_chunk=2,
            loss_compute_dtype=torch.float64,
            global_token_count=global_token_count,
            ddp_world_size=2,
        ).loss
        rank_gradients.append(torch.autograd.grad(rank_loss, rank_weight)[0])

    ddp_averaged_gradient = torch.stack(rank_gradients).mean(dim=0)
    torch.testing.assert_close(ddp_averaged_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_ddp_allows_empty_local_shard_when_global_count_is_positive() -> None:
    hidden = torch.randn((1, 4, 3), dtype=torch.float64, requires_grad=True)
    weight = torch.randn((5, 3), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3]])
    empty_mask = torch.zeros((1, 4), dtype=torch.bool)
    output = chunked_masked_causal_linear_cross_entropy(
        hidden,
        weight,
        labels,
        empty_mask,
        global_token_count=3,
        ddp_world_size=2,
    )
    assert output.local_token_count == 0
    assert output.loss.item() == 0.0
    output.loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None
    assert torch.count_nonzero(hidden.grad).item() == 0
    assert torch.count_nonzero(weight.grad).item() == 0


def test_d01_mask_and_budget_feed_d02_without_recounting() -> None:
    labels = torch.tensor([[-100, 3, 4, 2, -100], [-100, 5, 6, 7, 2]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    objective_mask = torch_assistant_target_loss_mask(
        labels,
        attention_mask=attention,
        eos_token_id=2,
    )
    selection = plan_torch_loss_budget(
        LossTokenBudget(5),
        objective_mask,
        sample_ids=["sample-a", "sample-b"],
        generation_indices=[0, 0],
    )
    logits = torch.randn((2, 5, 8), requires_grad=True)
    output = masked_causal_cross_entropy(logits, labels, selection.loss_mask)
    assert output.local_token_count == selection.selected_tokens == 5
    output.loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    ("labels", "mask", "message"),
    [
        (
            torch.tensor([[-100, -100, 1]]),
            torch.tensor([[False, True, False]]),
            "ignore_index",
        ),
        (
            torch.tensor([[-100, 8, 1]]),
            torch.tensor([[False, True, False]]),
            "selected labels",
        ),
        (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[True, False, False]]),
            "position 0",
        ),
    ],
)
def test_invalid_selected_targets_fail_loudly(
    labels: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(MaskValidationError, match=message):
        masked_causal_cross_entropy(torch.randn((1, 3, 5)), labels, mask)


def test_empty_local_mask_requires_explicit_distributed_normalization() -> None:
    logits = torch.randn((1, 3, 5), requires_grad=True)
    labels = torch.tensor([[-100, 1, 2]])
    mask = torch.zeros((1, 3), dtype=torch.bool)
    with pytest.raises(MaskValidationError, match="selects no"):
        masked_causal_cross_entropy(logits, labels, mask)


def test_normalization_metadata_must_be_complete_and_consistent() -> None:
    logits = torch.randn((1, 3, 5))
    labels = torch.tensor([[-100, 1, 2]])
    mask = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="provided together"):
        masked_causal_cross_entropy(logits, labels, mask, global_token_count=1)
    two_token_mask = torch.tensor([[False, True, True]])
    with pytest.raises(ValueError, match="smaller"):
        masked_causal_cross_entropy(
            logits,
            labels,
            two_token_mask,
            global_token_count=1,
            ddp_world_size=1,
        )


def test_unselected_sentinel_labels_never_reach_cross_entropy() -> None:
    logits = torch.randn((1, 4, 5), requires_grad=True)
    labels = torch.tensor([[-999, 3, 999, -777]])
    mask = torch.tensor([[False, True, False, False]])
    output = masked_causal_cross_entropy(logits, labels, mask)
    assert output.local_token_count == 1
    assert torch.isfinite(output.loss)
