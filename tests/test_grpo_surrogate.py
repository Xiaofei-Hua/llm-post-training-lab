from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from posttrain_lab.train import (
    LossTokenBudget,
    MaskValidationError,
    compute_exact_group_advantages,
    dr_grpo_token_surrogate,
    plan_torch_loss_budget,
    torch_completion_loss_mask,
)


def independent_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    std_epsilon: float,
) -> torch.Tensor:
    result = torch.zeros_like(rewards, dtype=torch.float64)
    for group_id in sorted(set(group_ids.tolist())):
        indices = [index for index, value in enumerate(group_ids.tolist()) if value == group_id]
        values = [float(rewards[index]) for index in indices]
        mean = sum(values) / len(values)
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if sample_std == 0:
            continue
        for index in indices:
            result[index] = (float(rewards[index]) - mean) / (sample_std + std_epsilon)
    return result


def independent_dr_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    epsilon: float,
    denominator: int,
) -> torch.Tensor:
    ratios = torch.exp(current_log_probs - old_log_probs)
    per_token_advantages = advantages.unsqueeze(1)
    unclipped = ratios * per_token_advantages
    clipped = ratios.clamp(1 - epsilon, 1 + epsilon) * per_token_advantages
    per_token_loss = -torch.minimum(unclipped, clipped)
    return per_token_loss[loss_mask].sum() / denominator


def test_exact_group_advantages_use_sample_std_and_skip_zero_variance_groups() -> None:
    group_ids = torch.tensor([20, 10, 20, 10, 20, 10, 20, 10])
    rewards = torch.tensor([0, 0, 1, 0, 1, 0, 0, 0])
    candidate_mask = torch.ones((8, 3), dtype=torch.bool)
    output = compute_exact_group_advantages(
        rewards,
        group_ids,
        candidate_mask,
        expected_group_size=4,
    )

    expected = independent_group_advantages(rewards, group_ids, std_epsilon=1e-4)
    torch.testing.assert_close(output.advantages, expected.to(dtype=torch.float32))
    assert output.group_ids.tolist() == [10, 20]
    torch.testing.assert_close(output.group_means, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(
        output.group_sample_stds,
        torch.tensor([0.0, math.sqrt(1.0 / 3.0)]),
    )
    assert output.active_group_ids.tolist() == [20]
    assert output.skipped_group_ids.tolist() == [10]
    assert output.active_row_mask.tolist() == [True, False, True, False, True, False, True, False]
    assert not output.loss_mask[~output.active_row_mask].any()
    assert output.total_completion_count == 8
    assert output.active_completion_count == 4
    assert output.skipped_completion_count == 4
    assert output.effective_group_rate == 0.5


def test_default_contract_requires_groups_of_eight() -> None:
    output = compute_exact_group_advantages(
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        torch.zeros(8, dtype=torch.int64),
        torch.ones((8, 2), dtype=torch.bool),
    )
    assert output.expected_group_size == 8
    assert output.active_group_count == 1
    assert output.active_completion_count == 8


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    group_count=st.integers(min_value=1, max_value=4),
    group_size=st.integers(min_value=2, max_value=6),
    sequence_length=st.integers(min_value=1, max_value=8),
)
def test_generated_batches_match_independent_advantage_and_loss_formula(
    seed: int,
    group_count: int,
    group_size: int,
    sequence_length: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    batch_size = group_count * group_size
    unshuffled_group_ids = torch.arange(group_count).repeat_interleave(group_size)
    group_rewards = []
    for _ in range(group_count):
        tail = torch.randint(2, (group_size - 2,), generator=generator)
        group_rewards.append(torch.cat((torch.tensor([0, 1]), tail)))
    unshuffled_rewards = torch.cat(group_rewards)
    permutation = torch.randperm(batch_size, generator=generator)
    group_ids = unshuffled_group_ids[permutation]
    rewards = unshuffled_rewards[permutation]
    candidate_mask = torch.rand((batch_size, sequence_length), generator=generator).ge(0.5)
    candidate_mask[:, 0] = True

    advantage_output = compute_exact_group_advantages(
        rewards,
        group_ids,
        candidate_mask,
        expected_group_size=group_size,
    )
    expected_advantages = independent_group_advantages(
        rewards,
        group_ids,
        std_epsilon=1e-4,
    ).to(dtype=torch.float32)
    torch.testing.assert_close(advantage_output.advantages, expected_advantages)

    current_log_probs = -1.0 - torch.rand(
        (batch_size, sequence_length),
        dtype=torch.float64,
        generator=generator,
    )
    log_ratios = (
        torch.rand(
            (batch_size, sequence_length),
            dtype=torch.float64,
            generator=generator,
        )
        - 0.5
    ) * 0.6
    old_log_probs = current_log_probs - log_ratios
    actual = dr_grpo_token_surrogate(
        current_log_probs,
        old_log_probs,
        advantage_output.advantages,
        advantage_output.loss_mask,
        global_active_completion_count=batch_size,
        max_completion_length=sequence_length,
        group_size=group_size,
        loss_compute_dtype=torch.float64,
    )
    expected_loss = independent_dr_grpo_loss(
        current_log_probs,
        old_log_probs,
        advantage_output.advantages,
        candidate_mask,
        epsilon=0.2,
        denominator=batch_size * sequence_length,
    )
    torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-12, atol=1e-12)


def test_surrogate_value_and_gradient_match_independent_dense_formula() -> None:
    current = torch.tensor(
        [[-1.2, -0.8, -2.0], [-0.7, -1.1, -1.4], [-2.1, -0.9, -1.3], [-1.7, -0.6, -0.5]],
        dtype=torch.float64,
    )
    old = torch.tensor(
        [[-1.1, -0.9, -1.8], [-0.8, -1.0, -1.5], [-2.0, -1.0, -1.2], [-1.8, -0.7, -0.6]],
        dtype=torch.float64,
    )
    advantages = torch.tensor([0.8, -0.8, 1.2, -1.2], dtype=torch.float64)
    mask = torch.tensor(
        [[True, True, False], [True, False, False], [True, True, True], [False, True, True]]
    )

    actual_current = current.clone().requires_grad_(True)
    actual = dr_grpo_token_surrogate(
        actual_current,
        old,
        advantages,
        mask,
        global_active_completion_count=4,
        max_completion_length=3,
        group_size=2,
        loss_compute_dtype=torch.float64,
    )
    actual_gradient = torch.autograd.grad(actual.loss, actual_current)[0]

    expected_current = current.clone().requires_grad_(True)
    expected_loss = independent_dr_grpo_loss(
        expected_current,
        old,
        advantages,
        mask,
        epsilon=0.2,
        denominator=12,
    )
    expected_gradient = torch.autograd.grad(expected_loss, expected_current)[0]
    torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_surrogate_passes_gradcheck_away_from_clip_boundaries() -> None:
    current = torch.tensor(
        [[-1.05, -1.80], [-2.10, -0.70]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old = torch.tensor([[-1.10, -1.90], [-2.00, -0.80]], dtype=torch.float64)
    advantages = torch.tensor([0.75, -0.75], dtype=torch.float64)
    mask = torch.ones((2, 2), dtype=torch.bool)

    def loss_function(log_probs: torch.Tensor) -> torch.Tensor:
        return dr_grpo_token_surrogate(
            log_probs,
            old,
            advantages,
            mask,
            global_active_completion_count=2,
            max_completion_length=2,
            group_size=2,
            loss_compute_dtype=torch.float64,
        ).loss

    assert torch.autograd.gradcheck(loss_function, (current,))


def test_on_policy_loss_can_be_zero_while_signed_policy_gradient_is_nonzero() -> None:
    current = torch.tensor([[-0.4], [-0.7]], dtype=torch.float64, requires_grad=True)
    old = current.detach().clone()
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    mask = torch.ones((2, 1), dtype=torch.bool)
    output = dr_grpo_token_surrogate(
        current,
        old,
        advantages,
        mask,
        global_active_completion_count=2,
        max_completion_length=1,
        group_size=2,
        loss_compute_dtype=torch.float64,
    )
    gradient = torch.autograd.grad(output.loss, current)[0]
    assert output.loss.item() == pytest.approx(0.0)
    torch.testing.assert_close(gradient, torch.tensor([[-0.5], [0.5]], dtype=torch.float64))


def test_clipping_zeroes_only_the_policy_gradient_that_crosses_the_bound() -> None:
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5], dtype=torch.float64)
    old = torch.full((4, 1), -2.0, dtype=torch.float64)
    current = (old.squeeze(1) + ratios.log()).unsqueeze(1).requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    output = dr_grpo_token_surrogate(
        current,
        old,
        advantages,
        torch.ones((4, 1), dtype=torch.bool),
        global_active_completion_count=4,
        max_completion_length=1,
        group_size=2,
        epsilon=0.2,
        loss_compute_dtype=torch.float64,
    )
    gradient = torch.autograd.grad(output.loss, current)[0].squeeze(1)
    assert gradient[0].item() == pytest.approx(0.0)
    assert gradient[1].item() == pytest.approx(0.0)
    assert gradient[2].item() < 0
    assert gradient[3].item() > 0
    assert output.low_clipped_token_count == 1
    assert output.high_clipped_token_count == 1
    assert output.clipped_token_count == 2
    assert output.local_clip_fraction == 0.5
    torch.testing.assert_close(output.importance_ratio_min, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(output.importance_ratio_max, torch.tensor(1.5, dtype=torch.float64))


def test_dr_grpo_denominator_is_completion_count_times_frozen_cap() -> None:
    current = torch.full((2, 4), -1.0, dtype=torch.float64)
    old = current.clone()
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    output = dr_grpo_token_surrogate(
        current,
        old,
        advantages,
        mask,
        global_active_completion_count=2,
        max_completion_length=4,
        group_size=2,
    )
    assert output.local_token_count == 4
    assert output.normalization_denominator == 8
    assert output.loss_sum.item() == pytest.approx(2.0)
    assert output.loss.item() == pytest.approx(0.25)
    assert output.loss.item() != pytest.approx(output.loss_sum.item() / output.local_token_count)


def test_skipped_groups_do_not_enter_active_completion_denominator() -> None:
    candidate_mask = torch.tensor(
        [[True, True], [True, True], [True, False], [True, True]],
        dtype=torch.bool,
    )
    advantages = compute_exact_group_advantages(
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([10, 10, 20, 20]),
        candidate_mask,
        expected_group_size=2,
    )
    current = torch.full((4, 2), -1.0, dtype=torch.float64)
    output = dr_grpo_token_surrogate(
        current,
        current.clone(),
        advantages.advantages,
        advantages.loss_mask,
        global_active_completion_count=advantages.active_completion_count,
        max_completion_length=2,
        group_size=2,
        loss_compute_dtype=torch.float64,
    )

    assert advantages.total_completion_count == 4
    assert advantages.active_completion_count == 2
    assert output.local_completion_count == 2
    assert output.normalization_denominator == 4
    assert output.loss.item() == pytest.approx(output.loss_sum.item() / 4)
    assert output.loss.item() != pytest.approx(output.loss_sum.item() / 8)


def test_accumulation_and_ddp_scaling_reconstruct_full_update_gradient() -> None:
    features = torch.tensor(
        [[0.2, -0.4, 0.1], [0.3, 0.5, -0.2], [-0.6, 0.4, 0.7], [0.8, -0.1, 0.2]],
        dtype=torch.float64,
    )
    old = torch.full((4, 3), -3.0, dtype=torch.float64)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)
    mask = torch.tensor(
        [[True, False, False], [True, True, False], [True, True, True], [False, True, False]]
    )

    full_parameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    full_current = -3.0 + full_parameter * features
    full_loss = dr_grpo_token_surrogate(
        full_current,
        old,
        advantages,
        mask,
        global_active_completion_count=4,
        max_completion_length=3,
        group_size=2,
        loss_compute_dtype=torch.float64,
    ).loss
    expected_gradient = torch.autograd.grad(full_loss, full_parameter)[0]

    accumulated_parameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    accumulated_losses = []
    for start, end in ((0, 1), (1, 4)):
        accumulated_losses.append(
            dr_grpo_token_surrogate(
                -3.0 + accumulated_parameter * features[start:end],
                old[start:end],
                advantages[start:end],
                mask[start:end],
                global_active_completion_count=4,
                max_completion_length=3,
                group_size=2,
                loss_compute_dtype=torch.float64,
            ).loss
        )
    accumulated_gradient = torch.autograd.grad(sum(accumulated_losses), accumulated_parameter)[0]
    torch.testing.assert_close(accumulated_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    rank_gradients = []
    for start, end in ((0, 2), (2, 4)):
        rank_parameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        rank_loss = dr_grpo_token_surrogate(
            -3.0 + rank_parameter * features[start:end],
            old[start:end],
            advantages[start:end],
            mask[start:end],
            global_active_completion_count=4,
            max_completion_length=3,
            group_size=2,
            ddp_world_size=2,
            loss_compute_dtype=torch.float64,
        ).loss
        rank_gradients.append(torch.autograd.grad(rank_loss, rank_parameter)[0])
    ddp_averaged_gradient = torch.stack(rank_gradients).mean()
    torch.testing.assert_close(ddp_averaged_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_empty_local_shard_returns_differentiable_zero() -> None:
    current = torch.tensor([[torch.nan, torch.inf]], requires_grad=True)
    output = dr_grpo_token_surrogate(
        current,
        torch.tensor([[-torch.inf, torch.nan]]),
        torch.tensor([0.0]),
        torch.zeros((1, 2), dtype=torch.bool),
        global_active_completion_count=2,
        max_completion_length=2,
        group_size=2,
        ddp_world_size=2,
    )
    assert output.loss.item() == 0.0
    assert output.local_token_count == 0
    assert output.local_completion_count == 0
    assert math.isnan(output.local_clip_fraction)
    assert torch.isnan(output.importance_ratio_min)
    assert torch.isnan(output.importance_ratio_max)
    output.loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.count_nonzero(current.grad).item() == 0


def test_bfloat16_log_probs_use_float32_surrogate_math() -> None:
    current = torch.tensor([[-1.0], [-1.5]], dtype=torch.bfloat16, requires_grad=True)
    old = current.detach().clone()
    output = dr_grpo_token_surrogate(
        current,
        old,
        torch.tensor([1.0, -1.0]),
        torch.ones((2, 1), dtype=torch.bool),
        global_active_completion_count=2,
        max_completion_length=1,
        group_size=2,
    )
    assert output.loss.dtype == torch.float32
    assert output.loss_sum.dtype == torch.float32
    output.loss.backward()
    assert current.grad is not None and current.grad.dtype == torch.bfloat16


def test_d01_completion_mask_group_filter_and_budget_feed_d03() -> None:
    input_ids = torch.tensor(
        [
            [9, 4, 2, 0],
            [9, 5, 6, 2],
            [8, 3, 2, 0],
            [8, 7, 1, 2],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    candidate_mask = torch_completion_loss_mask(
        input_ids,
        completion_start=1,
        attention_mask=attention,
        eos_token_id=2,
    )
    advantage_output = compute_exact_group_advantages(
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([10, 10, 20, 20]),
        candidate_mask,
        expected_group_size=2,
    )
    selection = plan_torch_loss_budget(
        LossTokenBudget(5),
        advantage_output.loss_mask,
        sample_ids=["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
        generation_indices=[0, 1, 0, 1],
    )
    current = torch.full((4, 4), -1.0, requires_grad=True)
    output = dr_grpo_token_surrogate(
        current,
        current.detach().clone(),
        advantage_output.advantages,
        selection.loss_mask,
        global_active_completion_count=advantage_output.active_completion_count,
        max_completion_length=3,
        group_size=2,
    )
    assert output.local_token_count == selection.selected_tokens == 5
    output.loss.backward()
    assert current.grad is not None


@pytest.mark.parametrize(
    ("rewards", "group_ids", "mask", "message"),
    [
        (
            torch.tensor([0.0, 0.5]),
            torch.tensor([0, 0]),
            torch.ones((2, 1), dtype=torch.bool),
            "only 0 or 1",
        ),
        (
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.ones((3, 1), dtype=torch.bool),
            "exactly 2",
        ),
        (
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[True], [False]]),
            "every generated completion",
        ),
    ],
)
def test_advantage_contract_rejects_invalid_batches(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(MaskValidationError, match=message):
        compute_exact_group_advantages(
            rewards,
            group_ids,
            mask,
            expected_group_size=2,
        )


def test_rewards_and_surrogate_metadata_must_be_stop_gradient() -> None:
    with pytest.raises(MaskValidationError, match="rewards must be stop-gradient"):
        compute_exact_group_advantages(
            torch.tensor([0.0, 1.0], requires_grad=True),
            torch.tensor([0, 0]),
            torch.ones((2, 1), dtype=torch.bool),
            expected_group_size=2,
        )

    current = torch.full((2, 1), -1.0)
    common = {
        "loss_mask": torch.ones((2, 1), dtype=torch.bool),
        "global_active_completion_count": 2,
        "max_completion_length": 1,
        "group_size": 2,
    }
    with pytest.raises(MaskValidationError, match="old_log_probs must be stop-gradient"):
        dr_grpo_token_surrogate(
            current,
            current.clone().requires_grad_(True),
            torch.tensor([1.0, -1.0]),
            **common,
        )
    with pytest.raises(MaskValidationError, match="advantages must be stop-gradient"):
        dr_grpo_token_surrogate(
            current,
            current.clone(),
            torch.tensor([1.0, -1.0], requires_grad=True),
            **common,
        )


def test_zero_advantage_positions_cannot_enter_loss_budget() -> None:
    with pytest.raises(MaskValidationError, match="zero-advantage"):
        dr_grpo_token_surrogate(
            torch.full((2, 1), -1.0),
            torch.full((2, 1), -1.0),
            torch.tensor([0.0, 1.0]),
            torch.ones((2, 1), dtype=torch.bool),
            global_active_completion_count=2,
            max_completion_length=1,
            group_size=2,
        )


def test_unselected_nonfinite_log_probs_are_ignored_but_selected_values_fail() -> None:
    current = torch.tensor([[torch.nan, -1.0], [torch.inf, -1.2]])
    old = torch.tensor([[torch.nan, -1.0], [-torch.inf, -1.2]])
    mask = torch.tensor([[False, True], [False, True]])
    output = dr_grpo_token_surrogate(
        current,
        old,
        torch.tensor([1.0, -1.0]),
        mask,
        global_active_completion_count=2,
        max_completion_length=2,
        group_size=2,
    )
    assert torch.isfinite(output.loss)

    with pytest.raises(FloatingPointError, match="selected current"):
        dr_grpo_token_surrogate(
            current,
            old,
            torch.tensor([1.0, -1.0]),
            torch.tensor([[True, False], [False, True]]),
            global_active_completion_count=2,
            max_completion_length=2,
            group_size=2,
        )


def test_invalid_log_probability_and_normalization_contracts_fail_loudly() -> None:
    current = torch.full((2, 2), -1.0)
    old = current.clone()
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones((2, 2), dtype=torch.bool)
    with pytest.raises(MaskValidationError, match="cannot be positive"):
        dr_grpo_token_surrogate(
            -current,
            old,
            advantages,
            mask,
            global_active_completion_count=2,
            max_completion_length=2,
            group_size=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        dr_grpo_token_surrogate(
            current,
            old,
            advantages,
            mask,
            global_active_completion_count=3,
            max_completion_length=2,
            group_size=2,
        )
    with pytest.raises(MaskValidationError, match="more tokens"):
        dr_grpo_token_surrogate(
            current,
            old,
            advantages,
            mask,
            global_active_completion_count=2,
            max_completion_length=1,
            group_size=2,
        )


def test_importance_ratio_underflow_is_not_silently_clamped() -> None:
    with pytest.raises(FloatingPointError, match="strictly positive"):
        dr_grpo_token_surrogate(
            torch.tensor([[-1_000.0], [-1.0]]),
            torch.tensor([[0.0], [-1.0]]),
            torch.tensor([1.0, -1.0]),
            torch.ones((2, 1), dtype=torch.bool),
            global_active_completion_count=2,
            max_completion_length=1,
            group_size=2,
        )
