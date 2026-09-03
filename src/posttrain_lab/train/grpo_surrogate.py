"""Exact-reward Dr.GRPO advantage and token-surrogate primitives.

The module consumes already aligned per-token log probabilities. Group
advantages must be computed on complete global prompt groups before data-
parallel or gradient-accumulation slicing. The loss mask is then the exact D01
budget mask for positions that really enter backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .loss_budget import MaskValidationError
from .torch_loss_budget import torch_exclude_zero_variance_grpo_groups

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@dataclass(frozen=True)
class ExactGroupAdvantageOutput:
    """Globally computed exact-reward group statistics and filtered mask."""

    advantages: Tensor
    loss_mask: Tensor
    group_ids: Tensor
    group_means: Tensor
    group_sample_stds: Tensor
    active_row_mask: Tensor
    active_group_ids: Tensor
    skipped_group_ids: Tensor
    expected_group_size: int
    std_epsilon: float

    @property
    def total_group_count(self) -> int:
        return self.group_ids.shape[0]

    @property
    def active_group_count(self) -> int:
        return self.active_group_ids.shape[0]

    @property
    def active_completion_count(self) -> int:
        return int(self.active_row_mask.sum(dtype=torch.int64).item())

    @property
    def total_completion_count(self) -> int:
        return self.active_row_mask.shape[0]

    @property
    def skipped_completion_count(self) -> int:
        return self.total_completion_count - self.active_completion_count

    @property
    def effective_group_rate(self) -> float:
        return self.active_group_count / self.total_group_count


@dataclass(frozen=True)
class DrGRPOSurrogateOutput:
    """Loss plus aggregation-safe local diagnostics for one update shard."""

    loss: Tensor
    loss_sum: Tensor
    local_token_count: int
    local_completion_count: int
    global_active_completion_count: int
    max_completion_length: int
    group_size: int
    ddp_world_size: int
    importance_ratio_sum: Tensor
    importance_ratio_min: Tensor
    importance_ratio_max: Tensor
    low_clipped_token_count: int
    high_clipped_token_count: int

    @property
    def normalization_denominator(self) -> int:
        return self.global_active_completion_count * self.max_completion_length

    @property
    def clipped_token_count(self) -> int:
        return self.low_clipped_token_count + self.high_clipped_token_count

    @property
    def local_clip_fraction(self) -> float:
        if self.local_token_count == 0:
            return math.nan
        return self.clipped_token_count / self.local_token_count

    @property
    def distributed_normalization(self) -> bool:
        return self.ddp_world_size > 1


def _require_float_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise MaskValidationError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise MaskValidationError(
            f"{name} must use float16, bfloat16, float32, or float64; got {value.dtype}"
        )


def _require_integer_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise MaskValidationError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _INTEGER_DTYPES:
        raise MaskValidationError(f"{name} must have an integer dtype, got {value.dtype}")


def _require_bool_mask(
    value: Tensor,
    *,
    name: str,
    shape: torch.Size | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise MaskValidationError(f"{name} must be a rank-2 torch.Tensor")
    if value.dtype != torch.bool:
        raise MaskValidationError(f"{name} must have Boolean dtype")
    if shape is not None and value.shape != shape:
        raise MaskValidationError(
            f"{name} shape {tuple(value.shape)} does not match {tuple(shape)}"
        )
    if device is not None and value.device != device:
        raise MaskValidationError(f"{name} must be on device {device}")


def _positive_integer(value: int | Tensor, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} tensor must be a scalar integer")
        normalized = int(value.detach().to(device="cpu", dtype=torch.int64).item())
    elif isinstance(value, int) and not isinstance(value, bool):
        normalized = value
    else:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return normalized


def _finite_fraction(value: float, *, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in ({lower}, {upper})")
    normalized = float(value)
    if not math.isfinite(normalized) or not lower < normalized < upper:
        raise ValueError(f"{name} must be a finite number in ({lower}, {upper})")
    return normalized


def _loss_compute_dtype(source_dtype: torch.dtype, requested: torch.dtype | None) -> torch.dtype:
    if requested is None:
        return torch.float64 if source_dtype == torch.float64 else torch.float32
    if requested not in {torch.float32, torch.float64}:
        raise ValueError("loss_compute_dtype must be torch.float32 or torch.float64")
    return requested


def compute_exact_group_advantages(
    rewards: Tensor,
    group_ids: Tensor,
    candidate_loss_mask: Tensor,
    *,
    expected_group_size: int = 8,
    std_epsilon: float = 1e-4,
) -> ExactGroupAdvantageOutput:
    """Compute group-scaled advantages for complete exact-reward groups.

    Rewards are restricted to the project's frozen verifier contract: every
    completion is scored exactly zero or one. Sample standard deviation
    (Bessel correction) matches the audited TRL group-scaling semantics.
    Groups with identical rewards receive zero advantage and an all-false loss
    mask; they are reported but never silently resampled.
    """

    _require_bool_mask(candidate_loss_mask, name="candidate_loss_mask")
    batch_size, sequence_length = candidate_loss_mask.shape
    if batch_size == 0 or sequence_length == 0:
        raise MaskValidationError("candidate_loss_mask dimensions must be non-zero")
    if bool(candidate_loss_mask.sum(dim=1).eq(0).any().item()):
        raise MaskValidationError("every generated completion must contain a candidate loss token")

    _require_integer_tensor(group_ids, name="group_ids", ndim=1)
    if group_ids.shape != (batch_size,):
        raise MaskValidationError("group_ids must contain one value per completion")
    if group_ids.device != candidate_loss_mask.device:
        raise MaskValidationError("group_ids and candidate_loss_mask must share a device")
    if bool(group_ids.lt(0).any().item()):
        raise MaskValidationError("group_ids must be non-negative")

    if not isinstance(rewards, Tensor) or rewards.ndim != 1:
        raise MaskValidationError("rewards must be a rank-1 torch.Tensor")
    if rewards.shape != (batch_size,):
        raise MaskValidationError("rewards must contain one value per completion")
    if rewards.device != candidate_loss_mask.device:
        raise MaskValidationError("rewards and candidate_loss_mask must share a device")
    if rewards.requires_grad:
        raise MaskValidationError("rewards must be stop-gradient")
    if rewards.dtype == torch.bool or not (
        rewards.dtype in _INTEGER_DTYPES or rewards.dtype in _SUPPORTED_FLOAT_DTYPES
    ):
        raise MaskValidationError("rewards must have a supported real numeric dtype")
    if rewards.dtype.is_floating_point and not bool(torch.isfinite(rewards).all().item()):
        raise MaskValidationError("rewards must be finite")
    if not bool(((rewards == 0) | (rewards == 1)).all().item()):
        raise MaskValidationError("exact verifier rewards must contain only 0 or 1")

    group_size = _positive_integer(expected_group_size, name="expected_group_size", minimum=2)
    epsilon = _finite_fraction(std_epsilon, name="std_epsilon", lower=0.0, upper=1.0)
    unique_group_ids, inverse, counts = torch.unique(
        group_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    if bool(counts.ne(group_size).any().item()):
        raise MaskValidationError(
            f"every global reward group must contain exactly {group_size} completions"
        )

    compute_dtype = torch.float64 if rewards.dtype == torch.float64 else torch.float32
    rewards_for_statistics = rewards.to(dtype=compute_dtype)
    group_sums = torch.zeros(
        unique_group_ids.shape,
        dtype=compute_dtype,
        device=rewards.device,
    )
    group_sums.scatter_add_(0, inverse, rewards_for_statistics)
    group_means = group_sums / counts.to(dtype=compute_dtype)
    centered_rewards = rewards_for_statistics - group_means[inverse]
    group_squared_deviation_sums = torch.zeros_like(group_sums)
    group_squared_deviation_sums.scatter_add_(0, inverse, centered_rewards.square())
    group_sample_stds = torch.sqrt(
        group_squared_deviation_sums / (counts.to(dtype=compute_dtype) - 1)
    )

    filtered = torch_exclude_zero_variance_grpo_groups(
        candidate_loss_mask,
        group_ids=group_ids,
        rewards=rewards,
    )
    advantages = centered_rewards / (group_sample_stds[inverse] + epsilon)
    advantages = torch.where(filtered.active_row_mask, advantages, torch.zeros_like(advantages))
    if not bool(torch.isfinite(advantages).all().item()):
        raise FloatingPointError("group advantage computation produced a non-finite value")

    return ExactGroupAdvantageOutput(
        advantages=advantages,
        loss_mask=filtered.masks,
        group_ids=unique_group_ids,
        group_means=group_means,
        group_sample_stds=group_sample_stds,
        active_row_mask=filtered.active_row_mask,
        active_group_ids=filtered.active_group_ids,
        skipped_group_ids=filtered.skipped_group_ids,
        expected_group_size=group_size,
        std_epsilon=epsilon,
    )


def dr_grpo_token_surrogate(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    loss_mask: Tensor,
    *,
    global_active_completion_count: int | Tensor,
    max_completion_length: int,
    group_size: int = 8,
    epsilon: float = 0.2,
    ddp_world_size: int = 1,
    loss_compute_dtype: torch.dtype | None = None,
) -> DrGRPOSurrogateOutput:
    """Compute the token-IS, clipped Dr.GRPO loss for one local shard.

    ``current_log_probs`` and ``old_log_probs`` are aligned to target-token
    positions already; this function does not perform a causal shift. The
    caller supplies the number of globally active completions before D01's
    final token-budget truncation. The returned loss is scaled so summing
    microbatch gradients and applying default DDP averaging reconstructs the
    exact global Dr.GRPO objective.
    """

    _require_float_tensor(current_log_probs, name="current_log_probs", ndim=2)
    _require_float_tensor(old_log_probs, name="old_log_probs", ndim=2)
    if current_log_probs.shape != old_log_probs.shape:
        raise MaskValidationError("current_log_probs and old_log_probs must have the same shape")
    if current_log_probs.device != old_log_probs.device:
        raise MaskValidationError("current_log_probs and old_log_probs must share a device")
    if old_log_probs.requires_grad:
        raise MaskValidationError("old_log_probs must be stop-gradient")

    batch_size, sequence_length = current_log_probs.shape
    if batch_size == 0 or sequence_length == 0:
        raise MaskValidationError("log-probability dimensions must be non-zero")
    _require_bool_mask(
        loss_mask,
        name="loss_mask",
        shape=current_log_probs.shape,
        device=current_log_probs.device,
    )

    _require_float_tensor(advantages, name="advantages", ndim=1)
    if advantages.shape != (batch_size,):
        raise MaskValidationError("advantages must contain one value per completion")
    if advantages.device != current_log_probs.device:
        raise MaskValidationError("advantages and log probabilities must share a device")
    if advantages.requires_grad:
        raise MaskValidationError("advantages must be stop-gradient")
    if not bool(torch.isfinite(advantages).all().item()):
        raise MaskValidationError("advantages must be finite")

    normalized_group_size = _positive_integer(group_size, name="group_size", minimum=2)
    global_completion_count = _positive_integer(
        global_active_completion_count,
        name="global_active_completion_count",
    )
    if global_completion_count % normalized_group_size != 0:
        raise ValueError("global_active_completion_count must be divisible by group_size")
    completion_length = _positive_integer(
        max_completion_length,
        name="max_completion_length",
    )
    world_size = _positive_integer(ddp_world_size, name="ddp_world_size")
    clip_epsilon = _finite_fraction(epsilon, name="epsilon", lower=0.0, upper=1.0)
    compute_dtype = _loss_compute_dtype(current_log_probs.dtype, loss_compute_dtype)

    per_row_token_count = loss_mask.sum(dim=1, dtype=torch.int64)
    if bool(per_row_token_count.gt(completion_length).any().item()):
        raise MaskValidationError("a completion selects more tokens than max_completion_length")
    local_completion_count = int(per_row_token_count.gt(0).sum(dtype=torch.int64).item())
    if local_completion_count > global_completion_count:
        raise ValueError(
            "global_active_completion_count cannot be smaller than contributing local completions"
        )

    selected_positions = loss_mask.nonzero(as_tuple=False)
    local_token_count = selected_positions.shape[0]
    if local_token_count == 0:
        anchor = current_log_probs[(0, 0)]
        finite_anchor = torch.where(torch.isfinite(anchor), anchor, torch.zeros_like(anchor))
        loss_sum = finite_anchor.to(dtype=compute_dtype) * 0.0
        zero = torch.zeros((), dtype=compute_dtype, device=current_log_probs.device)
        ratio_sum = zero
        ratio_min = torch.full_like(zero, torch.nan)
        ratio_max = torch.full_like(zero, torch.nan)
        low_clipped_count = 0
        high_clipped_count = 0
    else:
        batch_indices = selected_positions[:, 0]
        token_indices = selected_positions[:, 1]
        selected_current = current_log_probs[batch_indices, token_indices].to(dtype=compute_dtype)
        selected_old = old_log_probs[batch_indices, token_indices].to(dtype=compute_dtype)
        selected_advantages = advantages[batch_indices].to(dtype=compute_dtype)
        if not bool(torch.isfinite(selected_current).all().item()):
            raise FloatingPointError("selected current_log_probs must be finite")
        if not bool(torch.isfinite(selected_old).all().item()):
            raise FloatingPointError("selected old_log_probs must be finite")
        if bool((selected_current > 0).any().item()) or bool((selected_old > 0).any().item()):
            raise MaskValidationError("selected log probabilities cannot be positive")
        if bool(selected_advantages.eq(0).any().item()):
            raise MaskValidationError(
                "loss_mask selects a zero-advantage completion; filter zero-variance groups first"
            )

        log_importance_ratio = selected_current - selected_old
        importance_ratio = torch.exp(log_importance_ratio)
        if not bool((torch.isfinite(importance_ratio) & importance_ratio.gt(0)).all().item()):
            raise FloatingPointError("importance ratios must be finite and strictly positive")
        clipped_ratio = torch.clamp(
            importance_ratio,
            min=1.0 - clip_epsilon,
            max=1.0 + clip_epsilon,
        )
        unclipped_objective = importance_ratio * selected_advantages
        clipped_objective = clipped_ratio * selected_advantages
        per_token_loss = -torch.minimum(unclipped_objective, clipped_objective)
        if not bool(torch.isfinite(per_token_loss).all().item()):
            raise FloatingPointError("Dr.GRPO token loss produced a non-finite value")
        loss_sum = per_token_loss.sum()

        low_clipped = (importance_ratio < 1.0 - clip_epsilon) & selected_advantages.lt(0)
        high_clipped = (importance_ratio > 1.0 + clip_epsilon) & selected_advantages.gt(0)
        low_clipped_count = int(low_clipped.sum(dtype=torch.int64).item())
        high_clipped_count = int(high_clipped.sum(dtype=torch.int64).item())
        detached_ratio = importance_ratio.detach()
        ratio_sum = detached_ratio.sum()
        ratio_min = detached_ratio.min()
        ratio_max = detached_ratio.max()

    denominator = global_completion_count * completion_length
    loss = loss_sum * world_size / denominator
    return DrGRPOSurrogateOutput(
        loss=loss,
        loss_sum=loss_sum,
        local_token_count=local_token_count,
        local_completion_count=local_completion_count,
        global_active_completion_count=global_completion_count,
        max_completion_length=completion_length,
        group_size=normalized_group_size,
        ddp_world_size=world_size,
        importance_ratio_sum=ratio_sum,
        importance_ratio_min=ratio_min,
        importance_ratio_max=ratio_max,
        low_clipped_token_count=low_clipped_count,
        high_clipped_token_count=high_clipped_count,
    )
