"""Training-contract primitives shared by SFT, GRPO, and OPD."""

from .loss_budget import (
    BudgetReservation,
    BudgetStateError,
    BudgetStepRecord,
    LossTokenBudget,
    MaskValidationError,
)
from .masked_ce import (
    MaskedCrossEntropyOutput,
    chunked_masked_causal_linear_cross_entropy,
    masked_causal_cross_entropy,
)
from .torch_loss_budget import (
    TorchBudgetSelection,
    TorchGroupMaskResult,
    commit_torch_loss_budget,
    plan_torch_loss_budget,
    torch_assistant_target_loss_mask,
    torch_completion_loss_mask,
    torch_exclude_zero_variance_grpo_groups,
    torch_intersect_masks,
)

__all__ = [
    "BudgetReservation",
    "BudgetStateError",
    "BudgetStepRecord",
    "LossTokenBudget",
    "MaskValidationError",
    "MaskedCrossEntropyOutput",
    "TorchBudgetSelection",
    "TorchGroupMaskResult",
    "chunked_masked_causal_linear_cross_entropy",
    "commit_torch_loss_budget",
    "masked_causal_cross_entropy",
    "plan_torch_loss_budget",
    "torch_assistant_target_loss_mask",
    "torch_completion_loss_mask",
    "torch_exclude_zero_variance_grpo_groups",
    "torch_intersect_masks",
]
