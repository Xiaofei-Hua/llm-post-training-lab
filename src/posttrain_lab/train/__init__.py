"""Training-contract primitives shared by SFT, GRPO, and OPD."""

from .loss_budget import (
    BudgetReservation,
    BudgetStateError,
    BudgetStepRecord,
    LossTokenBudget,
    MaskValidationError,
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
    "TorchBudgetSelection",
    "TorchGroupMaskResult",
    "commit_torch_loss_budget",
    "plan_torch_loss_budget",
    "torch_assistant_target_loss_mask",
    "torch_completion_loss_mask",
    "torch_exclude_zero_variance_grpo_groups",
    "torch_intersect_masks",
]
