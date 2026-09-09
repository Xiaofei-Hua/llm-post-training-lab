"""Training-contract primitives shared by SFT, GRPO, and OPD."""

from .grpo_surrogate import (
    DrGRPOSurrogateOutput,
    ExactGroupAdvantageOutput,
    compute_exact_group_advantages,
    dr_grpo_token_surrogate,
)
from .loop import StageBudgetIncomplete, Trainer, TrainerConfig
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
from .model_losses import causal_lm_opd_loss, causal_lm_sft_loss
from .opd_reverse_kl import (
    MaskedReverseKLOutput,
    chunked_masked_causal_linear_reverse_kl,
    masked_causal_reverse_kl,
)
from .rollout import RolloutConfig, TrainingBatch, TrainingExample, sample_completions
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
    "DrGRPOSurrogateOutput",
    "ExactGroupAdvantageOutput",
    "LossTokenBudget",
    "MaskValidationError",
    "MaskedCrossEntropyOutput",
    "MaskedReverseKLOutput",
    "RolloutConfig",
    "StageBudgetIncomplete",
    "TorchBudgetSelection",
    "TorchGroupMaskResult",
    "Trainer",
    "TrainerConfig",
    "TrainingBatch",
    "TrainingExample",
    "causal_lm_opd_loss",
    "causal_lm_sft_loss",
    "chunked_masked_causal_linear_cross_entropy",
    "chunked_masked_causal_linear_reverse_kl",
    "commit_torch_loss_budget",
    "compute_exact_group_advantages",
    "dr_grpo_token_surrogate",
    "masked_causal_cross_entropy",
    "masked_causal_reverse_kl",
    "plan_torch_loss_budget",
    "sample_completions",
    "torch_assistant_target_loss_mask",
    "torch_completion_loss_mask",
    "torch_exclude_zero_variance_grpo_groups",
    "torch_intersect_masks",
]
