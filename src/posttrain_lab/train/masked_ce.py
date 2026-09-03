"""Production masked causal cross-entropy primitives for SFT.

Loss masks use absolute target-token positions. A target at position ``t`` is
predicted by the model output at position ``t - 1``; position zero can never be
selected. The project-wide reduction is a mean over all selected tokens, never
an equal-weight mean over sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .loss_budget import MaskValidationError

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
class MaskedCrossEntropyOutput:
    """Loss and accounting metadata for one local logical-update shard.

    ``loss`` is the scalar to backpropagate. Without explicit normalization it
    is the local token mean. Otherwise it is scaled so that summing gradient-
    accumulation shards and applying default DDP gradient averaging produces
    the exact global logical-update token mean.
    """

    loss: Tensor
    loss_sum: Tensor
    local_token_count: int
    normalization_token_count: int
    ddp_world_size: int

    @property
    def distributed_normalization(self) -> bool:
        return self.ddp_world_size > 1


@dataclass(frozen=True)
class _SelectedTargets:
    batch_indices: Tensor
    prediction_positions: Tensor
    targets: Tensor

    @property
    def count(self) -> int:
        return self.targets.shape[0]


def _validate_float_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise MaskValidationError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise MaskValidationError(
            f"{name} must use float16, bfloat16, float32, or float64; got {value.dtype}"
        )


def _validate_labels_and_mask(
    labels: Tensor,
    loss_mask: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    vocabulary_size: int,
    device: torch.device,
    ignore_index: int,
) -> _SelectedTargets:
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise MaskValidationError("labels must be a rank-2 torch.Tensor")
    if labels.shape != (batch_size, sequence_length):
        raise MaskValidationError(
            f"labels shape {tuple(labels.shape)} does not match {(batch_size, sequence_length)}"
        )
    if labels.device != device:
        raise MaskValidationError("labels must be on the same device as model activations")
    if labels.dtype not in _INTEGER_DTYPES:
        raise MaskValidationError(f"labels must have an integer dtype, got {labels.dtype}")
    if not isinstance(loss_mask, Tensor) or loss_mask.ndim != 2:
        raise MaskValidationError("loss_mask must be a rank-2 torch.Tensor")
    if loss_mask.shape != labels.shape:
        raise MaskValidationError("loss_mask and labels must have the same shape")
    if loss_mask.device != device:
        raise MaskValidationError("loss_mask must be on the same device as model activations")
    if loss_mask.dtype != torch.bool:
        raise MaskValidationError("loss_mask must have Boolean dtype")
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise MaskValidationError("ignore_index must be an integer")
    if batch_size == 0 or sequence_length == 0 or vocabulary_size == 0:
        raise MaskValidationError("batch, sequence, and vocabulary dimensions must be non-zero")
    if bool(loss_mask[:, 0].any().item()):
        raise MaskValidationError("loss_mask cannot select target position 0 in a causal LM")

    selected_positions = loss_mask[:, 1:].nonzero(as_tuple=False)
    batch_indices = selected_positions[:, 0]
    prediction_positions = selected_positions[:, 1]
    target_positions = prediction_positions + 1
    targets = labels[batch_indices, target_positions].to(dtype=torch.int64)
    if targets.numel() > 0:
        if bool(targets.eq(ignore_index).any().item()):
            raise MaskValidationError("loss_mask selects an ignore_index target")
        if bool(((targets < 0) | (targets >= vocabulary_size)).any().item()):
            raise MaskValidationError(f"selected labels must be in [0, {vocabulary_size - 1}]")
    return _SelectedTargets(
        batch_indices=batch_indices,
        prediction_positions=prediction_positions,
        targets=targets,
    )


def _positive_integer(value: int | Tensor, *, name: str) -> int:
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} tensor must be a scalar integer")
        normalized = int(value.detach().to(device="cpu", dtype=torch.int64).item())
    elif isinstance(value, int) and not isinstance(value, bool):
        normalized = value
    else:
        raise ValueError(f"{name} must be a positive integer")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _normalization_contract(
    local_token_count: int,
    *,
    global_token_count: int | Tensor | None,
    ddp_world_size: int | None,
) -> tuple[int, int]:
    if global_token_count is None and ddp_world_size is None:
        if local_token_count == 0:
            raise MaskValidationError("loss_mask selects no causal target tokens")
        return local_token_count, 1
    if global_token_count is None or ddp_world_size is None:
        raise ValueError("global_token_count and ddp_world_size must be provided together")

    normalized_global_count = _positive_integer(global_token_count, name="global_token_count")
    normalized_world_size = _positive_integer(ddp_world_size, name="ddp_world_size")
    if normalized_global_count < local_token_count:
        raise ValueError("global_token_count cannot be smaller than the local token count")
    return normalized_global_count, normalized_world_size


def _loss_compute_dtype(source_dtype: torch.dtype, requested: torch.dtype | None) -> torch.dtype:
    if requested is None:
        return torch.float64 if source_dtype == torch.float64 else torch.float32
    if requested not in {torch.float32, torch.float64}:
        raise ValueError("loss_compute_dtype must be torch.float32 or torch.float64")
    return requested


def _chunk_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_tokens_per_chunk must be a positive integer")
    return value


def _differentiable_zero(*values: Tensor, dtype: torch.dtype) -> Tensor:
    terms = [value[(0,) * value.ndim].to(dtype=dtype) * 0.0 for value in values]
    return torch.stack(terms).sum()


def _validate_linear_dtypes(
    hidden_states: Tensor,
    lm_head_weight: Tensor,
    lm_head_bias: Tensor | None,
) -> None:
    dtypes = {hidden_states.dtype, lm_head_weight.dtype}
    if lm_head_bias is not None:
        dtypes.add(lm_head_bias.dtype)
    if len(dtypes) == 1:
        return

    device_type = hidden_states.device.type
    if torch.is_autocast_enabled(device_type):
        autocast_dtype = torch.get_autocast_dtype(device_type)
        if dtypes <= {torch.float32, autocast_dtype}:
            return
    raise MaskValidationError(
        "hidden states, LM-head weight, and bias must share a dtype unless active "
        "autocast reconciles float32 with its configured lower-precision dtype"
    )


def _linear_cross_entropy_sum(
    hidden_states: Tensor,
    lm_head_weight: Tensor,
    lm_head_bias: Tensor | None,
    targets: Tensor,
    compute_dtype: torch.dtype,
) -> Tensor:
    logits = F.linear(hidden_states, lm_head_weight, lm_head_bias).to(dtype=compute_dtype)
    return F.cross_entropy(logits, targets, reduction="sum")


def _finalize_output(
    loss_sum: Tensor,
    *,
    local_token_count: int,
    global_token_count: int | Tensor | None,
    ddp_world_size: int | None,
) -> MaskedCrossEntropyOutput:
    normalization_count, world_size = _normalization_contract(
        local_token_count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )
    loss = loss_sum * world_size / normalization_count
    return MaskedCrossEntropyOutput(
        loss=loss,
        loss_sum=loss_sum,
        local_token_count=local_token_count,
        normalization_token_count=normalization_count,
        ddp_world_size=world_size,
    )


def masked_causal_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    *,
    ignore_index: int = -100,
    max_tokens_per_chunk: int = 128,
    loss_compute_dtype: torch.dtype | None = None,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedCrossEntropyOutput:
    """Compute token-mean causal CE from precomputed logits.

    Selected rows are gathered and processed in bounded chunks. Low-precision
    logits are converted to float32 for log-sum-exp unless float64 is requested
    explicitly. Unselected labels may contain ``ignore_index`` or other sentinel
    values; selected labels must be valid vocabulary IDs.
    """

    _validate_float_tensor(logits, name="logits", ndim=3)
    batch_size, sequence_length, vocabulary_size = logits.shape
    selected = _validate_labels_and_mask(
        labels,
        loss_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        device=logits.device,
        ignore_index=ignore_index,
    )
    chunk_size = _chunk_size(max_tokens_per_chunk)
    compute_dtype = _loss_compute_dtype(logits.dtype, loss_compute_dtype)

    if selected.count == 0:
        loss_sum = _differentiable_zero(logits, dtype=compute_dtype)
    else:
        chunk_sums = []
        for start in range(0, selected.count, chunk_size):
            end = min(start + chunk_size, selected.count)
            chunk_logits = logits[
                selected.batch_indices[start:end],
                selected.prediction_positions[start:end],
            ].to(dtype=compute_dtype)
            chunk_targets = selected.targets[start:end]
            chunk_sums.append(F.cross_entropy(chunk_logits, chunk_targets, reduction="sum"))
        loss_sum = torch.stack(chunk_sums).sum()

    return _finalize_output(
        loss_sum,
        local_token_count=selected.count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )


def chunked_masked_causal_linear_cross_entropy(
    hidden_states: Tensor,
    lm_head_weight: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    *,
    lm_head_bias: Tensor | None = None,
    ignore_index: int = -100,
    max_tokens_per_chunk: int = 128,
    loss_compute_dtype: torch.dtype | None = None,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedCrossEntropyOutput:
    """Compute masked causal CE without materializing full-sequence logits.

    The function selects only hidden states that predict active target tokens,
    projects at most ``max_tokens_per_chunk`` positions through the LM head, and
    activation-checkpoints differentiable chunks so vocabulary logits are
    recomputed instead of retained for backward. Gradients flow to hidden
    states, LM-head weight, and optional bias.
    """

    _validate_float_tensor(hidden_states, name="hidden_states", ndim=3)
    _validate_float_tensor(lm_head_weight, name="lm_head_weight", ndim=2)
    batch_size, sequence_length, hidden_size = hidden_states.shape
    vocabulary_size, weight_hidden_size = lm_head_weight.shape
    if hidden_size == 0:
        raise MaskValidationError("hidden dimension must be non-zero")
    if hidden_size != weight_hidden_size:
        raise MaskValidationError(
            f"LM-head input size {weight_hidden_size} does not match hidden size {hidden_size}"
        )
    if lm_head_weight.device != hidden_states.device:
        raise MaskValidationError("LM-head weight and hidden states must share a device")
    if lm_head_bias is not None:
        _validate_float_tensor(lm_head_bias, name="lm_head_bias", ndim=1)
        if lm_head_bias.shape != (vocabulary_size,):
            raise MaskValidationError("LM-head bias must have shape [vocabulary]")
        if lm_head_bias.device != hidden_states.device:
            raise MaskValidationError("LM-head bias and hidden states must share a device")
    _validate_linear_dtypes(hidden_states, lm_head_weight, lm_head_bias)

    selected = _validate_labels_and_mask(
        labels,
        loss_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        device=hidden_states.device,
        ignore_index=ignore_index,
    )
    chunk_size = _chunk_size(max_tokens_per_chunk)
    compute_dtype = _loss_compute_dtype(hidden_states.dtype, loss_compute_dtype)

    if selected.count == 0:
        differentiable_values = [hidden_states, lm_head_weight]
        if lm_head_bias is not None:
            differentiable_values.append(lm_head_bias)
        loss_sum = _differentiable_zero(*differentiable_values, dtype=compute_dtype)
    else:
        chunk_sums = []
        for start in range(0, selected.count, chunk_size):
            end = min(start + chunk_size, selected.count)
            chunk_hidden = hidden_states[
                selected.batch_indices[start:end],
                selected.prediction_positions[start:end],
            ]
            chunk_targets = selected.targets[start:end]
            requires_backward = torch.is_grad_enabled() and any(
                value is not None and value.requires_grad
                for value in (chunk_hidden, lm_head_weight, lm_head_bias)
            )
            if requires_backward:
                chunk_sum = checkpoint(
                    _linear_cross_entropy_sum,
                    chunk_hidden,
                    lm_head_weight,
                    lm_head_bias,
                    chunk_targets,
                    compute_dtype,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_sum = _linear_cross_entropy_sum(
                    chunk_hidden,
                    lm_head_weight,
                    lm_head_bias,
                    chunk_targets,
                    compute_dtype,
                )
            chunk_sums.append(chunk_sum)
        loss_sum = torch.stack(chunk_sums).sum()

    return _finalize_output(
        loss_sum,
        local_token_count=selected.count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )
