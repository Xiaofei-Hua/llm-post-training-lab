"""Full-vocabulary reverse-KL primitives for on-policy distillation.

Loss masks use absolute target-token positions. A target at position ``t`` is
scored from the Student and Teacher states at ``t - 1``. Only selected
positions are projected through the two LM heads, in bounded chunks, while the
divergence still covers every vocabulary item.
"""

from __future__ import annotations

import math
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
class MaskedReverseKLOutput:
    """Reverse-KL loss and aggregation-safe diagnostics for one local shard."""

    loss: Tensor
    loss_sum: Tensor
    student_entropy_sum: Tensor
    local_token_count: int
    normalization_token_count: int
    ddp_world_size: int
    max_tokens_per_chunk: int
    temperature: float

    @property
    def distributed_normalization(self) -> bool:
        return self.ddp_world_size > 1

    @property
    def local_student_entropy_mean(self) -> float:
        if self.local_token_count == 0:
            return math.nan
        return float((self.student_entropy_sum / self.local_token_count).item())


@dataclass(frozen=True)
class _SelectedPredictionPositions:
    batch_indices: Tensor
    prediction_positions: Tensor

    @property
    def count(self) -> int:
        return self.batch_indices.shape[0]


@dataclass(frozen=True)
class _LogitTransform:
    scale: float
    final_softcap: float | None
    temperature: float


def _validate_float_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise MaskValidationError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise MaskValidationError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise MaskValidationError(
            f"{name} must use float16, bfloat16, float32, or float64; got {value.dtype}"
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


def _positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


def _optional_positive_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _loss_compute_dtype(
    source_dtypes: tuple[torch.dtype, ...],
    requested: torch.dtype | None,
) -> torch.dtype:
    if requested is None:
        return torch.float64 if torch.float64 in source_dtypes else torch.float32
    if requested not in {torch.float32, torch.float64}:
        raise ValueError("loss_compute_dtype must be torch.float32 or torch.float64")
    return requested


def _chunk_size(value: int) -> int:
    return _positive_integer(value, name="max_tokens_per_chunk")


def _logit_transform(
    *,
    scale: float,
    final_softcap: float | None,
    temperature: float,
    prefix: str,
) -> _LogitTransform:
    return _LogitTransform(
        scale=_positive_float(scale, name=f"{prefix}_logit_scale"),
        final_softcap=_optional_positive_float(
            final_softcap,
            name=f"{prefix}_final_logit_softcapping",
        ),
        temperature=temperature,
    )


def _selected_prediction_positions(
    loss_mask: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> _SelectedPredictionPositions:
    if not isinstance(loss_mask, Tensor) or loss_mask.ndim != 2:
        raise MaskValidationError("loss_mask must be a rank-2 torch.Tensor")
    if loss_mask.shape != (batch_size, sequence_length):
        raise MaskValidationError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match "
            f"{(batch_size, sequence_length)}"
        )
    if loss_mask.device != device:
        raise MaskValidationError("loss_mask must share the Student/Teacher device")
    if loss_mask.dtype != torch.bool:
        raise MaskValidationError("loss_mask must have Boolean dtype")
    if batch_size == 0 or sequence_length == 0:
        raise MaskValidationError("batch and sequence dimensions must be non-zero")
    if bool(loss_mask[:, 0].any().item()):
        raise MaskValidationError("loss_mask cannot select target position 0 in a causal LM")

    positions = loss_mask[:, 1:].nonzero(as_tuple=False)
    return _SelectedPredictionPositions(
        batch_indices=positions[:, 0],
        prediction_positions=positions[:, 1],
    )


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


def _safe_differentiable_zero(*values: Tensor, dtype: torch.dtype) -> Tensor:
    terms = []
    for value in values:
        anchor = value.reshape(-1)[0]
        finite_anchor = torch.where(torch.isfinite(anchor), anchor, torch.zeros_like(anchor))
        terms.append(finite_anchor.to(dtype=dtype) * 0.0)
    return torch.stack(terms).sum()


def _transform_logits(
    logits: Tensor,
    transform: _LogitTransform,
    *,
    compute_dtype: torch.dtype,
    name: str,
) -> Tensor:
    transformed = logits.to(dtype=compute_dtype)
    if transform.scale != 1.0:
        transformed = transformed * transform.scale
    if transform.final_softcap is not None:
        transformed = transform.final_softcap * torch.tanh(transformed / transform.final_softcap)
    if transform.temperature != 1.0:
        transformed = transformed / transform.temperature
    if not bool(torch.isfinite(transformed).all().item()):
        raise FloatingPointError(f"selected {name} logits must remain finite after transforms")
    return transformed


def _reverse_kl_sums_from_logits(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_transform: _LogitTransform,
    teacher_transform: _LogitTransform,
    compute_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    student_logits = _transform_logits(
        student_logits,
        student_transform,
        compute_dtype=compute_dtype,
        name="Student",
    )
    teacher_logits = _transform_logits(
        teacher_logits,
        teacher_transform,
        compute_dtype=compute_dtype,
        name="Teacher",
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    per_token_reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    per_token_student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    if not bool(torch.isfinite(per_token_reverse_kl).all().item()):
        raise FloatingPointError("reverse-KL computation produced a non-finite value")
    if not bool(torch.isfinite(per_token_student_entropy).all().item()):
        raise FloatingPointError("Student entropy computation produced a non-finite value")
    return per_token_reverse_kl.sum(), per_token_student_entropy.sum()


def _linear_reverse_kl_sums(
    student_hidden: Tensor,
    student_weight: Tensor,
    student_bias: Tensor | None,
    teacher_hidden: Tensor,
    teacher_weight: Tensor,
    teacher_bias: Tensor | None,
    student_transform: _LogitTransform,
    teacher_transform: _LogitTransform,
    compute_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    student_logits = F.linear(
        student_hidden.to(dtype=compute_dtype),
        student_weight.to(dtype=compute_dtype),
        None if student_bias is None else student_bias.to(dtype=compute_dtype),
    )
    with torch.no_grad():
        teacher_logits = F.linear(
            teacher_hidden.to(dtype=compute_dtype),
            teacher_weight.to(dtype=compute_dtype),
            None if teacher_bias is None else teacher_bias.to(dtype=compute_dtype),
        )
    return _reverse_kl_sums_from_logits(
        student_logits,
        teacher_logits,
        student_transform,
        teacher_transform,
        compute_dtype,
    )


def _finalize_output(
    loss_sum: Tensor,
    student_entropy_sum: Tensor,
    *,
    local_token_count: int,
    global_token_count: int | Tensor | None,
    ddp_world_size: int | None,
    max_tokens_per_chunk: int,
    temperature: float,
) -> MaskedReverseKLOutput:
    normalization_count, world_size = _normalization_contract(
        local_token_count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )
    loss = loss_sum * world_size / normalization_count
    return MaskedReverseKLOutput(
        loss=loss,
        loss_sum=loss_sum,
        student_entropy_sum=student_entropy_sum.detach(),
        local_token_count=local_token_count,
        normalization_token_count=normalization_count,
        ddp_world_size=world_size,
        max_tokens_per_chunk=max_tokens_per_chunk,
        temperature=temperature,
    )


def masked_causal_reverse_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    loss_mask: Tensor,
    *,
    student_logit_scale: float = 1.0,
    teacher_logit_scale: float = 1.0,
    student_final_logit_softcapping: float | None = None,
    teacher_final_logit_softcapping: float | None = None,
    temperature: float = 1.0,
    max_tokens_per_chunk: int = 128,
    loss_compute_dtype: torch.dtype | None = None,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedReverseKLOutput:
    """Compute token-mean full-vocabulary reverse KL from materialized logits.

    Inputs are raw logits before optional model-specific scaling/softcapping and
    distillation temperature. The mask uses absolute target-token coordinates;
    only the prediction rows needed by selected targets are gathered.
    """

    _validate_float_tensor(student_logits, name="student_logits", ndim=3)
    _validate_float_tensor(teacher_logits, name="teacher_logits", ndim=3)
    if student_logits.shape != teacher_logits.shape:
        raise MaskValidationError("Student and Teacher logits must have identical shapes")
    if student_logits.device != teacher_logits.device:
        raise MaskValidationError("Student and Teacher logits must share a device")
    if teacher_logits.requires_grad:
        raise MaskValidationError("Teacher logits must be stop-gradient")

    batch_size, sequence_length, vocabulary_size = student_logits.shape
    if vocabulary_size == 0:
        raise MaskValidationError("vocabulary dimension must be non-zero")
    selected = _selected_prediction_positions(
        loss_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=student_logits.device,
    )
    chunk_size = _chunk_size(max_tokens_per_chunk)
    normalized_temperature = _positive_float(temperature, name="temperature")
    student_transform = _logit_transform(
        scale=student_logit_scale,
        final_softcap=student_final_logit_softcapping,
        temperature=normalized_temperature,
        prefix="student",
    )
    teacher_transform = _logit_transform(
        scale=teacher_logit_scale,
        final_softcap=teacher_final_logit_softcapping,
        temperature=normalized_temperature,
        prefix="teacher",
    )
    compute_dtype = _loss_compute_dtype(
        (student_logits.dtype, teacher_logits.dtype),
        loss_compute_dtype,
    )

    if selected.count == 0:
        loss_sum = _safe_differentiable_zero(student_logits, dtype=compute_dtype)
        entropy_sum = torch.zeros((), dtype=compute_dtype, device=student_logits.device)
    else:
        loss_sums = []
        entropy_sums = []
        for start in range(0, selected.count, chunk_size):
            end = min(start + chunk_size, selected.count)
            batch_indices = selected.batch_indices[start:end]
            prediction_positions = selected.prediction_positions[start:end]
            chunk_loss, chunk_entropy = _reverse_kl_sums_from_logits(
                student_logits[batch_indices, prediction_positions],
                teacher_logits[batch_indices, prediction_positions],
                student_transform,
                teacher_transform,
                compute_dtype,
            )
            loss_sums.append(chunk_loss)
            entropy_sums.append(chunk_entropy.detach())
        loss_sum = torch.stack(loss_sums).sum()
        entropy_sum = torch.stack(entropy_sums).sum()

    return _finalize_output(
        loss_sum,
        entropy_sum,
        local_token_count=selected.count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
        max_tokens_per_chunk=chunk_size,
        temperature=normalized_temperature,
    )


def chunked_masked_causal_linear_reverse_kl(
    student_hidden_states: Tensor,
    student_lm_head_weight: Tensor,
    teacher_hidden_states: Tensor,
    teacher_lm_head_weight: Tensor,
    loss_mask: Tensor,
    *,
    student_lm_head_bias: Tensor | None = None,
    teacher_lm_head_bias: Tensor | None = None,
    student_logit_scale: float = 1.0,
    teacher_logit_scale: float = 1.0,
    student_final_logit_softcapping: float | None = None,
    teacher_final_logit_softcapping: float | None = None,
    temperature: float = 1.0,
    max_tokens_per_chunk: int = 128,
    loss_compute_dtype: torch.dtype | None = None,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedReverseKLOutput:
    """Compute reverse KL without materializing full-sequence vocabulary logits.

    Student and Teacher may have different hidden widths, but both LM heads
    must map to the same full vocabulary. Each selected-position chunk is
    activation-checkpointed so Student vocabulary logits are recomputed during
    backward instead of retained. Teacher tensors are strict frozen targets.
    """

    _validate_float_tensor(student_hidden_states, name="student_hidden_states", ndim=3)
    _validate_float_tensor(teacher_hidden_states, name="teacher_hidden_states", ndim=3)
    _validate_float_tensor(student_lm_head_weight, name="student_lm_head_weight", ndim=2)
    _validate_float_tensor(teacher_lm_head_weight, name="teacher_lm_head_weight", ndim=2)

    student_batch, student_sequence, student_hidden_size = student_hidden_states.shape
    teacher_batch, teacher_sequence, teacher_hidden_size = teacher_hidden_states.shape
    if (student_batch, student_sequence) != (teacher_batch, teacher_sequence):
        raise MaskValidationError("Student and Teacher hidden states must align in batch/sequence")
    if student_hidden_size == 0 or teacher_hidden_size == 0:
        raise MaskValidationError("Student and Teacher hidden dimensions must be non-zero")

    student_vocab, student_weight_hidden = student_lm_head_weight.shape
    teacher_vocab, teacher_weight_hidden = teacher_lm_head_weight.shape
    if student_vocab == 0 or teacher_vocab == 0:
        raise MaskValidationError("Student and Teacher vocabularies must be non-zero")
    if student_vocab != teacher_vocab:
        raise MaskValidationError("Student and Teacher vocabulary sizes must match exactly")
    if student_hidden_size != student_weight_hidden:
        raise MaskValidationError("Student LM-head input size does not match Student hidden size")
    if teacher_hidden_size != teacher_weight_hidden:
        raise MaskValidationError("Teacher LM-head input size does not match Teacher hidden size")

    tensors = (
        teacher_hidden_states,
        student_lm_head_weight,
        teacher_lm_head_weight,
    )
    if any(value.device != student_hidden_states.device for value in tensors):
        raise MaskValidationError(
            "all Student/Teacher hidden states and LM heads must share a device"
        )

    if student_lm_head_bias is not None:
        _validate_float_tensor(student_lm_head_bias, name="student_lm_head_bias", ndim=1)
        if student_lm_head_bias.shape != (student_vocab,):
            raise MaskValidationError("Student LM-head bias must have shape [vocabulary]")
        if student_lm_head_bias.device != student_hidden_states.device:
            raise MaskValidationError("Student LM-head bias must share the model device")
    if teacher_lm_head_bias is not None:
        _validate_float_tensor(teacher_lm_head_bias, name="teacher_lm_head_bias", ndim=1)
        if teacher_lm_head_bias.shape != (teacher_vocab,):
            raise MaskValidationError("Teacher LM-head bias must have shape [vocabulary]")
        if teacher_lm_head_bias.device != student_hidden_states.device:
            raise MaskValidationError("Teacher LM-head bias must share the model device")

    frozen_teacher_tensors = {
        "teacher_hidden_states": teacher_hidden_states,
        "teacher_lm_head_weight": teacher_lm_head_weight,
        "teacher_lm_head_bias": teacher_lm_head_bias,
    }
    for name, value in frozen_teacher_tensors.items():
        if value is not None and value.requires_grad:
            raise MaskValidationError(f"{name} must be stop-gradient")

    selected = _selected_prediction_positions(
        loss_mask,
        batch_size=student_batch,
        sequence_length=student_sequence,
        device=student_hidden_states.device,
    )
    chunk_size = _chunk_size(max_tokens_per_chunk)
    normalized_temperature = _positive_float(temperature, name="temperature")
    student_transform = _logit_transform(
        scale=student_logit_scale,
        final_softcap=student_final_logit_softcapping,
        temperature=normalized_temperature,
        prefix="student",
    )
    teacher_transform = _logit_transform(
        scale=teacher_logit_scale,
        final_softcap=teacher_final_logit_softcapping,
        temperature=normalized_temperature,
        prefix="teacher",
    )
    source_dtypes = [
        student_hidden_states.dtype,
        student_lm_head_weight.dtype,
        teacher_hidden_states.dtype,
        teacher_lm_head_weight.dtype,
    ]
    if student_lm_head_bias is not None:
        source_dtypes.append(student_lm_head_bias.dtype)
    if teacher_lm_head_bias is not None:
        source_dtypes.append(teacher_lm_head_bias.dtype)
    compute_dtype = _loss_compute_dtype(tuple(source_dtypes), loss_compute_dtype)

    if selected.count == 0:
        student_values = [student_hidden_states, student_lm_head_weight]
        if student_lm_head_bias is not None:
            student_values.append(student_lm_head_bias)
        loss_sum = _safe_differentiable_zero(*student_values, dtype=compute_dtype)
        entropy_sum = torch.zeros((), dtype=compute_dtype, device=student_hidden_states.device)
    else:
        loss_sums = []
        entropy_sums = []
        for start in range(0, selected.count, chunk_size):
            end = min(start + chunk_size, selected.count)
            batch_indices = selected.batch_indices[start:end]
            prediction_positions = selected.prediction_positions[start:end]
            student_hidden = student_hidden_states[batch_indices, prediction_positions]
            teacher_hidden = teacher_hidden_states[batch_indices, prediction_positions]
            requires_backward = torch.is_grad_enabled() and any(
                value is not None and value.requires_grad
                for value in (
                    student_hidden,
                    student_lm_head_weight,
                    student_lm_head_bias,
                )
            )
            if requires_backward:
                chunk_loss, chunk_entropy = checkpoint(
                    _linear_reverse_kl_sums,
                    student_hidden,
                    student_lm_head_weight,
                    student_lm_head_bias,
                    teacher_hidden,
                    teacher_lm_head_weight,
                    teacher_lm_head_bias,
                    student_transform,
                    teacher_transform,
                    compute_dtype,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_loss, chunk_entropy = _linear_reverse_kl_sums(
                    student_hidden,
                    student_lm_head_weight,
                    student_lm_head_bias,
                    teacher_hidden,
                    teacher_lm_head_weight,
                    teacher_lm_head_bias,
                    student_transform,
                    teacher_transform,
                    compute_dtype,
                )
            loss_sums.append(chunk_loss)
            entropy_sums.append(chunk_entropy.detach())
        loss_sum = torch.stack(loss_sums).sum()
        entropy_sum = torch.stack(entropy_sums).sum()

    return _finalize_output(
        loss_sum,
        entropy_sum,
        local_token_count=selected.count,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
        max_tokens_per_chunk=chunk_size,
        temperature=normalized_temperature,
    )
