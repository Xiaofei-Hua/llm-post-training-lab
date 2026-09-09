"""Connect causal LM adapters to the existing D02/D04 loss kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from posttrain_lab.models import CausalLMAdapter
from posttrain_lab.models.causal_lm import causal_attention_mask

from .loss_budget import MaskValidationError
from .masked_ce import MaskedCrossEntropyOutput, chunked_masked_causal_linear_cross_entropy
from .opd_reverse_kl import MaskedReverseKLOutput, chunked_masked_causal_linear_reverse_kl


def _validate_loss_positions(
    input_ids: Tensor, attention_mask: Tensor | None, loss_mask: Tensor
) -> Tensor:
    valid = causal_attention_mask(input_ids, attention_mask)
    if (
        loss_mask.shape != input_ids.shape
        or loss_mask.dtype != torch.bool
        or loss_mask.device != input_ids.device
    ):
        raise MaskValidationError("loss_mask must be Boolean and match input_ids shape and device")
    predictable = torch.zeros_like(valid)
    predictable[:, 1:] = valid[:, 1:] & valid[:, :-1]
    if bool((loss_mask & ~predictable).any()):
        raise MaskValidationError(
            "selected targets and their preceding positions must be nonpadding"
        )
    return valid


def causal_lm_sft_loss(
    model: CausalLMAdapter,
    input_ids: Tensor,
    loss_mask: Tensor,
    *,
    attention_mask: Tensor | None = None,
    max_tokens_per_chunk: int = 128,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedCrossEntropyOutput:
    """Teacher-forced SFT; mask coordinates are absolute input target positions."""
    valid = _validate_loss_positions(input_ids, attention_mask, loss_mask)
    features = model.forward_features(input_ids, valid)
    return chunked_masked_causal_linear_cross_entropy(
        features.hidden_states,
        features.lm_head_weight,
        input_ids,
        loss_mask,
        lm_head_bias=features.lm_head_bias,
        max_tokens_per_chunk=max_tokens_per_chunk,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )


def causal_lm_opd_loss(
    student: CausalLMAdapter,
    teacher: CausalLMAdapter,
    input_ids: Tensor,
    loss_mask: Tensor,
    *,
    attention_mask: Tensor | None = None,
    temperature: float = 1.0,
    max_tokens_per_chunk: int = 128,
    global_token_count: int | Tensor | None = None,
    ddp_world_size: int | None = None,
) -> MaskedReverseKLOutput:
    """Exact KL(Student || Teacher) on shared token IDs/prefixes.

    Callers supply Student-sampled tokens for on-policy use (D10). These models
    must share token ID semantics, not merely vocabulary size. The tiny fixture
    shares a synthetic integer vocabulary; real tokenizer alignment is D14.
    """
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Teacher must be frozen and in eval mode; call freeze_model first")
    valid = _validate_loss_positions(input_ids, attention_mask, loss_mask)
    student_features = student.forward_features(input_ids, valid)
    with torch.no_grad():
        teacher_features = teacher.forward_features(input_ids, valid)
    return chunked_masked_causal_linear_reverse_kl(
        student_features.hidden_states,
        student_features.lm_head_weight,
        teacher_features.hidden_states,
        teacher_features.lm_head_weight,
        loss_mask,
        student_lm_head_bias=student_features.lm_head_bias,
        teacher_lm_head_bias=teacher_features.lm_head_bias,
        temperature=temperature,
        max_tokens_per_chunk=max_tokens_per_chunk,
        global_token_count=global_token_count,
        ddp_world_size=ddp_world_size,
    )
