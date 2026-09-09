"""Synchronous device-local sampling from the current policy, including its log probs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from posttrain_lab.models import CausalLMAdapter, CausalLMFeatures

from .torch_loss_budget import torch_completion_loss_mask


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    prompt: tuple[int, ...]
    completion: tuple[int, ...]


@dataclass(frozen=True)
class RolloutConfig:
    max_new_tokens: int = 4
    group_size: int = 8
    eos_token_id: int = 2
    pad_token_id: int = 0
    head_projection: Literal["dense", "last"] = "last"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1 or self.group_size < 2:
            raise ValueError("max_new_tokens >= 1 and group_size >= 2 are required")
        if min(self.eos_token_id, self.pad_token_id) < 0:
            raise ValueError("token IDs must be nonnegative")
        if self.eos_token_id == self.pad_token_id:
            raise ValueError("the tiny fixture uses distinct EOS and padding IDs")
        if self.head_projection not in {"dense", "last"}:
            raise ValueError("head_projection must be dense or last")


@dataclass(frozen=True)
class TrainingBatch:
    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    completion_starts: Tensor
    sample_ids: tuple[str, ...]
    generation_indices: tuple[int, ...]
    group_ids: Tensor
    completions: tuple[tuple[int, ...], ...]
    old_log_probs: Tensor | None
    policy_version: int
    truncated: Tensor


def _check_examples(examples: Sequence[TrainingExample]) -> None:
    if not examples or any(not example.prompt for example in examples):
        raise ValueError("a nonempty batch of nonempty prompts is required")
    if len({example.sample_id for example in examples}) != len(examples):
        raise ValueError("sample IDs must be unique within a prompt batch")


def supervised_batch(
    examples: Sequence[TrainingExample],
    config: RolloutConfig,
    *,
    device: str | torch.device = "cpu",
) -> TrainingBatch:
    _check_examples(examples)
    if any(not example.completion for example in examples):
        raise ValueError("SFT requires nonempty reference completions")
    length = max(len(example.prompt) + len(example.completion) for example in examples)
    ids = torch.full((len(examples), length), config.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    starts = torch.tensor([len(example.prompt) for example in examples], device=device)
    for row, example in enumerate(examples):
        tokens = example.prompt + example.completion
        ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
        attention[row, : len(tokens)] = True
    mask = torch_completion_loss_mask(
        ids, completion_start=starts, attention_mask=attention, eos_token_id=config.eos_token_id
    )
    return TrainingBatch(
        ids,
        attention,
        mask,
        starts,
        tuple(e.sample_id for e in examples),
        (0,) * len(examples),
        torch.arange(len(examples), device=device),
        tuple(tuple(ids[row, mask[row]].tolist()) for row in range(len(examples))),
        None,
        -1,
        torch.zeros(len(examples), dtype=torch.bool, device=device),
    )


@torch.no_grad()
def sample_completions(
    model: CausalLMAdapter,
    examples: Sequence[TrainingExample],
    config: RolloutConfig,
    *,
    generator: torch.Generator,
    policy_version: int,
    generations_per_prompt: int = 1,
    greedy: bool = False,
) -> TrainingBatch:
    """Sample the full, unmodified policy at temperature 1, without hidden retries.

    Log probabilities are recorded from the actual sampling distribution before
    any update. Each row terminates on its first EOS or the length cap. Padding
    IDs sampled before EOS remain genuine generated tokens, not masked padding.
    """
    _check_examples(examples)
    device = next(model.parameters()).device
    if generations_per_prompt < 1 or generator.device != device:
        raise ValueError(
            "positive generation count and a generator on the model device are required"
        )
    expanded = [example for example in examples for _ in range(generations_per_prompt)]
    batch_size = len(expanded)
    length = max(len(example.prompt) for example in examples) + config.max_new_tokens
    ids = torch.full((batch_size, length), config.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    starts = torch.tensor([len(example.prompt) for example in expanded], device=device)
    for row, example in enumerate(expanded):
        ids[row, : len(example.prompt)] = torch.tensor(example.prompt, device=device)
        attention[row, : len(example.prompt)] = True
    old_log_probs = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rows = torch.arange(batch_size, device=device)
    was_training = model.training
    model.eval()
    try:
        for step in range(config.max_new_tokens):
            write_positions = starts + step
            prefix_length = int(write_positions.max())
            if config.head_projection == "last":
                features = model.forward_features(
                    ids[:, :prefix_length], attention[:, :prefix_length]
                )
                last_hidden = features.hidden_states[rows, write_positions - 1]
                logits = torch.nn.functional.linear(
                    last_hidden, features.lm_head_weight, features.lm_head_bias
                )
            else:
                logits = model(ids[:, :prefix_length], attention[:, :prefix_length])[
                    rows, write_positions - 1
                ]
            logp = logits.to(
                torch.float64 if logits.dtype == torch.float64 else torch.float32
            ).log_softmax(-1)
            if old_log_probs is None:
                old_log_probs = torch.zeros((batch_size, length), dtype=logp.dtype, device=device)
            next_ids = (
                logp.argmax(-1)
                if greedy
                else torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            )
            active_rows = rows[~finished]
            positions = write_positions[~finished]
            ids[active_rows, positions] = next_ids[~finished]
            attention[active_rows, positions] = True
            old_log_probs[active_rows, positions] = logp[active_rows, next_ids[~finished]]
            finished |= next_ids.eq(config.eos_token_id)
            if bool(finished.all()):
                break
    finally:
        model.train(was_training)
    mask = torch_completion_loss_mask(
        ids, completion_start=starts, attention_mask=attention, eos_token_id=config.eos_token_id
    )
    return TrainingBatch(
        ids,
        attention,
        mask,
        starts,
        tuple(e.sample_id for e in expanded),
        tuple(g for _ in examples for g in range(generations_per_prompt)),
        torch.arange(len(examples), device=device).repeat_interleave(generations_per_prompt),
        tuple(tuple(ids[row, mask[row]].tolist()) for row in range(batch_size)),
        old_log_probs,
        policy_version,
        ~finished,
    )


def selected_token_statistics(
    features: CausalLMFeatures, input_ids: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Target-aligned policy log probs with detached entropy; only selected rows."""
    rows, positions = mask.nonzero(as_tuple=True)
    hidden = features.hidden_states[rows, positions - 1]
    logits = torch.nn.functional.linear(hidden, features.lm_head_weight, features.lm_head_bias)
    logp = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32).log_softmax(
        -1
    )
    selected = logp.gather(-1, input_ids[rows, positions, None]).squeeze(-1)
    aligned = torch.zeros_like(input_ids, dtype=logp.dtype).index_put((rows, positions), selected)
    entropy = -(logp.detach().exp() * logp.detach()).sum(-1)
    return aligned, entropy
