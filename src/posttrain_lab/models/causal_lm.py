"""A hidden-state/linear-head boundary and a locally initialized tiny decoder.

The tiny decoder is a development fixture, not a Gemma implementation. It uses
learned positions, pre-norm causal attention and a gated MLP, without dropout,
KV caching, a tokenizer, or checkpoint downloads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CausalLMFeatures:
    """Unshifted hidden states and the actual, possibly tied, linear LM head.

    No vocabulary logits are allocated until ``logits`` is called. D02/D04
    consume these tensors directly and perform their own causal target shift.
    This boundary currently describes models with an untransformed linear head.
    """

    hidden_states: Tensor
    lm_head_weight: Tensor
    lm_head_bias: Tensor | None = None

    def logits(self) -> Tensor:
        return F.linear(self.hidden_states, self.lm_head_weight, self.lm_head_bias)


class CausalLMAdapter(nn.Module, ABC):
    def set_activation_checkpointing(self, enabled: bool) -> None:
        if enabled:
            raise NotImplementedError("this adapter does not support activation checkpointing")

    @abstractmethod
    def forward_features(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> CausalLMFeatures:
        """Return final hidden states without projecting the whole vocabulary."""

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Dense reference/inference path; training can use forward_features."""
        return self.forward_features(input_ids, attention_mask).logits()


@dataclass(frozen=True)
class TinyCausalLMConfig:
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_sequence_length: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_heads",
            "max_sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings must be Boolean")


class _CausalAttention(nn.Module):
    def __init__(self, config: TinyCausalLMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                self,
                name,
                nn.Linear(
                    config.hidden_size, config.hidden_size, bias=False, device="cpu", dtype=dtype
                ),
            )

    def forward(self, hidden: Tensor, allowed: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        query, key, value = (
            projection(hidden).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        scores = (query @ key.transpose(-2, -1)) * self.head_dim**-0.5
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        attended = (probabilities @ value).transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(attended)


class _GatedMLP(nn.Module):
    def __init__(self, config: TinyCausalLMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device="cpu", dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device="cpu", dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, device="cpu", dtype=dtype
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _DecoderBlock(nn.Module):
    def __init__(self, config: TinyCausalLMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, device="cpu", dtype=dtype)
        self.attention = _CausalAttention(config, dtype)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, device="cpu", dtype=dtype)
        self.mlp = _GatedMLP(config, dtype)

    def forward(self, hidden: Tensor, allowed: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), allowed)
        return hidden + self.mlp(self.mlp_norm(hidden))


def causal_attention_mask(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Validate the shared token boundary without implicit device transfers."""
    if input_ids.ndim != 2 or input_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError("input_ids must be a rank-2 int32/int64 tensor")
    if 0 in input_ids.shape:
        raise ValueError("input_ids must contain a nonempty batch and sequence")
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    if (
        attention_mask.shape != input_ids.shape
        or attention_mask.dtype != torch.bool
        or attention_mask.device != input_ids.device
    ):
        raise ValueError("attention_mask must be Boolean and match input_ids shape and device")
    return attention_mask


class TinyCausalLM(CausalLMAdapter):
    def __init__(
        self, config: TinyCausalLMConfig, *, seed: int = 0, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(
                "initialize tiny parameters in float32 or float64; use autocast for BF16"
            )
        self.config = config
        self.activation_checkpointing = False
        # Restore the caller's CPU RNG and never initialize an accelerator RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.token_embedding = nn.Embedding(
                config.vocab_size, config.hidden_size, device="cpu", dtype=dtype
            )
            self.position_embedding = nn.Embedding(
                config.max_sequence_length, config.hidden_size, device="cpu", dtype=dtype
            )
            self.blocks = nn.ModuleList(
                _DecoderBlock(config, dtype) for _ in range(config.num_layers)
            )
            self.final_norm = nn.LayerNorm(config.hidden_size, device="cpu", dtype=dtype)
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False, device="cpu", dtype=dtype
            )
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.position_embedding.weight, std=0.02)
            if config.tie_word_embeddings:
                self.lm_head.weight = self.token_embedding.weight

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = enabled

    def forward_features(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> CausalLMFeatures:
        valid = causal_attention_mask(input_ids, attention_mask)
        if self.token_embedding.weight.device != input_ids.device:
            raise ValueError("model parameters and input_ids must be on the same device")
        length = input_ids.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("input_ids must be valid vocabulary IDs, including padding slots")

        positions = (valid.long().cumsum(dim=-1) - 1).clamp_min(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = torch.ones((length, length), dtype=torch.bool, device=input_ids.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        # A padding query may have no valid keys (left/all padding). Give it a
        # dummy self key to avoid softmax(-inf, ...); valid queries cannot see it.
        diagonal = torch.eye(length, dtype=torch.bool, device=input_ids.device)
        allowed = allowed | (~valid[:, None, :, None] & diagonal[None, None])
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Non-reentrant checkpointing also supports frozen LoRA inputs.
                # The tiny decoder has no dropout or other stochastic blocks.
                hidden = checkpoint(
                    block, hidden, allowed, use_reentrant=False, preserve_rng_state=False
                )
            else:
                hidden = block(hidden, allowed)
        hidden = self.final_norm(hidden).masked_fill(~valid.unsqueeze(-1), 0.0)
        return CausalLMFeatures(hidden, self.lm_head.weight, self.lm_head.bias)
