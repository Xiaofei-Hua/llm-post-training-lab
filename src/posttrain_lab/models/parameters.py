"""Text/LoRA parameter selection and explicit parameter-storage estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .causal_lm import TinyCausalLM


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target_modules must be nonempty and unique")


class LoRALinear(nn.Module):
    """Frozen W plus (alpha / rank) B A; zero B preserves initial outputs.

    Adapters are placed only in decoder attention/MLP projections. The linear
    LM-head contract therefore stays intact and never needs a dense BA merge.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        freeze_model(self.base)
        self.scaling = config.alpha / config.rank
        self.lora_A = nn.Parameter(base.weight.new_empty(config.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * (
            self.scaling
        )


def freeze_model(model: nn.Module) -> None:
    """Freeze a Teacher/base, clear stale gradients, and disable training mode."""
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    model.eval()


def configure_trainable_parameters(
    model: TinyCausalLM,
    mode: Literal["text", "lora"],
    *,
    lora: LoRAConfig | None = None,
    seed: int = 0,
) -> ParameterSummary:
    """Configure an unadapted tiny text model on CPU before device placement.

    ``text`` trains the entire text decoder, including embeddings and LM head.
    ``lora`` trains only A/B inside explicitly selected decoder projections.
    Real multimodal model discovery/freeze rules belong to D14.
    """
    if mode not in {"text", "lora"}:
        raise ValueError("mode must be 'text' or 'lora'")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("configure trainable parameters on CPU before device placement")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("configure an unadapted model; LoRA is already installed")
    if mode == "text":
        if lora is not None:
            raise ValueError("LoRA config is only valid in lora mode")
        model.zero_grad(set_to_none=True)
        model.requires_grad_(True)
    else:
        config = lora or LoRAConfig()
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if name.startswith("blocks.")
            and name.rsplit(".", 1)[-1] in config.target_modules
            and isinstance(module, nn.Linear)
        ]
        missing = set(config.target_modules) - {name.rsplit(".", 1)[-1] for name, _ in targets}
        if missing:
            raise ValueError(f"unknown decoder LoRA targets: {sorted(missing)}")
        freeze_model(model)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for name, module in targets:
                parent_name, attribute = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), attribute, LoRALinear(module, config))
    model.train()
    return summarize_parameters(model)


@dataclass(frozen=True)
class ParameterSummary:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_fraction: float
    parameter_bytes: int
    trainable_parameter_bytes: int
    estimated_gradient_bytes: int
    estimated_adamw_moment_bytes: int
    estimated_parameter_gradient_adamw_bytes: int


def summarize_parameters(model: nn.Module) -> ParameterSummary:
    """Count unique Parameters; tied embedding/head weights count only once.

    Byte estimates assume dense gradients and two AdamW moments in parameter
    dtype, with no master weights. They exclude activations, buffers, optimizer
    temporaries/step counters and runtime overhead; they are not peak RSS.
    """
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in parameters)
    active = sum(parameter.numel() for parameter in trainable)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for parameter in trainable)
    return ParameterSummary(
        total_parameters=total,
        trainable_parameters=active,
        frozen_parameters=total - active,
        trainable_fraction=active / total if total else 0.0,
        parameter_bytes=parameter_bytes,
        trainable_parameter_bytes=trainable_bytes,
        estimated_gradient_bytes=trainable_bytes,
        estimated_adamw_moment_bytes=2 * trainable_bytes,
        estimated_parameter_gradient_adamw_bytes=parameter_bytes + 3 * trainable_bytes,
    )
