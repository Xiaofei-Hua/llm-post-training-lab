"""Model logit transforms, shared by sampling and differentiable losses."""

import math

import torch
from torch import Tensor


def validate_logit_transform(scale: float, softcap: float | None) -> None:
    for name, value in (("logit_scale", scale), ("final_logit_softcapping", softcap)):
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")


def transform_logits(
    logits: Tensor,
    *,
    scale: float = 1.0,
    softcap: float | None = None,
    temperature: float = 1.0,
    compute_dtype: torch.dtype,
    name: str = "model",
) -> Tensor:
    """Scale → softcap → temperature in the loss dtype (usually FP32).

    Test finiteness before tanh: clipping must not conceal projection overflow.
    Callers validate scalar arguments once, outside vocabulary chunk loops.
    """
    result = logits.to(compute_dtype)
    if scale != 1.0:
        result = result * scale
    if softcap is not None:
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError(f"selected {name} logits must be finite before softcap")
        result = softcap * torch.tanh(result / softcap)
    if temperature != 1.0:
        result = result / temperature
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError(f"selected {name} logits must remain finite after transforms")
    return result
