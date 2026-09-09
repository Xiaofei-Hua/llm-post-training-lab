"""Single-process CPU/CUDA training with token-normalized accumulation and resume."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Literal

import torch

from posttrain_lab.models import CausalLMAdapter, LoRALinear

from .grpo_surrogate import compute_exact_group_advantages, dr_grpo_token_surrogate
from .loss_budget import LossTokenBudget
from .masked_ce import chunked_masked_causal_linear_cross_entropy
from .opd_reverse_kl import chunked_masked_causal_linear_reverse_kl
from .rollout import (
    RolloutConfig,
    TrainingExample,
    sample_completions,
    selected_token_statistics,
    supervised_batch,
)
from .torch_loss_budget import commit_torch_loss_budget, plan_torch_loss_budget

Objective = Literal["sft", "grpo", "opd"]
RewardFunction = Callable[[tuple[int, ...], TrainingExample], float]


@dataclass(frozen=True)
class TrainerConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    learning_rates: tuple[float, float, float] = (0.01, 0.003, 0.003)
    max_grad_norm: float = 1.0
    max_tokens_per_chunk: int = 32
    final_lr_fraction: float = 0.1
    device: str = "cpu"
    precision: Literal["full", "bf16"] = "full"
    microbatch_size: int | None = None  # Completion rows, including expanded GRPO rows.
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if len(self.learning_rates) != 3 or any(
            not math.isfinite(rate) or rate <= 0 for rate in self.learning_rates
        ):
            raise ValueError("three finite positive objective learning rates are required")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.max_tokens_per_chunk < 1 or not 0 < self.final_lr_fraction <= 1:
            raise ValueError("invalid chunk size or final_lr_fraction")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("only explicit CPU or single CUDA device execution is supported")
        if self.precision not in {"full", "bf16"}:
            raise ValueError("precision must be full or bf16")
        if self.precision == "bf16" and torch.device(self.device).type != "cuda":
            raise ValueError("this runtime validates BF16 on CUDA only")
        if self.microbatch_size is not None and (
            isinstance(self.microbatch_size, bool)
            or not isinstance(self.microbatch_size, int)
            or self.microbatch_size < 1
        ):
            raise ValueError("microbatch_size must be a positive integer or None")


class StageBudgetIncomplete(RuntimeError):
    """A bounded stage exhausted its attempt limit, possibly from zero reward variance."""


class Trainer:
    def __init__(
        self,
        student: CausalLMAdapter,
        *,
        config: TrainerConfig | None = None,
        teacher: CausalLMAdapter | None = None,
        reward_fn: RewardFunction | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("requested CUDA device is unavailable")
            self.device = torch.device("cuda", self.device.index or 0)
            with torch.cuda.device(self.device):
                if self.config.precision == "bf16" and not torch.cuda.is_bf16_supported(
                    including_emulation=False
                ):
                    raise ValueError("native BF16 support is required")
        if teacher is student:
            raise ValueError("Teacher and Student must be distinct model instances")
        if self.config.precision == "bf16" and any(
            p.dtype != torch.float32
            for model in (student, teacher)
            if model is not None
            for p in model.parameters()
        ):
            raise ValueError("BF16 autocast requires FP32 master parameters")
        self.student = student.to(self.device)
        self.student.set_activation_checkpointing(self.config.activation_checkpointing)
        self.teacher = teacher.to(self.device) if teacher is not None else None
        self.reward_fn = reward_fn
        self.parameters = [p for p in student.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("Student must have trainable parameters")
        self.stage = 0
        self.policy_version = 0
        self.records: list[dict] = []
        self.budget: LossTokenBudget | None = None
        self.last_batch = None

    def autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.config.precision == "bf16"
            else nullcontext()
        )

    def _timestamp(self) -> float:
        # Synchronization makes the CUDA phase measurements completed wall time.
        # These instrumented timings include Python/launch/synchronization overhead.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start_stage(self, objective: Objective, *, loss_tokens: int, seed: int) -> None:
        if objective not in {"sft", "grpo", "opd"}:
            raise ValueError("objective must be sft, grpo or opd")
        if self.budget is not None and not self.budget.complete:
            raise StageBudgetIncomplete("finish the current stage budget before switching stages")
        if objective == "grpo" and self.reward_fn is None:
            raise ValueError("GRPO requires an exact 0/1 reward function")
        if objective == "opd" and (
            self.teacher is None
            or self.teacher.training
            or any(p.requires_grad for p in self.teacher.parameters())
        ):
            raise ValueError("OPD requires a frozen Teacher in eval mode")
        self.budget = LossTokenBudget(loss_tokens)
        self.objective = objective
        self.stage += 1
        self.attempt = 0
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.student.zero_grad(set_to_none=True)
        lr = self.config.learning_rates[("sft", "grpo", "opd").index(objective)]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=0.0)
        # Only successful backward tokens age the schedule; new stages reset moments/LR.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda _: (
                1
                - (1 - self.config.final_lr_fraction)
                * (self.budget.consumed_tokens / self.budget.target_tokens)
            ),
        )

    def step(self, examples: Sequence[TrainingExample]) -> dict:
        if self.budget is None or self.budget.complete:
            raise ValueError("start a stage with a nonempty remaining token budget")
        start = self._timestamp()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.attempt += 1
        rollout = self.config.rollout
        with self.autocast():
            if self.objective == "sft":
                batch = supervised_batch(examples, rollout, device=self.device)
            else:
                batch = sample_completions(
                    self.student,
                    examples,
                    rollout,
                    generator=self.generator,
                    policy_version=self.policy_version,
                    generations_per_prompt=rollout.group_size if self.objective == "grpo" else 1,
                )
        self.last_batch = batch
        sampled = self._timestamp()
        candidate_mask = batch.completion_mask
        reward_mean = effective_group_rate = None
        group = None
        if self.objective == "grpo":
            examples_by_id = {example.sample_id: example for example in examples}
            rewards = torch.tensor(
                [
                    self.reward_fn(completion, examples_by_id[sample_id])
                    for completion, sample_id in zip(
                        batch.completions, batch.sample_ids, strict=True
                    )
                ],
                dtype=torch.float32,
                device=self.device,
            )
            # Group statistics and the Dr.GRPO denominator precede any microbatch/budget trim.
            group = compute_exact_group_advantages(
                rewards, batch.group_ids, candidate_mask, expected_group_size=rollout.group_size
            )
            candidate_mask = group.loss_mask
            reward_mean = rewards.mean().item()
            effective_group_rate = group.effective_group_rate
        scored = self._timestamp()
        selection = plan_torch_loss_budget(
            self.budget,
            candidate_mask,
            sample_ids=batch.sample_ids,
            generation_indices=batch.generation_indices,
        )
        result = {
            "stage": self.stage,
            "objective": self.objective,
            "attempt": self.attempt,
            "policy_version_before": self.policy_version,
            "policy_version_after": self.policy_version,
            "status": "skipped_zero_variance",
            "loss": None,
            "loss_tokens": 0,
            "candidate_loss_tokens": selection.candidate_tokens,
            "budget_truncated": selection.truncated,
            "rollout_tokens": int(batch.completion_mask.sum()) if self.objective != "sft" else 0,
            "reward_mean": reward_mean,
            "effective_group_rate": effective_group_rate,
            "entropy": None,
            "kl": None,
            "clip_fraction": None,
            "importance_ratio_mean": None,
            "mean_completion_length": batch.completion_mask.sum(-1).float().mean().item(),
            "truncation_rate": batch.truncated.float().mean().item(),
            "gradient_norm": None,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "microbatches": 0,
            "timings_ms": {"sample": (sampled - start) * 1000, "score": (scored - sampled) * 1000},
        }
        self.optimizer.zero_grad(set_to_none=True)
        loss_ms = backward_ms = update_ms = 0.0
        loss_sum = entropy_sum = clip_sum = ratio_sum = 0.0
        executed = False
        if selection.selected_tokens:
            self.student.train()
            width = self.config.microbatch_size or len(batch.sample_ids)
            for first in range(0, len(batch.sample_ids), width):
                rows = slice(first, first + width)
                mask = selection.loss_mask[rows]
                if not bool(mask.any()):
                    continue
                ids, attention = batch.input_ids[rows], batch.attention_mask[rows]
                teacher_features = None
                if self.objective == "opd":
                    teacher_start = self._timestamp()
                    with torch.no_grad(), self.autocast():
                        teacher_features = self.teacher.forward_features(ids, attention)
                    result["timings_ms"]["score"] += (self._timestamp() - teacher_start) * 1000
                loss_start = self._timestamp()
                with self.autocast():
                    features = self.student.forward_features(ids, attention)
                    if self.objective == "sft":
                        output = chunked_masked_causal_linear_cross_entropy(
                            features.hidden_states,
                            features.lm_head_weight,
                            ids,
                            mask,
                            lm_head_bias=features.lm_head_bias,
                            max_tokens_per_chunk=self.config.max_tokens_per_chunk,
                            global_token_count=selection.selected_tokens,
                            ddp_world_size=1,
                        )
                        with torch.no_grad():
                            _, entropy = selected_token_statistics(features, ids, mask)
                        entropy_sum += entropy.sum().item()
                    elif self.objective == "opd":
                        output = chunked_masked_causal_linear_reverse_kl(
                            features.hidden_states,
                            features.lm_head_weight,
                            teacher_features.hidden_states,
                            teacher_features.lm_head_weight,
                            mask,
                            student_lm_head_bias=features.lm_head_bias,
                            teacher_lm_head_bias=teacher_features.lm_head_bias,
                            max_tokens_per_chunk=self.config.max_tokens_per_chunk,
                            global_token_count=selection.selected_tokens,
                            ddp_world_size=1,
                        )
                        entropy_sum += output.student_entropy_sum.item()
                    else:
                        if batch.policy_version != self.policy_version:
                            raise RuntimeError("refuse stale-policy rollout")
                        log_probs, entropy = selected_token_statistics(features, ids, mask)
                        output = dr_grpo_token_surrogate(
                            log_probs,
                            batch.old_log_probs[rows],
                            group.advantages[rows],
                            mask,
                            global_active_completion_count=group.active_completion_count,
                            max_completion_length=rollout.max_new_tokens,
                            group_size=rollout.group_size,
                        )
                        entropy_sum += entropy.sum().item()
                        clip_sum += output.local_clip_fraction * output.local_token_count
                        ratio_sum += output.importance_ratio_sum.item()
                loss_end = self._timestamp()
                loss_ms += (loss_end - loss_start) * 1000
                if not bool(torch.isfinite(output.loss)):
                    result["status"] = "skipped_nonfinite_loss"
                    break
                loss_sum += output.loss.item()
                # Each loss already has the logical-batch denominator. No 1 / microbatches.
                output.loss.backward()
                backward_ms += (self._timestamp() - loss_end) * 1000
                result["microbatches"] += 1
                del features, teacher_features, output
            else:
                result["loss"] = loss_sum
                result["entropy"] = entropy_sum / selection.selected_tokens
                if self.objective == "opd":
                    result["kl"] = loss_sum
                if self.objective == "grpo":
                    result["clip_fraction"] = clip_sum / selection.selected_tokens
                    result["importance_ratio_mean"] = ratio_sum / selection.selected_tokens
                gradient_start = self._timestamp()
                finite = torch.stack(
                    [torch.isfinite(p.grad).all() for p in self.parameters if p.grad is not None]
                ).all()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
                finite = bool(finite & torch.isfinite(norm))
                backward_end = self._timestamp()
                backward_ms += (backward_end - gradient_start) * 1000
                if finite:
                    result["gradient_norm"] = norm.item()
                    self.optimizer.step()
                    executed = True
                    self.policy_version += 1
                    result["status"] = "updated"
                    result["policy_version_after"] = self.policy_version
                    update_ms = (self._timestamp() - backward_end) * 1000
                else:
                    result["status"] = "skipped_nonfinite_gradient"
        ledger = commit_torch_loss_budget(
            self.budget,
            selection,
            optimizer_step_executed=executed,
            objective=self.objective,
            step_id=f"stage-{self.stage}/attempt-{self.attempt}",
        )
        if executed:
            self.scheduler.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        result["loss_tokens"] = ledger.counted_tokens
        result["cumulative_loss_tokens"] = self.budget.consumed_tokens
        result["timings_ms"].update({"loss": loss_ms, "backward": backward_ms, "update": update_ms})
        result["timings_ms"]["total"] = (self._timestamp() - start) * 1000
        result["effective_tokens_per_second"] = ledger.counted_tokens / (
            result["timings_ms"]["total"] / 1000
        )
        if self.device.type == "cuda":
            result["memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
        self.records.append(result)
        return result

    def run_stage(
        self, examples: Sequence[TrainingExample], *, batch_size: int = 4, max_attempts: int = 256
    ) -> list[dict]:
        if not examples or batch_size < 1 or max_attempts < 1 or self.budget is None:
            raise ValueError("run_stage requires examples, positive limits and an active stage")
        first_record = len(self.records)
        for _ in range(max_attempts):
            if self.budget.complete:
                break
            offset = self.attempt * min(batch_size, len(examples))
            self.step(
                [
                    examples[(offset + index) % len(examples)]
                    for index in range(min(batch_size, len(examples)))
                ]
            )
        if not self.budget.complete:
            raise StageBudgetIncomplete(
                f"stage {self.stage} consumed {self.budget.consumed_tokens}/"
                f"{self.budget.target_tokens} tokens in {max_attempts} attempts"
            )
        return self.records[first_record:]

    @staticmethod
    def _model_contract(model: CausalLMAdapter | None):
        if model is None:
            return None
        config = getattr(model, "config", None)
        return {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": asdict(config) if is_dataclass(config) else None,
            "lora_scaling": {
                name: module.scaling
                for name, module in model.named_modules()
                if isinstance(module, LoRALinear)
            },
            "parameters": [
                (name, tuple(p.shape), str(p.dtype), p.requires_grad)
                for name, p in model.named_parameters()
            ],
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save at an attempted-step boundary; no partially accumulated gradients.

        The caller owns examples/reward code. Resume requires the same runtime,
        model topology and trainable parameter set, on the same device kind.
        """
        if self.budget is None:
            raise ValueError("start a stage before saving")
        state = {
            "config": asdict(self.config),
            "student_contract": self._model_contract(self.student),
            "teacher_contract": self._model_contract(self.teacher),
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict() if self.teacher is not None else None,
            "student_training": self.student.training,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "budget": self.budget.state_dict(),
            "objective": self.objective,
            "stage": self.stage,
            "attempt": self.attempt,
            "policy_version": self.policy_version,
            "generator": self.generator.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "records": self.records,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a locally generated checkpoint into a fresh Trainer."""
        if self.budget is not None:
            raise ValueError("load into a fresh Trainer before starting a stage")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint runtime configuration does not match")
        for name in ("student", "teacher"):
            if state[f"{name}_contract"] != self._model_contract(getattr(self, name)):
                raise ValueError(f"checkpoint {name} topology/trainable parameters do not match")
        budget = LossTokenBudget.from_state_dict(state["budget"])
        self.start_stage(state["objective"], loss_tokens=budget.target_tokens, seed=0)
        self.student.load_state_dict(state["student"])
        self.student.train(state["student_training"])
        if self.teacher is not None:
            self.teacher.load_state_dict(state["teacher"])
        self.budget = budget
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.stage, self.attempt = state["stage"], state["attempt"]
        self.policy_version = state["policy_version"]
        self.generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        self.records = state["records"]
        self.last_batch = None
