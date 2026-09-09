"""Bounded CPU model-integration experiment; not the D10 training framework."""

from __future__ import annotations

import copy
import platform
from dataclasses import asdict, replace

import torch

from posttrain_lab.train import causal_lm_opd_loss, causal_lm_sft_loss

from .causal_lm import TinyCausalLM, TinyCausalLMConfig
from .parameters import (
    LoRAConfig,
    configure_trainable_parameters,
    freeze_model,
    summarize_parameters,
)


def model_adapter_smoke_config() -> dict:
    student = TinyCausalLMConfig()
    return {
        "evidence_scope": "CPU synthetic fixed-prefix model integration; no on-policy rollout",
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 1,
        "student": asdict(student),
        "teacher": asdict(replace(student, hidden_size=48, intermediate_size=96)),
        "lora": asdict(LoRAConfig()),
        "seeds": {"student": 7, "teacher": 13, "adapters": 17, "synthetic_data": 11},
        "batch_size": 4,
        "sequence_length": 8,
        "valid_lengths": [8, 7, 6, 5],
        "prompt_length": 2,
        "steps": 12,
        "optimizer": {"name": "SGD", "lr": 0.1},
        "max_tokens_per_chunk": 3,
        "tolerances": {"rtol": 1e-4, "atol": 2e-6},
        "baseline": "full logits with independently expressed CE or KL(Student || Teacher)",
    }


def _dense_reference_loss(model, teacher, objective, ids, attention, mask):
    logp = model(ids, attention)[:, :-1].log_softmax(-1)
    if objective == "sft":
        per_token = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    else:
        with torch.no_grad():
            logq = teacher(ids, attention)[:, :-1].log_softmax(-1)
        per_token = (logp.exp() * (logp - logq)).sum(-1)
    return per_token[mask[:, 1:]].mean()


def _chunked_loss(model, teacher, objective, ids, attention, mask, chunk_size):
    kwargs = {"attention_mask": attention, "max_tokens_per_chunk": chunk_size}
    if objective == "sft":
        return causal_lm_sft_loss(model, ids, mask, **kwargs).loss
    return causal_lm_opd_loss(model, teacher, ids, mask, **kwargs).loss


def _run_case(config, objective, mode, teacher, ids, attention, mask):
    student = TinyCausalLM(TinyCausalLMConfig(**config["student"]), seed=config["seeds"]["student"])
    summary = configure_trainable_parameters(
        student,
        mode,
        lora=LoRAConfig(**config["lora"]) if mode == "lora" else None,
        seed=config["seeds"]["adapters"],
    )
    reference = copy.deepcopy(student)
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    optimizer = torch.optim.SGD(
        (p for p in student.parameters() if p.requires_grad), lr=config["optimizer"]["lr"]
    )
    reference_optimizer = torch.optim.SGD(
        (p for p in reference.parameters() if p.requires_grad), lr=config["optimizer"]["lr"]
    )
    losses, a_grad_norms, b_grad_norms = [], [], []
    errors = {"loss_abs": 0.0, "gradient_max_abs": 0.0, "parameter_max_abs": 0.0}
    for _ in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        loss = _chunked_loss(
            student, teacher, objective, ids, attention, mask, config["max_tokens_per_chunk"]
        )
        expected = _dense_reference_loss(reference, teacher, objective, ids, attention, mask)
        torch.testing.assert_close(loss, expected, **config["tolerances"])
        errors["loss_abs"] = max(errors["loss_abs"], abs(loss.item() - expected.item()))
        losses.append(loss.item())
        loss.backward()
        expected.backward()
        for name, parameter in student.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise AssertionError(f"frozen parameter received a gradient: {name}")
                continue
            reference_gradient = reference.get_parameter(name).grad
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"missing or nonfinite gradient: {name}")
            torch.testing.assert_close(parameter.grad, reference_gradient, **config["tolerances"])
            errors["gradient_max_abs"] = max(
                errors["gradient_max_abs"], (parameter.grad - reference_gradient).abs().max().item()
            )
        for suffix, norms in (("lora_A", a_grad_norms), ("lora_B", b_grad_norms)):
            norms.append(
                sum(
                    p.grad.square().sum().item()
                    for name, p in student.named_parameters()
                    if name.endswith(suffix)
                )
                ** 0.5
            )
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in student.named_parameters():
            target = reference.get_parameter(name)
            torch.testing.assert_close(parameter, target, **config["tolerances"])
            errors["parameter_max_abs"] = max(
                errors["parameter_max_abs"],
                (parameter.detach() - target.detach()).abs().max().item(),
            )

    with torch.no_grad():
        losses.append(
            _chunked_loss(
                student, teacher, objective, ids, attention, mask, config["max_tokens_per_chunk"]
            ).item()
        )
    changed, frozen_changed = [], []
    for name, parameter in student.named_parameters():
        if not torch.equal(parameter, before[name]):
            (changed if parameter.requires_grad else frozen_changed).append(name)
    if not changed or frozen_changed:
        raise AssertionError("expected trainable updates with unchanged frozen parameters")
    if not losses[-1] < losses[0]:
        raise AssertionError("the fixed-prefix smoke objective did not decrease")
    return {
        "objective": objective,
        "mode": mode,
        "parameters": asdict(summary),
        "loss_by_completed_update": losses,
        "max_dense_reference_errors": errors,
        "changed_trainable_tensors": len(changed),
        "changed_frozen_tensors": len(frozen_changed),
        "trainable_parameter_names": [n for n, p in student.named_parameters() if p.requires_grad],
        "lora_A_gradient_norm_by_update": a_grad_norms if mode == "lora" else None,
        "lora_B_gradient_norm_by_update": b_grad_norms if mode == "lora" else None,
    }


def run_model_adapter_smoke() -> dict:
    """Run four matched tiny-model cases and retain loss/gradient/update evidence."""
    config = model_adapter_smoke_config()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config["num_threads"])
    try:
        teacher = TinyCausalLM(
            TinyCausalLMConfig(**config["teacher"]), seed=config["seeds"]["teacher"]
        )
        freeze_model(teacher)
        teacher_before = [p.detach().clone() for p in teacher.parameters()]
        generator = torch.Generator(device="cpu").manual_seed(config["seeds"]["synthetic_data"])
        ids = torch.randint(
            1,
            config["student"]["vocab_size"],
            (config["batch_size"], config["sequence_length"]),
            generator=generator,
            device="cpu",
        )
        positions = torch.arange(config["sequence_length"], device="cpu")[None]
        attention = positions < torch.tensor(config["valid_lengths"], device="cpu")[:, None]
        ids = ids.masked_fill(~attention, 0)
        mask = attention & (positions >= config["prompt_length"])
        cases = [
            _run_case(config, objective, mode, teacher, ids, attention, mask)
            for objective in ("sft", "opd")
            for mode in ("text", "lora")
        ]
        for parameter, initial in zip(teacher.parameters(), teacher_before, strict=True):
            if parameter.grad is not None or not torch.equal(parameter, initial):
                raise AssertionError("Teacher changed or received gradients")
        return {
            "config": config,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "system": platform.system(),
                "machine": platform.machine(),
                "threads": torch.get_num_threads(),
            },
            "synthetic_input_ids": ids.tolist(),
            "selected_loss_tokens_per_update": int(mask.sum()),
            "teacher_parameters": asdict(summarize_parameters(teacher)),
            "teacher_unchanged_and_no_gradients": True,
            "memory_scope": "analytical parameter + gradient + AdamW moment bytes; not peak RSS; "
            "smoke updates use SGD, so AdamW moments are hypothetical",
            "cases": cases,
        }
    finally:
        torch.set_num_threads(previous_threads)
