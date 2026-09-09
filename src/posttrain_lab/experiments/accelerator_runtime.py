"""D13: seeded local tiny-model CUDA witnesses, numerical checks and profiles.

No model/data downloads. This measures a single-device development runtime,
not Gemma quality, real-model memory feasibility or distributed communication.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from posttrain_lab.models import (
    TinyCausalLM,
    TinyCausalLMConfig,
    configure_trainable_parameters,
    freeze_model,
)
from posttrain_lab.train import RolloutConfig, Trainer, TrainerConfig, TrainingExample
from posttrain_lab.train.grpo_surrogate import (
    compute_exact_group_advantages,
    dr_grpo_token_surrogate,
)
from posttrain_lab.train.model_losses import causal_lm_opd_loss, causal_lm_sft_loss
from posttrain_lab.train.rollout import selected_token_statistics, supervised_batch

SMALL = TinyCausalLMConfig(
    vocab_size=16,
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    max_sequence_length=16,
)
EXAMPLES = [
    TrainingExample("c", (1, 3), (3, 4, 2)),
    TrainingExample("a", (1, 5, 4), (4, 2)),
    TrainingExample("b", (1, 6), (2,)),
]


def witness() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("D13 requires an explicitly provided CUDA GPU")
    generator = torch.Generator(device="cuda:0").manual_seed(13)
    value = torch.randn(32, 32, generator=generator, device="cuda:0", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        projected = value @ value.T
        loss = projected.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert projected.dtype == torch.bfloat16 and torch.isfinite(value.grad).all()
    model = TinyCausalLM(SMALL, seed=13)
    trainer = Trainer(
        model,
        config=TrainerConfig(
            device="cuda:0", precision="bf16", microbatch_size=2, activation_checkpointing=True
        ),
    )
    trainer.start_stage("sft", loss_tokens=5, seed=13)
    before = model.token_embedding.weight.detach().clone()
    record = trainer.step(EXAMPLES)
    assert record["status"] == "updated" and record["loss_tokens"] == 5
    assert not torch.equal(before, model.token_embedding.weight)
    return {
        "sentinel": "D13_CUDA_WITNESS",
        "shape": list(projected.shape),
        "dtype": str(projected.dtype),
        "device": torch.cuda.get_device_name(0),
        "loss": loss.item(),
        "gradient_norm": value.grad.norm().item(),
        "tiny_sft_loss": record["loss"],
        "updated_loss_tokens": record["loss_tokens"],
    }


def make_trainer(
    *,
    objective=None,
    mode="text",
    precision="full",
    microbatch=None,
    checkpoint=False,
    seed=13,
    budget=257,
) -> Trainer:
    student = TinyCausalLM(SMALL, seed=seed)
    teacher = TinyCausalLM(replace(SMALL, hidden_size=48, intermediate_size=96), seed=seed + 10)
    configure_trainable_parameters(student, mode, seed=9)
    freeze_model(teacher)
    trainer = Trainer(
        student,
        teacher=teacher,
        reward_fn=lambda c, e: float(c[0] >= 8),
        config=TrainerConfig(
            device="cuda:0",
            precision=precision,
            rollout=RolloutConfig(max_new_tokens=3),
            microbatch_size=microbatch,
            activation_checkpointing=checkpoint,
            max_tokens_per_chunk=5,
        ),
    )
    if objective:
        trainer.start_stage(objective, loss_tokens=budget, seed=23)
    return trainer


def vector(model, *, gradients=False):
    return torch.cat(
        [
            (p.grad if gradients else p).detach().flatten().float()
            for p in model.parameters()
            if p.requires_grad
        ]
    )


def difference(actual, reference) -> dict:
    delta = actual - reference
    return {
        "max_absolute": delta.abs().max().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-12)).item(),
    }


def numerical_precision() -> list[dict]:
    """FP32 and BF16 see identical prefixes, masks, old policy and advantages."""
    batch = supervised_batch(
        [TrainingExample(str(i), (1, 3 + i % 2), (3 + i % 2, 2)) for i in range(16)],
        RolloutConfig(),
        device="cuda:0",
    )
    group = compute_exact_group_advantages(
        torch.arange(16, device="cuda:0").remainder(2).float(),
        torch.arange(16, device="cuda:0") // 8,
        batch.completion_mask,
    )
    old_model = TinyCausalLM(SMALL, seed=13).cuda()
    with torch.no_grad():
        old, _ = selected_token_statistics(
            old_model.forward_features(batch.input_ids), batch.input_ids, batch.completion_mask
        )
    results = []
    for mode in ("text", "lora"):
        for objective in ("sft", "opd", "grpo"):
            losses, gradients = [], []
            for precision in ("full", "bf16"):
                trainer = make_trainer(mode=mode, precision=precision)
                with trainer.autocast():
                    if objective == "sft":
                        output = causal_lm_sft_loss(
                            trainer.student, batch.input_ids, batch.completion_mask
                        )
                    elif objective == "opd":
                        output = causal_lm_opd_loss(
                            trainer.student, trainer.teacher, batch.input_ids, batch.completion_mask
                        )
                    else:
                        logp, _ = selected_token_statistics(
                            trainer.student.forward_features(batch.input_ids),
                            batch.input_ids,
                            batch.completion_mask,
                        )
                        output = dr_grpo_token_surrogate(
                            logp,
                            old,
                            group.advantages,
                            group.loss_mask,
                            global_active_completion_count=16,
                            max_completion_length=4,
                        )
                output.loss.backward()
                losses.append(output.loss.item())
                gradients.append(vector(trainer.student, gradients=True).clone())
            error = difference(gradients[1], gradients[0])
            # Global L2 avoids meaningless relative errors on near-zero individual weights.
            assert error["relative_l2"] < 0.03, (mode, objective, error)
            assert abs(losses[1] - losses[0]) < 0.003, (mode, objective, losses)
            results.append(
                {
                    "mode": mode,
                    "objective": objective,
                    "fp32_loss": losses[0],
                    "bf16_loss": losses[1],
                    "gradient_error": error,
                    "tolerance": {"gradient_relative_l2": 0.03, "loss_absolute": 0.003},
                }
            )
    return results


def accumulation_checks() -> list[dict]:
    results = []
    for mode in ("text", "lora"):
        for objective in ("sft", "grpo", "opd"):
            full = make_trainer(objective=objective, mode=mode, budget=11)
            split = make_trainer(
                objective=objective, mode=mode, microbatch=2, checkpoint=True, budget=11
            )
            a, b = full.step(EXAMPLES), split.step(EXAMPLES)
            assert a["status"] == b["status"] == "updated"
            assert a["loss_tokens"] == b["loss_tokens"]
            assert torch.equal(full.last_batch.input_ids, split.last_batch.input_ids)
            grad_error = difference(
                vector(split.student, gradients=True), vector(full.student, gradients=True)
            )
            parameter_error = difference(vector(split.student), vector(full.student))
            assert grad_error["relative_l2"] < 5e-5, (mode, objective, grad_error)
            assert parameter_error["max_absolute"] < 2e-5, (mode, objective, parameter_error)
            results.append(
                {
                    "mode": mode,
                    "objective": objective,
                    "gradient_error": grad_error,
                    "parameter_error": parameter_error,
                    "loss_error": abs(a["loss"] - b["loss"]),
                    "microbatches": b["microbatches"],
                    "loss_tokens": b["loss_tokens"],
                    "tolerance": {"gradient_relative_l2": 5e-5, "parameter_absolute": 2e-5},
                }
            )
    return results


def resume_checks(checkpoint_directory: Path) -> list[dict]:
    results = []
    for mode in ("text", "lora"):
        for objective in ("sft", "grpo", "opd"):
            trainer = make_trainer(
                objective=objective, mode=mode, precision="bf16", microbatch=2, checkpoint=True
            )
            initial = copy.deepcopy(trainer.student.state_dict())
            teacher_before = copy.deepcopy(trainer.teacher.state_dict())
            trainer.step(EXAMPLES)
            path = checkpoint_directory / f"{objective}-{mode}.pt"
            trainer.save_checkpoint(path)
            restored = make_trainer(
                mode=mode, precision="bf16", microbatch=2, checkpoint=True, seed=91
            )
            restored.load_checkpoint(path)
            trainer.run_stage(EXAMPLES, batch_size=2)
            restored.run_stage(EXAMPLES, batch_size=2)
            assert trainer.budget.complete and restored.budget.complete
            equal = torch.equal(vector(trainer.student), vector(restored.student))
            assert equal
            for left, right in zip(trainer.parameters, restored.parameters, strict=True):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    assert torch.equal(
                        trainer.optimizer.state[left][key], restored.optimizer.state[right][key]
                    )
            for a, b in zip(trainer.records, restored.records, strict=True):
                for key in ("loss", "loss_tokens", "reward_mean", "gradient_norm", "learning_rate"):
                    assert a[key] == b[key], (mode, objective, key, a[key], b[key])
            assert torch.equal(trainer.generator.get_state(), restored.generator.get_state())
            assert torch.equal(trainer.last_batch.input_ids, restored.last_batch.input_ids)
            changed = [
                name
                for name, p in trainer.student.named_parameters()
                if not torch.equal(p, initial[name])
            ]
            assert changed
            if mode == "lora":
                assert all(name.endswith(("lora_A", "lora_B")) for name in changed)
            assert all(
                p.grad is None and torch.equal(p, teacher_before[name])
                for name, p in trainer.teacher.named_parameters()
            )
            # Switching after a restored stage must reset moments and LR while retaining weights.
            for current in (trainer, restored):
                current.start_stage("opd", loss_tokens=7, seed=31)
                assert not current.optimizer.state
                current.run_stage(EXAMPLES)
            assert torch.equal(vector(trainer.student), vector(restored.student))
            results.append(
                {
                    "mode": mode,
                    "objective": objective,
                    "precision": "bf16",
                    "activation_checkpointing": True,
                    "microbatch_size": 2,
                    "resume_parameters_and_optimizer_exact": equal,
                    "teacher_unchanged": True,
                    "changed_parameters": len(changed),
                    "stages_loss_tokens": [257, 7],
                    "checkpoint_bytes": path.stat().st_size,
                    "records": trainer.records,
                }
            )
    return results


VARIANTS = {
    "fp32_full": ("full", None, False),
    "bf16_full": ("bf16", None, False),
    "bf16_accum2": ("bf16", 2, False),
    "bf16_accum2_checkpoint": ("bf16", 2, True),
}


def profile_case(objective, variant, rounds, steps, warmup):
    """Fresh process per variant. Round seeds are paired across the four variants."""
    config = TinyCausalLMConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=384,
        num_layers=4,
        num_heads=4,
        max_sequence_length=96,
    )
    teacher_config = replace(config, hidden_size=192, intermediate_size=576)
    data = [
        TrainingExample(
            f"profile-{i}", tuple([1] + [3 + i % 8] * 47), tuple([4 + i % 8] * 15 + [2])
        )
        for i in range(8)
    ]
    precision, microbatch, checkpoint = VARIANTS[variant]
    records = []
    for repeat in range(rounds):
        student, teacher = (
            TinyCausalLM(config, seed=repeat + 50),
            TinyCausalLM(teacher_config, seed=repeat + 60),
        )
        freeze_model(teacher)
        trainer = Trainer(
            student,
            teacher=teacher,
            reward_fn=lambda c, e: float(c[0] >= 2048),
            config=TrainerConfig(
                device="cuda:0",
                precision=precision,
                microbatch_size=microbatch,
                activation_checkpointing=checkpoint,
                max_tokens_per_chunk=64,
                rollout=RolloutConfig(max_new_tokens=8),
                learning_rates=(1e-4, 1e-4, 1e-4),
            ),
        )
        trainer.start_stage(objective, loss_tokens=10_000_000, seed=repeat + 100)
        # GRPO expands each of two prompts to eight completions; SFT/OPD use eight prompts.
        active_data = data[:2] if objective == "grpo" else data
        for _ in range(warmup):
            trainer.step(active_data)
        for _ in range(steps):
            record = trainer.step(active_data)
            assert record["status"] == "updated"
            records.append({"round": repeat, **record})
        del trainer, student, teacher
        gc.collect()
        torch.cuda.empty_cache()
    medians = [
        statistics.median(r["timings_ms"]["total"] for r in records if r["round"] == repeat)
        for repeat in range(rounds)
    ]
    totals = sorted(r["timings_ms"]["total"] for r in records)
    return {
        "objective": objective,
        "variant": variant,
        "student": asdict(config),
        "teacher": asdict(teacher_config),
        "prompt_length": 48,
        "sft_completion_length": 16,
        "max_new_tokens": 8,
        "prompt_batch_size": 2 if objective == "grpo" else 8,
        "group_size": 8,
        "rounds": rounds,
        "warmup_steps_per_round": warmup,
        "timed_steps_per_round": steps,
        "median_step_ms": statistics.median(totals),
        "p95_step_ms": totals[int(0.95 * (len(totals) - 1))],
        "round_median_range_ms": [min(medians), max(medians)],
        "mean_update_ms": statistics.mean(r["timings_ms"]["update"] for r in records),
        "phase_mean_ms": {
            phase: statistics.mean(r["timings_ms"][phase] for r in records)
            for phase in ("sample", "score", "loss", "backward", "update")
        },
        "effective_loss_tokens_per_second": sum(r["loss_tokens"] for r in records)
        / (sum(totals) / 1000),
        "peak_allocated_mib": max(r["memory"]["peak_allocated_bytes"] for r in records) / 2**20,
        "peak_reserved_mib": max(r["memory"]["peak_reserved_bytes"] for r in records) / 2**20,
        "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/gpu/d13_runtime.json"))
    parser.add_argument("--profile-child", choices=("sft", "grpo", "opd"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.rounds, args.steps, args.warmup) < 1:
        parser.error("rounds, steps and warmup must be positive")
    if args.profile_child and args.variant is None:
        parser.error("--profile-child requires --variant")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "scope": "planned single-GPU tiny-model validation; no GPU accessed by dry-run",
                    "numerical_model": asdict(SMALL),
                    "variants": VARIANTS,
                    "objectives": ["sft", "grpo", "opd"],
                    "modes": ["text", "lora"],
                    "rounds": args.rounds,
                    "warmup": args.warmup,
                    "steps": args.steps,
                    "output": str(args.output),
                    "real_models_and_data": False,
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.profile_child:
        result = profile_case(
            args.profile_child, args.variant, args.rounds, args.steps, args.warmup
        )
    else:
        proof = witness()
        print(json.dumps(proof), flush=True)
        if args.witness_only:
            return
        result = {
            "scope": "D13 synthetic tiny-model single-GPU runtime; no real-model/data training",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_build": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                "threads": torch.get_num_threads(),
                "tf32": False,
            },
            "measurement_scope": "synchronized wall time including Python and phase sync; "
            "memory is per-process PyTorch allocator peak, excluding driver/context; "
            "fresh process per profile variant; initialization/save excluded",
            "communication": {
                "status": "not_applicable",
                "world_size": 1,
                "reason": "single GPU; no FSDP/ZeRO requirement or distributed claim",
            },
            "witness": proof,
            "numerical_precision": numerical_precision(),
        }
        print("BF16 numerical checks passed", flush=True)
        result["accumulation"] = accumulation_checks()
        print("FP32 accumulation/checkpoint comparisons passed", flush=True)
        checkpoint_dir = Path("outputs/d13/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result["resume"] = resume_checks(checkpoint_dir)
        print("BF16 exact resume and stage transition checks passed", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        result["profiles"] = []
        # The parent is idle while a child profiles; all variants use the same one GPU serially.
        with tempfile.TemporaryDirectory(prefix="d13-profile-") as temporary:
            for objective in ("sft", "grpo", "opd"):
                for variant in VARIANTS:
                    child_output = Path(temporary) / f"{objective}-{variant}.json"
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "posttrain_lab.experiments.accelerator_runtime",
                            "--profile-child",
                            objective,
                            "--variant",
                            variant,
                            "--rounds",
                            str(args.rounds),
                            "--steps",
                            str(args.steps),
                            "--warmup",
                            str(args.warmup),
                            "--output",
                            str(child_output),
                        ],
                        check=True,
                        timeout=600,
                        env=os.environ.copy(),
                    )
                    result["profiles"].append(json.loads(child_output.read_text()))
                    print(f"Profile complete: {objective}/{variant}", flush=True)
        result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
