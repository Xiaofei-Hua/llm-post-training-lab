"""Five-arm CPU learning example plus a deliberately incorrect Teacher control."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from posttrain_lab.models import TinyCausalLM, TinyCausalLMConfig, freeze_model
from posttrain_lab.train import RolloutConfig, Trainer, TrainerConfig, TrainingExample
from posttrain_lab.train.loop import StageBudgetIncomplete
from posttrain_lab.train.rollout import sample_completions, supervised_batch

ARMS = {
    "A0": ("sft", "sft"),
    "A1": ("grpo", "grpo"),
    "A2": ("opd", "opd"),
    "A3": ("opd", "grpo"),
    "A4": ("grpo", "opd"),
}


def learning_config():
    return {
        "scope": "CPU synthetic copy-bit task; no Gemma recipe selection or research claims",
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "paired_cpu_seeds": [11, 22, 33],
        "student": asdict(
            TinyCausalLMConfig(
                vocab_size=9,
                hidden_size=24,
                intermediate_size=48,
                num_layers=1,
                num_heads=4,
                max_sequence_length=8,
            )
        ),
        "teacher": asdict(
            TinyCausalLMConfig(
                vocab_size=9,
                hidden_size=32,
                intermediate_size=64,
                num_layers=1,
                num_heads=4,
                max_sequence_length=8,
            )
        ),
        "anchor_loss_tokens": 64,
        "teacher_loss_tokens": 512,
        "stage_loss_tokens": 128,
        "trainer": asdict(
            TrainerConfig(
                rollout=RolloutConfig(max_new_tokens=2),
                learning_rates=(0.02, 0.005, 0.005),
            )
        ),
        "arms": ARMS,
        "max_attempts_per_stage": 256,
        "evaluate_every_updates": 4,
        "task": {
            "prompt": "[BOS=1, context, bit]",
            "reference": "[bit, EOS=2]",
            "bits": [3, 4],
            "training_contexts": [5, 6],
            "evaluation_contexts": [7, 8],
            "reward": "exact completion-token equality; malformed/extra tokens receive zero",
        },
    }


def copy_bit_examples(contexts, *, flip=False):
    return [
        TrainingExample(
            f"context-{context}/bit-{bit}", (1, context, bit), (7 - bit if flip else bit, 2)
        )
        for context in contexts
        for bit in (3, 4)
    ]


def exact_completion_reward(completion, example):
    return float(completion == example.completion)


def _trainer_config(config):
    values = dict(config["trainer"])
    values["rollout"] = RolloutConfig(**values["rollout"])
    return TrainerConfig(**values)


@torch.no_grad()
def evaluate_copy_task(student, teacher, examples, rollout):
    """Held-out synthetic contexts; evaluation never uses the trainer RNG."""
    was_training = student.training
    student.eval()
    try:
        batch = supervised_batch(examples, rollout)
        mask = batch.completion_mask[:, 1:]
        logits = student(batch.input_ids, batch.attention_mask)[:, :-1]
        logp = logits.log_softmax(-1)
        chosen = logp.gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
        sequence_logp = chosen.masked_fill(~mask, 0).sum(-1)
        teacher_logp = teacher(batch.input_ids, batch.attention_mask)[:, :-1].log_softmax(-1)
        kl = (logp.exp() * (logp - teacher_logp)).sum(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        greedy = sample_completions(
            student,
            examples,
            rollout,
            generator=torch.Generator(device="cpu").manual_seed(999),
            policy_version=0,
            greedy=True,
        )
        return {
            "mean_correct_completion_probability": sequence_logp.exp().mean().item(),
            "greedy_accuracy": float(
                np.mean(
                    [
                        exact_completion_reward(c, e)
                        for c, e in zip(greedy.completions, examples, strict=True)
                    ]
                )
            ),
            "reference_token_nll": -chosen[mask].mean().item(),
            "reference_prefix_entropy": entropy[mask].mean().item(),
            "reference_prefix_reverse_kl": kl[mask].mean().item(),
            "greedy_mean_completion_length": greedy.completion_mask.sum(-1).float().mean().item(),
            "greedy_truncation_rate": greedy.truncated.float().mean().item(),
        }
    finally:
        student.train(was_training)


def _fit_sft(model, config, examples, tokens, seed):
    trainer = Trainer(model, config=_trainer_config(config))
    trainer.start_stage("sft", loss_tokens=tokens, seed=seed)
    trainer.run_stage(examples, max_attempts=256)
    return trainer.records


def _run_arm(anchor, teacher, config, train, evaluation, seed, arm, objectives):
    student = copy.deepcopy(anchor)
    trainer = Trainer(
        student, teacher=teacher, reward_fn=exact_completion_reward, config=_trainer_config(config)
    )
    curves = [
        {
            "loss_tokens": 0,
            "stage": 0,
            **evaluate_copy_task(student, teacher, evaluation, trainer.config.rollout),
        }
    ]
    stage_summaries = []
    for stage, objective in enumerate(objectives, start=1):
        trainer.start_stage(
            objective, loss_tokens=config["stage_loss_tokens"], seed=seed + stage * 1000
        )
        first_record = len(trainer.records)
        updates = 0
        for _ in range(config["max_attempts_per_stage"]):
            record = trainer.step(train)
            updates += record["status"] == "updated"
            if (
                record["status"] == "updated" and updates % config["evaluate_every_updates"] == 0
            ) or trainer.budget.complete:
                curves.append(
                    {
                        "loss_tokens": (stage - 1) * config["stage_loss_tokens"]
                        + trainer.budget.consumed_tokens,
                        "stage": stage,
                        **evaluate_copy_task(student, teacher, evaluation, trainer.config.rollout),
                    }
                )
            if trainer.budget.complete:
                break
        if not trainer.budget.complete:
            raise StageBudgetIncomplete(
                f"{arm} seed={seed} stage={stage} did not close its CPU budget"
            )
        records = trainer.records[first_record:]
        stage_summaries.append(
            {
                "stage": stage,
                "objective": objective,
                "loss_tokens": trainer.budget.consumed_tokens,
                "attempts": len(records),
                "updates": updates,
                "skipped_zero_variance": sum(
                    r["status"] == "skipped_zero_variance" for r in records
                ),
                "rollout_tokens": sum(r["rollout_tokens"] for r in records),
                "training_step_seconds": sum(r["timings_ms"]["total"] for r in records) / 1000,
            }
        )
    return {
        "seed": seed,
        "arm": arm,
        "objectives": objectives,
        "stages": stage_summaries,
        "curves": curves,
        "updates": trainer.records,
        "final": curves[-1],
    }


def run_cpu_learning(output: Path):
    config = learning_config()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    start = time.perf_counter()
    report = {
        "config": config,
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "anchors": [],
        "runs": [],
        "failure_control": None,
    }
    train = copy_bit_examples(config["task"]["training_contexts"])
    evaluation = copy_bit_examples(config["task"]["evaluation_contexts"])
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        for seed in config["paired_cpu_seeds"]:
            print(f"CPU seed {seed}: fitting local Student anchor and Teacher", flush=True)
            anchor = TinyCausalLM(TinyCausalLMConfig(**config["student"]), seed=seed)
            anchor_records = _fit_sft(anchor, config, train, config["anchor_loss_tokens"], seed)
            teacher = TinyCausalLM(TinyCausalLMConfig(**config["teacher"]), seed=seed + 100)
            teacher_records = _fit_sft(teacher, config, train, config["teacher_loss_tokens"], seed)
            freeze_model(teacher)
            teacher_before = copy.deepcopy(teacher.state_dict())
            report["anchors"].append(
                {
                    "seed": seed,
                    "student_training": anchor_records,
                    "teacher_training": teacher_records,
                    "student": evaluate_copy_task(
                        anchor, teacher, evaluation, _trainer_config(config).rollout
                    ),
                    "teacher": evaluate_copy_task(
                        teacher, teacher, evaluation, _trainer_config(config).rollout
                    ),
                }
            )
            for arm, objectives in ARMS.items():
                print(f"CPU seed {seed}: {arm} {' -> '.join(objectives)}", flush=True)
                report["runs"].append(
                    _run_arm(anchor, teacher, config, train, evaluation, seed, arm, objectives)
                )
                save()
            for name, parameter in teacher.named_parameters():
                if parameter.grad is not None or not torch.equal(parameter, teacher_before[name]):
                    raise AssertionError("Teacher was modified during the five-arm run")
            if seed == config["paired_cpu_seeds"][0]:
                print(
                    "CPU failure control: Teacher trained on deliberately inverted labels",
                    flush=True,
                )
                wrong_teacher = TinyCausalLM(
                    TinyCausalLMConfig(**config["teacher"]), seed=seed + 100
                )
                wrong_training = _fit_sft(
                    wrong_teacher,
                    config,
                    copy_bit_examples(config["task"]["training_contexts"], flip=True),
                    config["teacher_loss_tokens"],
                    seed,
                )
                freeze_model(wrong_teacher)
                report["failure_control"] = {
                    "description": "deliberate incorrect-Teacher stress test; same OPD recipe, "
                    "anchor and two-stage budget as A2 for the first CPU seed",
                    "teacher_training": wrong_training,
                    "teacher_on_correct_task": evaluate_copy_task(
                        wrong_teacher, teacher, evaluation, _trainer_config(config).rollout
                    ),
                    "run": _run_arm(
                        anchor,
                        wrong_teacher,
                        config,
                        train,
                        evaluation,
                        seed,
                        "wrong_teacher_opd",
                        ("opd", "opd"),
                    ),
                }
                save()
        report["wall_seconds_including_anchor_evaluation_and_json_writes"] = (
            time.perf_counter() - start
        )
        report["complete"] = True
        save()
    finally:
        torch.set_num_threads(previous_threads)
    return report


def plot_cpu_learning(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(ARMS, ("#526477", "#bd5e36", "#2c8b78", "#7258a5", "#b48a20"), strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    metrics = [
        ("mean_correct_completion_probability", "Correct completion probability"),
        ("greedy_accuracy", "Held-out context greedy accuracy"),
        ("reference_prefix_entropy", "Entropy on reference prefixes (nats)"),
        ("reference_prefix_reverse_kl", "KL(Student || Teacher) on reference prefixes"),
        ("greedy_mean_completion_length", "Greedy completion length"),
        ("greedy_truncation_rate", "Greedy truncation rate"),
    ]
    grid = np.linspace(0, 2 * report["config"]["stage_loss_tokens"], 33)
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for arm in ARMS:
            runs = [run for run in report["runs"] if run["arm"] == arm]
            values = np.array(
                [
                    np.interp(
                        grid,
                        [point["loss_tokens"] for point in run["curves"]],
                        [point[metric] for point in run["curves"]],
                    )
                    for run in runs
                ]
            )
            ax.plot(grid, values.mean(0), label=arm, color=colors[arm])
            ax.fill_between(grid, values.min(0), values.max(0), color=colors[arm], alpha=0.12)
        ax.axvline(
            report["config"]["stage_loss_tokens"], color="#888888", linestyle="--", linewidth=1
        )
        ax.set(title=title, xlabel="Cumulative Student backward loss tokens")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylim(0, 1.04)
    axes[0, 1].set_ylim(0, 1.04)
    axes[1, 2].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(ncol=5, fontsize=8)
    fig.suptitle(
        "CPU synthetic copy-bit learning | 3 paired seeds | bands = seed min/max", fontsize=14
    )
    fig.savefig(directory / "d12_learning.png", dpi=160)
    fig.savefig(directory / "d12_learning.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, objective, metric, title in (
        (axes[0, 0], "sft", "loss", "SFT token CE"),
        (axes[0, 1], "opd", "kl", "On-policy OPD reverse KL"),
        (axes[0, 2], "grpo", "effective_group_rate", "GRPO effective-group rate"),
    ):
        for run in report["runs"]:
            for stage in (1, 2):
                records = [
                    r
                    for r in run["updates"]
                    if r["objective"] == objective and r["stage"] == stage and r[metric] is not None
                ]
                if records:
                    ax.plot(
                        [
                            (stage - 1) * report["config"]["stage_loss_tokens"]
                            + r["cumulative_loss_tokens"]
                            for r in records
                        ],
                        [r[metric] for r in records],
                        color=colors[run["arm"]],
                        alpha=0.5,
                    )
        ax.set(title=title, xlabel="Cumulative Student backward loss tokens")
        ax.grid(alpha=0.2)
    for arm in ARMS:
        runs = [run for run in report["runs"] if run["arm"] == arm]
        axes[1, 0].scatter(
            [sum(s["training_step_seconds"] for s in run["stages"]) for run in runs],
            [run["final"]["mean_correct_completion_probability"] for run in runs],
            label=arm,
            color=colors[arm],
        )
    axes[1, 0].set(
        title="Observed learning vs CPU step cost",
        xlabel="Training step seconds (single runs)",
        ylabel="Correct completion probability",
    )
    axes[1, 0].legend(ncol=3, fontsize=8)
    failure = report["failure_control"]["run"]
    control = next(
        run for run in report["runs"] if run["seed"] == failure["seed"] and run["arm"] == "A2"
    )
    for run, label, color in (
        (control, "Correct Teacher", "#2c8b78"),
        (failure, "Deliberately wrong Teacher", "#bd5e36"),
    ):
        axes[1, 1].plot(
            [p["loss_tokens"] for p in run["curves"]],
            [p["mean_correct_completion_probability"] for p in run["curves"]],
            label=label,
            color=color,
        )
    axes[1, 1].set(
        title="Failure control: Teacher error transfer",
        xlabel="Student backward loss tokens",
        ylabel="Correct completion probability",
    )
    axes[1, 1].legend(fontsize=8)
    for arm in ARMS:
        runs = [run for run in report["runs"] if run["arm"] == arm]
        axes[1, 2].bar(
            arm,
            np.mean([sum(s["rollout_tokens"] for s in run["stages"]) for run in runs]),
            color=colors[arm],
        )
    axes[1, 2].set(
        title="Generated tokens (including discarded groups)", ylabel="Mean across 3 CPU seeds"
    )
    fig.suptitle(
        "CPU diagnostics | fixed 256 backward tokens per arm | no Gemma inference", fontsize=14
    )
    fig.savefig(directory / "d12_diagnostics.png", dpi=160)
    fig.savefig(directory / "d12_diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cpu/d12_learning.json"))
    parser.add_argument(
        "--plot-only", action="store_true", help="Render plots from an existing result"
    )
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(learning_config(), indent=2))
        return
    report = (
        json.loads(args.output.read_text()) if args.plot_only else run_cpu_learning(args.output)
    )
    plot_cpu_learning(report, args.output.parent)
    print(f"Saved CPU learning evidence and figures in {args.output.parent}")


if __name__ == "__main__":
    main()
