"""Isolated CPU loss/step benchmarks and a profiler-guided rollout comparison."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from posttrain_lab.models import TinyCausalLM, TinyCausalLMConfig, freeze_model
from posttrain_lab.train import (
    RolloutConfig,
    Trainer,
    TrainerConfig,
    TrainingExample,
    chunked_masked_causal_linear_cross_entropy,
    chunked_masked_causal_linear_reverse_kl,
)

WARMUP, REPEATS, ROUNDS = 10, 30, 5
SHAPES = {
    "small": {"batch": 2, "length": 16, "vocab": 256, "hidden": 32, "teacher_hidden": 48},
    "large": {"batch": 4, "length": 64, "vocab": 16384, "hidden": 64, "teacher_hidden": 96},
}


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _summary(values):
    array = np.asarray(values)
    medians = np.median(array, axis=-1)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "round_median_min_ms": float(medians.min()),
        "round_median_max_ms": float(medians.max()),
    }


def _kernel(shape, objective, variant):
    generator = torch.Generator(device="cpu").manual_seed(29)
    b, t, v, h, ht = (
        shape[key] for key in ("batch", "length", "vocab", "hidden", "teacher_hidden")
    )
    hidden = torch.randn(b, t, h, generator=generator, device="cpu", requires_grad=True)
    weight = (torch.randn(v, h, generator=generator, device="cpu") * 0.02).requires_grad_()
    teacher_hidden = torch.randn(b, t, ht, generator=generator, device="cpu")
    teacher_weight = torch.randn(v, ht, generator=generator, device="cpu") * 0.02
    ids = torch.randint(v, (b, t), generator=generator, device="cpu")
    mask = torch.zeros((b, t), dtype=torch.bool, device="cpu")
    mask[:, t // 2 :] = True
    rows, positions = mask.nonzero(as_tuple=True)

    def loss():
        if variant == "chunked_recompute":
            if objective == "ce":
                return chunked_masked_causal_linear_cross_entropy(
                    hidden, weight, ids, mask, max_tokens_per_chunk=32
                ).loss
            return chunked_masked_causal_linear_reverse_kl(
                hidden, weight, teacher_hidden, teacher_weight, mask, max_tokens_per_chunk=32
            ).loss
        selected = variant == "selected"
        student_logits = F.linear(
            hidden[rows, positions - 1] if selected else hidden[:, :-1], weight
        )
        if objective == "ce":
            targets = ids[rows, positions] if selected else ids[:, 1:]
            losses = -student_logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        else:
            with torch.no_grad():
                teacher_logits = F.linear(
                    teacher_hidden[rows, positions - 1] if selected else teacher_hidden[:, :-1],
                    teacher_weight,
                )
                teacher_logp = teacher_logits.log_softmax(-1)
            student_logp = student_logits.log_softmax(-1)
            losses = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
        return losses.mean() if selected else losses[mask[:, 1:]].mean()

    return loss, (hidden, weight), int(mask.sum())


def _kernel_worker(spec):
    shape = SHAPES[spec["shape"]]
    baseline_rss = _rss_bytes()
    compute, parameters, tokens = _kernel(shape, spec["objective"], spec["variant"])
    results = {}
    for backward in (False, True):

        def run(backward=backward):
            for parameter in parameters:
                parameter.grad = None
            value = compute()
            if backward:
                value.backward()

        for _ in range(WARMUP):
            run()
        rounds = []
        for _ in range(ROUNDS):
            times = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                run()
                times.append((time.perf_counter() - start) * 1000)
            rounds.append(times)
        key = "forward_backward" if backward else "forward"
        results[key] = {**_summary(rounds), "rounds_ms": rounds}
    return {
        "spec": spec,
        "shape": shape,
        "selected_tokens": tokens,
        "timing": results,
        "runtime_baseline_peak_rss_bytes": baseline_rss,
        "process_peak_rss_bytes": _rss_bytes(),
        "kernel_effective_tokens_per_second": tokens
        / (results["forward_backward"]["median_ms"] / 1000),
    }


def _oracle_worker(spec):
    expected_loss = expected_grads = None
    comparisons = []
    for variant in ("dense", "selected", "chunked_recompute"):
        compute, parameters, _ = _kernel(SHAPES[spec["shape"]], spec["objective"], variant)
        loss = compute()
        loss.backward()
        grads = [parameter.grad.clone() for parameter in parameters]
        if expected_loss is None:
            expected_loss, expected_grads = loss.item(), grads
        torch.testing.assert_close(loss.item(), expected_loss, rtol=1e-4, atol=2e-6)
        for gradient, reference in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(gradient, reference, rtol=1e-4, atol=2e-6)
        comparisons.append(
            {
                "variant": variant,
                "loss_abs_error": abs(loss.item() - expected_loss),
                "gradient_max_abs_error": max(
                    (a - b).abs().max().item() for a, b in zip(grads, expected_grads, strict=True)
                ),
            }
        )
    return {"spec": spec, "comparisons": comparisons}


def _step_setup(variant):
    model_config = TinyCausalLMConfig(
        vocab_size=4096,
        hidden_size=48,
        intermediate_size=96,
        num_layers=2,
        num_heads=4,
        max_sequence_length=40,
    )
    teacher = TinyCausalLM(model_config, seed=2)
    freeze_model(teacher)
    trainer = Trainer(
        TinyCausalLM(model_config, seed=1),
        teacher=teacher,
        config=TrainerConfig(rollout=RolloutConfig(max_new_tokens=4, head_projection=variant)),
    )
    trainer.start_stage("opd", loss_tokens=1000000, seed=7)
    examples = [
        TrainingExample(str(i), (1, *[3 + (i + j) % 16 for j in range(23)]), (3, 2))
        for i in range(4)
    ]
    return trainer, examples


def _step_worker(spec):
    baseline_rss = _rss_bytes()
    trainer, examples = _step_setup(spec["variant"])
    for _ in range(WARMUP):
        trainer.step(examples)
    rounds = [[trainer.step(examples) for _ in range(REPEATS)] for _ in range(ROUNDS)]
    timings = {
        phase: _summary([[record["timings_ms"][phase] for record in run] for run in rounds])
        for phase in rounds[0][0]["timings_ms"]
    }
    records = [record for run in rounds for record in run]
    total_seconds = sum(record["timings_ms"]["total"] for record in records) / 1000
    return {
        "spec": spec,
        "timing": timings,
        "rounds": rounds,
        "runtime_baseline_peak_rss_bytes": baseline_rss,
        "process_peak_rss_bytes": _rss_bytes(),
        "effective_tokens_per_second": sum(r["loss_tokens"] for r in records) / total_seconds,
        "rollout_tokens_per_second": sum(r["rollout_tokens"] for r in records) / total_seconds,
    }


def _profile_worker(spec):
    trainer, examples = _step_setup(spec["variant"])
    for _ in range(WARMUP):
        trainer.step(examples)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profile:
        record = trainer.step(examples)
    events = profile.key_averages(group_by_input_shape=True)
    top = sorted(events, key=lambda event: event.self_cpu_time_total, reverse=True)[:12]
    lm_heads = [
        event
        for event in events
        if event.key == "aten::mm"
        and len(event.input_shapes) > 1
        and event.input_shapes[1] == [48, 4096]
    ]

    def describe(event):
        return {
            "operation": event.key,
            "input_shapes": event.input_shapes,
            "self_cpu_us": event.self_cpu_time_total,
            "calls": event.count,
        }

    return {
        "spec": spec,
        "profiled_step_timings_ms": record["timings_ms"],
        "top_operations": [describe(event) for event in top],
        "lm_head_projections": [describe(event) for event in lm_heads],
    }


def _workload_specs():
    return [
        *[
            {"kind": "oracle", "shape": shape, "objective": objective}
            for shape in SHAPES
            for objective in ("ce", "kl")
        ],
        *[
            {"kind": "kernel", "shape": shape, "objective": objective, "variant": variant}
            for shape in SHAPES
            for objective in ("ce", "kl")
            for variant in ("dense", "selected", "chunked_recompute")
        ],
        *[
            {"kind": kind, "variant": variant}
            for kind in ("step", "profile")
            for variant in ("dense", "last")
        ],
    ]


def run_cpu_benchmarks(output: Path):
    report = {
        "scope": "CPU FP32 synthetic workloads; kernel and end-to-end results kept separate",
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "threads": 1,
        },
        "protocol": {
            "warmup_per_phase": WARMUP,
            "repeats_per_round": REPEATS,
            "rounds": ROUNDS,
            "timing_scope": "kernel includes zero-grad, forward and optional backward; "
            "step includes sampling, Teacher, loss, backward, update and accounting",
            "memory_scope": "fresh process peak RSS includes Python/Torch, inputs and parameters; "
            "correctness oracle runs in a separate process",
            "dtype": "float32",
            "seed": 29,
            "chunk_size": 32,
        },
        "step_workload": {
            "objective": "opd",
            "batch": 4,
            "prompt_length": 24,
            "max_new_tokens": 4,
            "vocab": 4096,
            "hidden": 48,
            "layers": 2,
            "teacher_hidden": 48,
            "student_seed": 1,
            "teacher_seed": 2,
            "sampling_seed": 7,
        },
        "measurements": [],
    }
    if sys.platform == "darwin":
        report["runtime"]["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    for spec in _workload_specs():
        print(f"Measuring {spec}", file=sys.stderr, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "posttrain_lab.performance.cpu", "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            check=True,
        )
        report["measurements"].append(json.loads(result.stdout.splitlines()[-1]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def plot_cpu_benchmarks(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measurements = report["measurements"]
    colors = ("#526477", "#2c8b78", "#bd5e36")
    variants = ("dense", "selected", "chunked_recompute")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, shape in zip(axes[0], SHAPES, strict=True):
        for offset, (variant, color) in enumerate(zip(variants, colors, strict=True)):
            cases = [
                next(
                    m
                    for m in measurements
                    if m["spec"]
                    == {
                        "kind": "kernel",
                        "shape": shape,
                        "objective": objective,
                        "variant": variant,
                    }
                )
                for objective in ("ce", "kl")
            ]
            summaries = [case["timing"]["forward_backward"] for case in cases]
            values = [s["median_ms"] for s in summaries]
            errors = [
                [s["median_ms"] - s["round_median_min_ms"] for s in summaries],
                [s["round_median_max_ms"] - s["median_ms"] for s in summaries],
            ]
            ax.bar(
                np.arange(2) + (offset - 1) * 0.24,
                values,
                0.24,
                yerr=errors,
                capsize=3,
                label=variant,
                color=color,
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["CE", "Reverse KL"],
            ylabel="Forward + backward (ms)",
            title=f"{shape.title()} kernel workload",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for offset, (variant, color) in enumerate(zip(variants, colors, strict=True)):
        values = [
            next(
                m["process_peak_rss_bytes"] / 2**20
                for m in measurements
                if m["spec"]
                == {"kind": "kernel", "shape": "large", "objective": objective, "variant": variant}
            )
            for objective in ("ce", "kl")
        ]
        axes[1, 0].bar(np.arange(2) + (offset - 1) * 0.24, values, 0.24, color=color)
    axes[1, 0].set(
        xticks=[0, 1],
        xticklabels=["CE", "Reverse KL"],
        ylabel="Peak process RSS (MiB)",
        title="Large workload: RSS includes Python/Torch runtime",
    )
    for i, variant in enumerate(("dense", "last")):
        case = next(m for m in measurements if m["spec"] == {"kind": "step", "variant": variant})
        s = case["timing"]["total"]
        axes[1, 1].bar(
            i,
            s["median_ms"],
            color=colors[i],
            yerr=[
                [s["median_ms"] - s["round_median_min_ms"]],
                [s["round_median_max_ms"] - s["median_ms"]],
            ],
            capsize=5,
        )
        axes[1, 1].text(i, 0.3, f"{s['median_ms']:.2f} ms", ha="center", color="white")
    axes[1, 1].set(
        xticks=[0, 1],
        xticklabels=["Dense rollout head", "Last-position head"],
        ylabel="Complete OPD step (ms)",
        title="Overlapping ranges: no robust speedup claim",
    )
    fig.suptitle(
        "Apple M3 Pro CPU, FP32, 1 thread | error bars: range of 5 round medians", fontsize=13
    )
    fig.savefig(directory / "d11_performance.png", dpi=160)
    fig.savefig(directory / "d11_performance.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cpu/d11_performance.json"))
    args = parser.parse_args()
    if args.worker:
        torch.set_num_threads(1)
        spec = json.loads(args.worker)
        worker = {
            "kernel": _kernel_worker,
            "oracle": _oracle_worker,
            "step": _step_worker,
            "profile": _profile_worker,
        }[spec["kind"]]
        print(json.dumps(worker(spec), allow_nan=False))
    elif args.dry_run:
        print(
            json.dumps(
                {
                    "workloads": _workload_specs(),
                    "shapes": SHAPES,
                    "warmup": WARMUP,
                    "repeats": REPEATS,
                    "rounds": ROUNDS,
                },
                indent=2,
            )
        )
    else:
        report = (
            json.loads(args.output.read_text())
            if args.plot_only
            else run_cpu_benchmarks(args.output)
        )
        plot_cpu_benchmarks(report, args.output.parent)
        print(f"Saved {len(report['measurements'])} measurements to {args.output}")


if __name__ == "__main__":
    main()
