"""D13 regression tests: accumulation math and exact stochastic resume on CPU."""

import copy
from dataclasses import replace

import pytest
import torch

from posttrain_lab.models import (
    TinyCausalLM,
    TinyCausalLMConfig,
    configure_trainable_parameters,
    freeze_model,
)
from posttrain_lab.train import RolloutConfig, Trainer, TrainerConfig, TrainingExample


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_trainer(objective, *, microbatch=None, checkpoint=False, mode="text", seed=1):
    def model(model_seed):
        return TinyCausalLM(
            TinyCausalLMConfig(
                vocab_size=9,
                hidden_size=12,
                intermediate_size=24,
                num_layers=2,
                num_heads=3,
                max_sequence_length=12,
            ),
            seed=model_seed,
            dtype=torch.float64,
        )

    student, teacher = model(seed), model(seed + 10)
    configure_trainable_parameters(student, mode, seed=7)
    freeze_model(teacher)
    trainer = Trainer(
        student,
        teacher=teacher,
        reward_fn=lambda c, e: float(c[0] >= 4),
        config=TrainerConfig(
            rollout=RolloutConfig(max_new_tokens=3),
            max_tokens_per_chunk=3,
            microbatch_size=microbatch,
            activation_checkpointing=checkpoint,
        ),
    )
    if objective is not None:
        trainer.start_stage(objective, loss_tokens=83, seed=9)
    return trainer


def examples():
    # Unequal completion and prompt lengths exercise token, rather than row, normalization.
    return [
        TrainingExample("c", (1, 3), (3, 4, 2)),
        TrainingExample("a", (1, 5, 4), (4, 2)),
        TrainingExample("b", (1, 6), (2,)),
    ]


@pytest.mark.parametrize("objective", ["sft", "grpo", "opd"])
@pytest.mark.parametrize("mode", ["text", "lora"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_microbatch_matches_full_batch_gradients_updates_and_global_denominator(
    objective, mode, checkpoint
):
    full = make_trainer(objective, mode=mode)
    split = make_trainer(objective, mode=mode, microbatch=2, checkpoint=checkpoint)
    # Partial final batch also leaves some microbatches entirely unselected.
    full.budget = type(full.budget)(11)
    split.budget = type(split.budget)(11)
    for _ in range(4):
        if full.budget.complete:
            break
        a, b = full.step(examples()), split.step(examples())
        for key in (
            "loss_tokens",
            "candidate_loss_tokens",
            "budget_truncated",
            "policy_version_after",
        ):
            assert a[key] == b[key]
        for key in ("loss", "entropy", "gradient_norm", "importance_ratio_mean", "kl"):
            assert a[key] == pytest.approx(b[key], abs=2e-10)
        assert torch.equal(full.last_batch.input_ids, split.last_batch.input_ids)
        for p, q in zip(full.student.parameters(), split.student.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-8)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-11, rtol=1e-8)
    assert full.budget.complete and split.budget.complete


@pytest.mark.parametrize("objective", ["sft", "grpo", "opd"])
@pytest.mark.parametrize("mode", ["text", "lora"])
def test_save_resume_reproduces_rollouts_moments_schedule_and_stage_transition(
    tmp_path, objective, mode
):
    full = make_trainer(objective, microbatch=2, checkpoint=True, mode=mode)
    full.step(examples())
    checkpoint = tmp_path / "resume.pt"
    full.save_checkpoint(checkpoint)
    # Different initial parameters must be replaced, including the frozen Teacher.
    resumed = make_trainer(None, microbatch=2, checkpoint=True, mode=mode, seed=999)
    resumed.load_checkpoint(checkpoint)
    assert resumed.budget.state_dict() == full.budget.state_dict()
    assert resumed.stage == full.stage
    assert resumed.attempt == full.attempt
    full.run_stage(examples(), batch_size=2)
    resumed.run_stage(examples(), batch_size=2)
    for a, b in zip(full.records, resumed.records, strict=True):
        for key in (
            "status",
            "loss",
            "loss_tokens",
            "learning_rate",
            "gradient_norm",
            "reward_mean",
        ):
            assert a[key] == b[key]
    for p, q in zip(full.student.parameters(), resumed.student.parameters(), strict=True):
        assert torch.equal(p, q)
    assert torch.equal(full.generator.get_state(), resumed.generator.get_state())
    for p, q in zip(full.parameters, resumed.parameters, strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(full.optimizer.state[p][key], resumed.optimizer.state[q][key])
    for trainer in (full, resumed):
        trainer.start_stage("opd", loss_tokens=5, seed=4)
        trainer.step(examples())
    assert full.records[-1]["loss"] == resumed.records[-1]["loss"]
    assert full.stage == resumed.stage == 2


def test_failed_later_microbatch_discards_all_accumulated_gradients_and_budget():
    trainer = make_trainer("sft", microbatch=1)
    before = copy.deepcopy(trainer.student.state_dict())
    calls = 0

    def corrupt_second(gradient):
        nonlocal calls
        calls += 1
        return gradient if calls == 1 else gradient * torch.nan

    hook = trainer.student.token_embedding.weight.register_hook(corrupt_second)
    result = trainer.step(examples())
    hook.remove()
    assert result["status"] == "skipped_nonfinite_gradient"
    assert trainer.budget.consumed_tokens == trainer.policy_version == 0
    assert trainer.scheduler.last_epoch == 0
    assert not trainer.optimizer.state
    for name, parameter in trainer.student.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, before[name])
    assert trainer.step(examples())["status"] == "updated"


def test_resume_rejects_changed_runtime_before_overwriting_student(tmp_path):
    trainer = make_trainer("sft")
    trainer.step(examples())
    trainer.save_checkpoint(tmp_path / "state.pt")
    fresh = make_trainer(None, microbatch=1, seed=99)
    before = copy.deepcopy(fresh.student.state_dict())
    with pytest.raises(ValueError, match="runtime configuration"):
        fresh.load_checkpoint(tmp_path / "state.pt")
    for name, p in fresh.student.named_parameters():
        assert torch.equal(p, before[name])


def test_cpu_default_does_not_touch_cuda(monkeypatch):
    def forbidden(name):
        # Dynamo registers functions by identity on the first optimizer import;
        # distinct sentinels keep this check independent of test execution order.
        def fail(*args, **kwargs):
            raise AssertionError(f"CPU execution must not call CUDA {name}")

        return fail

    for name in (
        "_lazy_init",
        "synchronize",
        "get_rng_state",
        "reset_peak_memory_stats",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden(name))
    monkeypatch.setattr(torch.accelerator, "current_stream", forbidden("current_stream"))
    # CPU checkpointing may read the compiled accelerator type/availability.
    # The optimizer must not run the upstream check that can open its stream.
    monkeypatch.setattr(
        torch.optim.AdamW,
        "_accelerator_graph_capture_health_check",
        forbidden("optimizer accelerator graph check"),
    )
    trainer = make_trainer("sft", microbatch=2, checkpoint=True)
    assert trainer.step(examples())["status"] == "updated"
    with pytest.raises(ValueError, match="CUDA only"):
        replace(trainer.config, precision="bf16")


def test_resume_rejects_lora_scaling_mismatch(tmp_path):
    trainer = make_trainer("sft", mode="lora")
    trainer.save_checkpoint(tmp_path / "state.pt")
    fresh = make_trainer(None, mode="lora")
    fresh.student.blocks[0].attention.q_proj.scaling *= 2
    with pytest.raises(ValueError, match="student topology"):
        fresh.load_checkpoint(tmp_path / "state.pt")


def test_skipped_group_and_completed_stage_checkpoints_resume(tmp_path):
    trainer = make_trainer("grpo", microbatch=2)
    trainer.reward_fn = lambda c, e: 1.0
    trainer.step(examples())
    assert trainer.policy_version == 0 and trainer.attempt == 1
    trainer.save_checkpoint(tmp_path / "skipped.pt")
    resumed = make_trainer(None, microbatch=2)
    resumed.load_checkpoint(tmp_path / "skipped.pt")
    resumed.reward_fn = lambda c, e: 1.0
    a, b = trainer.step(examples()), resumed.step(examples())
    assert a["status"] == b["status"] == "skipped_zero_variance"
    assert torch.equal(trainer.last_batch.input_ids, resumed.last_batch.input_ids)
    assert trainer.scheduler.last_epoch == resumed.scheduler.last_epoch == 0
    for current in (trainer, resumed):
        current.reward_fn = lambda c, e: float(c[0] >= 4)
        current.run_stage(examples())
    trainer.save_checkpoint(tmp_path / "complete.pt")
    complete = make_trainer(None, microbatch=2)
    complete.load_checkpoint(tmp_path / "complete.pt")
    assert complete.budget.complete
    complete.start_stage("sft", loss_tokens=3, seed=11)
    assert complete.stage == 2 and not complete.optimizer.state
    assert complete.step(examples())["loss_tokens"] == 3
