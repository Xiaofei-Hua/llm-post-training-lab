from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from posttrain_lab.models import (
    CausalLMAdapter,
    CausalLMFeatures,
    TinyCausalLM,
    TinyCausalLMConfig,
    configure_trainable_parameters,
    freeze_model,
)
from posttrain_lab.train.grpo_surrogate import compute_exact_group_advantages
from posttrain_lab.train.loop import StageBudgetIncomplete, Trainer, TrainerConfig
from posttrain_lab.train.loss_budget import LossTokenBudget
from posttrain_lab.train.rollout import (
    RolloutConfig,
    TrainingExample,
    sample_completions,
    selected_token_statistics,
)
from posttrain_lab.train.torch_loss_budget import plan_torch_loss_budget


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(seed=1):
    return TinyCausalLM(
        TinyCausalLMConfig(
            vocab_size=9,
            hidden_size=12,
            intermediate_size=24,
            num_layers=1,
            num_heads=3,
            max_sequence_length=10,
        ),
        seed=seed,
        dtype=torch.float64,
    )


def examples():
    return [TrainingExample(f"item-{i}", (1, 5 + i, 3 + i % 2), (3 + i % 2, 2)) for i in range(4)]


def reward(completion, example):
    return float(completion[0] >= 4)


def config():
    return TrainerConfig(rollout=RolloutConfig(max_new_tokens=2), max_tokens_per_chunk=3)


class ConstantPolicy(CausalLMAdapter):
    def __init__(self, chosen):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((9, 1), -10.0, device="cpu"))
        with torch.no_grad():
            self.weight[chosen] = 10

    def forward_features(self, ids, attention_mask=None):
        return CausalLMFeatures(torch.ones((*ids.shape, 1), device="cpu"), self.weight)


@pytest.mark.parametrize("chosen,expected_length,truncated", [(2, 1, False), (0, 3, True)])
def test_rollout_first_eos_and_sampled_padding_token_semantics(chosen, expected_length, truncated):
    student = ConstantPolicy(chosen)
    batch = sample_completions(
        student,
        examples()[:2],
        RolloutConfig(max_new_tokens=3),
        generator=torch.Generator().manual_seed(3),
        policy_version=7,
        greedy=True,
    )
    assert all(len(completion) == expected_length for completion in batch.completions)
    assert all(completion == (chosen,) * expected_length for completion in batch.completions)
    assert batch.completion_mask.sum().item() == 2 * expected_length
    assert batch.truncated.tolist() == [truncated, truncated]
    assert batch.policy_version == 7
    assert not batch.old_log_probs.requires_grad
    assert student.training


def test_rollout_probabilities_match_current_policy_and_refresh_after_update():
    trainer = Trainer(model(), config=config(), reward_fn=reward)
    trainer.start_stage("grpo", loss_tokens=100, seed=5)
    before = copy.deepcopy(trainer.student)
    first = trainer.step(examples())
    first_batch = trainer.last_batch
    with torch.no_grad():
        expected, _ = selected_token_statistics(
            before.forward_features(first_batch.input_ids, first_batch.attention_mask),
            first_batch.input_ids,
            first_batch.completion_mask,
        )
    torch.testing.assert_close(
        first_batch.old_log_probs[first_batch.completion_mask],
        expected[first_batch.completion_mask],
        rtol=1e-12,
        atol=1e-12,
    )
    assert first["status"] == "updated"
    assert first["importance_ratio_mean"] == pytest.approx(1)
    assert first["clip_fraction"] == 0
    snapshot = first_batch.old_log_probs.clone()
    second = trainer.step(examples())
    assert trainer.last_batch.policy_version == first["policy_version_after"] == 1
    assert second["policy_version_after"] == 2
    assert torch.equal(first_batch.old_log_probs, snapshot)


def test_last_hidden_head_projection_preserves_sampling_and_avoids_full_logits(monkeypatch):
    student = model()
    mixed_prompts = [*examples()[:2], TrainingExample("short", (1, 3), (3, 2))]
    rollout = RolloutConfig(max_new_tokens=3, head_projection="dense")
    dense = sample_completions(
        student,
        mixed_prompts,
        rollout,
        generator=torch.Generator().manual_seed(8),
        policy_version=0,
        generations_per_prompt=4,
    )

    def fail(*args, **kwargs):
        raise AssertionError("the last-position sampler must use hidden states directly")

    monkeypatch.setattr(student, "forward", fail)
    last = sample_completions(
        student,
        mixed_prompts,
        replace(rollout, head_projection="last"),
        generator=torch.Generator().manual_seed(8),
        policy_version=0,
        generations_per_prompt=4,
    )
    assert torch.equal(dense.input_ids, last.input_ids)
    assert torch.equal(dense.completion_mask, last.completion_mask)
    torch.testing.assert_close(dense.old_log_probs, last.old_log_probs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("objective", ["sft", "grpo", "opd"])
@pytest.mark.parametrize("mode", ["text", "lora"])
def test_three_objectives_update_parameters_and_close_an_exact_partial_budget(objective, mode):
    student, teacher = model(), model(seed=8)
    configure_trainable_parameters(student, mode)
    freeze_model(teacher)
    teacher_before = copy.deepcopy(teacher.state_dict())
    before = copy.deepcopy(student.state_dict())
    trainer = Trainer(student, teacher=teacher, reward_fn=reward, config=config())
    trainer.start_stage(objective, loss_tokens=17, seed=3)
    records = trainer.run_stage(examples(), max_attempts=16)
    assert sum(record["loss_tokens"] for record in records) == 17
    assert trainer.budget.complete
    assert records[-1]["budget_truncated"]
    changed = [name for name, p in student.named_parameters() if not torch.equal(p, before[name])]
    assert changed
    if mode == "lora":
        assert all(name.endswith(("lora_A", "lora_B")) for name in changed)
    for name, parameter in teacher.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, teacher_before[name])
    assert all(record["entropy"] is not None for record in records if record["loss_tokens"])
    assert all(record["timings_ms"]["total"] > 0 for record in records)


def test_grpo_full_update_matches_independent_clipped_objective_and_pretrim_denominator():
    student = model()
    reference = copy.deepcopy(student)
    trainer = Trainer(student, reward_fn=reward, config=config())
    trainer.start_stage("grpo", loss_tokens=5, seed=5)
    batch = sample_completions(
        reference,
        examples(),
        config().rollout,
        generator=torch.Generator().manual_seed(5),
        policy_version=0,
        generations_per_prompt=8,
    )
    rewards = torch.tensor([reward(c, None) for c in batch.completions])
    group = compute_exact_group_advantages(rewards, batch.group_ids, batch.completion_mask)
    selection = plan_torch_loss_budget(
        LossTokenBudget(5),
        group.loss_mask,
        sample_ids=batch.sample_ids,
        generation_indices=batch.generation_indices,
    )
    logp = reference(batch.input_ids, batch.attention_mask)[:, :-1].log_softmax(-1)
    chosen = logp.gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
    ratios = (chosen - batch.old_log_probs[:, 1:]).exp()
    advantages = group.advantages[:, None]
    token_loss = -torch.minimum(ratios * advantages, ratios.clamp(0.8, 1.2) * advantages)
    expected = token_loss[selection.loss_mask[:, 1:]].sum() / (group.active_completion_count * 2)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0)
    expected.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    optimizer.step()
    record = trainer.step(examples())
    assert record["loss"] == pytest.approx(expected.item(), abs=1e-12)
    for parameter, target in zip(student.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, target, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("constant_reward", [0.0, 1.0])
def test_zero_variance_groups_skip_without_budget_scheduler_or_parameter_updates(constant_reward):
    trainer = Trainer(model(), config=config(), reward_fn=lambda c, e: constant_reward)
    trainer.start_stage("grpo", loss_tokens=17, seed=1)
    before = copy.deepcopy(trainer.student.state_dict())
    with pytest.raises(StageBudgetIncomplete, match="0/17"):
        trainer.run_stage(examples(), max_attempts=2)
    assert len(trainer.records) == 2
    assert trainer.budget.consumed_tokens == 0
    assert trainer.policy_version == 0
    assert trainer.scheduler.last_epoch == 0
    assert not trainer.optimizer.state
    assert all(record["effective_group_rate"] == 0 for record in trainer.records)
    assert all(record["rollout_tokens"] > 0 for record in trainer.records)
    for name, parameter in trainer.student.named_parameters():
        assert torch.equal(parameter, before[name])


def test_nonfinite_gradient_skips_update_and_a_later_valid_attempt_can_learn():
    student = model()
    trainer = Trainer(student, config=config())
    trainer.start_stage("sft", loss_tokens=8, seed=1)
    before = copy.deepcopy(student.state_dict())
    hook = student.token_embedding.weight.register_hook(lambda grad: grad * torch.nan)
    record = trainer.step(examples())
    hook.remove()
    assert record["status"] == "skipped_nonfinite_gradient"
    assert trainer.budget.consumed_tokens == trainer.policy_version == 0
    assert trainer.scheduler.last_epoch == 0
    assert all(p.grad is None for p in student.parameters())
    for name, parameter in student.named_parameters():
        assert torch.equal(parameter, before[name])
    assert trainer.step(examples())["status"] == "updated"
    assert trainer.budget.complete


def test_stage_switch_resets_optimizer_scheduler_rng_and_keeps_student_parameters():
    teacher = model(seed=8)
    freeze_model(teacher)
    trainer = Trainer(model(), teacher=teacher, config=config())
    trainer.start_stage("sft", loss_tokens=8, seed=1)
    with pytest.raises(StageBudgetIncomplete):
        trainer.start_stage("opd", loss_tokens=8, seed=2)
    trainer.step(examples())
    before = copy.deepcopy(trainer.student.state_dict())
    old_optimizer = trainer.optimizer
    assert old_optimizer.state
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    trainer.start_stage("opd", loss_tokens=8, seed=2)
    assert trainer.optimizer is not old_optimizer
    assert not trainer.optimizer.state
    assert trainer.scheduler.last_epoch == 0
    assert trainer.optimizer.param_groups[0]["lr"] == 0.003
    assert trainer.budget.consumed_tokens == 0
    assert trainer.policy_version == 1
    for name, parameter in trainer.student.named_parameters():
        assert torch.equal(parameter, before[name])
    assert torch.equal(trainer.generator.get_state(), torch.Generator().manual_seed(2).get_state())
