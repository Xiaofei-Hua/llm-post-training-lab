from __future__ import annotations

import copy

import pytest
import torch

from posttrain_lab.models import (
    CausalLMFeatures,
    LoRAConfig,
    LoRALinear,
    TinyCausalLM,
    TinyCausalLMConfig,
    configure_trainable_parameters,
    freeze_model,
    summarize_parameters,
)
from posttrain_lab.train import MaskValidationError, causal_lm_opd_loss, causal_lm_sft_loss


def make_student(*, tied: bool = True, seed: int = 3) -> TinyCausalLM:
    return TinyCausalLM(
        TinyCausalLMConfig(
            vocab_size=13,
            hidden_size=12,
            intermediate_size=20,
            num_layers=2,
            num_heads=3,
            max_sequence_length=12,
            tie_word_embeddings=tied,
        ),
        seed=seed,
        dtype=torch.float64,
    )


def make_teacher() -> TinyCausalLM:
    teacher = TinyCausalLM(
        TinyCausalLMConfig(
            vocab_size=13,
            hidden_size=16,
            intermediate_size=28,
            num_layers=2,
            num_heads=4,
            max_sequence_length=12,
        ),
        seed=9,
        dtype=torch.float64,
    )
    freeze_model(teacher)
    return teacher


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 1, 3, 5, 7], [1, 4, 7, 0, 0, 0]])
    attention = ids.ne(0)
    mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, False, False, True, False, True],
            [False, True, True, False, False, False],
        ]
    )
    return ids, attention, mask


def dense_loss(model, teacher, objective, ids, attention, mask):
    # Independent full-logit formulas: shift once, then select target positions.
    student_logp = model(ids, attention)[:, :-1].log_softmax(-1)
    if objective == "sft":
        nll = -student_logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        return nll[mask[:, 1:]].mean()
    with torch.no_grad():
        teacher_logp = teacher(ids, attention)[:, :-1].log_softmax(-1)
    reverse_kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
    return reverse_kl[mask[:, 1:]].mean()


def adapter_loss(model, teacher, objective, ids, attention, mask, **kwargs):
    if objective == "sft":
        return causal_lm_sft_loss(
            model, ids, mask, attention_mask=attention, max_tokens_per_chunk=2, **kwargs
        ).loss
    return causal_lm_opd_loss(
        model, teacher, ids, mask, attention_mask=attention, max_tokens_per_chunk=2, **kwargs
    ).loss


def test_causal_prefix_is_independent_of_future_tokens() -> None:
    model = make_student()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed = torch.tensor([[1, 2, 3, 10, 11]])
    torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(model(ids)[:, :3], model(ids[:, :3]), rtol=1e-12, atol=1e-12)
    assert not torch.allclose(model(ids)[:, 3:], model(changed)[:, 3:])


def test_left_right_and_all_padding_are_finite_and_do_not_change_real_tokens() -> None:
    model = make_student(tied=False)
    ids = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 0, 0], [0, 0, 0, 0, 0]])
    logits = model(ids, ids.ne(0))
    expected = model(torch.tensor([[1, 2, 3]]))[0]
    torch.testing.assert_close(logits[0, 2:], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logits[1, :3], expected, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(logits).all()
    assert logits[2].count_nonzero() == 0
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.token_embedding.weight.grad[0].count_nonzero() == 0


def test_initialization_is_reproducible_and_restores_cpu_rng() -> None:
    before = torch.random.get_rng_state()
    model = make_student()
    configure_trainable_parameters(model, "lora", seed=7)
    assert torch.equal(torch.random.get_rng_state(), before)
    other = make_student()
    configure_trainable_parameters(other, "lora", seed=7)
    for parameter, other_parameter in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(parameter, other_parameter, rtol=0, atol=0)
    assert model.lm_head.weight is model.token_embedding.weight


def test_lora_initial_outputs_and_parameter_selection() -> None:
    model = make_student()
    ids, attention, _ = batch()
    expected = model(ids, attention).detach()
    base_count = summarize_parameters(model).total_parameters
    report = configure_trainable_parameters(
        model, "lora", lora=LoRAConfig(rank=2, alpha=4, target_modules=("q_proj", "v_proj"))
    )
    torch.testing.assert_close(model(ids, attention), expected, rtol=0, atol=0)
    # 2 layers * 2 projections * rank * (input width + output width).
    assert report.trainable_parameters == 2 * 2 * 2 * (12 + 12)
    assert report.frozen_parameters == base_count
    assert model.lm_head.weight is model.token_embedding.weight
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.endswith((".lora_A", ".lora_B"))


def test_unknown_or_head_lora_targets_fail_before_freezing() -> None:
    model = make_student()
    with pytest.raises(ValueError, match="unknown decoder LoRA targets"):
        configure_trainable_parameters(
            model, "lora", lora=LoRAConfig(target_modules=("q_proj", "lm_head"))
        )
    assert all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


@pytest.mark.parametrize("objective", ["sft", "opd"])
@pytest.mark.parametrize("mode", ["text", "lora"])
@pytest.mark.parametrize("tied", [True, False])
def test_model_loss_gradients_and_two_updates_match_dense_reference(objective, mode, tied):
    student = make_student(tied=tied)
    configure_trainable_parameters(student, mode)
    reference = copy.deepcopy(student)
    teacher = make_teacher()
    teacher_before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    ids, attention, mask = batch()
    optimizer = torch.optim.SGD((p for p in student.parameters() if p.requires_grad), lr=0.1)
    reference_optimizer = torch.optim.SGD(
        (p for p in reference.parameters() if p.requires_grad), lr=0.1
    )
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = adapter_loss(student, teacher, objective, ids, attention, mask)
        expected = dense_loss(reference, teacher, objective, ids, attention, mask)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        actual.backward()
        expected.backward()
        for name, parameter in student.named_parameters():
            reference_parameter = reference.get_parameter(name)
            if not parameter.requires_grad:
                assert parameter.grad is None
                continue
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, rtol=1e-9, atol=1e-11
            )
        if mode == "lora":
            a_grad = sum(
                p.grad.abs().sum().item()
                for n, p in student.named_parameters()
                if n.endswith("lora_A")
            )
            assert (a_grad == 0) if step == 0 else (a_grad > 0)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in student.named_parameters():
            torch.testing.assert_close(
                parameter, reference.get_parameter(name), rtol=1e-10, atol=1e-12
            )

    changed = [name for name, p in student.named_parameters() if not torch.equal(p, before[name])]
    assert changed
    if mode == "lora":
        assert all(name.endswith(("lora_A", "lora_B")) for name in changed)
        assert any(name.endswith("lora_A") for name in changed)
        assert any(name.endswith("lora_B") for name in changed)
    else:
        assert "token_embedding.weight" in changed
    for name, parameter in teacher.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, teacher_before[name])
    # The final training path survives state_dict reload with the same adapter setup.
    restored = make_student(tied=tied, seed=88)
    configure_trainable_parameters(restored, mode)
    restored.load_state_dict(student.state_dict())
    torch.testing.assert_close(restored(ids, attention), student(ids, attention), rtol=0, atol=0)


@pytest.mark.parametrize("objective", ["sft", "opd"])
def test_training_adapter_does_not_use_dense_forward_or_features_logits(monkeypatch, objective):
    student, teacher = make_student(), make_teacher()
    ids, attention, mask = batch()

    def fail(*args, **kwargs):
        raise AssertionError("full vocabulary logits must not be materialized by the model")

    monkeypatch.setattr(student, "forward", fail)
    monkeypatch.setattr(teacher, "forward", fail)
    monkeypatch.setattr(CausalLMFeatures, "logits", fail)
    adapter_loss(student, teacher, objective, ids, attention, mask).backward()
    assert student.token_embedding.weight.grad is not None


@pytest.mark.parametrize("objective", ["sft", "opd"])
def test_uneven_microbatches_preserve_model_gradient_with_global_normalizer(objective):
    student, teacher = make_student(), make_teacher()
    configure_trainable_parameters(student, "lora")
    reference = copy.deepcopy(student)
    ids, attention, mask = batch()
    for selection in (slice(0, 1), slice(1, 3)):
        adapter_loss(
            student,
            teacher,
            objective,
            ids[selection],
            attention[selection],
            mask[selection],
            global_token_count=int(mask.sum()),
            ddp_world_size=1,
        ).backward()
    dense_loss(reference, teacher, objective, ids, attention, mask).backward()
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad, reference.get_parameter(name).grad, rtol=1e-9, atol=1e-11
            )


def test_teacher_freeze_clears_stale_gradients_and_unfrozen_teacher_is_rejected():
    student, teacher = make_student(), make_student(seed=9)
    ids, attention, mask = batch()
    with pytest.raises(ValueError, match="Teacher must be frozen"):
        causal_lm_opd_loss(student, teacher, ids, mask, attention_mask=attention)
    teacher(ids, attention).sum().backward()
    freeze_model(teacher)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert not teacher.training
    causal_lm_opd_loss(student, teacher, ids, mask, attention_mask=attention).loss.backward()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_padding_and_first_real_token_cannot_be_loss_targets(position):
    ids = torch.tensor([[0, 0, 1, 2]])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, position] = True
    with pytest.raises(MaskValidationError, match="nonpadding"):
        causal_lm_sft_loss(make_student(), ids, mask, attention_mask=ids.ne(0))


def test_parameter_estimates_deduplicate_ties_and_match_allocated_adamw_moments():
    tied, untied = make_student(), make_student(tied=False)
    assert (
        summarize_parameters(untied).total_parameters - summarize_parameters(tied).total_parameters
        == 13 * 12
    )
    report = configure_trainable_parameters(tied, "lora")
    optimizer = torch.optim.AdamW((p for p in tied.parameters() if p.requires_grad), lr=0.01)
    ids, attention, mask = batch()
    causal_lm_sft_loss(tied, ids, mask, attention_mask=attention).loss.backward()
    optimizer.step()
    moment_bytes = sum(
        state[key].numel() * state[key].element_size()
        for state in optimizer.state.values()
        for key in ("exp_avg", "exp_avg_sq")
    )
    gradient_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in tied.parameters() if p.grad is not None
    )
    assert report.estimated_adamw_moment_bytes == moment_bytes
    assert report.estimated_gradient_bytes == gradient_bytes
    assert report.estimated_parameter_gradient_adamw_bytes == (
        report.parameter_bytes + gradient_bytes + moment_bytes
    )
