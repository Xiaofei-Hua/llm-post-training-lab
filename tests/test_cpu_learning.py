from __future__ import annotations

import copy

import pytest
import torch

from posttrain_lab.experiments.cpu_learning import copy_bit_examples, evaluate_copy_task
from posttrain_lab.models import CausalLMAdapter, CausalLMFeatures, freeze_model
from posttrain_lab.train import RolloutConfig


class ConstantDistribution(CausalLMAdapter):
    def __init__(self):
        super().__init__()
        probabilities = torch.tensor(
            [0.02, 0.02, 0.4, 0.3, 0.2, 0.01, 0.02, 0.01, 0.02], dtype=torch.float64, device="cpu"
        )
        self.weight = torch.nn.Parameter(probabilities.log().unsqueeze(1))

    def forward_features(self, input_ids, attention_mask=None):
        return CausalLMFeatures(
            torch.ones((*input_ids.shape, 1), dtype=torch.float64, device="cpu"), self.weight
        )


def test_evaluation_measures_sequence_probability_and_does_not_mutate_training_state():
    student = ConstantDistribution()
    teacher = copy.deepcopy(student)
    freeze_model(teacher)
    initial = student.weight.detach().clone()
    rng = torch.random.get_rng_state()
    result = evaluate_copy_task(
        student, teacher, copy_bit_examples([7, 8]), RolloutConfig(max_new_tokens=2)
    )
    # The two references have probabilities p(3)*p(EOS) and p(4)*p(EOS).
    assert result["mean_correct_completion_probability"] == pytest.approx((0.3 + 0.2) / 2 * 0.4)
    assert result["reference_prefix_reverse_kl"] == pytest.approx(0, abs=1e-12)
    assert result["greedy_accuracy"] == 0  # Greedy immediately emits EOS.
    assert result["greedy_mean_completion_length"] == 1
    assert result["greedy_truncation_rate"] == 0
    assert student.training
    assert student.weight.grad is None
    assert torch.equal(student.weight, initial)
    assert torch.equal(rng, torch.random.get_rng_state())
