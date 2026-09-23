"""Actual Transformers Gemma architecture at reduced sizes; no downloads."""

import copy

import pytest
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from posttrain_lab.models import LoRAConfig, LoRALinear, freeze_model
from posttrain_lab.models.gemma4 import Gemma4TextAdapter
from posttrain_lab.train import causal_lm_opd_loss, causal_lm_sft_loss
from posttrain_lab.train.rollout import selected_token_statistics


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_gemma(seed=4, *, dtype=torch.float32):
    text = Gemma4TextConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=41,
        num_kv_shared_layers=2,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        final_logit_softcapping=0.3,
        max_position_embeddings=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text)).to(dtype)
    return Gemma4TextAdapter(model)


def text_batch():
    ids = torch.tensor([[2, 5, 8, 9, 1, 0], [2, 4, 7, 8, 9, 1]])
    mask = ids.ne(0)
    mask[:, :2] = False
    return ids, ids.ne(0), mask


def test_text_features_match_native_wrapper_and_preserve_ple():
    adapter = small_gemma()
    ids, attention, _ = text_batch()
    with torch.no_grad():
        native = adapter.model(ids, attention_mask=attention, use_cache=False).logits
        actual = adapter(ids, attention)
    torch.testing.assert_close(actual, native, atol=2e-7, rtol=2e-6)
    assert adapter.model.lm_head.weight is adapter.text_model.embed_tokens.weight
    # PLE omission is a real numerical change; the adapter must retain it.
    ple = adapter.text_model.embed_tokens_per_layer.weight
    with torch.no_grad():
        ple.zero_()
        changed = adapter(ids, attention)
    assert (changed - actual).abs().max() > 1e-4


@pytest.mark.parametrize("objective", ["sft", "opd", "policy"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_loss_and_lora_gradients_match_dense_native_reference(objective, checkpoint):
    student = small_gemma()
    student.configure_lora(config=LoRAConfig(rank=2, alpha=4))
    # Nonzero B exercises gradients through A and all shared KV dependencies.
    with torch.no_grad():
        for module in student.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.fill_(0.02)
    reference = copy.deepcopy(student)
    student.set_activation_checkpointing(checkpoint)
    teacher = small_gemma(seed=9)
    freeze_model(teacher)
    ids, attention, mask = text_batch()
    native_logits = reference.model(ids, attention_mask=attention, use_cache=False).logits
    native_logp = native_logits[:, :-1].log_softmax(-1)
    if objective in {"sft", "policy"}:
        expected = -native_logp.gather(-1, ids[:, 1:, None]).squeeze(-1)[mask[:, 1:]].mean()
        if objective == "sft":
            actual = causal_lm_sft_loss(
                student, ids, mask, attention_mask=attention, max_tokens_per_chunk=2
            ).loss
        else:
            features = student.forward_features(ids, attention)
            actual = -selected_token_statistics(features, ids, mask)[0][mask].mean()
    else:
        with torch.no_grad():
            q = teacher.model(ids, attention_mask=attention, use_cache=False).logits
            teacher_logp = q[:, :-1].log_softmax(-1)
        expected = (native_logp.exp() * (native_logp - teacher_logp)).sum(-1)[mask[:, 1:]].mean()
        actual = causal_lm_opd_loss(
            student, teacher, ids, mask, attention_mask=attention, max_tokens_per_chunk=2
        ).loss
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-5)
    pairs = zip(student.named_parameters(), reference.named_parameters(), strict=True)
    for (name, parameter), (_, oracle) in pairs:
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, oracle.grad, atol=2e-6, rtol=3e-4)
        else:
            assert parameter.grad is None, name
    assert all(p.grad is None for p in teacher.parameters())


def test_bf16_base_fp32_lora_updates_and_freeze():
    adapter = small_gemma(dtype=torch.bfloat16)
    before = {n: p.clone() for n, p in adapter.named_parameters()}
    adapter.configure_lora(config=LoRAConfig(rank=2, alpha=4))
    adapter.place_for_training(torch.device("cpu"))
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    ids, attention, mask = text_batch()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_sft_loss(adapter, ids, mask, attention_mask=attention).loss
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            original_name = name.replace(".base.", ".")
            assert torch.equal(parameter, before[original_name]), name
            assert parameter.grad is None
    # Shared-KV layers deliberately have no K/V projections to adapt.
    assert len(adapter.lora_targets) == 24
    assert all(not n.startswith("layers.2.self_attn.k_proj") for n in adapter.lora_targets)


def test_multimodal_tokens_rejected_before_text_execution():
    adapter = small_gemma()
    adapter.config.image_token_id = 15
    with pytest.raises(ValueError, match="multimodal"):
        adapter(torch.tensor([[2, 15, 1]]))


def small_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "Answer": 4, "five": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def test_tokenizer_alignment_and_explicit_text_boundary():
    from posttrain_lab.models.gemma4 import check_tokenizer_alignment, encode_text_pair

    student, teacher = small_tokenizer(), small_tokenizer()
    assert check_tokenizer_alignment(student, teacher)["vocab_size"] == 6
    assert encode_text_pair(student, "Answer", "five") == ([2, 4], [5, 1])
    with pytest.raises(ValueError, match="special/control"):
        encode_text_pair(student, "Answer <bos>", "five")


def test_tokenizer_alignment_rejects_missing_special_ids_and_different_normalization():
    from tokenizers.normalizers import Lowercase

    from posttrain_lab.models.gemma4 import check_tokenizer_alignment

    student, teacher = small_tokenizer(), small_tokenizer()
    teacher.backend_tokenizer.normalizer = Lowercase()
    with pytest.raises(ValueError, match="normalizer"):
        check_tokenizer_alignment(student, teacher)
    student, teacher = small_tokenizer(), small_tokenizer()
    student.bos_token = teacher.bos_token = None
    with pytest.raises(ValueError, match="must be defined"):
        check_tokenizer_alignment(student, teacher)


def test_offline_loader_requires_tokenizer_configuration(monkeypatch):
    from posttrain_lab.models import gemma4

    def absent(*args, **kwargs):
        raise FileNotFoundError("tokenizer_config.json")

    monkeypatch.setattr(gemma4, "hf_hub_download", absent)
    with pytest.raises(FileNotFoundError, match="tokenizer_config"):
        gemma4.load_gemma4_tokenizer("google/gemma-4-E2B", local_files_only=True)
