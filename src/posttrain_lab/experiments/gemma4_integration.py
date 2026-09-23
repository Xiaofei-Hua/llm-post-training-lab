"""D14 bounded real-Base integration witness; no benchmark data or recipe selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
import transformers

from posttrain_lab.models import summarize_parameters
from posttrain_lab.models.gemma4 import (
    GEMMA4_BASE_REVISIONS,
    Gemma4TextAdapter,
    check_tokenizer_alignment,
    encode_text_pair,
    load_gemma4_tokenizer,
)
from posttrain_lab.train import RolloutConfig, Trainer, TrainerConfig, TrainingExample
from posttrain_lab.train.model_losses import causal_lm_sft_loss
from posttrain_lab.train.rollout import (
    sample_completions,
    selected_token_statistics,
    supervised_batch,
)


def frozen_digest(model) -> str:
    """Stream every frozen parameter, including unused encoders; no weight clones."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        flat = parameter.detach().reshape(-1)
        for chunk in flat.split(1 << 20):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_snapshot(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def gradient_error(model, oracle):
    error_sq = norm_sq = max_error = 0.0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            actual = parameter.grad.detach().cpu()
            expected = oracle[name]
            delta = actual - expected
            error_sq += delta.double().square().sum().item()
            norm_sq += expected.double().square().sum().item()
            max_error = max(max_error, delta.abs().max().item())
    return {"relative_l2": (error_sq / max(norm_sq, 1e-30)) ** 0.5, "max_abs": max_error}


def control_prefix_checks(model, trainer) -> dict:
    """Exercise rare full-policy IDs without claiming naturally sampled accuracy."""
    controls = {
        key: getattr(model.config, key)
        for key in (
            "image_token_id", "audio_token_id", "video_token_id", "boi_token_id",
            "eoi_token_id", "boa_token_id", "eoa_token_index",
        )
    }
    rollout = trainer.config.rollout
    completion = (*dict.fromkeys(controls.values()), rollout.pad_token_id, rollout.eos_token_id)
    ids = torch.tensor([[2, 5, *completion]], device=trainer.device)
    attention = torch.ones_like(ids, dtype=torch.bool)
    mask = attention.clone()
    mask[:, :2] = False
    original_lookup = model.text_model.get_per_layer_inputs

    def cpu_lookup(input_ids, inputs_embeds):
        # Native wrapper independently maps placeholders. Only move its frozen
        # PLE lookup to CPU, preserving the native text/marker interpretation.
        assert input_ids is not None
        return original_lookup(input_ids.cpu(), None).to(trainer.device)

    model.set_activation_checkpointing(False)
    model.zero_grad(set_to_none=True)
    model.train()
    with trainer.autocast():
        with patch.object(model.text_model, "get_per_layer_inputs", side_effect=cpu_lookup):
            hidden = model.model.model(
                input_ids=ids, attention_mask=attention, use_cache=False
            ).last_hidden_state
        with torch.no_grad():
            actual = model.forward_features(ids, attention).hidden_states
            hidden_error = (hidden.float() - actual.float()).abs().max().item()
            assert hidden_error == 0.0
        raw = torch.nn.functional.linear(hidden[:, :-1], model.model.lm_head.weight).float()
        cap = model.text_config.final_logit_softcapping
        logp = (cap * (raw / cap).tanh()).log_softmax(-1)
        dense = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)[mask[:, 1:]].mean()
    dense.backward()
    expected_loss, oracle = dense.item(), gradient_snapshot(model)
    model.zero_grad(set_to_none=True)
    del hidden, actual, raw, logp, dense
    model.set_activation_checkpointing(True)
    with trainer.autocast():
        chunked = causal_lm_sft_loss(
            model, ids, mask, attention_mask=attention, max_tokens_per_chunk=2
        ).loss
    chunked.backward()
    error = gradient_error(model, oracle)
    loss_error = abs(chunked.item() - expected_loss)
    assert loss_error < 2e-5 and error["relative_l2"] < 0.02
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    model.zero_grad(set_to_none=True)
    del oracle, chunked
    draws = iter(completion)

    def controlled_draw(probabilities, count, *, generator):
        token = next(draws)
        assert count == 1 and bool((probabilities[:, token] > 0).all())
        return torch.full(
            (len(probabilities), 1), token, device=trainer.device, dtype=torch.long
        )

    with patch("torch.multinomial", side_effect=controlled_draw), trainer.autocast():
        batch = sample_completions(
            model, [TrainingExample("control-prefix", (2, 5), (rollout.eos_token_id,))],
            replace(rollout, max_new_tokens=len(completion)),
            generator=torch.Generator(device=trainer.device).manual_seed(101), policy_version=0,
        )
    assert batch.completions == (completion,)
    with torch.no_grad(), trainer.autocast():
        recomputed, _ = selected_token_statistics(
            model.forward_features(batch.input_ids, batch.attention_mask),
            batch.input_ids, batch.completion_mask,
        )
    logp_error = (
        recomputed[batch.completion_mask] - batch.old_log_probs[batch.completion_mask]
    ).abs().max().item()
    assert logp_error < 0.1, f"control-prefix sampling/recompute log-prob error: {logp_error}"
    model.set_activation_checkpointing(trainer.config.activation_checkpointing)
    return {
        "scope": "deterministic positive-probability draws; no optimizer update or accuracy claim",
        "control_token_ids": controls, "completion_ids": list(completion),
        "native_hidden_max_abs": hidden_error, "dense_loss": expected_loss,
        "chunked_checkpoint_loss_abs_error": loss_error, "lora_gradient_error": error,
        "all_trainable_gradients_finite": True, "frozen_gradients_absent": True,
        "rollout_recomputed_logp_max_abs": logp_error, "sampling_logp_bound": 0.1,
        "passed": True,
    }


def run(model_id: str) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("D14 real witness requires native BF16 CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(101)
    device = torch.device("cuda:0")
    tokenizers = [
        load_gemma4_tokenizer(repo, local_files_only=True) for repo in GEMMA4_BASE_REVISIONS
    ]
    alignment = check_tokenizer_alignment(*tokenizers)
    tokenizer = tokenizers[list(GEMMA4_BASE_REVISIONS).index(model_id)]
    pairs = [("Compute 2 + 3.\nAnswer:", " 5"), ("Compute 7 - 4.\nAnswer:", " 3")]
    examples = []
    for i, (prompt, completion) in enumerate(pairs):
        ids = [encode_text_pair(t, prompt, completion) for t in tokenizers]
        if ids[0] != ids[1]:
            raise AssertionError("tokenizer segmentation differs on the actual workload")
        p, c = ids[0]
        examples.append(TrainingExample(f"d14-{i}", tuple(p), tuple(c)))
    rollout = RolloutConfig(
        max_new_tokens=max(len(e.completion) for e in examples),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"Loading {model_id} (BF16, complete checkpoint)", flush=True)
    model = Gemma4TextAdapter.from_pretrained(model_id, local_files_only=True)
    model.configure_lora(seed=101)
    trainer = Trainer(
        model,
        config=TrainerConfig(
            device=str(device),
            precision="bf16",
            activation_checkpointing=False,
            max_tokens_per_chunk=2,
            rollout=rollout,
            learning_rates=(1e-4, 1e-4, 1e-4),
        ),
    )
    report = {
        "model_id": model_id,
        "revision": GEMMA4_BASE_REVISIONS[model_id],
        "loading_info": model.loading_info,
        "scope": "real Base weights; two handwritten arithmetic strings; integration only",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "tokenizer": alignment,
        "precision": "BF16 frozen base/GEMMs; FP32 accumulation, LoRA masters, transforms/losses",
        "placement": "CPU frozen PLE lookup + unused multimodal modules; CUDA text decoder/head",
        "lora": {"rank": 32, "alpha": 64, "dropout": 0, "targets": model.lora_targets},
        "parameters": asdict(summarize_parameters(model)),
        "parameter_bytes_by_device": {
            kind: sum(
                p.numel() * p.element_size() for p in model.parameters() if p.device.type == kind
            )
            for kind in ("cpu", "cuda")
        },
        "workload": {
            "seed": 101,
            "pairs": pairs,
            "input_ids": [list(e.prompt + e.completion) for e in examples],
            "batch_size": 2,
            "chunk_tokens": 2,
            "updates": 2,
        },
    }
    print("Checking frozen weights and native text projection", flush=True)
    frozen_before = frozen_digest(model)
    report["control_prefix_checks"] = control_prefix_checks(model, trainer)
    batch = supervised_batch(examples, rollout, device=device)
    ids, attention, mask = batch.input_ids, batch.attention_mask, batch.completion_mask
    model.eval()
    with torch.no_grad(), trainer.autocast():
        features = model.forward_features(ids, attention)
        actual = features.logits()
        # Same native text architecture, using the public precomputed-PLE API.
        embeds = model.text_model.embed_tokens(ids)
        ple = model.text_model.get_per_layer_inputs(ids.cpu(), None).to(device)
        unique_ids, inverse = torch.unique(ids.cpu(), return_inverse=True)
        lookup = model.text_model.embed_tokens_per_layer
        gpu_rows = lookup.weight[unique_ids].to(device)
        gpu_lookup = torch.nn.functional.embedding(inverse.to(device), gpu_rows)
        gpu_lookup = (
            gpu_lookup * lookup.embed_scale.to(device=device, dtype=gpu_rows.dtype)
        ).reshape_as(ple)
        torch.testing.assert_close(ple, gpu_lookup, atol=0, rtol=0)
        report["cpu_ple_lookup_matches_cuda"] = True
        del unique_ids, inverse, gpu_rows, gpu_lookup
        native = model.model(
            inputs_embeds=embeds,
            per_layer_inputs=ple,
            attention_mask=attention,
            use_cache=False,
        ).logits.float()
        raw = torch.nn.functional.linear(features.hidden_states, features.lm_head_weight).float()
        canonical = 30.0 * torch.tanh(raw / 30.0)
        torch.testing.assert_close(actual, canonical, atol=0, rtol=0)
        native_error = (actual - native).abs()
        # Native HF applies tanh in BF16; our loss/rollout contract promotes it
        # to FP32. This bound covers BF16 division/tanh/multiplication rounding.
        if native_error.max().item() > 0.3:
            raise AssertionError("native BF16 logit difference exceeds rounding bound")
        report["native_logit_comparison"] = {
            "fp32_transform_max_abs": (actual - canonical).abs().max().item(),
            "native_bf16_softcap_max_abs": native_error.max().item(),
            "native_bf16_softcap_mean_abs": native_error.mean().item(),
            "native_bf16_rounding_bound": 0.3,
        }
    del features, actual, native, embeds, ple, raw, canonical, native_error
    model.train()
    print("Comparing dense/chunked/checkpointed loss and gradients", flush=True)
    # Dense full-vocabulary oracle on this deliberately short sequence only.
    with trainer.autocast():
        logits = model(ids, attention)
        logp = logits[:, :-1].log_softmax(-1)
        dense_loss = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)[mask[:, 1:]].mean()
    dense_loss.backward()
    dense_value = dense_loss.item()
    oracle = gradient_snapshot(model)
    del logits, logp, dense_loss
    report["loss_gradient_checks"] = []
    for checkpoint in (False, True):
        model.zero_grad(set_to_none=True)
        model.set_activation_checkpointing(checkpoint)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with trainer.autocast():
            loss = causal_lm_sft_loss(
                model, ids, mask, attention_mask=attention, max_tokens_per_chunk=2
            ).loss
        torch.cuda.synchronize(device)
        forward_peak = torch.cuda.max_memory_allocated(device)
        loss.backward()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        error = gradient_error(model, oracle)
        if abs(loss.item() - dense_value) > 2e-5 or error["relative_l2"] > 0.02:
            raise AssertionError(f"dense/chunked/checkpointed gradient mismatch: {error}")
        report["loss_gradient_checks"].append(
            {
                "activation_checkpointing": checkpoint,
                "loss": loss.item(),
                "dense_loss": dense_value,
                "gradient_error": error,
                "gradient_relative_l2_bound": 0.02,
                "loss_abs_bound": 2e-5,
                "forward_backward_seconds": seconds,
                "baseline_allocated_bytes": baseline,
                "forward_peak_allocated_bytes": forward_peak,
                "forward_activation_and_temporary_increment_bytes": forward_peak - baseline,
                "forward_backward_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
        del loss
    del oracle
    model.zero_grad(set_to_none=True)
    model.set_activation_checkpointing(True)
    trainer.config = replace(trainer.config, activation_checkpointing=True)
    report["trainer_config"] = asdict(trainer.config)
    before_lora = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    token_count = int(mask.sum())
    trainer.start_stage("sft", loss_tokens=2 * token_count, seed=101)
    print("Running two token-budgeted Trainer updates", flush=True)
    report["updates"] = [trainer.step(examples) for _ in range(2)]
    if not trainer.budget.complete or trainer.policy_version != 2:
        raise AssertionError("real-model Trainer did not complete both parameter updates")
    changed = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not torch.isfinite(parameter).all() or parameter.grad is None:
                raise AssertionError(f"invalid LoRA update: {name}")
            if not torch.equal(parameter.detach().cpu(), before_lora[name]):
                changed.append(name)
        elif parameter.grad is not None:
            raise AssertionError(f"frozen parameter acquired a gradient: {name}")
    if len(changed) != len(before_lora):
        raise AssertionError("not every targeted LoRA matrix updated after two steps")
    report["changed_lora_matrices"] = len(changed)
    del before_lora
    print("Checking current-policy sampling and frozen parameters", flush=True)
    model.eval()
    with torch.no_grad(), trainer.autocast():
        after = causal_lm_sft_loss(
            model, ids, mask, attention_mask=attention, max_tokens_per_chunk=2
        ).loss.item()
        sampled = sample_completions(
            model,
            examples[:1],
            rollout,
            generations_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(101),
            policy_version=2,
        )
        features = model.forward_features(sampled.input_ids, sampled.attention_mask)
        new_logp, _ = selected_token_statistics(
            features, sampled.input_ids, sampled.completion_mask
        )
        logp_error = (new_logp - sampled.old_log_probs)[sampled.completion_mask].abs().max().item()
    # BF16 prefix/full-sequence GEMMs can round differently even for the same
    # policy. Record the difference; no filtered HF generate defaults are used.
    if logp_error > 0.1:
        raise AssertionError("current-policy sampling log probs are inconsistent")
    report["learning_smoke"] = {
        "nll_before": dense_value,
        "nll_after": after,
        "improved": after < dense_value,
    }
    if after >= dense_value:
        raise AssertionError("two-step learning smoke did not lower its training NLL")
    report["rollout"] = {
        "tokens": sampled.input_ids.cpu().tolist(),
        "current_policy_logp_max_abs": logp_error,
        "logp_abs_bound": 0.1,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
    }
    frozen_after = frozen_digest(model)
    if frozen_before != frozen_after:
        raise AssertionError("a frozen checkpoint parameter changed")
    report["frozen_parameters"] = {
        "before_sha256": frozen_before,
        "after_sha256": frozen_after,
        "unchanged": True,
        "includes_unused_multimodal_encoders": True,
    }
    report["status"] = "PASS"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(GEMMA4_BASE_REVISIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS {args.model}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
