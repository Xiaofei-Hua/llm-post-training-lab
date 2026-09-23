"""Gemma 4 Base text adapter, with exact frozen PLE lookup on CPU.

The complete checkpoint stays registered, including unused multimodal modules.
Only decoder attention/MLP LoRA parameters can train. No quantization, tokenizer
remapping, KV cache, or approximate head is introduced here.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from .causal_lm import CausalLMAdapter, CausalLMFeatures, causal_attention_mask
from .parameters import LoRAConfig, LoRALinear, freeze_model

GEMMA4_BASE_REVISIONS = {
    "google/gemma-4-E2B": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    "google/gemma-4-E4B": "411aa17b749aa952df1359d2dcea73917a544d9a",
}


class Gemma4TextAdapter(CausalLMAdapter):
    def __init__(self, model: Gemma4ForConditionalGeneration, *, offload_ple: bool = True):
        super().__init__()
        if not isinstance(model, Gemma4ForConditionalGeneration):
            raise TypeError("expected the complete Gemma4ForConditionalGeneration checkpoint")
        self.model = model
        self.config = model.config
        self.offload_ple = offload_ple
        self.lora_targets: tuple[str, ...] = ()
        if self.text_config.use_bidirectional_attention is not None:
            raise ValueError("text training requires a causal Base configuration")
        freeze_model(self)

    @classmethod
    def from_pretrained(cls, model_id: str, *, local_files_only: bool = False):
        if model_id not in GEMMA4_BASE_REVISIONS:
            raise ValueError("D14 accepts only the pinned E2B/E4B Base checkpoints")
        model, loading = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            revision=GEMMA4_BASE_REVISIONS[model_id],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"incomplete Gemma checkpoint load: {loading}")
        adapter = cls(model)
        adapter.loading_info = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading.items()
        }
        return adapter

    @property
    def text_model(self):
        return self.model.model.language_model

    @property
    def text_config(self):
        return self.config.get_text_config()

    @property
    def input_device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    def configure_lora(self, *, seed: int = 101, config: LoRAConfig | None = None) -> None:
        if self.lora_targets or any(isinstance(m, LoRALinear) for m in self.modules()):
            raise ValueError("configure an unadapted Gemma checkpoint")
        if any(p.device.type != "cpu" for p in self.parameters()):
            raise ValueError("configure LoRA on CPU before device placement")
        config = config or LoRAConfig(rank=32, alpha=64)
        targets = [
            (name, module)
            for name, module in self.text_model.named_modules()
            if name.startswith("layers.")
            and name.rsplit(".", 1)[-1] in config.target_modules
            and isinstance(module, nn.Linear)
            and name.split(".")[-2] in {"self_attn", "mlp"}
        ]
        missing = set(config.target_modules) - {name.rsplit(".", 1)[-1] for name, _ in targets}
        if missing:
            raise ValueError(f"unknown decoder LoRA targets: {sorted(missing)}")
        freeze_model(self)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for name, module in targets:
                parent, attr = name.rsplit(".", 1)
                setattr(
                    self.text_model.get_submodule(parent),
                    attr,
                    LoRALinear(module, config, adapter_dtype=torch.float32),
                )
        self.lora_targets = tuple(name for name, _ in targets)
        self.train()

    def train(self, mode: bool = True):
        # nn.Module.train recursively enables encoders too; keep those in eval.
        super().train(mode)
        self.model.eval()
        self.text_model.train(mode)
        return self

    def place_for_training(self, device: torch.device):
        device = torch.device(device)
        text = self.text_model
        # Avoid ever staging the large frozen PLE table on CUDA. All other text
        # modules, including context-dependent PLE projection, run on the GPU.
        for name, module in text.named_children():
            module.to("cpu" if self.offload_ple and name == "embed_tokens_per_layer" else device)
        if self.text_config.tie_word_embeddings:
            self.model.lm_head.weight = text.embed_tokens.weight
        else:
            self.model.lm_head.to(device)
        return self

    def set_activation_checkpointing(self, enabled: bool) -> None:
        if enabled:
            self.text_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.text_model.gradient_checkpointing_disable()

    def forward_features(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        valid = causal_attention_mask(input_ids, attention_mask)
        if input_ids.device != self.input_device:
            raise ValueError("input_ids must be on the text embedding device")
        if bool(((input_ids < 0) | (input_ids >= self.text_config.vocab_size)).any()):
            raise ValueError("input token IDs are outside the text vocabulary")
        # The multimodal wrapper normally interprets these IDs as feature slots.
        # A text-only call must fail rather than silently treat them as text.
        forbidden = [
            getattr(self.config, key, None)
            for key in (
                "image_token_id",
                "audio_token_id",
                "video_token_id",
                "boi_token_id",
                "eoi_token_id",
                "boa_token_id",
                "eoa_token_index",
            )
        ]
        if any(value is not None and bool((input_ids == value).any()) for value in forbidden):
            raise ValueError("multimodal token IDs are not allowed in the text adapter")
        embeds = self.text_model.embed_tokens(input_ids)
        per_layer = None
        if self.text_config.hidden_size_per_layer_input:
            ple_device = self.text_model.embed_tokens_per_layer.weight.device
            per_layer = self.text_model.get_per_layer_inputs(input_ids.to(ple_device), None)
            per_layer = per_layer.to(embeds.device)
        hidden = self.text_model(
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            attention_mask=valid,
            use_cache=False,
        ).last_hidden_state
        return CausalLMFeatures(
            hidden,
            self.model.lm_head.weight,
            self.model.lm_head.bias,
            final_logit_softcapping=self.text_config.final_logit_softcapping,
        )


def load_gemma4_tokenizer(model_id: str, *, local_files_only: bool = False):
    if model_id not in GEMMA4_BASE_REVISIONS:
        raise ValueError("expected a pinned E2B/E4B Base tokenizer")
    # AutoTokenizer can silently fall back to a generic backend without special
    # IDs when this file is absent in an offline cache. Require it explicitly.
    hf_hub_download(
        model_id,
        "tokenizer_config.json",
        revision=GEMMA4_BASE_REVISIONS[model_id],
        local_files_only=local_files_only,
    )
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=GEMMA4_BASE_REVISIONS[model_id],
        local_files_only=local_files_only,
    )


def check_tokenizer_alignment(student, teacher) -> dict:
    """Require full token→ID and backend agreement before exact same-ID KL.

    BOS/EOS insertion is explicit in ``encode_text_pair``; model-specific
    tokenizer defaults and the raw tokenizer_config bytes need not match.
    """
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Student/Teacher vocabulary IDs differ")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if not isinstance(getattr(student, name), int) or not isinstance(
            getattr(teacher, name), int
        ):
            raise ValueError(f"Student/Teacher {name} must be defined")
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError(f"Student/Teacher {name} differs")
    # post_processor can contain different default BOS insertion; the core
    # normalization and segmentation must still agree for arbitrary text.
    import hashlib
    import json

    backends = [json.loads(t.backend_tokenizer.to_str()) for t in (student, teacher)]
    for key in ("model", "normalizer", "pre_tokenizer", "decoder", "added_tokens"):
        if backends[0].get(key) != backends[1].get(key):
            raise ValueError(f"Student/Teacher tokenizer backend {key} differs")
    core = {k: v for k, v in backends[0].items() if k != "post_processor"}
    return {
        "vocab_size": len(student),
        "bos_token_id": student.bos_token_id,
        "eos_token_id": student.eos_token_id,
        "pad_token_id": student.pad_token_id,
        "tokenizer_core_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "policy": "explicit BOS + separately encoded raw prompt/completion + explicit EOS",
    }


def encode_text_pair(tokenizer, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """D14 raw Base smoke format, not a chat template or the D15 data recipe."""
    parts = [tokenizer.encode(text, add_special_tokens=False) for text in (prompt, completion)]
    special = set(tokenizer.all_special_ids)
    if any(special.intersection(part) for part in parts):
        raise ValueError("raw text must not inject special/control tokens")
    if not parts[0] or not parts[1]:
        raise ValueError("prompt and completion must both contain text tokens")
    return [tokenizer.bos_token_id, *parts[0]], [*parts[1], tokenizer.eos_token_id]
