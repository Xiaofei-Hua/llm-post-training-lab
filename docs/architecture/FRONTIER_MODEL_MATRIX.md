# 前沿模型矩阵（2026-09-03）

## 结论

主线采用 2026 年 Gemma 4，而不是 Qwen3。最新性、可训练性与同数据谱系同时作为 gate：选择 E2B Base 做 Student、E4B Base 经同源 SFT 后做 Teacher，避免把未知 instruction recipe 当成算法收益。

## 已核对的官方配置

| Role | Checkpoint | Text config | Total/effective | Vocab | 状态 |
|---|---|---|---|---:|---|
| Main Student | `google/gemma-4-E2B` | 35 layers, hidden 1536, 8 heads/1 KV head | ~5.1B / 2.3B | 262,144 | 锁定 |
| Primary Teacher base | `google/gemma-4-E4B` | 42 layers, hidden 2560, 8 heads/2 KV heads | ~8B / 4.5B | 262,144 | same-lineage SFT + capability gate |
| Stretch Teacher | `google/gemma-4-12B` | 48-layer Unified family | ~12B | 262,144 | 高预算、不同 subtype |
| Stretch Teacher | `google/gemma-4-31B-it` | 60-layer dense family | ~30.7B | 262,144 | 仅独立 Teacher server |

官方文档说明 Gemma 4 E2B/E4B 支持 text、image、audio，采用 local sliding-window 与 global attention 交错、p-RoPE 和 per-layer embeddings。配置与文件大小来自官方 model card/`config.json`；真正运行时仍需锁定 revision。

## 为什么不以“最新最大模型”作为 Student

- 算法项目需要多分支、多 seed；一次 31B LoRA 不能回答 GRPO 与 OPD 的因果问题。
- E2B 的架构与发布时间仍处于前沿，且官方 TRL 将 Gemma 4 列为 GRPO tested family。这里不声称它晚于所有 Qwen3.x，只声称它是启动时更适合受控轻量 Student/Teacher 实验的前沿组合。
- E2B 的 embeddings 使总参数约 5.1B，不能按“2B 普通 dense LM”估算显存；主线锁定 LoRA，QLoRA 只能另建实验族。
- 更大模型的正确角色是 Teacher quality/scale，不是替代实验设计。

## G0 必须实测

1. E2B/E4B Base tokenizer 文件 SHA-256 和随机文本 token IDs 完全一致；
2. Transformers、TRL、vLLM 正确识别 `gemma4` 与 VLM wrapper；
3. 非文本 encoder 的 zero-gradient 与 checksum 断言通过；
4. LoRA target modules 完整列出，trainable parameter count 可复核；
5. packed/unpacked text logits 对齐；
6. 512/2k/4k 长度下测 rollout、Student backward 与 Teacher divergence 吞吐；
7. E4B Base 用同一 `D_anchor` SFT 后通过 capability/NLL gate；E4B-it 不得作为静默替代。

## 官方入口

- [Gemma 4 release history](https://ai.google.dev/gemma/docs/releases)
- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B)
- [Gemma 4 E4B Base](https://huggingface.co/google/gemma-4-E4B)
- [Gemma 4 E4B-it（仅 sensitivity）](https://huggingface.co/google/gemma-4-E4B-it)
- [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B)
- [Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it)
