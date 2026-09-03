# Gemma 4 训练兼容性 Gate

前沿 checkpoint 的框架支持仍快速变化。项目不因兼容失败退回旧模型，而是先用最小测试锁定可用版本和执行路径。

## 已知风险（截至 2026-09-03）

| 风险 | 影响 | 规划处置 |
|---|---|---|
| `Gemma4ForConditionalGeneration` 是多模态 wrapper | TRL/vLLM 权重路径、FSDP wrap 可能与纯 CausalLM 不同 | 先做 2-step weight sync；显式 text decoder wrap class |
| E2B 约 5.1B total、E4B 约 8B total，而名称是 effective size | 按“2B/4B”估算会严重低估显存；两者 BF16 静态加载约 29 GB | 预算按实际权重、optimizer、KV 与 logits profile |
| per-layer embeddings 占大量参数 | full FT/OPD 成本高，LoRA target 选择影响结论 | 主矩阵统一 LoRA，PLE 与 embeddings 冻结；QLoRA 只能新建实验族 |
| VLM wrapper 含视觉和音频 encoder | text-only 训练可能误更新非文本参数 | zero-gradient + before/after checksum |
| DistillationTrainer 读取 nested vocab | OPD loss 维度或显存异常 | 对 262,144 vocab 做 shape test；锁定含修复的版本 |
| GRPO 默认 completion 上限可能截断 CoT | reward 被截断主导 | 显式设置 2k/4k；记录 P95 length 与 truncation |
| 12B Unified 与 E2B/E4B architecture subtype 不同 | Teacher 比较混入架构差异 | 只作为 stretch teacher，报告 checkpoint transfer |

## 兼容性测试矩阵

### C0：模型与 processor

- E2B Base 与 E4B Base/same-lineage SFT Teacher text-only forward 通过；
- 不插入 image/audio tokens；
- 非文本 encoder 参数冻结且 checksum 可复核；
- tokenizer hash/token IDs/vocab size 检查通过。

### C1：SFT

- 8 条样本 overfit，loss 连续下降；
- assistant-only mask 的非 assistant token gradient 为零；
- LoRA 只命中预注册 text projection modules；
- packed/unpacked logits 差异低于预注册容差，否则禁用 packing。

### C2：GRPO

- `use_vllm=false` 完成 2 steps，先验证正确性；
- vLLM server/colocate 完成 2 steps，权重同步后生成分布变化；
- group size 8 的峰值显存、rollout tokens/s 与有效 group rate 可记录；
- 2k/4k completion cap 的截断率经小样本测量。
- `loss_type=dr_grpo`、`beta=0`、`num_iterations=1`、token-level importance sampling 与 group reward scaling 的数值/日志语义经固定 TRL revision 验证；
- generation batch 后权重同步生效，old-policy age 为 1 batch；prompt RNG 与 rollout RNG 可独立复现。

### C3：OPD

- Student/Teacher 同一文本 token IDs 完全一致；
- logits 最后一维均为 262,144；
- `num_iterations=1`，每次 update 前由当前 policy 新生成 completion；
- KL 只 mask 生成 completion 首 token 至第一个 EOS（含 EOS），prompt/padding/EOS 后位置为零；
- reverse KL 的 CPU 数值、极限与梯度 oracle 已由 D04 通过，且按全 batch 有效 completion token 数归一化；真实模型 parity 仍待验证；
- full-vocab chunked loss 已与独立的小型精确 full-logit reference 对齐；E2B/E4B 接线仍待验证；
- Teacher 永久 stop-gradient，Student rollout token IDs stop-gradient。

### C4：Teacher quality

- E4B Base 使用与 Student 相同的 `D_anchor`、template、mask、样本顺序与 token/epoch budget 完成 SFT；只允许 `D_select` 选 checkpoint；
- 在从未参与选择、始终 sealed 的 `D_teacher_gate` 上要求 paired bootstrap CI 下界 >0、accuracy 点估计至少 +5pp、verified-solution NLL 更低且 parse rate 不下降；
- Teacher 错误样本不从主 OPD prompt 池中静默删除；按 Teacher-correctness 做事后 slice；
- same-lineage Teacher 不过 gate 则 primary OPD 停止；更换 Teacher 必须产生新 experiment family，不能覆盖旧结果。

### C5：真实算力闭合

- 分别 profile E2B LoRA backward、group-8 rollout、E4B SFT 和 E2B+E4B full-vocab chunked reverse KL；
- 记录 100-step steady-state tokens/s、峰值显存、通信与 checkpoint 开销；
- 以 3 seeds × A0–A4 × 4M Student loss tokens 重算总预算并保留 30% 余量；
- 2×80 GB 或 4×48 GB 不可得且 C5 不闭合时，停止主矩阵；不得用量化 Teacher、top-k KL 或旧模型静默改变问题。

### C6：Loss-token 与 objective invariant

- 人工构造含 prompt、EOS、padding、EOS 后 token 与 zero-variance group 的 batch，验证 `U` 只数实际进入已执行 update 的 objective positions；
- 同一 objective 在不同 arm/stage 加载同一 config hash；只允许 parent checkpoint、stage ID 和预注册 RNG seed 不同；
- SFT、GRPO、OPD 在最后一个 update 按 `(sample_id, generation_index, token_index)` 稳定排序并施加 budget mask，使累计 `U` 精确等于 2M；manifest 保存候选/保留位置数及最终计数。

## 版本冻结规则

- 环境记录 Python、CUDA、PyTorch、Transformers、TRL、vLLM 与 kernel 的精确版本/commit。
- 只在 compatibility branch 升级依赖；主实验期间不滚动升级。
- 升级后重复 C0–C4，旧结果与新结果不能直接拼表。
- 保存官方 issue/release 状态快照；“最新版”不是兼容性证明。

## 参考

- [TRL GRPO supported models](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
- [TRL VLM weight-sync issue covering Gemma 4](https://github.com/huggingface/trl/issues/6028)
- [TRL releases](https://github.com/huggingface/trl/releases)
- [Gemma 4 official overview](https://ai.google.dev/gemma/docs/core)
