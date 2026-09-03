# 公开资料阅读地图

## 模型与训练框架

- [Gemma 4 release history](https://ai.google.dev/gemma/docs/releases)：核对 2026 发布时间与更新。
- [Gemma 4 architecture overview](https://ai.google.dev/gemma/docs/core)：E2B/E4B effective/total parameters、hybrid attention、PLE 与模态。
- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B)：Main Student checkpoint。
- [Gemma 4 E4B Base](https://huggingface.co/google/gemma-4-E4B)：Primary Teacher 起点，与 Student 使用同一 `D_anchor` SFT。
- [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it)：仅用于 external-post-training sensitivity，不进入主 Teacher。
- [Open-R1](https://github.com/huggingface/open-r1)：公开的 SFT/GRPO reasoning 训练配方。
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)：GRPO 配置、自定义 reward 与日志接口。
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer)：Student on-policy generation 与 generalized JSD。
- [TRL trainers taxonomy](https://huggingface.co/docs/trl/main/index)：SFT、DPO、GRPO、reward modeling 和 distillation 的实现入口。

## 数据与评测

- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)：SFT 候选数据；必须去污染后抽样。
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)：主数学 benchmark。
- [IFEval](https://huggingface.co/datasets/google/IFEval)：可验证的 instruction-following retention benchmark。
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)：通用能力 retention benchmark。
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：统一通用评测入口。
- [MathArena](https://github.com/eth-sri/matharena)：2026 年更新的数学竞赛与 ArXivMath 评测。

## 阅读产出规则

每篇论文或文档只记录四项：

1. 它解决什么 bottleneck；
2. loss/reward/采样机制是什么；
3. 哪个消融真正支持 claim；
4. 哪个失败模式可迁移到本项目。

不要把摘要改写当成学习完成。
