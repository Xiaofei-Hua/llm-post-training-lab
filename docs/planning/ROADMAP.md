# 技术开发与训练路线

> 2026-09-23 更新。D01–D14 已完成；D13 为 tiny GPU 实测，D14 为真实 E2B/E4B Base 接入与两步学习 smoke。下一模块 D15 为数据和 Base 评测准备。后续周数为资源与授权闭合后的参考安排。

## 已完成：CPU 算法、框架与单卡运行时

| 顺序 | 主要目标 | 实际工作 | 可展示交付 |
|---|---|---|---|
| D09（已完成） | 模型适配与参数更新 | 本地 tiny causal LM；hidden states/LM head 接 CE/KL；LoRA/text 参数选择；forward/backward/update | 四组 CPU 更新、dense reference 梯度对照、参数量与内存估算；见 `docs/architecture/MODEL_ADAPTER.md` |
| D10（已完成） | 统一训练循环 | 当前策略采样→reward/Teacher→objective→backward→optimizer；SFT/GRPO/OPD 切换及 stage reset | 见 `docs/architecture/CPU_TRAINING_LOOP.md` |
| D11（已完成） | 性能剖析与优化 | CE/KL/完整 step baseline；rollout last-position head 优化与复测 | 时间/RSS、重计算负结果；见 `docs/performance/CPU_BENCHMARK.md` |
| D12（已完成） | CPU 学习实验 | 合成任务五臂两阶段三 seed、学习动态与错误 Teacher 对照 | 见 `docs/experiments/CPU_LEARNING_EXPERIMENT.md` |
| D13（已完成） | 单 GPU runtime | CUDA/BF16、累积/重计算、精确续训与失败恢复通过；12 组性能实测，保留负结果和波动 | 见 `docs/architecture/ACCELERATOR_RUNTIME.md` |
| D14（已完成） | 真实 Gemma 接入 | BF16 基座、FP32 LoRA、CPU PLE lookup、统一 softcap 与 tokenizer；两模型各两次更新 | 梯度、冻结、学习 NLL 和显存实测，见 `docs/architecture/GEMMA4_INTEGRATION.md` |

每轮围绕一个主要技术目标推进，可以修改必要依赖。开发 smoke 和 microbenchmark 不要求新建确认性 claim；完成有用测量后进入下一目标。D05–D08 按需修复，不再为扩充 schema、hash 或审计报告安排独立开发周期。

CPU/GPU tiny 模型用于验证训练算法和框架，不承担真实 Gemma 的选参、性能预测或 C1/C2 结论。D13 实测见 `docs/architecture/ACCELERATOR_RUNTIME.md`；D14 已完成真实模型接入，见 `docs/architecture/GEMMA4_INTEGRATION.md`；下一步 D15 准备真实数据与 Base 评测，正式训练尚未开始。

## 后续：资源就绪后的约 12 周参考安排

| 参考时间 | 模块 | 技术重点 | 退出条件 |
|---|---|---|---|
| 第 1–2 周 | D13–D15 | GPU/distributed runtime、Gemma text/LoRA 接入、真实任务分布与 Base 指标 | 资源授权，真实 forward/backward 可用，数据隔离和 evaluator 可用 |
| 第 3–4 周 | D16–D17 | 64-example overfit、2k sanity、10k 双模型 SFT；分析 NLL/准确率与 Teacher 能力 | D_select 选 recipe，独立 D_teacher_gate 通过 |
| 第 5–6 周 | D18–D20 | GRPO 有效组/探索动态，OPD 大词表开销，rollout/Teacher/backward profile 与优化 | objective pilot 稳定；四类 100-step profile；五臂预算有 30% 余量 |
| 第 7–8 周 | D21 | 全部 A0–A4 × 3 seeds 的 Stage 1 | 15 个中点、每臂每 seed 2M Student loss tokens、动态与成本曲线 |
| 第 9–10 周 | D22 | 统一 optimizer/scheduler reset 后 Stage 2 | 15 个终点、累计 60M Student loss tokens、完整 paired-seed 矩阵 |
| 第 11 周 | D23 | C1/C2、accuracy/pass@k、retention、错误迁移与 accuracy–cost | 主表、性能/成本表、顺序效应分析、至少一个有解释力的负结果 |
| 第 12 周 | D24 | 框架演示、技术报告、简历与面试表达 | 讲清公式→实现→瓶颈→实测→结论，量化表述有结果支持 |

同一 objective 跨 arm/stage 复用 recipe 与实现；不因初步效果删臂。Teacher 不合格或资源不足时保留诊断结论，按既定方法边界重新规划，不能用单 seed 代替主研究。各阶段已经完成的 CPU/框架/性能成果可先写入作品，注明验证范围。

## 延后研究

X01–X08 继续排在 D24 之后：DPO、ORPO/KTO、GSPO/TIS、PRM、替代蒸馏、scale/transfer、多模态与 Agent RL。只为明确暴露的算法限制安排新实验；现有 exact CE/KL 优化、rollout 性能和训练框架完善属于当前核心范围。
