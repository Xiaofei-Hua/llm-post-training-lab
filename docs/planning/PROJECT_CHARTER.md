# 项目契约

## 目标用户与用途

本仓库首先服务于项目作者本人，用于：

- 系统掌握 base model post-training 的算法与训练信号；
- 形成一段能在后训练/基模算法实习面试中深入讲解的项目经历；
- 训练 claim、ablation、failure analysis 和实验复现能力；
- 在不使用内部资产的前提下形成可公开作品。

## Problem Anchor

- **Bottom-line problem（immutable）**：在有限算力和完全公开、可追溯的数据条件下，从同一前沿学生模型的 SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。
- **Must-solve bottleneck**：现有学习项目常把模型、数据、训练 token、reward、解码和框架同时改变，最终无法回答收益到底来自哪里。
- **Non-goals**：从零预训练、训练系统性能竞赛、服务部署、榜单 SOTA、多模态和大规模 Agent RL。
- **Frontier constraint**：不使用 Qwen3；主线固定 2026 Gemma 4 E2B Base→SFT Student 与 E4B Base→同源 SFT Teacher。
- **Method Thesis**：用 exact-reward GRPO 实例化稀疏可验证反馈，用 fully on-policy reverse-KL OPD 实例化稠密 Teacher 反馈，并用对称五臂识别单独作用与顺序。
- **Constraints**：当前 GPU 型号、数量与可用时长未知；先完成 24 GB 代码路径和 80 GB/多卡预算校准；只使用可公开发布的数据和模型。
- **Success condition**：完成 A0–A4 两阶段五臂的 3 paired seeds、双预算/三成本视角核算、一次负结果复盘，并输出可复现配置、结果表、技术报告与面试讲稿。

## 两个核心 Claim

### Claim boundary

全部确认性结论只估计：**指定 same-lineage E4B Teacher 与已冻结 GRPO/OPD recipe，在相等 Student backward-token 预算下，对这个 E2B Student 的 intervention effect。** 不把结果外推为抽象 loss 的性质、Teacher 容量的纯效应、任意 Teacher 的 OPD 效果，或所有 dense/sparse signal 的普遍优劣。

### C1：Sparse 与 dense 训练信号的可分辨贡献

在相同 Gemma 4 E2B SFT parent、canonical prompt distribution、每阶段 2M Student backward loss-token 预算和评测协议下，GRPO 与 OPD 对准确率、探索性、输出长度、策略熵和能力保持产生可重复、可诊断的影响。

最低证据：A1−A0、A2−A0 在 MATH-500 上的三个预注册 paired seeds、item-conditional bootstrap CI、Holm 校正与双成本表。

### C2：阶段顺序是否形成可解释交互

比较 `SFT→OPD→GRPO` 与 `SFT→GRPO→OPD`；若差异落入预注册 practical-equivalence band，也接受“该预算下顺序影响有限”的负结论。

最低证据：A3−A4 的 paired seeds、95% superiority CI 或完全落入 ±2pp 的 TOST 90% equivalence CI、双预算和错误类型迁移分析。

### Anti-claims

必须排除以下解释：

- 只是 Student 更新更多或总计算更多；
- 只是 Teacher 或参数量更大；
- 只是 temperature、max tokens 或 answer parser 更宽松；
- benchmark 已进入训练数据；
- 只优化了格式或长度，却没有提高答案正确率；
- 单一 seed 的偶然波动。

## 完成定义

### 学习完成

- 能从公式推导并解释 CE、DPO、GRPO、forward/reverse KL 与 generalized JSD；DPO 是学习/附录，不是主 claim。
- 能解释 on-policy/off-policy、importance ratio、credit assignment、reward hacking、exposure bias。
- 能手写最小 reward、answer verifier 和 distillation loss，并有单元测试。

### 实验完成

- G0–G6 gate 全部记录结论。
- Must-run 实验完成，失败实验也保留配置、日志与分析。
- 主结果可以由一个干净环境从 manifest 重建。

### 作品完成

- 一页项目摘要、完整技术报告、主表、训练曲线、失败案例图。
- 10 分钟讲稿和至少 20 个追问题答案。
- 简历表述只包含实测且可由结果文件支持的数字。
