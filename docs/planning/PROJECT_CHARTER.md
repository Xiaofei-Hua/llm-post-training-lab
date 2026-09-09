# 项目契约

## 目标用户与用途

本仓库首先服务于项目作者本人，用于：

- 系统掌握 base model post-training 的算法与训练信号；
- 形成一段能在后训练/基模算法实习面试中深入讲解的项目经历；
- 掌握模型适配、on-policy 训练循环、性能剖析与优化；
- 训练消融设计、指标分析、failure analysis 和技术表达能力；
- 在不使用内部资产的前提下形成可公开作品。

## Problem Anchor

- **研究问题**：在有限算力和公开数据下，稀疏 reward 与稠密 Teacher 信号如何改变同一 Student 的学习动态、推理能力与训练效率，其顺序交互是否有可解释的差异？
- **必须解决的技术瓶颈**：从独立 loss 函数走向真实学习循环，处理策略刷新、梯度归一化、大词表蒸馏内存和 rollout 成本，再用受控指标解释效果。
- **Non-goals**：从零预训练、通用治理/审计平台、生产服务、榜单 SOTA、多模态和大规模 Agent RL。服务于当前后训练 workload 的框架与性能优化属于核心范围。
- **Frontier constraint**：不使用 Qwen3；主线固定 2026 Gemma 4 E2B Base→SFT Student 与 E4B Base→同源 SFT Teacher。
- **Method Thesis**：用 exact-reward GRPO 实例化稀疏可验证反馈，用 fully on-policy reverse-KL OPD 实例化稠密 Teacher 反馈，并用对称五臂识别单独作用与顺序。
- **Constraints**：当前只授权 CPU 开发与合成验证，GPU 型号、数量与可用时长未知；真实模型与训练的资源校准在授权后进行；只使用公开数据和模型。
- **Success condition**：形成算法推导与实现、可执行训练框架、性能优化对照、能力/动态/成本分析四类成果；最终完成 A0–A4 五臂两阶段的 3 paired seeds，输出技术报告与面试讲稿。无显著提升或优化无收益也可形成完整技术结论。

## 研发投入与技术价值

参考投入为算法 35%、框架 30%、性能 20%、指标分析 15%；用于排序任务，不建立工时审批。每个主要任务应回答一个具体技术问题，并交付代码、最小验证/测量和解释。当前最优先的是 D09–D12 的模型接入、完整更新循环、profile 与学习实验。

性能是独立成果方向：先测 baseline，再选择分块、重计算、batch/packing 或 rollout 优化，报告时间/内存收益及代价。引入 vLLM/FSDP 本身不等于贡献；定位瓶颈、实施适配/优化并给出同条件实测对照，才形成可讲解的技术成果。

已有 D05–D08 提供数据、评测和统计支持，默认进入维护模式。哈希、schema、溯源和一致性检查不再作为下一阶段的主交付；只有实际错误、泄漏或实验阻塞才补必要修复。

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
- 能实现 CE、GRPO、reverse-KL 的数值和梯度路径，并解释 mask、normalizer、stop-gradient 与 chunking 的取舍。
- 能把采样、打分、loss、backward、optimizer update 接成会学习的训练循环。
- 能读懂 profile，分清核函数、模型 update 和端到端训练的瓶颈，并用同条件对照验证优化。

### 实验完成

- 正式研究完成需要 G0–G6 与主五臂三 seed；CPU 开发实验不等待真实训练 gate。
- 训练行为、性能 baseline/优化对照和失败分析完成；保留配置、seed 与关键结果即可定位和复跑。
- 主结果能用锁定环境、配置和数据版本复跑；现有记录由工具产生，不新建证据平台。

### 作品完成

- 一页项目摘要、算法/框架技术报告、能力主表、训练动态、性能对照与失败案例图。
- 10 分钟讲稿和至少 20 个追问题答案。
- 简历表述只包含实测且可由结果文件支持的数字。
- 阶段成果可提前展示：CPU 算法实现与梯度验证、训练循环、CPU 性能测量分别注明范围；真实模型效果待训练完成后补充，不把计划写成成果。
