# Refinement Report

## 目标如何收敛

最初需求是为大模型训练/后训练实习准备一个覆盖算法、架构、数据和 benchmark 的仓库。第一版容易变成“把所有热门算法各跑一次”的清单，无法形成可信的算法贡献。四轮 refinement 将它收敛为一个单一、可证伪问题：同一 Student anchor 下，exact-reward GRPO 与 fully on-policy OPD 的单独作用及先后顺序。

## 迭代轨迹

### Round 0 → 1：从百科全书变成研究

删除 DPO/ORPO/PRM 等主矩阵扩张，只保留 A0–A4。DPO 降为核心完成后的 shadow。把笼统 matched-token 拆成 E1 signal efficiency 与 E2 practical efficiency。用户明确拒绝 Qwen3 后，模型主线改为 2026 Gemma 4 E2B/E4B。

### Round 1 → 2：消除 Teacher 与阶段混杂

E4B-it 带有未知 instruction/post-training 数据，不能作为干净主 Teacher，因此改为 E4B Base 使用同一 `D_anchor` SFT。A1/A2 与 A3/A4 原先 stage 数不同，统一为两阶段五臂。E1 只保留 Student backward loss tokens 作为唯一严格匹配量。GRPO 配置、三 seed 覆盖与确认性统计被明确冻结。

### Round 2 → 3：定义真正的实验单位

发现 `D_calib` 同时选 checkpoint 与证明 Teacher 更强，拆成 `D_select` 和 sealed `D_teacher_gate`。纠正 training seed 嵌套于 item 的统计错误：推断条件于三个预注册 seeds，bootstrap/randomization 只重采样 item。补齐 OPD refresh/mask/normalization 与 `U` 的逐 token 计数、最后一批精确闭合。

### Round 3 → 4：边界与一致性

恢复与解法无关的 immutable Problem Anchor，把 Gemma 4/GRPO/OPD 放入 Method Thesis。明确主张只适用于指定 Teacher/recipe/E2B。删除 QLoRA silent fallback，统一 stretch 描述并修复数据文档三处残余。最终独立评分 9.06，Planning READY。

## 最重要的设计判断

### 为什么不是“算法越多越好”

五臂已是两个 claims 的最小闭合图。再增加算法会消耗 seed 与评测预算，却不增强 GRPO–OPD 交互的因果识别。面试价值来自能解释对照、失败和统计，而不是方法名数量。

### 为什么 Teacher 必须自己同源 SFT

直接使用 instruction Teacher 会把它先前见过的数据、模板、数学训练与安全调优全部带入。same-lineage 不能隔离纯容量效应，但在主动限制 claim 后，足以回答“这个冻结 Teacher 下 OPD recipe 的 intervention effect”。

### 为什么只匹配 `U`

SFT、GRPO 和 OPD 的 rollout multiplicity、completion length、Teacher/old-policy forward 不同，prompt exposure、Student FLOPs和端到端成本无法同时严格相等。选择 Student 实际接收梯度的有效 positions 作为唯一 E1 denominator，其他成本透明报告，比使用含义模糊的 matched-token 更诚实。

### 为什么统计条件于三个 seeds

三个 training seeds 不足以支持对随机 seed 总体的强显著性推断。把完整三-seed vector 固定在每个 item 上，只对 MATH item population 做配对推断，既利用全部结果，又避免伪重复。每 seed 结果仍必须单列。

### 为什么不能现在写 execution READY

Gemma 4 E2B/E4B 名称低估总权重，262K vocab exact KL 与 group-8 长 rollout 的真实吞吐只能靠硬件 profile。当前用户只要求规划；虚构兼容性、显存或 GPU-hours 会破坏整个项目的可信度。

## 最终边界

- Method READY；
- Execution 等待 C0/C3/C4/C5；
- 不下载、不训练、不声称结果；
- 资源不足时只允许全臂同步缩小预算并重新审查；
- 任何进入简历的数字都必须经过结果到 claim 审计。
