# Round 2 Raw Review

> Reviewer output preserved by the primary agent. No edits were made by the reviewer.

CALIBRATION: none

COMPOSITE: **8.10 / 10**

| 维度 | 权重 | 分数 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 8 | 2.00 |
| Contribution Quality | 25% | 8 | 2.00 |
| Frontier Leverage | 15% | 9 | 1.35 |
| Feasibility | 10% | 6 | 0.60 |
| Validation Focus | 5% | 8 | 0.40 |
| Venue/Portfolio Readiness | 5% | 8 | 0.40 |

**GAP:** 没有人工精选的 proposal anchors，因此不能进行 exemplar-relative calibration。相对 READY=9 的标准，本轮已经解决了上一轮最主要的贡献发散和 matched-token 问题：五臂图聚焦、双 estimand 合理、GRPO/OPD 合同已可实施、评测也加入了真正新鲜的数据。剩余差距主要是三个尚未闭合的因果/资源接口：E4B-it 带入未知 post-training recipe；五臂的 stage/scheduler 结构并非完全对称；E1 同时声称匹配 prompt exposure、loss tokens 和 Student FLOPs，但输出长度和 rollout 数不同使三者无法严格同时匹配。Gemma 4 的实际参数和全词表 KL 成本又使未确认硬件成为真正 blocker，而非普通工程细节。

## Round 2 判断

- **Problem Anchor:** preserved。模型从 Qwen3.5 改为 Gemma 4 是对用户“禁用 Qwen3、选择启动时最前沿可训练模型”原意的恢复，不是研究问题漂移。
- **Dominant contribution:** 明显更锋利。核心已经统一为 sparse verifier reward 与 dense on-policy teacher signal 的单独作用和顺序交互。
- **Simplicity:** 接近合格。五臂对于单独作用与两种顺序是最小闭合图；DPO 必须继续留在 shadow/nice-to-have，而不能悄悄回到主表。
- **Frontier leverage:** 合理且自然。Gemma 4 不是装饰性替换，而是当前小规模、同族 Base/Teacher、TRL 可训练组合。
- **Blocking issue:** primary Teacher 构造和真实 compute closure 尚未最终确定。

## Gemma 4 是否优于 Qwen3.5/3.6 作为主线

**结论：是。Gemma 4 E2B/E4B 应替代 Qwen3.5 2B/4B 成为 primary pair，但原因是“前沿性 × 可训练性 × 同族 teacher compatibility”的综合最优，不是单纯发布时间最新。**

Google 的发布记录显示 Gemma 4 E2B/E4B 于 2026-03-31 发布；它晚于 Qwen3.5，但并不晚于 2026 年 4 月的 Qwen3.6。因此不能写成“Gemma 4 比所有 Qwen3.x 都更新”。更准确的说法是：Gemma 4 是当前更新且最适合受控小模型实验的 Base/Teacher 配对。[Google Gemma releases](https://ai.google.dev/gemma/docs/releases), [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)

| 因素 | Gemma 4 E2B/E4B | Qwen3.5 2B/4B | Qwen3.6 |
|---|---|---|---|
| 可用配对 | 同族 Base E2B、Base/IT E4B | 同族 2B Base、4B IT | 没有同等轻量的核心 Student/Teacher pair |
| GRPO 支持 | TRL 明确列为 tested | TRL 也列为 tested | TRL 列为 tested，但当前公开主力尺寸较重 |
| 训练内核 | local/global attention，较常规 | Gated DeltaNet 需要额外高效内核 | 同类 hybrid 风险 |
| 已知 runtime 风险 | 生态较新，仍需 C0 | 缺少 `flash-linear-attention` 时可能极慢；packing recurrent state 还有边界风险 | 相似 hybrid 风险 |
| OPD tokenizer | E2B/E4B 均为 262K，同族兼容性强，仍需 hash gate | 同族兼容性也强 | 必须逐项 gate |
| 实际成本 | E2B/E4B 实际约 5.1B/8B，名字低估显存 | 更轻 | 27B/35B 总权重明显更重 |

TRL 当前 GRPO 文档明确将 Gemma 4、Qwen3.5 和 Qwen3.6 都列为 tested；但其异步 GRPO 文档也专门指出 Qwen3.5/3.6 的线性注意力内核和 packing 状态风险。[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer), [TRL async GRPO](https://huggingface.co/docs/trl/main/async_grpo_trainer)

代价是 Gemma 4 的 `E` 是 effective parameters。Google 给出的 BF16 推理内存约为 E2B 11.4 GB、E4B 17.9 GB；二者静态加载已约 29 GB，尚未包含训练激活、rollout 副本和 262K-vocab KL。[Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)

因此：

- 有 2×80 GB 或 4×48 GB：Gemma 4 是更好的主线。
- 只有单卡 24 GB：当前五臂、全词表 OPD 方案不闭合；不能靠 “E2B” 名字推断它等价于普通 2B 模型。
- Qwen3.6 更适合作为未来扩展，不适合作为本项目核心 teacher。
- 不建议仅为省算力退回 Qwen3.5；应先缩小 `U`、completion cap 和 nice-to-have。

## Primary Teacher 应如何构造

**不建议把 `google/gemma-4-E4B-it` 作为主 Teacher。推荐：`google/gemma-4-E4B` Base → 与 E2B Student 相同的 `D_anchor` SFT → 冻结为 Teacher。**

E4B-it 虽然省去 teacher construction，但它带入了未知的 instruction/post-training 数据、模板、数学训练和安全调优。它不会破坏固定 Teacher 下 A3/A4 的内部顺序比较，却会把论文结论变成“对某个厂商 post-trained Teacher 的蒸馏效果”，而非同数据 lineage 下的 capacity-controlled dense signal。

精确 Teacher recipe 应为：

1. 加载 [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) Base。
2. 使用与 E2B Student 完全相同的 immutable `D_anchor`、chat template、assistant-only loss mask、max length 和样本顺序。
3. 使用相同 LoRA module classes、rank 32、alpha 64、dropout 0；不同层数带来的 trainable-parameter 差异视为模型容量的一部分。
4. 使用相同 SFT token/epoch budget；学习率可在同一两档预注册 pilot 中按模型分别选择。
5. 只用 `D_calib` 选择 checkpoint；禁止查看 `D_core` gold trace、`D_dev` 或 test。
6. 合并或固定 adapter，保存 checkpoint/config/tokenizer hash，此后所有 OPD arms 使用同一只读 Teacher。
7. Teacher gate 应要求：paired CI 下界 > 0、accuracy point estimate 至少 +5pp、verified-solution NLL 更低、parse rate 不下降。Student-error coverage 应作为诊断，不能替代 +5pp 门槛。
8. 若 same-lineage E4B Teacher 不过 gate，primary OPD 问题应停止。E4B-it 只能作为明确标注的“external post-trained Teacher sensitivity”，不能静默替换主 Teacher。

### Teacher SFT 成本如何计入

定义三个成本：

- `C_anchor`：所有 arm 共用的 E2B SFT anchor；
- `C_teacher`：E4B Base → SFT Teacher 的一次性成本；
- `C_arm`：每个 post-anchor arm 的训练和 rollout 成本。

E1 从固定 Student anchor 和固定 Teacher 后开始，因此不把 `C_teacher` 混入 Student signal efficiency。

E2 必须同时报告：

- **Warm-start/marginal:** `C_arm`，其中 OPD teacher forward 已计入；
- **Cold-start pipeline:** 非 OPD 为 `C_anchor + C_arm`，OPD 为 `C_anchor + C_teacher + C_arm`；
- **Campaign total:** `C_anchor + C_teacher + ΣC_arm`。

不能把 Teacher SFT 完全隐藏，也不应在三个 OPD arms 中重复计三次。450–900 GPUh 估计必须在加入 `C_teacher` 的 100-step profile 后重算。

## 剩余方法学修订

### 1. 将五臂统一为两阶段网格

当前 A1/A2 是一个 4M stage，而 A3/A4 是两个 2M stage，并在中间重置 optimizer/scheduler。这使 A1/A2 与顺序臂的比较混入 scheduler restart。

应改成：

| Arm | Stage 1 | Stage 2 |
|---|---|---|
| A0 | SFT 2M | SFT 2M |
| A1 | GRPO 2M | GRPO 2M |
| A2 | OPD 2M | OPD 2M |
| A3 | OPD 2M | GRPO 2M |
| A4 | GRPO 2M | OPD 2M |

所有 arm 在 2M 边界执行同样的 optimizer/scheduler reset。这样 A3/A4 是纯顺序对照，A1/A2 又是同 stage-count 的单信号对照，不需新增第六臂。

### 2. 修正 E1 的“同时匹配”表述

不同 objective 产生不同 completion length 和 rollout multiplicity，无法同时严格匹配完全相同的 prompt exposure、non-padding loss tokens 和 Student forward/backward FLOPs。

应选一个 primary denominator：

- E1 主匹配项：**Student backward loss tokens `U`**；
- prompt 只保证来自相同 registry、相同 sampling distribution 和预冻结的循环顺序，不宣称 exposure count 完全相等；
- Student update FLOPs 作为审计值报告，不再同时称为 matched；
- 所有 rollout/old-policy/Teacher forward 归入 E2。

### 3. 补齐 GRPO 最后几个决定性配置

除 group size 和 clip 外，还需冻结 reference KL `beta`、`loss_type`/advantage scaling、每批 policy update 次数、old-policy refresh 频率、generation backend、training/rollout weight-sync 频率，以及 prompt RNG 与 generation RNG 的独立可复现 stream。

### 4. 避免 single-seed screening 产生选择偏差

三 paired seeds 必须对 A0–A4 全部五臂执行，不能只重复单 seed 中看起来有效的 arm。单 seed 只能筛 learning rate/config，不可筛结果臂。

### 5. 冻结统计 estimand

需预注册各数据集是 macro-average 还是按 item pooled、两个 A1/A2-vs-A0 对比的 multiplicity、A3-vs-A4 的唯一 primary order contrast、pass@8 固定采样次数和 estimator，以及 equivalence CI 必须完全落入 ±2pp。

[MathArena ArXivMath 06/2026](https://matharena.ai/competitions) 是很好的新鲜 contamination sentinel：它有 49 道来自 2026 年 6 月论文的 final-answer 题，晚于 Gemma 4 发布且支持规则解析。但它规模小且对 E2B 可能出现 floor effect；G0 必须先确认有非零区分度。规则 parser 无法判定的响应应盲化人工复核，不能在主结果中悄悄引入未冻结的 LLM judge。[ArXivMath methodology](https://matharena.ai/arxivmath/)

## 低于 7 分的维度

### Feasibility — 6/10

- **Weakness:** Gemma 4 E2B/E4B 实际为约 5.1B/8B 总参数，262K 词表的 exact reverse-KL、Teacher SFT、五臂三 seed 和长至 2k/4k 的 group-8 rollout 尚未经过真实显存/吞吐闭合；硬件仍未确认。
- **Concrete fix:** 将 C0/G0 设为执行前硬 gate：分别实测 E2B LoRA backward、group-8 rollout、E4B Teacher SFT、E2B+E4B chunked KL 的峰值显存和 tokens/s；按同一比例降低五臂 `U`；2×80/4×48 不可得时停止主矩阵，而不是用量化 Teacher 或更换 loss 静默改变研究问题。
- **Priority:** **CRITICAL**

## Simplification Opportunities

1. 在 A0–A4 三 seed 完成前完全不运行 DPO shadow。
2. 只保留 same-lineage E4B Base→SFT Teacher；E4B-it sensitivity 和 larger Teacher 都是结果完成后的可选项。
3. 使用统一两阶段五臂网格，不新增 scheduler-reset control arm。

## Modernization Opportunities

**NONE at the primitive level.** Gemma 4、GRPO、on-policy distribution distillation 和双预算 Pareto 已足够现代。继续添加 PRM、Agent RL、异步系统贡献或多模态任务只会稀释主张。需要的是收紧 teacher lineage 和计算合同，而不是增加新组件。

## Drift Warning

**NONE.** 模型切换没有改变原始后训练归因问题；拒绝为了 Gemma 4 的多模态能力加入视觉/音频训练也是正确的。

## Verdict

**REVISE**

本轮已经从“宽泛作品集清单”升级为一个可信、现代、可证伪的研究设计。阻塞 READY 的内容很少但关键：把主 Teacher 改为 same-lineage E4B Base→SFT、统一两阶段五臂、选择唯一 E1 匹配量，并用真实硬件关闭全词表 OPD 的成本风险。完成这些后，方案有望进入高 8 分区；只有实际 C0/G0 证明 Gemma 4 + OPD 路径可运行且预算闭合，才接近 READY。
