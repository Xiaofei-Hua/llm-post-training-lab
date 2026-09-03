# Round 1 Review

<details open>
<summary>Independent reviewer raw response</summary>

CALIBRATION: none

COMPOSITE: **6.55 / 10**

| 维度 | 权重 | 分数 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8 | 1.20 |
| Method Specificity | 25% | 6 | 1.50 |
| Contribution Quality | 25% | 6 | 1.50 |
| Frontier Leverage | 15% | 8 | 1.20 |
| Feasibility | 10% | 5 | 0.50 |
| Validation Focus | 5% | 6 | 0.30 |
| Venue/Portfolio Readiness | 5% | 7 | 0.35 |

**GAP:** 没有可用的人工精选 known-good/known-bad proposal anchors，因此不能做 exemplar-relative calibration。相对于 READY=9 的明确标准，本方案的差距不在模型是否前沿，而在因果 estimand 尚未锁定：当前把不同数据形态、在线采样成本、teacher/reference forward、阶段数和 loss 一并改变，却用 matched-token 概括为公平比较；同时 SFT/DPO/GRPO/OPD、两种顺序、两种 reward、多个 teacher 和多个尺度形成了过宽矩阵。它已经是一个不错的实习作品集蓝图，但还不是一个聚焦、可证伪且预算闭合的研究设计。

## 核心判断

### 1. Qwen3.5/3.6：确实前沿，但角色需要收紧

模型选择是真实且现代的。需修正三点：

- 4B/9B 是 post-trained teacher，而 2B 是 Base student。这个差异合理，但必须写成“teacher capability/instruction-tuning gap”，不能只解释为参数规模差异。
- Teacher 必须通过预注册 gate：在独立 calibration split 上显著优于 SFT student，且对 verified solution 的 token NLL 更低；4B 不过 gate 就直接换 9B，否则 OPD 可能只是向弱 teacher 拟合。
- Qwen3.6-27B 与 Qwen3.5 共享架构族，把它作为 teacher 不能称为“architecture portability”；它只能验证 teacher-scale/checkpoint transfer。真正的架构 portability 需要不同架构，但本项目无需为此扩张。

Qwen3.6-27B 作为 stretch 是合理的，作为核心依赖则不合理。约 28B BF16 参数仅权重就约 56 GB，尚未计 KV cache、student、optimizer 和全词表 logits。4–8×80 GB 档位可以尝试，24 GB 路线不应承诺 logit-level 27B OPD。若必须实际使用 Qwen3.6，可将一个小规模 teacher-quality/sequence-generation calibration 设为必做，在线 logit OPD 保持 stretch。

### 2. 文本训练多模态 hybrid-attention 模型：可以方法学干净

这本身不是缺陷，也不需要为了“充分利用模型”而加入多模态任务；那会偏离明确的 non-goal。条件是把研究对象准确表述为 text path 的后训练，并满足：

- 不插入 image/video token，禁用视觉预处理；
- 视觉塔始终 `requires_grad=False`，训练前后 checksum 一致，并加入 zero-gradient assertion；
- 明确“固定语言主干”是固定架构还是冻结参数；当前措辞有歧义；
- 指定 LoRA/full-FT 的确切 text-module target，并统一冻结或移除辅助部分；
- 固定 chat template、thinking/non-thinking mode、特殊 token 和解码协议。

不做多模态 retention 评测是可以接受的，但不能声称保持了完整多模态能力。

### 3. 比较范围过宽

当前至少包含 Base、SFT、DPO、GRPO、OPD、两个顺序、两种 reward、off-policy KD、三个 teacher 档位和多 seed。对 12 周、算力未知的项目过宽，而且“阶段顺序”与“训练信号地图”实际上是两个研究问题。

建议将主问题收缩为：

> 从同一个 SFT checkpoint 出发，dense on-policy teacher signal 与 sparse verifier reward 单独及按不同顺序施加时，是否产生可重复的能力、探索和 retention 差异？

核心五臂即可：

1. SFT continuation；
2. GRPO；
3. OPD；
4. OPD→GRPO；
5. GRPO→OPD。

五臂使用相同总预算；单阶段臂使用完整预算，双阶段臂各分一半。DPO 降为单 seed 的 offline shadow baseline 或作品集附录，不再是并列贡献。`exact+format` 只作为 reward-hacking 负例，不跑完整矩阵。

### 4. matched-token 不是跨算法公平计算控制

这是当前最严重的因果缺陷。相同 token 数无法等价控制：

- GRPO 的 group rollouts 和在线生成；
- DPO 的 chosen/rejected 双序列及 reference forward；
- OPD 的 student rollout、teacher forward 和超大词表分布；
- SFT 的单次 teacher-forced backward。

应预注册两个不同 estimand，而不是追求一个“公平数字”：

- **Signal/sample efficiency**：匹配 canonical prompt IDs、最大输出长度、student non-padding loss tokens 和 student backward FLOPs。
- **Practical compute efficiency**：记录并匹配或绘制总计算 Pareto，包括 rollout tokens、student/reference/teacher forward、student backward、实测 accelerator-seconds/GPU-hours。

主表可以按 student-update budget 比较算法行为，附表按总 FLOPs/GPU-hours 比较实用效率。二者不能混称 matched-token。

阶段顺序实验还必须固定：

- 每个 objective 的 token/FLOP 配额；
- 相同 parent SFT checkpoint 和 paired seeds；
- 每阶段重置 optimizer/scheduler，或明确声明研究的是“连同优化器状态的 pipeline order”；
- 相同 prompt exposure、采样温度、最大长度和 early-stop rule。

### 5. 数据污染与评测仍不足以支撑强因果表述

现有 exact hash、n-gram 和 MinHash 是良好起点，但 MATH-500、GSM8K、AIME 都是高频公开集合。需要：

- 建立唯一 canonical prompt registry，让 SFT target、DPO pair、GRPO/OPD prompt 都来自同一训练 ID 集合；
- 按来源、题型/template family 分组切分，不能只随机按题切分；
- 同时对题目、参考解答、生成 reasoning trace 做规范化和近重复检测；
- 冻结 test manifest 后才调参，并保存污染审计及 removed-pair provenance；
- 明确只能保证“项目后训练数据不包含测试集”，无法证明模型预训练或 Teacher 未见过公开 benchmark；
- 增加一个在模型发布后公开、项目开始前冻结的困难数学 test set，作为低污染 sanity check；
- 报告 paired item bootstrap CI、seed variance；AIME 样本很小，不应单独承担决定性结论；
- pass@k 必须匹配采样次数并同时报告 pass@1，否则可能把更高熵误读为能力提升。

## 所有低于 7 分的项目

### Method Specificity — 6/10

- **Weakness:** OPD 的 KL/JSD 方向、温度、采样 stop-gradient、词表处理和 logits 实现未确定；GRPO 的 group size、clip/KL、zero-variance prompt 规则未确定；DPO pair 构造与 adapter 策略也未具体化。
- **Concrete fix:** 写出每个分支的执行合同。OPD 固定一种 divergence、温度和 teacher checkpoint；使用 chunked/fused full-vocab loss，或经小规模等价性验证后的 top-k+residual-mass target，避免持久化 `[B,L,V]` logits。GRPO 固定 group size、reward formula、advantage normalization、clip/KL 和无有效 advantage 时的 drop rule。明确统一采用 LoRA、QLoRA 或 full FT，并列出 text-module targets。
- **Priority:** **CRITICAL**

### Contribution Quality — 6/10

- **Weakness:** “训练信号因果地图”过大，协议贡献、reward audit 和阶段顺序都在争夺主叙事；现设计能比较完整 recipe，却不能隔离“loss 本身”。
- **Concrete fix:** 将 dominant contribution 收缩为 GRPO 与 OPD 的单独/顺序交互；SFT 是 parent/control，DPO 是 offline shadow baseline。把结论限定为“受控 recipe intervention 的效果”，而不是纯 loss causal effect。
- **Priority:** **CRITICAL**

### Feasibility — 5/10

- **Weakness:** 算力未确认，却同时承诺 2B 三 seed、在线 RL、全词表 OPD、9B/27B teacher。大词表 teacher logits、reference 模型和 rollout cache 的显存/吞吐没有进入预算；100–550 GPUh 仍是未经实测的估计。
- **Concrete fix:** 设置硬 gate：小模型只验证代码路径，不能用于选最终超参；Main Student + default Teacher 完成 100-step profile 后才承诺主矩阵；更大 Teacher 只在 capability gate 需要时进入核心；最大 Teacher 保持独立 stretch。profile 必须分别测峰值显存、rollout tokens/s、student backward tokens/s、teacher KL tokens/s，并据此裁剪臂数和 seed 数。
- **Priority:** **CRITICAL**

### Validation Focus — 6/10

- **Weakness:** 核心矩阵和消融数量偏多，而现有公开数学 benchmark 污染风险高；缺少预注册统计规则和低污染 held-out set。
- **Concrete fix:** 主验证限制为上述五臂、一个 teacher、一个 exact reward；format reward 仅做负例。采用 family-level split、post-release held-out test、paired bootstrap CI 和预声明的最小效应阈值；三 seed 只给最终核心臂。
- **Priority:** **IMPORTANT**

## Simplification Opportunities

1. 将 DPO 从主矩阵降为同一 frozen SFT rollout bank 上的 offline shadow baseline。
2. 核心只保留一个通过能力 gate 的 teacher；最大 Teacher 不进入成败关键路径。
3. exact verifier reward 是唯一主 reward；format reward 只做一次受控 hack 负例。

## Modernization Opportunities

1. 用“shared frozen rollout bank + online refresh”统一接口：DPO/off-policy KD 使用冻结 bank，GRPO/OPD 使用当前 policy，由此把核心差异表述为 offline/online freshness 与 sparse/dense supervision，而非算法名堆叠。
2. 以 student-update efficiency 和 end-to-end compute Pareto 双视角替代粗糙 matched-token。
3. 若做大 Teacher OPD，采用 chunked fused divergence 或经验证的 sparse teacher targets；不要保存完整 Teacher logits。

## Drift Warning

**NONE.** 当前 text-only 路线仍然解决原始的受控后训练归因问题。为了迎合原生多模态架构而加入视觉训练、VLM benchmark 或跨架构主矩阵，反而会发生 drift。

## Verdict

**REVISE**

方向正确、模型足够前沿，而且作为实习作品集很有潜力；目前阻塞 READY 的不是技术过时，而是因果控制定义、主贡献聚焦和计算闭合。先把主问题缩成 GRPO–OPD 顺序/交互、用双预算协议替代 matched-token，并完成前沿模型 trainer/teacher 的 100-step 兼容性与吞吐 gate。

</details>
