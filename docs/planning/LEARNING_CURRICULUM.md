# 后训练算法学习课程与验收

项目执行和知识学习并行。每个模块必须同时完成“能推导、能实现、能实验、能表达”四层验收。

## Module 1：Transformer 与 Gemma 4 架构

### 必须掌握

- decoder-only causal LM、pre-norm、residual、MLP、GQA/MQA；
- RoPE、p-RoPE、local sliding-window 与 global attention；
- Gemma 4 E2B 的 effective vs total parameters、per-layer embeddings；
- KV/state、长序列复杂度、chat template 与 special tokens；
- 原生多模态 wrapper 和 text path 的参数边界。

### 验收产物

- 一张 E2B text stack 图；
- 参数量/激活/optimizer 显存手算表；
- 打印每层类型、LoRA target 和 trainable parameter 的脚本；
- 回答：“为什么 E2B 不能按普通 2B 模型估算训练显存？”

## Module 2：SFT 与数据分布

### 必须掌握

- token-level maximum likelihood、teacher forcing、assistant-only mask；
- packing、truncation、loss normalization、exposure bias；
- data mixture、quality filtering、difficulty curriculum；
- 伪 CoT、后验 rationale 和格式清洗的收益/风险。

### 验收产物

- 手写 masked CE 与官方实现数值对齐；
- 64 样本 overfit；
- raw/clean/quality-filter 三臂小实验；
- 数据卡、拒绝原因图和 20 条人工审计。

## Module 3：Preference Learning 与 Reward Modeling

### 必须掌握

- Bradley–Terry、reward model、KL-regularized RLHF；
- DPO 隐式 reward、β、reference policy；
- ORPO/KTO 的目标差异与适用条件；
- pair construction、length bias、position bias、judge bias。

### 验收产物

- 从 KL-RL 目标推导 DPO；
- 手写 DPO loss/gradient sanity test；
- 同 prompt、长度受控的 frozen pair bank；
- DPO 单 seed shadow result，不挤占主五臂。

## Module 4：Policy Gradient、PPO 与 GRPO

### 必须掌握

- policy gradient theorem、importance ratio、baseline/advantage；
- PPO clip、KL、GAE 与 critic；
- GRPO group-relative normalization、无 critic 的代价；
- zero-variance groups、on-policy freshness、entropy collapse；
- response/token/sequence-level loss normalization 与长度偏差。
- `dr_grpo` normalization、reference-KL beta、policy age、weight sync 与 RNG 隔离。

### 验收产物

- 从 REINFORCE 推到 GRPO surrogate；
- 两动作 Bernoulli policy 的解析梯度与实现梯度对齐；
- group size/有效组率诊断；
- 一张 reward、accuracy、entropy、length 联合曲线。

## Module 5：Reward Specification 与 Verifier

### 必须掌握

- outcome vs process reward、dense vs sparse reward；
- exact/symbolic verification、Rubrics、reward model；
- reward hacking、Goodhart、parser exploit、credit assignment；
- 为什么直接奖励长度或格式可能破坏真实目标。

### 验收产物

- 100–300 条 adversarial verifier tests；
- trainer reward 与独立 evaluator 分离；
- 一次过宽 parser/format reward 的受控负例；
- reward card：输入、输出、失败域、版本与审计统计。

## Module 6：Knowledge Distillation 与 OPD

### 必须掌握

- sequence KD、logit KD、forward/reverse KL、JSD；
- mode covering vs mode seeking；
- Student-generated prefix、distribution mismatch、exposure bias；
- on-policy/off-policy mixture、Teacher capacity 与 support；
- full vocab、top-k approximation 与 residual mass。

### 验收产物

- 手工分布推导 KL/JSD 极限；
- chunked reverse KL 与 tiny full reference 的数值/梯度对齐；
- Teacher capability/NLL gate；
- Student 错误、Teacher 正确/错误四象限分析。

## Module 7：训练顺序与 Optimization Dynamics

### 必须掌握

- curriculum、stagewise/joint objectives；
- optimizer state carryover、scheduler reset 与路径依赖；
- RL 扩展 support、KD 收缩/重分配 probability mass 的直觉；
- catastrophic forgetting 与 retention trade-off。

### 验收产物

- A0–A4 统一 2M+2M stage、阶段边界统一 reset 的 paired-seed 对照；
- 每阶段前后 KL/entropy/error transition；
- practical equivalence band 和负结论写法；
- 回答：“顺序差异是算法交互还是训练预算差异？”

## Module 8：On-policy、Off-policy 与重要性采样

### 必须掌握

- behavior/target policy mismatch；
- token-level 与 sequence-level importance ratio；
- clipping/truncation 的 bias–variance trade-off；
- GSPO/TIS 等扩展为何改善长序列 RL 稳定性。

### 验收产物

- 两个可枚举离散分布的 importance-sampling 精确仿真；
- ratio/ESS/clip fraction 监控说明；
- 读完 TIS-GSPO 后写一页“何时值得从 GRPO 升级”。

## Module 9：数据治理与 Contamination

### 必须掌握

- provenance、license、immutable revision、lineage；
- exact/fuzzy/semantic dedup；
- family-level split、benchmark leakage、Teacher contamination；
- 只能证明本项目数据去污染，不能证明预训练无污染。

### 验收产物

- canonical prompt registry；
- contamination report 和 removed-pair provenance；
- dataset card 与 checksum manifest；
- 30 个 borderline duplicate 人工复核。

## Module 10：Benchmark 与统计

### 必须掌握

- pass@1/pass@k、temperature confound、paired evaluation；
- bootstrap CI、seed variance、multiple comparisons；
- slice analysis、error taxonomy、equivalence testing；
- offline metric 与真实目标不一致。

### 验收产物

- evaluator 99% 人工一致率；
- paired bootstrap 实现和 synthetic unit test；
- 预注册 metric/threshold；
- MATH/GSM8K/MathArena/AIME/IFEval/MMLU-Pro result schema。

## Module 11：Scaling、效率与实验归因

### 必须掌握

- 参数、token、FLOPs 与 wall-clock 的不同含义；
- rollout、Teacher/reference forward、Student backward 的成本分解；
- LoRA/QLoRA/full FT 的优化空间差异；
- 唯一 E1 denominator（Student backward loss tokens）、E2 practical efficiency 与 `C_anchor/C_teacher/C_arm` 三类成本。

### 验收产物

- 100-step profile；
- accuracy/retention–compute Pareto；
- 每个结果的 config/data/model/git hash；
- 回答：“为什么只可匹配 Student loss tokens，而 prompt exposure、FLOPs 与端到端成本必须另报？”

## Module 12：研究表达与面试

### 必须掌握

- claim–evidence 对齐、ablation、negative result；
- 不夸大、不过度归因、清楚说明适用边界；
- 从公式、实现、系统、数据和结果五个角度接受追问。

### 验收产物

- 30 秒、2 分钟、10 分钟三个版本；
- 4–6 页技术报告；
- 20+ 追问题库；
- 每个简历数字的 evidence link。

## 总验收矩阵

| 模块 | 公式推导 | 最小实现 | 单测/实验 | 面试表达 |
|---|---|---|---|---|
| Architecture | TODO | TODO | TODO | TODO |
| SFT/Data | TODO | TODO | TODO | TODO |
| Preference | TODO | TODO | TODO | TODO |
| GRPO | TODO | TODO | TODO | TODO |
| Reward | TODO | TODO | TODO | TODO |
| OPD | TODO | TODO | TODO | TODO |
| Order | TODO | TODO | TODO | TODO |
| Off-policy | TODO | TODO | TODO | TODO |
| Data governance | TODO | TODO | TODO | TODO |
| Evaluation | TODO | TODO | TODO | TODO |
| Compute | TODO | TODO | TODO | TODO |
| Communication | TODO | TODO | TODO | TODO |
