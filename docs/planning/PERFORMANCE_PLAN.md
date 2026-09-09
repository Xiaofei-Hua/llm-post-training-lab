# 后训练性能实验计划

> 2026-09-09 新增；所有条目为计划，尚无本计划的性能实测。当前只执行 CPU 开发，GPU 项待单独授权。

## 要回答的技术问题

1. masked CE 与 full-vocab reverse KL 的时间和内存主要消耗在哪里？有效位置分块与重计算在什么形状下有收益？
2. rollout、Teacher forward、Student backward、权重同步分别占多少时间，局部优化能转化成多少端到端收益？
3. batch、序列长度和有效 token 比例如何影响吞吐与峰值内存？优化是否改变原目标函数或学习动态？
4. 在相同 Student 更新预算下，准确率与实际计算成本如何权衡？

性能实验不要求新建 C1/C2 式预注册 claim。每次只写清瓶颈假设、对照、测量指标和结论；复用现有配置与日志。

## CPU 首轮实验（D11，D12 集成复测）

| ID | 对照与单次控制变量 | 主要测量 | 技术决策 |
|---|---|---|---|
| PERF-CPU-CE | 可容纳的 dense CE reference vs D02 selected-position chunking；逐次改变有效位置数、vocab 或 chunk size | loss/gradient 误差、forward/backward ms、有效 tokens/s、峰值 RSS | 找到分块减少中间 logits 的收益和额外开销 |
| PERF-CPU-KL | 可容纳的 dense full-vocab reverse-KL reference vs D04 exact chunk/recompute；逐次改变 chunk size 或 vocab | KL/gradient 误差、时间、峰值 RSS、Teacher/Student LM-head 占比 | 选择内存–时间折中；记录无收益的小形状 |
| PERF-CPU-STEP | D10 固定 tiny 模型与采样设置下的 baseline vs 一项 profile 指向的优化 | rollout/score/loss/backward/update 分段时间、完整 step latency、有效 tokens/s | 判断局部优化是否改善完整学习循环 |

从能快速完成的两档形状起步，按瓶颈增加一个维度，不做全组合 sweep。CPU tensor 始终在 `cpu`，不下载真实 checkpoint，不开启 MPS/CUDA。需要真实采样的测量单独列出生成长度；kernel 对照则复用固定 token/prefix 输入。

## GPU 后续实验（D18–D20）

| ID | 工作负载 | 优化候选 | 主要输出 |
|---|---|---|---|
| PERF-GPU-SFT | E2B backward 与 E4B SFT | microbatch、gradient checkpointing、经语义验证的 packing | update tokens/s、显存、step p50/p95、padding 比例 |
| PERF-GPU-ROLLOUT | group-8 GRPO，固定采样和 cap | generation backend、batch 调度、weight sync 开销 | rollout tokens/s、有效组率、sync 占比、完整 step 时间 |
| PERF-GPU-OPD | E2B Student + E4B Teacher exact KL | 有效位置 chunk size、重计算、模型放置/分片 | Teacher/KL/backward 时间、显存、有效 loss tokens/s |
| PERF-GPU-E2E | 固定 recipe 的完整训练循环 | 选择最大瓶颈的一项优化并复测 | baseline/优化对照、成本预测、accuracy–cost 图的数据 |

这些实验复用 C5-001..005 的 profile，避免另跑一套相同的算力审计。正式矩阵前固定选定实现；同一 objective 跨 arm/stage 复用。若改变 KL 近似、策略新鲜度、有效 batch 或监督信号，属于算法变量，不能当作等价性能优化混入主矩阵。top-k KL、Teacher 量化和异步陈旧策略仍在独立扩展范围。

## 测量方法

- 同一对照固定设备、线程/进程数、dtype、输入形状、有效 token mask、模型/算法设置；性能选型只用合成输入或 D_dev，不查看正式 test。
- 先 warm-up；CPU 初始 10 次 warm-up、每轮 30 次测量、至少 5 轮，若耗时过长则缩小 workload 并记录。GPU C5 各类至少 100 个 steady-state steps。保存各轮结果，报告中位数、p95 与轮间范围。
- kernel 对照分开测 forward 与 forward+backward；端到端测量包含采样、打分和 optimizer。GPU 计时在区间边界同步，端到端 wall time 单独测；记录 profiler 的额外开销。
- `update tokens/s = 实际参与已执行更新的 Student loss tokens / update 秒数`；`end-to-end effective tokens/s` 的分母包含 rollout/Teacher/sync；`rollout tokens/s` 包含实际生成但最终被丢弃的 token。三者分列。
- CPU 内存用独立进程峰值 RSS，注明包含模型/运行时基线；GPU 同时报 peak allocated/reserved。理论 tensor 字节估算另列，不能当作实测峰值。基线 OOM 时记录 OOM，仅在双方可跑形状报告实测比率。
- loss/gradient 容差按 dtype 与 reference 在比较前确定，复用 D02–D04 oracle。先保证数学目标与梯度正确，再解释性能结果；不要求整段随机训练轨迹逐位相同。
- FLOPs 标注解析估算或 profiler 来源及覆盖范围；无法可靠测量时填 unavailable。CPU 时间和内存不能换算为 Gemma GPU 加速或显存收益。

## 验收与成果表达

D11/D20 各完成一轮“baseline→瓶颈→优化尝试→复测→解释”。不预设 2×、50% 等收益门槛；无加速或内存下降但耗时上升也记录为有效结论。可运行基准、对照表和瓶颈解释为主要交付，性能图按需要在对应模块生成。

| Workload/设备/dtype | 变量 | baseline | 优化后 | 比率/差值 | 波动 | 数值误差 | 解释 |
|---|---|---|---|---|---|---|---|
| 待测 | 待测 | TBD | TBD | TBD | TBD | TBD | 不填预期数字 |

最终展示分别包含核函数/内存优化、完整训练吞吐、达到实测能力所花成本。加速只作用于某一 kernel 时，简历明确该范围；算法准确率提升由正式五臂结果支持。
