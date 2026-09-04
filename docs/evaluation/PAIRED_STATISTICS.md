# D08 Paired Statistics Core

## 模块边界

D08 把 D07 冻结的逐题 greedy correctness 转成可审计的确认性推断。它不读取 prompt、generation、candidate 或 sealed answer，也不重新运行 verifier。唯一额外的题目级信息是由同一 public benchmark snapshot 提供并受 hash 约束的 `level` 分层。

本模块是 CPU 统计框架，不是模型实验。仓库内的 8-item panel、强效应、零效应和等效结果全部是人工构造的 regression oracle，不能解释为 Gemma 4、MATH-500 或任何后训练算法的实测表现。

## 输入合同

`build_paired_panel` 接收：

1. 一个已通过 D07 schema 校验的 public snapshot；
2. 一个 greedy generation protocol；
3. 按 `A0..A4 × (101, 202, 303)` 排列的 15 个 D07 `EvaluationReport`。

构建器逐项拒绝以下混用：

- benchmark descriptor、public/sealed bytes、generation protocol 不一致；
- evaluator contract 或 evaluator version 不一致；
- 非 greedy、每题不止一个 correctness、题目数或 item ID/order 不一致；
- 缺少 `level`、重复 report hash 或不完整 arm-seed 网格。

产出的 `PairedPanel` 只含 report/checkpoint hashes、run ID、item ID/index、stratum 和 `5 arms × 3 seeds` Boolean correctness。`panel_sha256` 绑定完整规范化内容。training seed 标签来自后续 D11 run ledger；D07 本身没有能力证明调用方给 checkpoint 标注了正确的训练 seed。

## 冻结 estimand

对 contrast 的第 (i) 题和预注册 seed (s\in\{101,202,303\})，令

\[
d_{is}=Y^{\mathrm{treat}}_{is}-Y^{\mathrm{control}}_{is}\in\{-1,0,1\}.
\]

题目级值为三 seed 平均：

\[
d_i=\frac{1}{3}\sum_s d_{is},\qquad
\hat\Delta=\frac{1}{N}\sum_i d_i.
\]

因此推断条件于这三个固定 training seeds，只对题目总体推断；不得把区间解释为对任意训练 seed population 的不确定性。输出仍单列三个

\[
\hat\Delta_s=\frac{1}{N}\sum_i d_{is}.
\]

若至少一个 seed effect 为正且另一个为负，`seed_instability=true`。零效应不会被伪装成反方向，但仍会在逐 seed 数值中显式出现。

## 分层 item bootstrap

正式协议在每个 MATH `level` 内独立、有放回抽取与原层相同数量的 item，重复 10,000 次。每次抽到的是整个 item，因此同一索引同时携带三 seed 的差值；实现不会把 seed 当成独立样本。

95% 与 90% percentile interval 复用同一批 bootstrap replicates，quantile 固定为 exact Hyndman–Fan type 7。replicate total、效应、区间端点和阈值判断均使用整数或 `Fraction`；JSON 同时保存最简分数与 half-away-from-zero 的整数 ppm 展示值。

正式 bootstrap seed 固定为 `2026090408`。每个 hypothesis 的实际 PCG64 stream 由 base seed、statistics protocol hash、operation、hypothesis ID 和不含 correctness/report/checkpoint 的冻结 benchmark/resampling design 做 SHA-256 domain separation；随机流因而在观察模型结果前已经确定。完整 panel hash 仍单独绑定输出。审计同时记录 stream hash 和 NumPy 版本。

## paired randomization / sign flip

对每个 item 以 0.5 概率整体交换 treatment/control 的三-seed vectors，等价于对该 item 的完整向量共同乘以 `+1/-1`。正式协议运行 100,000 次，base seed 为 `2026090409`，同样按 hypothesis 分离随机流。

- C1a/C1b 是预注册正向 superiority 假设，使用 one-sided `greater` statistic；
- C2 同时允许两种顺序方向，记录 two-sided statistic，但 C2 的预注册结论由区间规则决定；
- Monte Carlo p-value 使用保守的 `(extreme + 1) / (B + 1)`；
- `exact_sign_flip_p_value` 为最多 20 items 的穷举参考 oracle，不会替代正式 100,000 次协议。

## C1、Holm 与 practical success

C1 family 只有：

- `C1a = A1 - A0`；
- `C1b = A2 - A0`。

两个 raw p-values 以精确有理数做 Holm step-down，排序 ties 由 hypothesis ID 稳定打破，adjusted p 强制单调并截断到 1。

`statistical_superiority=true` 当且仅当：

1. Holm-adjusted `p < 0.05`；
2. 95% bootstrap CI lower `> 0`。

`practical_success=true` 还要求点估计 `>= +0.02`。代码不会把四舍五入后的 ppm 用于阈值比较。

## C2 superiority 与 TOST 顺序

唯一顺序 contrast 为 `C2 = A3 - A4`：

1. 先判断 95% CI 是否严格位于 0 的一侧，且点估计绝对值至少 2pp；满足时输出 `superior_A3` 或 `superior_A4`，并将 equivalence 标为未评估；
2. 只有 superiority 不成立才评估等效性；90% bootstrap CI 的 lower 必须严格大于 -2pp，upper 必须严格小于 +2pp，二者同时通过才输出 `practical_equivalence`；
3. 其余情况输出 `inconclusive`。

这里的 TOST 使用预注册的 90% CI dual decision：边界相等不算通过；“点估计在 ±2pp 内”本身不能推出等效。`equivalence_assessed` 与 nullable margin decisions 防止同一结果同时声称 superiority 和 equivalence。

## 冻结 protocol

正式 `d08-statistics-protocol-v1` 固定：

| 字段 | 值 |
|---|---:|
| training seeds | 101, 202, 303 |
| stratum | `level` |
| bootstrap repetitions | 10,000 |
| randomization repetitions | 100,000 |
| superiority CI | 95% |
| equivalence CI | 90% |
| C1 family alpha | 0.05 |
| practical threshold | 2pp |
| equivalence margin | ±2pp |
| RNG | NumPy PCG64，hash-domain-separated |
| quantile | exact linear type 7 |
| Monte Carlo correction | plus-one |

`run_confirmatory_analysis(..., require_preregistered=True)` 会拒绝任何安静修改。缩短 repetition 的配置只允许用于明确标记的 unit tests。

## 审计与使用

```bash
uv sync --frozen --all-groups
uv run pytest -q tests/test_paired_statistics.py tests/test_statistics_audit.py
uv run python scripts/audit_paired_statistics.py
```

正式 audit 要求 implementation/runtime source、CLI、`.python-version`、`pyproject.toml`、`uv.lock`、panel、protocol 和 expectation 全部被 Git 跟踪且与同一 HEAD 一致。它校验实际 import source、loader 消费 bytes、执行前后文件 hash 与 HEAD，并把完整 text-free analysis 嵌入 audit artifact。

真实 D23 运行仍必须使用 D15 冻结的 MATH-500 snapshot 和 D22 的 15 个合法 endpoint reports。D08 完成不会通过 G1/G5/G6，不会产生任何可写进简历的模型效果数字。
