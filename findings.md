# Findings

## 2026-09-03 — Planning baseline

- 项目需要优先展示算法归因，而不是训练系统吞吐。
- 用户明确把前沿 backbone 设为硬约束并排除 Qwen3。进一步核对 2026 官方发布后，主线升级为更新的 Gemma 4 E2B/E4B；核心实验内固定同一 E2B Student。
- Round 1 reviewer 指出 matched-token 无法公平控制 GRPO rollout 与 OPD Teacher forward；项目改为 signal efficiency 与 end-to-end compute 两套 estimand。
- 内部近期案例反复显示：数据清洗、reward specification、伪 CoT 和 offline/online 指标偏差是后训练成败的关键。
- 所有内部材料只作为个人思考来源，不进入可公开 Git 历史。

## 2026-09-03 — Round 2 refinement

- E4B-it 会引入未知 instruction/post-training recipe；主 Teacher 改为 E4B Base 使用与 E2B Student 相同的 `D_anchor` SFT 后冻结。
- 五臂统一为两个 2M Student loss-token stage，并统一在阶段边界重置 optimizer/scheduler，消除 stage-count/restart 混杂。
- E1 只匹配 Student backward loss tokens；prompt exposure 与 FLOPs改为审计项，端到端资源全部进入 E2。
- A0–A4 必须全部运行 3 paired seeds；single-seed pilot 只选 objective 配置，不能按结果筛 arm。
- 确认性 endpoint 收缩为 MATH-500；MathArena 06/2026 是 49 题的 freshness sentinel，而不是可单独承载 claim 的主集。
- 真实 blocker 是 5.1B/8B total weights 与 262K-vocab full-KL 的硬件闭合；单卡 24 GB 只能验证代码路径，不能支撑主矩阵。

## 2026-09-03 — Round 3 refinement

- Teacher checkpoint selection 与 Teacher qualification 必须拆分为 `D_select` 和始终 sealed 的 `D_teacher_gate`，避免 gate 乐观偏差。
- 统计推断条件于三个预注册 training seeds，只重采样 item 并携带完整三-seed prediction vector；不声称推广到所有随机种子总体。
- `U` 被定义为真正进入已执行 optimizer update 的有效 objective positions；prompt、padding、EOS 后 token、纯 rollout 与 skipped groups 不计。
- OPD 固定每 update 刷新 rollout、`num_iterations=1`、completion-only mask 和全 batch token normalization。
- LoRA 是主矩阵唯一参数高效训练形式；QLoRA 不能作为资源不足时的静默 fallback。

## 2026-09-03 — D05 exact/symbolic verifier

- Math-Verify 0.9.0 可以承担 symbolic comparison backend，但其 best-effort extraction 不能直接作为冻结 reward 合同；项目在外层固定 last-surface-wins、malformed-last fail-closed、prediction/reference 分离失败语义和有界输入。
- 第一轮对抗审计发现上游可将 finite set 与 tuple/interval 跨结构判为等价；初始 guard 令 corpus 从 122/128 提升到 128/128。随后独立攻防继续发现 malformed/escaped command precedence、trailing-math/numeric-gap/nested-container hijack、任意 text-unit 消隐、constant juxtaposition 语义漂移、Markdown fence 误闭合，以及 scalar/伪等式/Boolean-relation/assignment-RHS-family false positive。最终用完整 candidate validation、原式/显式乘法双解析和 structural guard 将冻结 corpus 扩为 257/257。这里记录的是 verifier correctness 修复，不是模型效果。
- prediction 不可提取、不可解析或 comparison timeout 可以安全计 0；reference 不可解析、backend exception 与非进程主线程调用必须让 logical batch fail fast，避免把基础设施或 gold 错误训练成模型负样本。
- D05 只完成 CPU reward/evaluator 共用语义及 adversarial audit；真实模型输出的至少 100 条盲审、数据 revision 校验、GRPO lifecycle 接入和所有 GPU 运行仍分别留给后续 gates/modules。

## 2026-09-04 — D06 data trust stack

- 单题随机切分不足以防止泄漏：source/problem/template family 必须构成传递 component，再用冻结 policy 确定性分配；命中污染后也必须整 component quarantine。
- 污染扫描不能只看 user prompt。system/tool、reference、response/assistant trace，以及跨 message 的 prompt/solution/full-record aggregate 都是可泄漏面；exact、fuzzy 和 review band 均采用 fail-closed。
- lineage 只有绑定 parent 的完整 payload 才能检测 split、families、source/revision 和上游 lineage 漂移；transform code/config 与外部 parent 因而分别进入 Git-bound registry 与 payload ledger。
- manifest self-hash 是 tamper evidence，不等于来源真实性证明。正式证据路径必须在捕获的单一 Git revision 上重算 split、污染与 manifest；HEAD 中途移动也必须失败。
- D06 的 96 个 tests 和 audit fixture 只验证 CPU 机制。真实 license/card bytes、family 标签、全量 scale profile、人工 pair review、sealed benchmark 和 G1 仍属于 D07/D15/EVAL-002。

## 2026-09-04 — D07 sealed evaluator

- test-answer 隔离必须表现为不同 capability 和进程输入，而不是同一 row 上“约定不读”的字段；generator request 因而完全不含 reference，vault 只进入 evaluator。
- sampling pairing 必须由 benchmark/item/sample 和冻结 protocol 派生，不能依赖 checkpoint 或遍历顺序；request ID 则必须包含 checkpoint，防止跨模型输出误合并。
- 只保存 aggregate accuracy 会丢失 D08 所需的配对单位。D07 同时冻结逐 sample generation hash/status、逐 item correctness vector 与精确组合式 pass@k。
- generation failure、invalid gold 和 verifier backend failure 都是基础设施错误，不能静默计 0；prediction mismatch/unparseable 才是模型错误。
- self-hash 仍不是来源证明。正式 D07 audit 将实际 `__main__`/module source origin、每个 loader 消费的 raw bytes、implementation/env/fixtures 与单一 Git revision 绑定，重跑 greedy/sampling oracle，并拒绝 foreign checkout、restore-after-read input TOCTOU 与 HEAD race。
- D07 formal synthetic audit 已在 implementation `c948fe2eae50289b78513a3a9513e188bff54a98` 上闭合：`passed=true`、0 failures、tracked inputs 与 Git 一致；它只证明 evaluator 机制和 provenance，不提供真实模型或 benchmark 结论。
- 6-item fixture 的 3/6 greedy、21/48 sampling 与 5/6 pass@8 只是 regression oracle；真实 benchmark materialization、官方 adapter、盲审和 G1 仍属于 D15。
