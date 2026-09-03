# Round 4 Finalization

## Problem Anchor

在有限算力和完全公开、可追溯的数据条件下，从同一前沿学生模型的 SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。

## Anchor Check

**Preserved.** 最终整理只修复跨文档一致性并生成交付物，没有改动问题、claims、模型、五臂或 estimand。

## Simplicity Check

**通过。** 主矩阵保持 A0–A4；没有新增算法或实验臂。DPO 等继续 deferred。

## Reviewer minor fixes completed

1. Benchmark 不再被描述为 dev calibration 数据；
2. `D_dev` 明确与 `D_select`、`D_teacher_gate`、`D_core` 和 test 隔离；
3. sealed evaluation 清单补齐 MMLU-Pro。

## Terminal status

- Final independent score：9.06/10；
- Planning/Method：READY；
- Execution：CONDITIONAL；
- 未下载模型、未占用 GPU、未伪造 C0/C3/C4/C5 结果；
- 后续以 `refine-logs/EXPERIMENT_TRACKER.md` 为唯一运行队列。
