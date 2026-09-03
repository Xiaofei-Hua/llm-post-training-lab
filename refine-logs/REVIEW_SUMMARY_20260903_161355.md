# Review Summary

## Final verdict

- **Planning / Method：READY**
- **Execution：CONDITIONAL — NOT YET READY**
- **Final independent score：9.06 / 10**
- **Training status：未下载模型、未安装环境、未启动 GPU**

## Score evolution

| Round | Overall | Verdict | 主要问题 | 结果 |
|---:|---:|---|---|---|
| 1 | 6.55 | REVISE | 范围太宽、DPO/GRPO/OPD主张发散、matched-token 不成立、算力低估 | 收缩为五臂 GRPO–OPD 主线，采用双预算 |
| 2 | 8.10 | REVISE | E4B-it 谱系混杂、stage/reset 不对称、E1 多重匹配、统计未冻结 | same-lineage E4B Base→SFT、两阶段网格、唯一 U、完整 GRPO/统计合同 |
| 3 | 8.37 | REVISE | checkpoint selection/gate 双用、seed/item 层级错误、OPD/U 语义缺失 | 拆 D_select/D_teacher_gate、item-conditional inference、精确 U 与 OPD 合同 |
| 4 | 9.06 | READY (planning) | 只剩三处数据文档措辞与真实执行证据 | 三处措辞已修；执行风险全部转化为 hard gates |

## 最终通过的关键审计

1. Problem Anchor 与具体解法解耦，模型/算法属于 Method Thesis；
2. 主张明确限制为 recipe-level intervention effect；
3. E2B/E4B 使用同一 `D_anchor` 谱系，且 checkpoint selection 与 Teacher qualification 数据隔离；
4. A0–A4 都有两段2M `U`、统一 reset、相同 objective config hash 和三个 seeds；
5. GRPO 的 loss、reward、policy refresh、sync、RNG、零方差与 cap 全部冻结；
6. OPD 的 on-policy refresh、mask、full-vocab reverse KL、normalization 和 Teacher freeze 全部冻结；
7. `U` 精确定义到实际 update positions，末批不 overshoot；
8. MATH-500 是唯一 confirmatory endpoint；统计条件于三 seeds、只在 item 层重采样；
9. Teacher 与算法成本没有隐藏或重复计算；
10. 真实硬件未知被明确标为 C0/C3/C4/C5 外部条件，没有用文档推演冒充实测。

## 已拒绝的扩张

- 不把 DPO、ORPO、KTO、PRM、GSPO/TIS 或 Agent RL 加回主矩阵；
- 不因 Gemma 4 原生多模态而增加图像/音频任务；
- 不使用 E4B-it 作为 primary Teacher；
- 不以 QLoRA、top-k KL、旧模型或删除“不好看”的 arm 救预算；
- 不拼接多个 benchmark 成语义不清的 primary composite；
- 不从三 training seeds 推断所有 seed population。

## Execution readiness 的外部条件

只有以下证据真实产生后，execution 才能转为 READY：

- C0：Gemma 4 processor/text path/tokenizer、LoRA target、非文本冻结和依赖版本通过；
- C3：exact full-vocab KL 的 value/gradient/mask oracle 通过；
- C4：same-lineage E4B 在独立 `D_teacher_gate` 通过；
- C5：E4B SFT、group-8 rollout、E2B+E4B OPD 与60M Student-token campaign 在已确认资源上闭合并留30% buffer。

这些条件超出本轮“只做前期规划”的授权范围，因此保留为真实 gate 是正确结论，不是未完成的文档工作。
