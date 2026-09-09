# 技术作品与简历交付

> 优先展示算法深度、可执行框架、性能实测和结果解释。数据隔离与最小复现记录是支撑材料，哈希数量、审计轮数和测试数量不作为主卖点。

## 四类核心成果

| 方向 | 需要讲清的技术内容 | 交付 | 当前证据范围 |
|---|---|---|---|
| 算法 | CE/GRPO/reverse-KL 梯度、normalization、mask、on-policy 与顺序作用 | 公式与实现、解析/reference 对照、失败案例 | D01–D04 CPU 实现已完成，真实模型效果待测 |
| 框架 | 模型/LoRA 接入、rollout–score–update、Teacher/old-policy、stage reset | 训练循环、时序图、可运行 CPU 示例及后续 Gemma 路径 | D09–D12 计划中 |
| 性能 | LM-head/KL 中间张量、分块/重计算、rollout 与通信瓶颈 | baseline/profile/优化对照、吞吐与内存图、取舍分析 | D11、D18–D20 待测，暂无加速数字 |
| 指标 | accuracy/pass@k、entropy/KL/clip/有效组率、retention、成本和错误迁移 | 主表、行为曲线、顺序图、accuracy–cost 图 | D05–D08 CPU 支持已完成，模型结果待测 |

## 代码与演示

- 从一个 objective 的公式定位到 forward、loss、gradient 与参数更新；解释 selected-position chunking 的内存行为。
- 运行 SFT/GRPO/OPD 共用的学习示例，展示 Student rollout 与 Teacher/old-policy 生命周期。
- 运行一个性能 baseline 与优化版本，说明 workload、测量范围、误差和波动。
- 展示一个 reward 不等于能力提升、有效组退化、Teacher 错误迁移或优化无收益的案例。

DPO 推导与 shadow 实现作为学习附录；不要求为了作品完整而提前实现 JSD、PRM 等扩展算法。

## 结果与材料

- E1 等 Student backward loss-token 主表和 C2 顺序对照；列出三个 seed 的效果及不确定性。
- 训练 reward/独立 accuracy、entropy/KL、有效组率、长度/截断与 retention 曲线。
- baseline/优化性能表：update 与端到端 tokens/s、p50/p95 latency、峰值内存，注明 CPU/GPU 和 workload。
- E2 accuracy–cost 图及 marginal/cold-start/campaign 成本；解释 Teacher 构建和推理开销。
- 至少一个负结果和三个能解释行为变化的样例。
- 一页摘要、4–6 页技术报告、30 秒/2 分钟/10 分钟讲稿、20 个技术追问与回答。

## 简历写法与阶段边界

可以在研究尚未完成时描述已交付的实现。例如当前可写“实现 masked causal CE、Dr.GRPO surrogate 与 full-vocab reverse-KL，完成 CPU 数值/梯度验证”；不能写“完成 Gemma 后训练框架”或填入尚未测量的加速/准确率。

后续量化表述采用以下结构，方括号是待实测字段，不是当前成果：

- 框架：“实现 SFT/GRPO/OPD 统一训练循环，支持 [已验证模型/阶段功能]，在 [合成任务或真实模型] 验证 [学习行为]。”
- 性能：“针对 [workload/硬件/精度] 的 [瓶颈] 实现 [优化]，相对 [baseline] 将 [kernel 或端到端指标] 从 [实测值] 改为 [实测值]，代价为 [时间/内存变化]。”
- 算法：“在 Gemma E2B、[数据/Student budget] 和三个 paired seeds 下比较五臂，观察到 [准确率差及区间/等效或不确定结果]，由 [行为/错误分析] 解释。”

每个量化字段附测量结果与配置入口即可。核函数、端到端吞吐、模型效果分别限定范围；不使用内部资产或他人成果，不把预期写成实测。G6 复核已有材料的准确性，不新建审计平台。
