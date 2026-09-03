# 面试与作品交付清单

## 必须能现场讲清楚

- 为什么 SFT 能建立 support，但会有 exposure bias。
- DPO 的隐式 reward 与 β 的意义。
- GRPO 为什么不用 critic、group advantage 何时退化为零。
- forward KL 与 reverse KL 的 mode-covering/mode-seeking 差异。
- OPD 为什么要在 Student 自己生成的 prefix 上训练。
- reward 上涨、真实准确率下降时如何诊断。
- 如何排除数据泄漏、训练 token 和 decoding confounder。

## 代码级证明

- answer verifier 与 adversarial tests；
- GRPO reward functions 与审计日志；
- 最小 DPO loss 数值测试；
- generalized JSD/OPD loss 的方向和梯度测试；
- experiment manifest 与可复现入口。

## 结果级证明

- 一张按 Student backward loss tokens 严格匹配的 E1 主表；
- 一张含 `C_anchor/C_teacher/C_arm` 的 E2 marginal/cold-start/campaign 成本表；
- 一张 `reward vs independent accuracy` 图；
- 一张 stage-order 对照图；
- 一张 error taxonomy 图；
- 至少一个有解释力的负结果；
- 三个代表性样例，展示算法改变而非只展示“答对了”。

## 对外材料

- 30 秒项目介绍；
- 2 分钟简历深挖版本；
- 10 分钟技术分享；
- 4–6 页 technical report；
- README 可复现命令与结果卡；
- 20 个追问题及答案。

## 简历数字规则

- 每个数字必须指向不可变结果 JSON 和配置。
- 明确模型尺寸、数据规模、benchmark、seed 和统计方法。
- 不把组合实验收益全部归因于某个算法。
- 不使用任何内部指标、未公开名称或同事成果作为个人成果。
