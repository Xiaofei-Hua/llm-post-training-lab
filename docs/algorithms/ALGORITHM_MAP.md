# 后训练算法地图

## 统一符号

- prompt：`x`
- response tokens：`y = (y_1, ..., y_T)`
- Student policy：`π_θ`
- reference policy：`π_ref`
- Teacher policy：`π_T`
- sequence reward：`r(x, y)`

## Base 与 SFT

SFT 在教师给定的轨迹上最小化交叉熵：

```text
L_SFT = - Σ_t log π_θ(y_t | x, y_<t)
```

它擅长建立行为支持集与格式先验，但训练 prefix 来自数据而非 Student 自己，存在 exposure bias。首期消融：原始数据、格式清洗数据、质量筛选数据，以及 2k/10k 数据规模曲线。

## DPO：offline shadow baseline

DPO 直接增大 chosen 相对 rejected 的隐式偏好 margin：

```text
L_DPO = -log σ(β[(log π_θ(y+|x)-log π_ref(y+|x))
                    -(log π_θ(y-|x)-log π_ref(y-|x))])
```

它不需要在线 rollout，是理解 offline/online 差异的重要附录。pair 必须来自同一 frozen rollout bank，且同时记录正确性、格式和长度，防止“chosen 只是更长”。它不进入主五臂 claim。

## GRPO：verifiable online RL

对同一 prompt 采样一组响应，用组内 reward 标准化形成 advantage，再使用 clipped policy ratio 更新：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
ρ_i,t = π_θ(y_i,t|x,y_i,<t) / π_old(y_i,t|x,y_i,<t)
L_GRPO ≈ -mean[min(ρA, clip(ρ,1-ε,1+ε)A)] + optional KL
```

首个主配置固定：`group_size=8`、`loss_type=dr_grpo`、`epsilon=0.2`、`beta=0`、`num_iterations=1`、token-level importance sampling、group reward scaling、`temperature=1.0`。old policy 与 rollout weights 每个 generation batch 刷新/同步；prompt/data RNG 和 generation RNG 独立。重点不只是公式，而是：

- 全组 reward 相同会产生零 advantage；必须监控有效组比例。
- 过宽 answer parser、格式奖励和长度奖励都可能被策略利用。
- on-policy freshness、clip fraction、entropy、response length 与 reward 方差必须进入日志。
- 不能只看 trainer reward，必须用独立 evaluator 复算正确率。
- 正式 run 的零方差组不重采样；有效组比例低于 30% 触发 stop gate，而不是临时改 reward。

## OPD / generalized knowledge distillation

Student 先从自身策略采样响应，Teacher 在相同 Student prefix 上给出稠密 token distribution：

```text
y ~ π_θ(.|x)
L_OPD = Σ_t D(π_θ(.|x,y_<t), π_T(.|x,y_<t))
```

主配置固定 fully on-policy、temperature 1.0 的 full-vocabulary chunked reverse KL；以下方向用于理解，只有主结论稳定后才做小型消融：

- forward KL `KL(π_T || π_θ)`：覆盖 Teacher 分布，较 mode-covering；
- reverse KL `KL(π_θ || π_T)`：惩罚 Student 放在 Teacher 低概率区域的质量，较 mode-seeking；
- generalized JSD：在两者之间插值，优先用稳定实现进行主实验。

核心对照：SFT continuation vs OPD，以及 `OPD→GRPO` vs `GRPO→OPD`。off-policy KD、其他 KL/JSD 方向与多个 Teacher 尺寸均为 nice-to-have。

## Reward 设计

首期主 reward 只有正确性：

```text
r_total = r_correct
```

- `r_correct`：答案抽取后进行精确/符号等价判断，正确为 1，其他为 0。
- `r_format`：不进入主训练；只在一次小规模受控负例中展示格式投机。
- 长度不进入默认正奖励；若出现 verbosity，再做有上限的惩罚消融。

Reward 必须通过以下攻击测试：空答案、重复答案、多重 boxed、复制题目、超长废话、NaN/异常表达式、prompt injection 字符串。

D05 已将该合同实现为 `posttrain_lab.rewards.ExactMathVerifier`：最后一个显式 terminal answer 生效，malformed last marker 不回退，prediction 不可解析为 0，reference/backend 错误阻断 batch；pinned Math-Verify 外另加 single-surface、normalization/juxtaposition cross-check 与 structural/assignment guard。冻结的 257-case CPU attack corpus 已全部通过，但真实模型输出的盲化人工一致率仍待 EVAL-002，因此 G1/G3 未据此宣告通过。

## 核心交互问题

- GRPO 提供稀疏但直接面向任务的 verifier reward；
- OPD 提供稠密但受 Teacher 支持集限制的 token signal；
- 五臂对照研究二者单独与前后顺序，而不是宣称比较所有后训练算法；
- SFT continuation 控制 Student 更新预算，DPO 只辅助解释 offline preference。

跨算法公平性采用两张账：E1 **只**匹配 Student non-padding backward loss tokens；prompt exposure 与 Student FLOPs只审计。E2 完整计入 rollout、Teacher/reference/old-policy forward、Student backward 与 GPU-hours，衡量 practical efficiency。

## 算法学习顺序

1. CE、mask 与 packing；
2. KL 方向、temperature 与 label smoothing；
3. Bradley–Terry 与 DPO；
4. REINFORCE、PPO、GRPO；
5. on/off-policy 与 importance sampling；
6. OPD/GKD 与 distribution mismatch；
7. GSPO/TIS、PRM 与多步 credit assignment（扩展）。

每一项都必须产出：一页公式推导、一个最小实现、一个 gradient/数值单测、一个失败案例。
