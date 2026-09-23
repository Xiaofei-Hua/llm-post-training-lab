# D01–D14 全进度复审（2026-09-23）

本轮技术目标：复查现有算法、训练框架、实测结果和研究方向，修复真实问题后重新计数，直到连续两轮没有新增问题。基线为 `c25272d`，D14 真实 Base 接入已完成。复审不启动 D15 数据或 D16+ 正式训练。

共完成三轮：第一轮修复两类代码/数值问题及过时状态，第二、三轮无新增 bug 或方向偏移，连续通过计数为 **2**。这是对下述实现、输入和环境的复审结果，不是所有输入与硬件均无缺陷的证明。

## 发现、复现与修复

| 问题 | 影响与修复前证据 | 修复与验证 |
|---|---|---|
| 完整词表采样与 Gemma 控制 token 拒绝冲突 | sampler 可以抽到 image/audio/video 等 ID，但适配器在续生成或整段重算时抛 `ValueError`。原生 wrapper 接受同一前缀；4 个新增 CPU 用例在旧实现均失败 | 使用原生无媒体特征语义：3 种占位符查 PAD embedding/PLE，边界标记保留本身 lookup；原 token、mask 和概率不变，不过滤或重采样。CPU 原生 loss/梯度、两种 head 采样路径及真实 E2B/E4B 验证通过 |
| BF16 精度策略未固定 BLAS/split-K，短前缀概率偏差超限 | 原配置关闭低精度 reduction 后仍允许 split-K。真实 E2B 控制前缀的采样/重算 log-prob 最大差 **0.188593 > 0.1**；相同前缀的原生/适配器 hidden states 一致，整段与逐步前向的 hidden 差可达 1.804688 | 固定 cuBLASLt，关闭 BF16 低精度 reduction 和 split-K；配置保存进 checkpoint，并提前拒绝不支持的组合。相同 E2B 回归误差降为 0；E4B 为 0.043503，均满足原界限 |
| 当前状态与完成边界有遗漏 | 模块总表仍写 13/32；系统设计仍称 Gemma 待做、下载未授权；D12 后续说明仍称真实接入未执行 | 当前状态统一为 D14 完成、14/24（完整目录 14/32）、下一模块 D15；带日期的历史实验与失败结果保留 |

实现涉及 `models/gemma4.py`、`train/loop.py`。`experiments/gemma4_integration.py` 的原验收入口加入控制前缀回归，使用有正概率的强制抽样覆盖罕见分支；它不估计自然发生率或任务准确率。原来的拒绝测试由捕获实际接线错误的回归替换。

精度诊断另比较了 math SDPA：原 BLAS 下该失败位置仍有约 0.1504 log-prob 差；cuBLASLt + 禁用 split-K 的组合使该位置差为 0。这里不声称单独某个开关解释全部变化，也不把有限样例上的零差推广成跨形状逐位等价。完整失败和修复记录见 [复审实测](../../artifacts/gpu/d14_review_validation.json)。

## 逐轮记录

| 轮次 | 复审重点与依据 | 结果 | 连续通过计数 |
|---|---|---|---:|
| 1 | D01–D14 契约和调用链、采样分布、mask/归一化/冻结、历史结果与当前状态；先复现控制 ID 失败，再定位真实权重数值偏差；修复后全量 CPU、两模型 D14 和 CUDA 回归 | 两类代码/数值问题及状态遗漏已修复 | 0 |
| 2 | 从独立公式和接口组合重查：CE/KL loss/gradient、GRPO 分段导数、预算、reward、隔离/评测/统计；Gemma 前缀与左 padding；核对既有结果的解释边界 | 无新增问题 | 1 |
| 3 | 从恢复和证据重查：Gemma 三 objective 中途 checkpoint/阶段切换、Teacher 与 optimizer 隔离；最终差异、原始实测、精度配置和文档逐项对照 | 无新增问题 | 2 |

第二、三轮使用不同的检查重点；没有用重复执行同一套测试充当两轮复审，也没有新增通用审计平台。

## 覆盖与实际验证

| 范围 | 本次验证 |
|---|---|
| D01–D04 | 80 组独立 EOS/mask/预算参考；48 组含 scale/softcap/temperature 的 FP64 CE/KL loss 与梯度对照；40 组逐 token GRPO loss/导数；group size 8 的全部 9 种正确数量与 sample std/零方差处理 |
| D05–D08 | 42 个有理数正负例；24 个随机 family 图对照独立传递闭包；330 个 pass@k 组合穷举；24 组整题三 seed 向量的 sign-flip/分位数参考；全量 CPU 覆盖 sealed/public 与失败语义 |
| D09–D10、D14 组合 | 12 个 Gemma 前缀/左 padding 对照，最大 prefix logit 差 1.06e-7；缩小的原生 Gemma 保留 PLE、shared KV、控制 ID、LoRA、microbatch=1 与重计算，三 objective 各完成 67-token 阶段中途续训，再重置为 3-token SFT 阶段；参数、AdamW、scheduler、采样 RNG 精确一致，Teacher 冻结 |
| D11–D12 | 重算既有 D11 的 26 个计时/吞吐聚合，D12 的 30 个阶段预算及学习/错误 Teacher 退化方向；未重跑性能，也未把合成学习解释成真实研究结果 |
| D13 | 原始 12 组 profile 的 p50/p95、吞吐和峰值内存可重算；新精度配置下，6 组数值、6 组累积、6 组精确续训、6 组非有限失败恢复 CUDA 回归通过 |
| D14 真实权重 | E2B/E4B 顺序复跑完整 C0：原生前向、PLE、softcap、分块/重计算 loss/梯度、12 tokens 的两次 LoRA 更新、冻结摘要、当前策略概率；新增控制前缀对照通过 |

修复后全量 CPU：**638 passed（20.30 s）**，Hypothesis seed `20260923`。Ruff、`git diff --check`、离线 uv lock 检查与依赖检查通过。缩小 Gemma 的续训证据是 CPU FP32；真实 E2B/E4B 证据是 RTX 5070 Ti 上的 BF16 frozen Base / FP32 LoRA，两者分开记录。

真实两条算术样例的 NLL：E2B **2.978141 → 0.370571**，E4B **3.522726 → 0.227566**；410/516 个 LoRA A/B 矩阵实际变化，全部冻结参数摘要相同、无梯度。dense/分块梯度相对误差在本次两模型样例上为 0，原 2% 容限不变。控制前缀的 log-prob 界限仍为 0.1，未因失败放宽。

## 方向与范围

当前主线仍是同一 Student SFT anchor 下，比较 exact-reward GRPO 与 same-lineage Teacher reverse-KL OPD 的单独和顺序作用；正式五臂、两阶段、三个 paired seeds、backward-token 预算与 C1/C2 未改变。DPO 和扩展模块继续延后，没有以通用审计建设替代算法/框架工作。

模块进度保持 **14/24**。D14 有界接入通过不表示真实数据、Teacher capability、双模型 OPD 显存、长序列 rollout、正式 GRPO/OPD 或 benchmark 准确率已经验证。E4B 控制前缀仍有有限数值偏差，后续 D18/D19 需按真实 batch/长度复验。新 BLAS 配置没有做稳定性能比较，不宣称加速；历史性能表仍对应原配置。

## 复跑

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q --hypothesis-seed=20260923
.venv/bin/python -m ruff check .
uv lock --check --offline
uv pip check
git diff --check

# 两个模型顺序运行；使用本地已下载的固定 revision。
HF_HUB_OFFLINE=1 .venv/bin/python -m posttrain_lab.experiments.gemma4_integration \
  --model google/gemma-4-E2B --output outputs/review_d14/e2b_integration.json
HF_HUB_OFFLINE=1 .venv/bin/python -m posttrain_lab.experiments.gemma4_integration \
  --model google/gemma-4-E4B --output outputs/review_d14/e4b_integration.json
```

额外独立探针位于忽略的 `outputs/review_d14/`；公开 JSON 保留结果、失败诊断、固定配置与复验指标。原 D01–D13 历史复审见 [前次记录](D01_D13_REVIEW.md)。
