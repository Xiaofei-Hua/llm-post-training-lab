# D13：单 GPU 运行时与可恢复训练

2026-09-22：用户确认当前 GPU 可用并恢复实验，已在 **RTX 5070 Ti / 16GB / WSL2** 完成 CUDA witness、数值/梯度、实际参数更新、精确续训、失败更新恢复和 12 组性能测量。**D13 COMPLETE；核心进度 13/24**。原始记录见 [d13_runtime.json](../../artifacts/gpu/d13_runtime.json)。只使用本地初始化 tiny 模型和合成输入；D14 真实 Gemma 接入、模型/数据下载及正式训练仍未执行。

2026-09-23 后续进展：D14 真实 Base 接入已完成，核心进度更新为 14/24，见 [D14](GEMMA4_INTEGRATION.md)。D14 增加 BF16 冻结基座/FP32 LoRA，并默认禁用 cuBLAS BF16 低精度中间累加；上文和下方性能表保留 D13 当日的历史测量范围。

## 实现与正确性

`Trainer` / `TrainerConfig` 在 D10 路径上支持显式 `cpu` 或单个 `cuda:N`，默认 CPU，没有第二套 trainer。模型与 LoRA 在 CPU 上按 seed 初始化、选定可训练参数，再迁移设备并创建 optimizer；采样 generator、tokens 和 mask 跟随模型设备。CPU 默认路径不初始化 CUDA。锁定的 PyTorch 2.14 在 AdamW 健康检查中会查询 accelerator/current stream，训练器对全 CPU 参数跳过该检查，CUDA 参数仍保留上游检查；优化公式和 state 不变。

- BF16 使用 autocast，参数和 AdamW moments 保持 FP32，概率归一化与 CE/KL 使用 FP32；CPU reference 仍可用 FP64。BF16 不使用 loss scaling。
- 在完整逻辑 batch 上计算 reward、GRPO advantage 和 D01 预算，再按 completion 行拆 microbatch。SFT/OPD 局部 loss 除以完整 batch 选中的 token 数；Dr.GRPO 保持预算截断前 active completions × 固定 cap 分母。局部 backward 直接求和，最后统一 clip、AdamW update、预算提交和 scheduler step，不再除以 microbatch 数。
- Teacher 按 microbatch forward，保持 frozen/eval/no_grad。生成仍为完整当前策略、每批一次 update、无 KV cache；GRPO 分组在拆分前完成。
- activation checkpointing 使用非 reentrant 的逐 decoder block 重计算，支持 embedding 冻结的 LoRA。tiny decoder 没有 dropout，故不保存 block RNG；LM-head 分块重计算继续复用 D02/D04。
- 任一 microbatch 的 loss 或累积 gradient 非有限时，整次更新丢弃并清空梯度，不扣 token、不更新 moments/LR。新阶段保留参数，重置 optimizer/scheduler 和采样 RNG。

`save_checkpoint(path)` 在完整 attempt 结束后原子替换文件，保存 Student/Teacher、可训练参数与 LoRA scaling、AdamW、scheduler、预算、stage/attempt/policy version、采样及 CPU/CUDA RNG、已有记录。`load_checkpoint(path)` 使用 `weights_only=True` 加载到同配置的新 Trainer，拒绝不匹配的运行时或模型拓扑。样本顺序和 reward 实现由调用方保持一致；精确续训只针对相同环境和配置，不承诺跨版本、设备或 batch 形状的随机训练轨迹一致。

## 本轮修复：失败 microbatch 的整批清理

原循环只检查返回的 loss 是否有限，但 GRPO/OPD 内核遇到非有限 log probabilities/logits 时会先抛出 `FloatingPointError`，绕过失败更新路径，留下此前 microbatch 的梯度。用旧实现复现了 GRPO/OPD 两项失败；SFT 原有返回 NaN 的路径正常。

现在捕获 loss 计算的数值异常，记录 `skipped_nonfinite_loss` 和失败原因，然后沿用统一的梯度清理与预算提交路径；类型、配置和编程错误仍会抛出。CPU 回归及 GPU BF16 故障注入都覆盖“首个 microbatch 已反传、后续 microbatch 失败”，确认参数、optimizer、LR、预算不变、梯度清空，下一次有效 attempt 能更新。

## 已运行的数值与恢复验收

环境：Python 3.12.3、torch 2.14.0+cu130、CUDA build 13.0、compute capability 12.0、4 CPU threads、TF32 关闭；显卡由桌面 WSL2 环境使用。依赖版本未修改。

- CUDA witness：32×32 BF16 矩阵 backward 有限，tiny SFT 实际更新 5 个 loss tokens。
- 六组 BF16/FP32 对照：SFT/GRPO/OPD × text/LoRA，固定 tokens、mask、old policy 与优势。
- 六组 CUDA FP32 累积对照：完整 batch vs microbatch=2 + checkpoint，保持同一采样及全局分母，包含末批预算截断。
- 六组 BF16 精确续训：每组第一阶段 257 tokens，随后切换 OPD 7 tokens；连续/恢复运行的参数、AdamW moments/step、LR、采样 RNG、rollout 和 loss 记录逐位相同。Teacher 保持冻结，LoRA 仅修改 adapter。
- 六组 BF16 失败恢复：三个 objective 分别注入后续 microbatch 的非有限 loss 或 gradient，全部跳过整次更新并成功恢复。
- 初次验收全量 603 项 CPU 测试通过；D01–D13 回顾修复后为 **620 项**，其中运行时回归 27 项。GPU 验收独立于默认 CPU pytest。

| 对照 | 实测最大误差 | 验收容限 |
|---|---:|---:|
| BF16 vs FP32 loss 绝对误差 | 2.0504e-4 | 3e-3 |
| BF16 vs FP32 gradient relative L2 | 1.0179e-2 | 3e-2 |
| FP32 累积 gradient relative L2 | 4.6448e-7 | 5e-5 |
| AdamW 实际更新 vs FP64 首步公式，绝对误差 | 5.9271e-8 | 2e-7 |

### 为什么不能用固定参数差判定梯度累积错误

首轮 GPU 验收在 LoRA/SFT 的参数差阈值 `2e-5` 处失败，实测差为 `9.5951e-5`，但梯度 relative L2 仅 `1.3971e-7`。改变 CUDA GEMM 的 batch 形状会改变 FP32 舍入；一个约 `1.7e-8` 的梯度分量相差约 `7.35e-10`，被 AdamW 的小 epsilon 放大。

零 weight decay、首步 moments 为零时，独立公式为：

```text
theta1 = theta0 - lr * g / (abs(g) + epsilon)
```

验收保留原梯度容限，改为分别用实际 clipped gradient 与该 FP64 公式核对两条路径的更新，绝对容限 `2e-7`；同时完整记录跨 batch 参数差，不放宽原参数阈值后宣称等价。公式预测参数差 `9.5952e-5`，与实际吻合。这是 FP32 梯度误差经自适应优化器放大的数值现象；不应据此声称不同 batch 形状的 AdamW 轨迹逐位一致。相同配置的精确续训仍单独通过。

### D01–D13 回顾补充：梯度裁剪不能替代归一化验收

回顾中确认上述更新公式只能验证 optimizer 是否正确使用了输入梯度。若累积分支梯度整体放大两倍，clipping 会消除大部分幅度差，旧六组验收仍全部通过。因此增加 loss 绝对误差 `<5e-6` 与**裁剪前** gradient norm 相对误差 `<5e-5` 的断言，保留裁剪后梯度及 AdamW 公式检查；CPU 负对照会拒绝两倍缩放错误。

修复后重新执行 CUDA witness、6 组精度、6 组累积、6 组续训与6组失败恢复，全部通过，见 [d13_review_validation.json](../../artifacts/gpu/d13_review_validation.json)。累积 loss 最大绝对误差 `2.6822e-7`，裁剪前 norm 最大相对误差 `1.3344e-7`。本次只复验正确性；下面的性能表仍引用原始计时。完整回顾与复跑方式见 [D01–D13 回顾](../experiments/D01_D13_REVIEW.md)。

## 单卡性能实测

Student 为 V4096/H128/4 layers，Teacher 为 H192/4 layers；prompt 48 tokens，SFT completion 16，rollout cap 8。SFT/OPD 为 8 prompts，GRPO 为 2 prompts × 8 completions；Teacher 在所有变体中均驻留。每个变体独立进程，3 个 paired round seeds，每轮 5 warm-up + 10 timed steps，共 360 个计时 steps。各变体记录实际生成长度及有效 tokens，不假设 FP32/BF16 采样完全一致。

| Objective | 配置 | p50 ms | p95 ms | 轮间 p50 范围 ms | 有效 loss tokens/s | peak allocated/reserved MiB |
|---|---|---:|---:|---:|---:|---:|
| SFT | `fp32_full` | 11.90 | 12.90 | 11.61–12.06 | 10787 | 121.1/128.0 |
| SFT | `bf16_full` | 13.01 | 15.32 | 12.27–13.12 | 9773 | 116.2/126.0 |
| SFT | `bf16_accum2` | 33.45 | 36.03 | 33.38–33.69 | 3779 | 104.3/114.0 |
| SFT | `bf16_accum2_checkpoint` | 56.24 | 64.91 | 51.96–59.58 | 2247 | 100.9/110.0 |
| GRPO | `fp32_full` | 31.57 | 35.71 | 30.96–32.71 | 3851 | 138.4/148.0 |
| GRPO | `bf16_full` | 37.50 | 49.58 | 36.45–38.61 | 3095 | 127.7/140.0 |
| GRPO | `bf16_accum2` | 89.74 | 97.25 | 88.09–93.15 | 1396 | 103.7/114.0 |
| GRPO | `bf16_accum2_checkpoint` | 127.34 | 132.60 | 126.92–128.02 | 992 | 102.9/114.0 |
| OPD | `fp32_full` | 34.73 | 38.58 | 34.50–37.67 | 1793 | 117.3/132.0 |
| OPD | `bf16_full` | 159.62 | 294.00 | 36.72–211.76 | 450 | 115.2/128.0 |
| OPD | `bf16_accum2` | 88.76 | 142.25 | 87.51–111.74 | 666 | 103.8/114.0 |
| OPD | `bf16_accum2_checkpoint` | 265.88 | 435.89 | 107.32–338.97 | 255 | 100.9/112.0 |

SFT/GRPO 的 FP32 full batch 在本次小负载下最快；BF16、累积和重计算没有吞吐收益。相对 FP32 full，累积+重计算将 SFT peak allocated 从 121.1 降至 100.9 MiB、GRPO 从 138.4 降至 102.9 MiB，但 p50 分别从 11.90 增至 56.24 ms、31.57 增至 127.34 ms。它们在这里体现显存与耗时取舍，不能外推为 Gemma 配置选择。

FP32 GRPO/OPD 的采样阶段均约 20–21 ms，是最大计时阶段；这里没有 KV cache，计时还包括 Python 校验与同步。该分段结果提示后续检查生成路径，但不能当作 kernel profiler 归因。OPD 的 BF16 首轮存在明显波动，尤其后两个 round；全部数据保留，不把单次聚合中位数作为稳定性能差距。

计时为 CUDA 同步后的 wall time，包含 Python、校验与同步开销，分段之和不含所有账本开销。显存为各子进程 PyTorch allocator 峰值，排除 driver/context，也不是整卡使用量。单卡通信记录 `not_applicable`；没有实现或声称验证 FSDP/ZeRO。D20 的真实模型 100-step 成本校准仍单独待完成。

### OPD 固定规模复测

保留首轮全部记录，再按相反变体顺序（checkpoint → accum2 → BF16 full → FP32 full）执行同配置、同 seeds、同样 3×(5 warm-up + 10 timed steps) 的四个独立进程。结果见 [d13_opd_repeat.json](../../artifacts/gpu/d13_opd_repeat.json)。

| 配置 | p50 ms | p95 ms | 轮间 p50 范围 ms | 有效 loss tokens/s | peak allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|
| `fp32_full` | 34.34 | 36.89 | 33.86–36.21 | 1841 | 117.3/132.0 |
| `bf16_full` | 41.20 | 53.43 | 39.68–44.10 | 1496 | 115.2/128.0 |
| `bf16_accum2` | 78.26 | 89.39 | 74.33–82.06 | 809 | 103.8/114.0 |
| `bf16_accum2_checkpoint` | 93.90 | 103.17 | 90.80–96.50 | 683 | 100.9/112.0 |

复测中 FP32 full 基线基本一致，BF16 full 和 checkpoint 的耗时波动明显收窄；不能将首轮 159.62/265.88 ms 当作稳定配置代价。复测仍没有 BF16 吞吐收益：full batch 为 34.34 vs 41.20 ms，累积+重计算以 93.90 ms 换取 117.3 → 100.9 MiB 的显存下降。原始数据不足以将首轮波动归因到某个 kernel、后台任务或 WSL 调度，保留为测量限制。这里只完成一次有明确原因的复测，没有反复挑选最快结果。

逐项复跑性能对照可用：

```bash
for variant in bf16_accum2_checkpoint bf16_accum2 bf16_full fp32_full; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/validate_accelerator.py --profile-child opd \
    --variant "$variant" --rounds 3 --warmup 5 --steps 10 \
    --output "outputs/d13/opd-repeat-${variant}.json"
done
```

## 复跑与镜像环境

`uv.lock` 提供依赖版本及 wheel hashes。`scripts/sync_environment_mirror.sh` 先 frozen export，再经镜像执行 `uv pip sync --require-hashes`，最后安装 editable 项目，不重新选择 torch/numpy 版本。环境规格见 `configs/environments/d13.json`。

```bash
bash scripts/sync_environment_mirror.sh
.venv/bin/python -m ruff check .
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_accelerator.py --dry-run
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python scripts/validate_accelerator.py --witness-only
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python scripts/validate_accelerator.py --output artifacts/gpu/d13_runtime.json
```

Checkpoint 写入被忽略的 `outputs/d13/checkpoints/`，公开 artifact 仅含合成配置和指标。D13 完成不代表真实 Gemma、真实数据、C1/C2 或正式五臂训练完成；下一模块是 D14。
