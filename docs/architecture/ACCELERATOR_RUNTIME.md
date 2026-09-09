# D13：单 GPU 运行时与可恢复训练

2026-09-09：实现与 CPU 验证已完成，**GPU 实测尚未执行，D13 保持 IN_PROGRESS**。用户提供单卡环境后授权 tiny 模型验证，随后通知 GPU 正被另一任务占用，要求先完成其余内容并提交。因此当前暂停所有 CUDA witness、训练和性能测量；真实模型/数据下载与 D14 尚未授权。

## 实现与正确性

`Trainer` / `TrainerConfig` 在现有 D10 路径上支持显式 `cpu` 或单个 `cuda:N`，默认 CPU；没有保留第二套 CPU trainer。模型与 LoRA 在 CPU 上按 seed 初始化、选定可训练参数，然后迁移设备并创建 optimizer；采样 generator、tokens 和 mask 跟随模型设备。CPU 默认路径不初始化 CUDA。

- BF16 使用 autocast，参数与 AdamW moments 保持 FP32，概率归一化与 CE/KL 使用 FP32；CPU 数值对照仍可用 FP64。BF16 不使用 loss scaling；CUDA 原生 BF16 支持与真实数值误差待 GPU 验收。
- 在完整逻辑 batch 上计算 reward、GRPO advantage 与 D01 预算，然后按 completion 行拆 microbatch。SFT/OPD 的每个局部 loss 除以完整 batch 选中的 token 数；Dr.GRPO 保持截断前 active completions × 固定 cap 分母。局部 backward 直接求和，最后统一 clip、AdamW update、预算提交和 scheduler step，不再除以 microbatch 数。
- Teacher 也按 microbatch forward，保持 frozen/eval/no_grad。生成仍为完整当前策略、每批一次 update、无 KV cache；GRPO 分组在拆分前完成。
- activation checkpointing 用非 reentrant 的逐 decoder block 重计算，支持输入 embedding 冻结的 LoRA；tiny decoder 没有 dropout，故无需保存 block RNG。D02/D04 的 LM-head 分块重计算继续复用。
- 任一 microbatch 的 loss/累积 gradient 非有限时，整次更新丢弃并清空梯度；不扣 token、不更新 moments/LR。新阶段保留参数、重置 optimizer/scheduler 与采样 RNG。

`save_checkpoint(path)` 在完整 attempt 结束后原子替换文件，保存 Student/Teacher、可训练参数与 LoRA scaling 配置、AdamW、scheduler、预算、stage/attempt/policy version、采样及 CPU/CUDA RNG、已有记录。`load_checkpoint(path)` 用 `weights_only=True` 加载到新建的同配置 Trainer，拒绝不匹配的运行时或模型拓扑。训练样本顺序和 reward 实现由调用方保持一致；这是同环境续训，不承诺跨 PyTorch 版本或设备的随机序列一致。

## 已运行的 CPU 验证

```bash
uv run --frozen pytest -q tests/test_training_runtime.py
uv run --frozen python scripts/validate_accelerator.py --dry-run
```

23 项 CPU regression 覆盖 SFT/GRPO/OPD × text/LoRA、不同 completion 长度、部分末批、完全未选中的 microbatch、重计算开关、完整 batch 与累积梯度/更新等价；同时覆盖随机 rollout 的精确续训、moments/LR/预算恢复、阶段切换、后续 microbatch NaN、零方差 skip 后恢复、完成阶段 checkpoint 与 LoRA scaling 不匹配。以上只证明 CPU 路径；GPU 验证脚本当前为待执行交付物。

## 镜像环境安装

依赖版本与哈希仍来自 `uv.lock`。锁文件中的 wheel URL 指向官方文件站，直接给 frozen sync 换 index 不保证 wheel 下载经过镜像。因此 `scripts/sync_environment_mirror.sh` 先 frozen export 全部依赖及 hashes，再通过指定镜像执行 `uv pip sync --require-hashes`，最后安装本地 editable 项目。不会重新选择 torch/numpy 版本；项目构建依赖仍按 `pyproject.toml` 的 build-system 约束安装。

```bash
# 在项目目录执行；远端解释器路径由本机环境决定。
UV_BIN=uv POSTTRAIN_BOOTSTRAP_PYTHON=python3.12 \
UV_PROJECT_ENVIRONMENT=.venv \
POSTTRAIN_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/sync_environment_mirror.sh
.venv/bin/python scripts/validate_accelerator.py --dry-run
```

环境规格见 `configs/environments/d13.json`；具体 SSH、环境路径和下载诊断只保存在忽略的 `notes/private/compute/`。镜像安装/CPU import 成功与 GPU kernel 可运行是两件事，后者暂未验证。

## GPU 恢复后待执行的验收

以下命令仅在 GPU 可用后运行，本次没有执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python scripts/validate_accelerator.py --witness-only
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python scripts/validate_accelerator.py --output artifacts/gpu/d13_runtime.json
```

计划检查：seeded BF16 矩阵 backward 与 tiny SFT 更新；固定相同 tokens/masks/old-policy 下 BF16 对 FP32 loss/gradient；CUDA FP32 完整 batch 对累积+重计算；六组 BF16 text/LoRA 训练保存恢复，要求参数、optimizer、采样和阶段切换一致。预设误差容限与配置在脚本内，运行失败必须诊断，不能只为通过而放宽。

性能脚本为 SFT/GRPO/OPD 各运行 FP32 full、BF16 full、BF16 microbatch=2、BF16 microbatch=2+checkpoint 四种变体；每种在新进程执行 3 轮，每轮 5 warm-up + 10 timed steps。Student 为 V4096/H128/4 layers，Teacher 为 H192/4 layers；prompt 48 tokens，SFT completion 16，rollout cap 8；SFT/OPD 8 prompts，GRPO 2 prompts × 8 completions。Teacher 在各变体中均驻留。实际生成长度和有效 token 数随策略记录，不假设 FP32/BF16 采样完全相同。

记录包含完成 CUDA 同步的 wall time、分段 sample/score/loss/backward/update、p50/p95、轮间中位数范围、有效 loss tokens/s、PyTorch peak allocated/reserved。该计时含 Python、校验和同步开销；分段之和不含所有账本开销，不能替代 profiler kernel time。显存不含 driver/context，也不是整卡使用量。单卡无通信占比，记录 not_applicable；当前没有实现或声称验证 FSDP/ZeRO。D20 的真实模型 100-step 成本校准仍单独待完成。

D13 退出需真实 GPU witness、数值/续训和性能结果齐备；仅完成代码和 CPU 测试不增加核心完成数。
