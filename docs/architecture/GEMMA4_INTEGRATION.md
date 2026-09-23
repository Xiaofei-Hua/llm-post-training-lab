# D14 — Gemma 4 Base 文本与 LoRA 接入

本轮技术目标：在真实 E2B/E4B Base 权重上，打通文本前向、分块 CE、反向与 LoRA 参数更新；验证 tokenizer 对齐、模型 logit 变换和冻结规则，并测量参数与运行时内存。2026-09-23 用户授权完成 D14 后提交推送。D14 使用两条手写算术字符串检查接线与梯度，不承担 D15 数据验收、SFT recipe 选择或 Teacher capability gate。

## 模型与输入合同

| 用途 | 官方 Base checkpoint | 固定 revision |
|---|---|---|
| Student 接入 | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| Teacher 架构接入 | `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` |

两者都是尚未进行本项目 SFT 的 Base 模型。加载完整的 `Gemma4ForConditionalGeneration`，保留 BF16 权重，不替换 IT checkpoint，也不量化。Transformers 固定为 5.17.0；所有 Python 依赖版本及哈希由 `uv.lock` 复现，安装沿用镜像。官方实现和模型配置见 [Transformers Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4)、[E2B](https://huggingface.co/google/gemma-4-E2B/tree/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)、[E4B](https://huggingface.co/google/gemma-4-E4B/tree/411aa17b749aa952df1359d2dcea73917a544d9a)。

两个 `tokenizer.json` 的 LFS SHA256 相同：`12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`。加载后还比较完整 token→ID 字典、分词与规范化后端，以及 BOS/EOS/PAD ID。仅相同 vocabulary size 不足以通过同 ID 的 exact KL 对齐。

E2B tokenizer 配置文件显式声明 `add_bos_token=true`，E4B 没有该字段；但在锁定的 Transformers 5.17.0 中，两者的真实默认编码都通过 tokenizer 后处理器插入 BOS，输出相同。不能仅凭配置字段或 Python 属性判断有效行为。D14 输入固定为：显式 BOS + 分别以 `add_special_tokens=False` 编码的 prompt/completion + 显式 EOS。completion loss 包含 EOS，prompt 和 padding 不计入。该手写 Base smoke 格式不是官方 chat template，也不是 D15 的正式数据 recipe。原始输入中的控制/多模态特殊 token 会被拒绝。

离线加载必须有 `tokenizer_config.json`：缺少它时，Transformers 会退回通用后端，BOS/EOS/PAD 可能变成 `None`。加载器显式检查该文件及有效 special IDs，防止不完整缓存静默改变训练输入。

## 文本路径与显存取舍

入口为 `models/gemma4.py:Gemma4TextAdapter`。模型仍注册完整的音频/视觉编码器与投影，但这些模块冻结、保持 eval，文本前向直接调用原生 language model。所有可训练参数来自 text decoder 中实际存在的 attention/MLP projection；共享 KV 的后部层没有独立 K/V projection，按真实模块枚举，不能套用每层七个 projection 的假设。

LoRA 固定 rank=32、alpha=64、dropout=0，只训练 A/B；embedding、LM head、PLE lookup、PLE context projection、norm 和多模态参数均冻结。BF16 基座配 FP32 LoRA 主参数与 AdamW 状态；LoRA 计算接受 autocast，残差返回基座输出 dtype。`Trainer` 的 BF16 检查相应要求**可训练**主参数为 FP32，允许冻结权重为 BF16。

E4B 的完整 BF16 文件约 16 GB，当前单卡不能把全部权重和训练临时空间都留在 GPU。适配器把冻结的 PLE lookup table 留在 CPU，先查出当前 token 的 per-layer embedding，再把小张量传到 CUDA。PLE 的 context-dependent projection、归一化、混合缩放和 decoder 仍由原生实现执行。没有删除 PLE 信号；本轮也没有使用量化或近似 lookup。

`place_for_training` 明确放置各模块，`input_device` 指向 text embedding；训练器和 rollout 不再假设模型的第一个参数就是输入设备。E2B/E4B 逐个加载验证，本轮不表示双模型联合 OPD 的显存方案已经闭合。

## 同一 logit 分布贯穿各目标

`CausalLMFeatures` 现在携带 `logit_scale` 与 `final_logit_softcapping`。dense forward、last-position sampling、GRPO selected log-prob、chunked SFT 和 OPD 均读取同一组参数。

Gemma Base 的变换为 `30 × tanh(raw_logits / 30)`。本框架在 BF16 GEMM 后将 logit 变换、log-softmax 与 loss 累加提升为 FP32；OPD 继续使用模型变换之后的 temperature，不添加 T² 系数。共用变换在 tanh 前检查有限值，避免 softcap 隐藏上游 overflow。

真实 E2B 暴露了原先 tiny 模型没有触发的精度问题：PyTorch 默认允许 cuBLAS 将 BF16 GEMM 的部分中间累加降低精度，dense 与分块 CE 的 LoRA 梯度相对误差达到 **2.344%**，超过预设的 **2%** 界限。关闭 `allow_bf16_reduced_precision_reduction` 后，误差降为 **1.108%**，无需放宽界限。同一分块下开启/关闭重计算的梯度完全一致；在真实 hidden states 上另做 FP32 head 独立公式检查，loss 差 `1.91e-6`、hidden-gradient 相对误差 `9.20e-6`。失败与修复对照保留在 [精度诊断](../../artifacts/gpu/d14_precision_diagnosis.json)。该选项含义见 [PyTorch CUDA 精度说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#reduced-precision-reduction-in-bf16-gemms)。

`TrainerConfig.bf16_reduced_precision_reduction` 默认设为 `False`，进入 BF16 forward 前显式应用，并保持到 backward/重计算结束。cuBLAS 选项是进程级状态，同一进程按顺序执行 Trainer，不能并发交错不同累加策略。配置进入 checkpoint 与本轮报告；性能比较需保持该策略一致。D13 的 6 组数值、6 组累积、6 组精确续训与 6 组失败恢复 CUDA 回归已复验，见 [运行时回归](../../artifacts/gpu/d14_runtime_regression.json)，不重写历史性能数据。

原生 Transformers 无 loss 的 BF16 forward 会在 BF16 中执行 softcap，因此与这里的 FP32 softcap 存在舍入差异。验收分别报告独立 FP32 softcap oracle 和原生 BF16 logits 差异；不把两种精度下的非零差值宣称为 bitwise parity。没有继承 checkpoint 中 top-k/top-p 的 generation 默认值，rollout 仍为温度 1 的完整策略。

## 验收与实测

CPU 测试使用 Transformers 的真实 Gemma 4 架构，缩小维度并保留 PLE、共享 KV、sliding/full attention 与 softcap。测试覆盖原生 wrapper 前向、SFT/KL/策略 log-prob 的 loss/gradient、重计算、LoRA 更新与冻结、tokenizer 后端差异和不完整离线缓存。这些是缩小架构测试，不是真实 Base 权重结果。

真实权重脚本为 `experiments/gemma4_integration.py`。每个模型固定两条算术字符串、seed 101、rank 32、alpha 64、初始学习率 1e-4、按已消耗 token 线性衰减至初始值的 0.1、chunk 2 tokens 和两次 Trainer 更新。该学习率仅用于有界接入 smoke，不据此选择正式训练参数。

脚本检查：

1. 原生文本与独立模型 softcap 对照；
2. dense full-vocab CE 与分块 CE 的 loss/LoRA 梯度，分别关闭和开启重计算；
3. 两次 token-budgeted AdamW 更新、所有目标 A/B 实际变化与训练 NLL；
4. 当前策略采样与重新计算 log-prob 的差异；
5. 全部冻结参数（含未执行的多模态编码器）更新前后逐字节摘要一致、无梯度。

运行时报告 CUDA parameter bytes、CPU parameter bytes、forward/backward 峰值、forward 相对参数基线的增量。该增量包含 activation 和临时 workspace；不是纯 activation 的精确分解。短 smoke 的单次延迟不支持吞吐加速结论，也不替代 D20 的稳定 profile。

**2026-09-23：两种真实 Base 均 PASS，D14/C0 完成，核心进度 14/24。** 硬件为 RTX 5070 Ti 16GB，Python 3.12.3 / PyTorch 2.14.0+cu130 / Transformers 5.17.0；两个模型顺序加载。模型权重与 tokenizer 最终通过官方 LFS SHA256 核对，网络中断后使用 ModelScope 镜像提供的同字节文件；模型卡声明 Apache-2.0，权重只保存在本地忽略的 cache。

| 模型 | 可训练 LoRA 参数 | 实际变化的 A/B 矩阵 | 两步训练 NLL（前 → 后） | dense/分块梯度相对 L2 | 采样/重算 log-prob 最大差 |
|---|---:|---:|---|---:|---:|
| E2B | 48,316,416 | 410 | 2.979266 → 0.382242 | 1.108% | 1.04e-07 |
| E4B | 69,763,072 | 516 | 3.547443 → 0.229452 | 1.011% | 5.96e-08 |

每个模型两次更新共消耗 **12 个 completion loss tokens**。dense/分块 loss 差为零，重计算开启/关闭均通过原 2% 梯度界限；独立 FP32 softcap oracle 差为零，原生 BF16 softcap 最大差约 0.1212。CPU PLE lookup 与同一 token 行在 CUDA 上的 lookup/缩放逐位一致。两模型全部冻结参数在更新前后摘要相同，且没有梯度。原生 checkpoint 加载均无 missing/unexpected/mismatched keys。

| 模型 | CPU 参数存储 / GiB | CUDA 参数存储 / GiB | forward activation+临时增量，重计算关 → 开 / MiB | 两次更新的最大 PyTorch allocated / GiB |
|---|---:|---:|---|---:|
| E2B | 5.261 | 4.426 | 238.5 → 103.9 | 5.230 |
| E4B | 6.141 | 8.911 | 351.6 → 148.5 | 10.014 |

参数存储包含 LoRA、按唯一 Parameter 计数，不含 buffers；运行时峰值包含梯度/优化器/临时张量，但不包含全部驱动、显示和库外分配。上述短序列不能证明 2k/4k context 或双模型同驻可行。首步 AdamW 初始化与主机调度使时间波动明显，保留全部原始计时，不据此宣称稳定吞吐或加速。

原始报告：[E2B](../../artifacts/gpu/d14_e2b.json)、[E4B](../../artifacts/gpu/d14_e4b.json)、[实际 tokenizer 行为](../../artifacts/gpu/d14_tokenizer_alignment.json)。两条训练样例的 NLL 下降只证明程序能更新并学习，不代表 benchmark 准确率、泛化或 Teacher 已合格。CPU 全量 **633 tests**、Ruff、依赖检查与 uv lock 检查通过。


## 复现

```bash
bash scripts/sync_environment_mirror.sh
.venv/bin/pytest -q tests/test_gemma4_adapter.py
# 先按上表 revision 将 config、tokenizer/config、generation_config 和
# model.safetensors 下载到 Hugging Face cache；权重不进入 Git。
.venv/bin/python -m posttrain_lab.experiments.gemma4_integration \
  --model google/gemma-4-E2B --output artifacts/gpu/d14_e2b.json
.venv/bin/python -m posttrain_lab.experiments.gemma4_integration \
  --model google/gemma-4-E4B --output artifacts/gpu/d14_e4b.json
```

C0 接入验收通过后仍需 D15 的真实数据/评测准备、D16/D17 的双方 SFT 和 Teacher 能力门槛，再推进正式 GRPO/OPD。这里的 NLL smoke 不能证明推理能力、准确率提升或 Teacher 已合格。
