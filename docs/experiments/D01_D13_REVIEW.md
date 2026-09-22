# D01–D13 回顾与修复（2026-09-22）

本轮目标：检查前十三个模块的算法、训练/评测接线、性能证据和研发方向；发现问题后修复，直到连续两轮无新增问题。范围是现有 CPU/单卡 tiny 实现及合成实验，未进入 D14、下载真实模型/数据或改变预注册 C1/C2。

共完成五轮审阅，最后两轮无新增 bug 或方向偏移，连续通过计数为 2。确认并修复三类代码/验收缺陷及过时文档。这个结果限定于本次已审阅的实现和输入范围，不表示证明所有可能输入均无缺陷。

## 发现并修复的问题

| 问题 | 修复前复现 | 修复与验收 |
|---|---|---|
| D02–D04 空分片的可微零值不可靠 | CE 读取未选中的 NaN/Inf 再乘零，loss 为 NaN；三个 objective 在 FP64 未选中极值转 FP32 后也会得到 NaN | 对空 tensor view 求和，保持梯度连接且不读取 sentinel；CPU 回归及 9 个 CUDA 边界案例得到有限零 loss、全零梯度 |
| D04 softcap 掩盖 logits 溢出 | 选中 Student/Teacher 的 ±Inf 或 scale 后溢出，经 tanh 变成有限 logits；Student Inf 的例子返回约 0.3278 的有限 KL 并反传 | 在 softcap 前检查有限性；6 个 Student/Teacher 回归案例修复前不抛异常，修复后正确拒绝；保留未选中 sentinel 的忽略语义 |
| D13 累积验收漏掉梯度整体缩放 | 人为把累积分支真实梯度乘 2，同时令两支触发 clipping，旧六组验收全部通过；裁剪后最大相对 L2 仅 1.3157e-5 | 同时断言 loss、裁剪前 norm、裁剪后梯度以及各自 AdamW 首步公式；负对照现在失败于裁剪前 norm 检查，正常 CPU/GPU 六组对照通过 |

对应实现为 `train/masked_ce.py`、`train/grpo_surrogate.py`、`train/opd_reverse_kl.py` 和 `experiments/accelerator_runtime.py`。保留已有 Trainer 的整次更新清理路径；未改变 CE、Dr.GRPO 或 reverse-KL 的有限输入目标函数。

新 GPU 累积断言的 loss 绝对容限为 `5e-6`、裁剪前 norm 相对容限为 `5e-5`；后者与原梯度容限相同。修复后实测最大值分别为 `2.6822e-7`、`1.3344e-7`。原 AdamW 公式容限 `2e-7` 不变。完整复验见 [d13_review_validation.json](../../artifacts/gpu/d13_review_validation.json)。

## 偏移与表述修正

- 当前主线仍符合“算法→可运行训练循环→测量→解释”。但部分旧文档仍要求未来建设“D11 provenance ledger”，与已接受的算法/框架/性能优先决策冲突。已删除这些过时模块依赖，正式实验继续保存必要的 checkpoint/arm/seed 记录，复用现有基础设施。
- 最终方案、实验计划、pipeline summary、兼容性说明曾保留 D01–D08 完成、D09–D12 待做及所有 CUDA 待授权等状态。已统一为 D01–D13 完成、下一模块 D14，tiny GPU 与真实模型授权分别表述。第三轮发现实验计划尾表还有遗漏，补齐后重新计数；旧日期快照保留历史语义。
- D04 文档原称所有低精度输入都做 FP32 projection；实际启用 autocast 时 GEMM 遵循 autocast，而 log-softmax/KL 用 FP32。已按已测实现更正文档。历史测试数量标记为对应验收时记录，当前全量是 620 项。
- D11 的核函数时间、完整 step 时间、RSS，以及 D13 的同步 wall time、PyTorch allocator 峰值各有自己的测量范围。原始 OPD 波动与复测均保留，不能宣称稳定端到端加速。本次未重跑或替换性能计时。

这里的 13/24 是模块交付数量，不能解释为 Gemma 研究效果已经完成一半。D12 的 copy-bit 学习与 D13 的 tiny GPU 结果都不用于真实 recipe 选择或 C1/C2 推断。

## 逐轮记录

| 轮次 | 审阅范围与新增验证 | 结果 | 连续无新增计数 |
|---|---|---|---:|
| 1 | D01–D13 目标函数、分母、mask、策略/Teacher、评测统计、性能及方案；先复现再修复，执行 CPU 与 CUDA 复验 | 3 类代码/验收问题及文档偏移，已修复 | 0 |
| 2 | 独立公式、分段导数、穷举统计、变长采样；重新生成 Hypothesis 输入；复跑 D11 数值 oracle 与 D12 五臂学习 | 无新增 bug | 1 |
| 3 | 故障后 checkpoint 续训、CUDA 空分片、原始性能聚合与文档依据 | 无新增代码 bug；发现实验计划尾部状态表遗漏，修复后重置 | 0 |
| 4 | 从接口重查 D01–D13 组合边界、状态/梯度隔离、研究口径与更新后的进度；30 个随机 family 图对照独立传递闭包 | 无新增问题 | 1 |
| 5 | 从公式/接口重核 D01–D13，审阅最终代码差异、修复测试、完整证据与文档；确认模块状态、tiny/正式研究与性能边界一致 | 无新增问题 | 2 |

每轮重新检查全部模块，表中的新增验证是该轮重点；未对已通过且未修改的路径反复跑性能或重建审计平台。

## 覆盖和实际结果

| 模块 | 重点检查与证据 |
|---|---|
| D01 | 首个 EOS（含 EOS）、prompt/padding 排除、稳定预算截断、失败更新不计数；额外 40 组独立 mask/排序参考 |
| D02 | causal shift、全局 token mean、hidden/head/bias 梯度、分块与空分片 |
| D03 | sample std、零方差组、预算截断前的 active-completion 分母、ratio/双向 clip；24 组逐 token 手算 loss/导数 |
| D04 | KL(Student\|\|Teacher) 方向、Teacher stop-gradient、scale→softcap→temperature、完整词表梯度；D02/D04 合计 48 组完整 logits 参考对照 |
| D05 | extraction/parse/compare 失败语义、错误 gold 与模型错误区分；既有 257 条冻结 adversarial corpus，另加 48 个分数正负例 |
| D06 | train/eval、family 隔离与传递 quarantine、normalization/检索路径；30 个随机 family 图与独立 all-pairs 闭包一致 |
| D07 | sealed/public 边界、generation 对齐、invalid gold/失败生成、accuracy/pass@k；240 个 pass@k 组合穷举对照 |
| D08 | 固定三个训练 seeds 的 estimand、整题向量重采样、分层 bootstrap、Holm、C2 区间判定；48 个 sign-flip 穷举对照 |
| D09 | tied/untied head、LoRA A/B 与 frozen base、因果/padding、实际更新；复用完整梯度与 SGD 参考测试 |
| D10 | 当前策略 rollout→score→loss→backward→update、阶段重置、精确预算；42 个变长 prefix log probability 独立核对 |
| D11 | 两档 CE/KL 的 dense/selected/recompute 四组数值 oracle 全通过；核函数与端到端、RSS 与张量内存口径分开 |
| D12 | 复跑 5 臂×3 seeds 的 30 个阶段，各完成 128 loss tokens；所有 run 的正确 completion 概率上升，错误 Teacher 对照下降 |
| D13 | CUDA witness；精度/累积/续训/非有限恢复各 6 组通过；另测 9 个 CUDA 空分片，三个 objective 的 CPU 故障后 checkpoint 精确续训；16 组原始 GPU profile 聚合可重算 |

全量测试：普通运行 **620 passed（20.77 s）**，独立 Hypothesis seed `20260922` 运行 **620 passed（18.23 s）**；Ruff、`uv pip check`、`git diff --check` 通过。CPU 正确性复跑中的 wall time 不作为新性能对照。

额外探针及学习复跑记录放在忽略的 `outputs/review/`，没有新增通用审计模块。当前 GPU 正确性结果另存为公开 artifact，原 D11/D12/D13 历史测量保留。

## 仍属后续范围

真实 tokenizer/template、Gemma text/PLE/freeze 接入、模型/真实数据、Teacher capability gate 和真实五臂仍待 D14 以后。当前 adapter 明确只表示未变换的线性 head；SFT entropy 和 GRPO selected log-probability 仍会一次投影全部选中位置，D02/D04 的 chunk 上限不能当作整个真实 Trainer 的显存上限。这是后续接入与性能工作，不用 tiny 结果提前宣称已解决。

本次只读核对了 [Google 官方模型概览](https://ai.google.dev/gemma/docs/core) 和 [E4B Base 模型卡](https://huggingface.co/google/gemma-4-E4B)，E2B/E4B 的模型身份与方案一致。官方加载内存估计仍约 11.4/17.9 GB，包含特定加载开销且不等于训练峰值；当前 16GB tiny 验证不能推出双模型正式训练可行。未下载权重。

## 复跑入口

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q --hypothesis-seed=20260922
.venv/bin/python -m ruff check .
uv pip check
git diff --check
```

只复验 GPU 正确性、保留已有性能记录：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python - <<'PY'
import json, platform
from pathlib import Path
import torch
from posttrain_lab.experiments import accelerator_runtime as runtime
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out = Path('outputs/review')
(out / 'd13-checkpoints').mkdir(parents=True, exist_ok=True)
report = {'scope': 'tiny CUDA correctness only; no performance rerun',
          'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                          'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                          'threads': 4, 'tf32': False}}
for name, check in [('witness', runtime.witness),
                    ('numerical_precision', runtime.numerical_precision),
                    ('accumulation', runtime.accumulation_checks),
                    ('resume', lambda: runtime.resume_checks(out / 'd13-checkpoints')),
                    ('nonfinite', runtime.nonfinite_checks)]:
    report[name] = check()
    print(name, 'passed', flush=True)
report['passed'] = True
(out / 'd13-review-validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
PY
```
