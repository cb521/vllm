# H20 Prompt Logprobs 对比报告

H20 上比较了四条路径：vLLM 原生、原 PR fused、新 fused 和 materialized。所有
性能数据都来自同一个模型、同一台 H20 节点和相同的 TP 配置。

## Kernel 实现

- fused 路径把 TP-local LM Head、local LSE、目标 Token Rank 统计和 local Top-K
  合到一个 SM90 WGMMA CuTeDSL Kernel。
- materialized 路径保留 BF16 GEMM，生成 TP-local Logits；后面的 local LSE 和
  Rank 统计合到一个 Triton Kernel，local Top-K 单独计算。
- TP 合并阶段把 global LSE、Rank 求和、Top-K 合并和 Logprob 归一化放到一个
  Triton Kernel。
- 目标 Token Logit 单独用 BF16 batched GEMM 计算，与 vLLM 原生 LM Head 的结果
  对齐。

## vLLM 接入

- `model_runner.py` 把 Hidden States 直接交给 Prompt Logprobs，不再先生成并
  AllGather 完整词表 Logits。
- `prompt_logprob.py` 增加按 Chunk 的 native、fused、materialized 路由；选择条件
  包括 GPU、TP、行数、K 和 256 MiB 临时显存上限。
- `logits_processor.py` 增加 TP-local 计算和 compact AllGather。卡间只传目标值、
  LSE、Rank 和 Top-K 候选，最终输出格式保持不变。
- 初始化时检查 LM Head 和数据类型，并预热会用到的 Kernel。

## 测试配置

- 模型：Qwen3.5-27B，BF16，`H=5120`，`V=248320`
- GPU：NVIDIA H20-3e
- TP：2、4、8
- Batch size：1
- Prompt：1K、4K、8K、16K、32K
- Prompt Logprobs K：0、20；主表使用 K=20
- 每个测试点：预热 1 次，正式测试 3 次，取中位数
- 基线：相同模型、Prompt、硬件和 TP 配置下的 vLLM 原生实现

## Benchmark

核心算子一共 15 个测试点：TP2/4/8 × Prompt 1K/4K/8K/16K/32K，K 固定为 20。
四条路径共用同一份输入，误差直接对 PyTorch。这里的“vLLM 原生”是完整词表
Logits 加 FP32 `log_softmax` 的原生计算方式。一个单元格里有三个数字时，顺序为
TP2 / TP4 / TP8。

| 评价维度 | 统计口径 | vLLM 原生 | 原 PR fused | 新 fused | materialized |
| --- | --- | --- | --- | --- | --- |
| Logprobs 数值正确性 | 最大绝对误差；各测试点平均绝对误差的最大值 | 最大 0<br>平均 0 | 最大 1.567e-2<br>平均 1.646e-3 | 最大 1.907e-6<br>平均 2.9e-7 | 最大 1.907e-6<br>平均 3.0e-7 |
| 峰值显存占用 | 全矩阵单卡临时峰值；相对原生降低范围 | 85367.9 MiB | 1069.2 MiB<br>降低 98.69%–99.68% | 1067.7 MiB<br>降低 98.75%–99.61% | 7929.7 MiB<br>降低 90.71%–97.64% |
| 跨卡通信开销 | 全矩阵最大每卡通信量；相对原生降低；通信耗时几何平均加速 | 7760 MiB<br>1.00x | 5.375 MiB<br>降低 99.723%–99.931%<br>201.12x | 5.375 MiB<br>降低 99.723%–99.931%<br>198.64x | 5.375 MiB<br>降低 99.723%–99.931%<br>198.64x |
| 设备负载均衡 | 全矩阵最大多卡计算耗时差；最大多卡峰值显存差 | 耗时差 0.034%<br>显存差 0 MiB | 耗时差 0.021%<br>显存差 0 MiB | 耗时差 0.035%<br>显存差 0 MiB | 耗时差 0.030%<br>显存差 0 MiB |
| Prompt 推理吞吐 | 32K Logprobs 耗时、吞吐（TP2 / TP4 / TP8）；全矩阵加速 | 651.96 / 513.46 / 444.10 ms<br>50,261 / 63,819 / 73,785 tokens/s<br>1.000x | 529.74 / 265.88 / 134.40 ms<br>61,857 / 123,244 / 243,809 tokens/s<br>1.911x（1.180x–3.310x） | 459.01 / 230.54 / 116.73 ms<br>71,388 / 142,138 / 280,714 tokens/s<br>2.246x（1.391x–3.821x） | 361.23 / 177.40 / 89.71 ms<br>90,713 / 184,715 / 365,253 tokens/s<br>2.920x（1.797x–5.001x） |

原 PR 虽然快，也省显存，但误差到了 `1e-2`，正确性不过关。两条新路径的最大误差
都是 `1.907e-6`。其中 fused 占用的临时显存更少，materialized 更快。

## 完整模型对比

整模型使用 Qwen3.5-27B、Batch 1、K=20，同样跑 15 个测试点。这里的第一列是
真正的 vLLM 原生路径。加速比按 `原生总耗时 / 对比路径总耗时` 计算。

| 核心指标 | vLLM 原生 | 原 PR fused | 新 fused | materialized |
| --- | ---: | ---: | ---: | ---: |
| 最大绝对误差 | 0 | 5.921e-2 | 3.815e-6 | 3.815e-6 |
| 各测试点平均绝对误差的最大值 | 0 | 7.265e-3 | 4.5e-7 | 4.0e-7 |
| 结果一致的测试点 | 15/15 | 0/15 | 15/15 | 15/15 |
| Prompt 吞吐几何平均加速 | 1.000x | 1.012x | 1.022x | 1.037x |
| TTFT 几何平均加速 | 1.000x | 1.009x | 1.025x | 1.043x |
| 32K 总耗时（ms，TP2 / TP4 / TP8） | 8798.5 / 5397.3 / 3705.1 | 8865.0 / 5339.0 / 3582.8 | 8771.4 / 5297.2 / 3534.1 | 8657.9 / 5217.4 / 3502.8 |
| 32K Prompt 吞吐（tokens/s，TP2 / TP4 / TP8） | 3724 / 6071 / 8844 | 3696 / 6138 / 9146 | 3736 / 6186 / 9272 | 3785 / 6281 / 9355 |
| 单卡峰值显存（全矩阵最大） | 64179.8 MiB | 63193.8 MiB | 63193.8 MiB | 62957.8 MiB |
| 相对原生显存降低范围 | — | 1.54%–3.95% | 1.54%–3.95% | 1.54%–3.90% |
| 多卡峰值显存差（全矩阵最大） | 0 MiB | 0 MiB | 0 MiB | 0 MiB |

## 测试范围

目前只测了 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8 和 H20。Batch 2/4、在线服务、
其他模型和其他 GPU 不在这次测试范围内。

原始数据在 `benchmark_artifacts/prompt-logprobs-adaptive-prompt-sweep-h20-20260825/`
和 `benchmark_artifacts/prompt-logprobs-fourway-h20-k20-20260826/`。
