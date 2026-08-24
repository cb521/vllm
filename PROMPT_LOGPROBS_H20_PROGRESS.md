# H20 Prompt Logprobs 优化：阶段进展与后续计划

更新时间：2026-08-24

## 一句话说明

目前已经完成一版可接入 vLLM 的 H20 Prompt Logprobs 优化：结果算得准，
在合成的 LM Head/Logprobs 测试中也明显省显存、少通信、速度更快。

当前还缺真实模型的完整 vLLM 端到端测试。因此现在可以确认“这个算子方案有效”，
但还不能把下面的加速数字当成整个模型的最终加速数字。

## 代码位置

- 仓库：<https://github.com/cb521/vllm>
- 分支：`optimize-prompt-logprobs`
- 当前提交：`75abd8cf33083aa72e64e7f4929abcc024d58a71`
- 详细性能报告：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.md`](benchmarks/kernels/benchmark_lm_head_logprobs.md)
- 基准测试程序：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.py`](benchmarks/kernels/benchmark_lm_head_logprobs.py)

## 已经完成的工作

### 1. 找到原始实现的问题

vLLM 原始路径会生成很大的词表 Logits。在 TP 多卡场景下，还需要处理或传输
大块词表数据。长 Prompt、词表较大时，这部分会占用大量显存和通信时间。

本次使用的测试形状为：

- Hidden Size：5120
- Vocabulary Size：248320
- 数据类型：BF16
- GPU：NVIDIA H20-3e
- TP：2、4、8
- Prompt：1K、4K、8K、16K、32K

### 2. 调研并复用了现有方案

上游已有一个相关实现：vLLM PR
[#53267](https://github.com/vllm-project/vllm/pull/53267)。它采用全融合方式，
显存非常低，但在 H20 的 TP2/TP4/TP8 测试中不一定是最快方案。

本分支保留该实现作为可选路径和对比对象，并针对 H20 增加了另一条实现路径。

### 3. 实现 H20 专用路径

新路径的做法可以简单理解为：

1. 每张卡只生成自己负责的那部分 BF16 Logits。
2. 在本卡上直接算出目标 Token、归一化所需统计量、排名和 Top-K 候选。
3. 多卡之间只交换这些很小的结果，不再交换完整词表 Logits。
4. 不再生成完整词表大小的 FP32 Logprobs 张量。

具体实现包括：

- 使用 vLLM 平台 GEMM 计算 TP 本地 BF16 Logits。
- 新增 Triton 归约，一次完成本地 LSE 和目标 Token 排名统计。
- 本地计算 Top-K，再通过紧凑通信合并各卡结果。
- 处理词表 Padding、某张卡没有有效词表行以及 K 大于本地有效词表等边界情况。
- 接入 vLLM Prompt Logprobs 流程，并保留不满足条件时的原始路径回退。

### 4. 增加 H20 自适应选择

不是所有输入都强行走新算子，目前的选择规则为：

- H20 + TP1：继续使用 vLLM 原始路径。
- H20 + TP2：当前 Chunk 至少 64 个 Token 时使用 H20 路径。
- H20 + TP4/TP8：当前 Chunk 至少 256 个 Token 时使用 H20 路径。
- 很小的尾部 Chunk：自动回退到原始路径，避免小任务反而变慢。
- 非 H20 GPU：保留已有的全融合紧凑路径。

功能开关仍为：

```bash
export VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS=1
```

### 5. 补充测试与基准程序

已经增加或扩展以下测试：

- BF16、FP16、FP32 数值测试。
- K=0、5、20、32。
- TP1、TP2 以及多卡基准。
- 空输入、词表 Padding、TP 边界和无有效词表行等边界情况。
- Prompt Logprobs 分块及尾部回退测试。
- H20 TP2、TP4、TP8 的性能、显存、通信和负载均衡测试。

当前测试结果：

- H20 Kernel 测试：45 个全部通过。
- H20 新路径专项测试：8 个全部通过。
- CPU/Sample/环境相关测试：24 个全部通过。
- Pre-commit 检查全部通过。

## 当前测试结果

### 数值正确性

以 PyTorch FP32 `logsumexp` 为参考，在 1024 Token、Top-20 下：

| TP | 最大绝对误差 | 平均绝对误差 | 目标 Token 排名 |
| ---: | ---: | ---: | :---: |
| 2 | 1.91e-6 | 2.68e-7 | 完全一致 |
| 8 | 1.91e-6 | 1.68e-7 | 完全一致 |

BF16 中可能出现完全相等的 Logits。此时不同 `topk` 实现可能用不同顺序返回
并列 Token，但它们的 Logprobs 相同，目标 Token 的结果和排名保持一致。

### 1024 Token 核心算子性能

| TP | vLLM 原始路径 | 上游全融合路径 | H20 新路径 | 相对原始路径加速 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 14.84 ms | 17.17 ms | 11.28 ms | 1.32x |
| 4 | 10.53 ms | 9.35 ms | 5.79 ms | 1.82x |
| 8 | 8.33 ms | 5.34 ms | 3.01 ms | 2.76x |

H20 路径比全融合路径占用更多显存，但它能使用 H20 上更快的 GEMM，所以速度更高；
与 vLLM 原始路径相比，它仍然大幅降低了显存。

### 32K Prompt 结果

以下数据只包含 LM Head 和 Prompt Logprobs 阶段：

| TP | 原始路径耗时 | H20 路径耗时 | 加速 | 峰值显存降低 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 475.36 ms | 360.26 ms | 1.32x | 85.2% |
| 4 | 335.83 ms | 184.27 ms | 1.82x | 91.8% |
| 8 | 266.20 ms | 136.05 ms | 1.96x | 95.5% |

### TP 通信

在 1024 Token、Top-20 下，紧凑通信每张卡只需要约 0.168 MiB 数据。
通信耗时相对原始路径降低约 13 至 16 倍。各卡增量峰值显存差异为 0 MiB，
核心 Chunk 的设备计算时间也更加均衡。

## 当前结论和限制

当前已经能够确认：

- 算子数值正确。
- H20 的 TP2、TP4、TP8 场景均能减少显存和多卡通信。
- LM Head/Prompt Logprobs 阶段有明确加速。
- 代码已经接入 vLLM 的 Prompt Logprobs 流程。

当前还不能确认：

- 完整模型 Prefill 的最终加速比例。
- 连续请求服务时的实际吞吐和 TTFT 改善。
- 其他 Hidden Size、词表大小以及模型架构上的最佳选择阈值。
- 其他 GPU 上是否也应该采用 H20 这套路径。

目前自动启用范围也有意保持保守：原始 Logprobs、未量化 LM Head、无 Bias、
无新增词表，并且 K 不大于 32。其他情况会回退，不会强行使用新路径。

## 下一步工作

### 第一步：真实模型端到端验证

需要先确定一个真实模型或本地权重路径。之前的 Demo 只给出了
Hidden Size=5120、Vocabulary Size=248320，没有给出模型名称和权重，
所以暂时无法完成这一项。

拿到模型后，在相同配置下分别测试：

- 基线：关闭紧凑 Prompt Logprobs。
- 优化：开启 `VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS=1`。
- TP：2、4、8。
- Prompt：1K、4K、8K、16K、32K。
- Batch Size：根据模型显存情况测试 1、2、4 或更高。
- Top-K：至少覆盖任务实际使用的 K，并保留 K=0 和 K=20 对照。

需要记录：

- 完整 Prefill 吞吐和端到端 Prompt 吞吐。
- TTFT。
- Prompt Logprobs 阶段耗时。
- 各卡峰值显存。
- NCCL 通信量和通信耗时。
- 各卡耗时差异。
- 基线和优化输出的最大、平均绝对误差以及排名一致性。

### 第二步：分析并调优真实模型结果

- 用 Nsight Systems 或 PyTorch Profiler 确认剩余瓶颈。
- 根据真实模型的 Hidden Size、词表和 Batch Size 调整切换阈值。
- 检查较小 Prompt 和尾部 Chunk，保证不会出现性能回退。
- 如果完整模型中的 Logprobs 占比很低，需要如实说明端到端收益上限。

### 第三步：按需求扩展支持范围

只有实际模型需要时再做以下扩展：

- K 大于 32。
- LM Head Bias。
- 量化 LM Head。
- 新增词表或 LoRA 词表。
- 更多模型尺寸和其他 GPU。

### 第四步：完成最终验收报告

最终报告需要同时给出：

- PyTorch 数值参考结果。
- vLLM 原始路径、上游全融合路径和 H20 路径三方对比。
- 1K、4K、8K、16K、32K 的完整数据。
- TP2、TP4、TP8 的显存、通信、负载和吞吐数据。
- 合成算子测试与真实模型端到端测试的区别。
- 适用条件、回退条件和已知限制。

## 下次继续前需要准备的信息

至少提供下面第一项即可继续：

1. 实际模型名称或本地权重路径。
2. 业务真正使用的 Prompt Logprobs 数量 K。
3. 期望测试的 Batch Size 和 TP 配置。
4. 如果有固定的 vLLM 启动参数或验收脚本，也一起提供。

恢复工作时使用：

```bash
cd /lustre/raplab/client/binc/workspace/vllm-logprobs
git checkout optimize-prompt-logprobs
git pull origin optimize-prompt-logprobs
```
