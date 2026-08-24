# H20 Prompt Logprobs 优化：三方对比与真实模型结果

更新时间：2026-08-25

## 先说结论

这项工作的基线有两个，作用不同：

- **验收基线**：vLLM 原生 Prompt Logprobs，用来判断数值、显存、通信和
  端到端性能是否真的改善；
- **竞争基线**：vLLM PR
  [#53267](https://github.com/vllm-project/vllm/pull/53267) 的全融合 CuTe
  实现，用来判断我们的 H20 专用方案有没有超过已有优化。

在 Qwen3.5-27B、Batch 1、BF16、K=0/20、Prompt 1K 至 32K、
TP2/TP4/TP8 的 30 个真实模型 case 上，结果是：

- 对 PR #53267：我们的引擎 TTFT **30/30 全部更快**；完整
  `LLM.generate` wall time 首轮 **29/30 更快**，唯一慢 `0.2%` 的
  TP8/32K/K=20 case 用 7 次重复复测后变为 **快 1.8%**，TTFT 快
  **3.5%**；
- 对 vLLM 原生：我们的 wall time 提升 `1.009x–1.062x`，TTFT 提升
  `1.009x–1.082x`，每卡整进程峰值显存降低 `972–1750 MiB`；
- 我们和原生的输出 **30/30 语义等价**，目标 Logprob 最大绝对误差
  `3.815e-6`；
- PR #53267 采用 FP32 累加语义，和原生 BF16 投影结果不同。在本项目要求的
  原生/PyTorch BF16 参考口径、`2e-5` 误差门槛下，PR 的 30 个 case
  均不等价，最大目标 Logprob 差值 `0.05921`。这不等于 PR 的数学计算错误，
  但它不是原生数值语义；
- PR 的全融合实现仍然是**算子临时显存最省**的方案。真实模型整进程峰值中，
  PR 和我们的差距不超过 `252 MiB/GPU`，相对 43–63 GiB 的总占用基本持平；
- 两个 compact 方案都不再跨卡聚合完整词表。我们的 Nsight 结果显示，
  相对原生路径 AllGather 累计时间降低 `96.8%`。

所以，就当前已经明确的 H20、Qwen3.5-27B、Batch 1、K≤20、TP2/4/8
验收范围而言：**文档里的正确性、显存、通信、负载均衡和吞吐目标都已达到，
而且端到端性能超过 PR #53267。** 尚未覆盖的是 Batch 2/4、在线并发、其他模型
形状和其他 GPU。

## 三个实现到底是什么

| 名称 | 对应实现 | 做法 |
| --- | --- | --- |
| `native` | vLLM 原生路径 | 物化并跨 TP 聚合完整词表 Logits，再做 LogSoftmax、Rank 和 Top-K |
| `fused` | PR #53267，提交 `fa087e6d8` | CuTe 内核把 LM Head、LSE、Rank 和 Top-K 融合，只通信紧凑结果 |
| `h20` | 本工作的 H20 专用实现 | 用 H20 上更快的平台 GEMM 物化 TP-local BF16 Logits，再做紧凑归约和通信 |

我们的 H20 路径建立在 PR #53267 的 compact 框架上，但做了不同取舍：保留本卡
BF16 Logits，换取更快的 H20 GEMM；同时仍不生成全词表 FP32 Logprobs，也不跨卡
AllGather 全词表 Logits。

生产环境的自动选择规则为：

- TP1：保留原生路径；
- TP2：当前 Chunk 至少 64 Token 时使用 H20 路径；
- TP4/TP8：当前 Chunk 至少 256 Token 时使用 H20 路径；
- 更小的尾部 Chunk 自动回退，避免小任务性能倒退；
- 非 H20 GPU 保留 PR #53267 的 fused 路径。

自动启用范围保持保守：Model Runner V2、raw logprobs、未量化且无 Bias 的标准
LM Head、无新增词表、K 不大于 32；不满足条件时回退原生路径。

## 代码和原始数据

- 仓库：<https://github.com/cb521/vllm>
- 分支：`optimize-prompt-logprobs`
- PR #53267 实现：`fa087e6d8`
- [H20 专用实现](https://github.com/cb521/vllm/commit/75abd8cf33083aa72e64e7f4929abcc024d58a71)
- 首轮真实模型验证与编译缓存修复：`39e7e4f73`
- 三方基准支持：`6ca7c88f5`
- E2E 基准：
  [`benchmarks/benchmark_prompt_logprobs_e2e.py`](benchmarks/benchmark_prompt_logprobs_e2e.py)
- 核心算子基准：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.py`](benchmarks/kernels/benchmark_lm_head_logprobs.py)
- 核心算子详细报告：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.md`](benchmarks/kernels/benchmark_lm_head_logprobs.md)

正式结果保存在工作区外的 `benchmark_artifacts/`，没有把大体积 NPZ 和 Nsight
文件提交到 Git：

| 配置 | 结果目录 | `comparison.json` SHA256 |
| --- | --- | --- |
| TP2 | `prompt-logprobs-qwen35-three-way-tp2-b1-20260825` | `d778b31af701220d97d70b8ef0d3d8ec50b5112bcd1749ccaec1870b5e7f0145` |
| TP4 | `prompt-logprobs-qwen35-three-way-tp4-b1-20260825` | `1fe79e26bce489aa3eebd6c4860ba61995ea7b73efae39e1c9e50c571cbf6d0e` |
| TP8 | `prompt-logprobs-qwen35-three-way-tp8-b1-20260825` | `73686d927ad15a01ff07bc404199184325859af571a6f76ff1a6ac4c5c170ed8` |

TP8/32K/K=20 的 7 次复测目录是
`prompt-logprobs-qwen35-three-way-tp8-32k-k20-r7-20260825`，其中
`comparison-fused-vs-h20.json` 的 SHA256 为
`efb095c2c5ea416572c9e075c4f0c9a143f3248f07a9e3c915fadf7972b1fe44`。

Nsight Systems 原始报告在
`benchmark_artifacts/prompt-logprobs-qwen35-nsys-tp8-20260824/`：

- `baseline.nsys-rep`：`ce5b713e895565f16937f5c4e501f02fe3d1e401aa34e68dc34eee429215d053`
- `optimized.nsys-rep`：`7d72d8399c37fad1bdd3be7e683c81de50924dd5f9dac2005ef7dff76f47c71a`

## 正式测试环境和方法

### 模型与软件

- 模型：`Qwen/Qwen3.5-27B`
- 固定权重 revision：
  `fc05daec18b0a78c049392ed2e771dde82bdf654`
- 架构：`Qwen3_5ForConditionalGeneration`
- Hidden Size：5120
- Vocabulary Size：248320
- 层数：64
- 权重类型：BF16
- GPU：NVIDIA H20-3e，143771 MiB/GPU
- PyTorch：`2.13.0+cu130`
- CUDA Runtime：13.0
- NCCL：2.29.7
- Triton：3.7
- 三方结果记录的 Git revision：`6ca7c88f5450c649cf7638d1603b1d842850c39f`

权重使用固定 revision 的节点本地副本：
`/tmp/binc-qwen3.5-27b-fc05daec`，避免每次启动都从 Lustre 读取约 52 GiB。

### 测试矩阵

- TP：2、4、8
- Batch Size：1
- Prompt：1024、4096、8192、16384、32768 Token
- Prompt Logprobs K：0、20
- 输出：每个请求生成 1 Token，greedy，固定 seed 17
- `max_num_batched_tokens=8192`
- 每卡固定 `kv_cache_memory_bytes=32 GiB`
- Prefix Cache 关闭
- 每个 case 1 次 warmup、3 次正式重复，表中取中位数
- 三个实现使用完全相同的随机 Token ID Prompt

三个实现分别在独立进程中加载模型。基准强制共用相同的 backbone AOT 编译产物，
只切换 Prompt Logprobs 后端，并同时保存：

- `LLM.generate` wall time 和 Prompt tokens/s；
- vLLM 请求指标里的 first-token latency，本文简称 TTFT；
- 10 ms 周期的 NVML 各卡显存；
- 目标 Token 分数和排名、Top-K、生成 Token 的压缩 NPZ；
- 三个实现之间的逐位置数值与语义对拍结果。

Wall time 还包括把大量 Prompt Logprobs 整理成 Python 对象并返回给调用者的时间；
TTFT 更接近模型执行路径。K=20、长 Prompt 时，这段前端整理可占 0.4–1.37 秒，
因此两个时间都报告。

## 我们和 PR #53267 的直接对比

下表中的加速倍率是 `PR 时间 / H20 时间`，大于 1 表示我们的 H20 路径更快。

| TP | Wall 加速范围 | Wall 中位数 | Wall 胜出 | TTFT 加速范围 | TTFT 中位数 | TTFT 胜出 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.020x–1.027x | 1.023x | 10/10 | 1.020x–1.029x | 1.024x | 10/10 |
| 4 | 1.019x–1.030x | 1.024x | 10/10 | 1.019x–1.032x | 1.025x | 10/10 |
| 8 | 0.998x–1.059x | 1.026x | 9/10 | 1.025x–1.076x | 1.033x | 10/10 |

### TP2 明细

| Prompt | K | PR→H20 Wall (ms) | 加速 | PR→H20 TTFT (ms) | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 233.02→228.40 | 1.020x | 231.38→226.83 | 1.020x |
| 1024 | 20 | 245.03→238.96 | 1.025x | 232.95→227.16 | 1.025x |
| 4096 | 0 | 876.87→857.51 | 1.023x | 870.62→851.40 | 1.023x |
| 4096 | 20 | 930.40→905.87 | 1.027x | 883.22→859.13 | 1.028x |
| 8192 | 0 | 1729.50→1691.15 | 1.023x | 1717.70→1679.19 | 1.023x |
| 8192 | 20 | 2119.62→2072.95 | 1.023x | 1742.52→1693.54 | 1.029x |
| 16384 | 0 | 3543.02→3468.96 | 1.021x | 3519.75→3446.00 | 1.021x |
| 16384 | 20 | 4053.28→3951.07 | 1.026x | 3567.83→3473.47 | 1.027x |
| 32768 | 0 | 7454.74→7302.98 | 1.021x | 7408.49→7256.55 | 1.021x |
| 32768 | 20 | 8840.15→8660.05 | 1.021x | 7502.48→7311.15 | 1.026x |

### TP4 明细

| Prompt | K | PR→H20 Wall (ms) | 加速 | PR→H20 TTFT (ms) | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 134.72→132.22 | 1.019x | 133.08→130.60 | 1.019x |
| 1024 | 20 | 147.27→143.83 | 1.024x | 135.24→131.84 | 1.026x |
| 4096 | 0 | 475.38→465.02 | 1.022x | 469.07→458.59 | 1.023x |
| 4096 | 20 | 525.97→511.58 | 1.028x | 477.92→464.51 | 1.029x |
| 8192 | 0 | 928.49→906.90 | 1.024x | 916.23→894.90 | 1.024x |
| 8192 | 20 | 1324.97→1287.13 | 1.029x | 933.49→904.50 | 1.032x |
| 16384 | 0 | 1898.79→1855.72 | 1.023x | 1874.79→1832.02 | 1.023x |
| 16384 | 20 | 2405.50→2336.51 | 1.030x | 1907.03→1849.04 | 1.031x |
| 32768 | 0 | 3978.41→3889.17 | 1.023x | 3931.24→3843.76 | 1.023x |
| 32768 | 20 | 5371.08→5224.45 | 1.028x | 3992.08→3875.64 | 1.030x |

### TP8 明细

| Prompt | K | PR→H20 Wall (ms) | 加速 | PR→H20 TTFT (ms) | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 103.70→100.78 | 1.029x | 102.10→99.17 | 1.029x |
| 1024 | 20 | 118.65→112.04 | 1.059x | 106.78→99.24 | 1.076x |
| 4096 | 0 | 272.63→265.28 | 1.028x | 266.36→258.75 | 1.029x |
| 4096 | 20 | 322.70→311.05 | 1.037x | 273.07→262.67 | 1.040x |
| 8192 | 0 | 519.20→505.52 | 1.027x | 507.25→492.86 | 1.029x |
| 8192 | 20 | 901.36→897.84 | 1.004x | 520.56→500.31 | 1.040x |
| 16384 | 0 | 1058.78→1033.24 | 1.025x | 1035.28→1008.66 | 1.026x |
| 16384 | 20 | 1544.83→1520.89 | 1.016x | 1060.95→1021.20 | 1.039x |
| 32768 | 0 | 2224.58→2172.65 | 1.024x | 2178.92→2125.08 | 1.025x |
| 32768 | 20 | 3571.23→3579.96 | 0.998x | 2223.53→2146.02 | 1.036x |

最后一行是首轮唯一一个 wall time 没赢的 case，但它的 TTFT 已经快 `3.6%`，
差异来自约 1.35 秒的前端 Python 输出整理。为避免拿 `0.2%` 的抖动下结论，单独做了
2 次 warmup、7 次正式重复：

| 配置 | PR | H20 | H20 加速 |
| --- | ---: | ---: | ---: |
| Wall time | 3580.03 ms | 3515.65 ms | 1.018x |
| TTFT | 2222.49 ms | 2147.77 ms | 1.035x |
| 整进程峰值显存 | 43365.9 MiB/GPU | 43365.9 MiB/GPU | 相同 |

因此，结合复测，当前所有已测配置中我们的 H20 路径都超过了 PR #53267。

## 对 vLLM 原生基线的验收结果

| TP | Wall 加速范围 | Wall 中位数 | TTFT 加速范围 | TTFT 中位数 | 显存降低 (MiB/GPU) | 语义等价 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1.009x–1.020x | 1.011x | 1.009x–1.017x | 1.013x | 972–1496 | 10/10 |
| 4 | 1.022x–1.039x | 1.028x | 1.022x–1.044x | 1.033x | 972–1646 | 10/10 |
| 8 | 1.023x–1.062x | 1.042x | 1.023x–1.082x | 1.051x | 972–1750 | 10/10 |

长上下文 32K、K=20 的结果为：

| TP | Native→H20 Wall (ms) | 加速 | Native→H20 TTFT (ms) | 加速 | 显存降低 (MiB/GPU) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8766.97→8660.05 | 1.012x | 7418.23→7311.15 | 1.015x | 1222 |
| 4 | 5370.72→5224.45 | 1.028x | 4024.23→3875.64 | 1.038x | 1348 |
| 8 | 3669.99→3579.96 | 1.025x | 2310.78→2146.02 | 1.077x | 1280 |

这里的显存是“模型权重 + 32 GiB KV Cache + CUDA Graph + 临时张量”的整进程
NVML 峰值。由于权重和 KV Cache 占了大头，降低比例不如算子微基准醒目，但绝对值
仍稳定减少约 1.0–1.7 GiB/GPU。

## 数值正确性：我们的结果和 PR 为什么不同

我们的 H20 路径使用与原生相同的 BF16 LM Head 投影语义，再用 FP32 做 LSE 和
Logprob：

- 30/30 个 case 的目标 Logprob 在绝对误差 `2e-5` 下全部一致；
- 最大绝对误差 `3.815e-6`；
- tie-aware 目标排名全部等价；
- Top-K 集合及同分边界替换后的语义全部等价；
- 生成 Token 全部一致。

PR #53267 的 fused CuTe 内核从 BF16 输入和权重直接以 FP32 累加计算后续结果，
中间不按原生路径物化并舍入 BF16 Logits。它自己的测试也明确把 FP32 累加作为契约，
并为 BF16 结果使用更宽的容差。因此它与原生/PyTorch BF16 参考有系统性差异：

| TP | 最大目标 Logprob 差值 | 平均绝对差值范围 | `2e-5` 严格等价 |
| ---: | ---: | ---: | :---: |
| 2 | 0.05921 | 0.00193–0.00203 | 0/10 |
| 4 | 0.04850 | 0.00191–0.00198 | 0/10 |
| 8 | 0.05280 | 0.00193–0.00205 | 0/10 |

PR 的目标分数只有约 `0.9%–1.4%` 的位置落在 `2e-5` 内，tie-aware 目标 Rank
等价率约 `5.2%–6.9%`；15 个 K=20 case 在该严格门槛下均不是 Top-K 语义等价。
Prompt Logprobs 不参与当前 greedy 生成，所以生成 Token 仍然全部相同。

这里的准确说法是：**PR 选择了不同的 FP32 数值语义，不是简单的“算错了”；但如果
项目按文档要求以 vLLM 原生 BF16 路径作为验收基线，它不满足严格对齐，我们满足。**

## 显存、通信和负载均衡

### 显存取舍

1024 行、K=20 的核心算子增量峰值显存如下：

| TP | Native | PR fused | H20 | H20 相对 Native 降低 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1214.5 MiB | 33.4 MiB | 247.8 MiB | 79.6% |
| 4 | 1094.0 MiB | 16.8 MiB | 125.3 MiB | 88.5% |
| 8 | 1032.6 MiB | 8.6 MiB | 62.9 MiB | 93.9% |

PR fused 因为完全不物化 local Logits，算子临时显存更少；H20 路径多占几十到两百
MiB 来换取更快 GEMM。放到完整 27B 模型中，两者峰值差异最多 `252 MiB/GPU`，
不到整进程占用的 `0.6%`；7 次复测中二者峰值完全相同。两者相对原生都节省约
1–1.7 GiB/GPU。

### 通信

1024 行、K=20 时，原生路径每卡 collective 输入随 TP 分别为
`242.500/121.250/60.625 MiB`；两个 compact 路径只传目标值、LSE、Rank 和
Top-K 候选，每卡约 `0.168 MiB`，不再随词表大小增长。

TP8、32K、K=20 的 Nsight 中，H20 相对原生：

- NCCL AllGather 累计时间：`342.86 ms → 10.88 ms`，降低 `96.8%`；
- Top-K `computeBlockDigitCounts`：`363.14 ms → 54.20 ms`；
- Top-K `gatherTopK`：`321.87 ms → 54.51 ms`；
- 原生 Top-K、LogSoftmax 和 Rank kernel 被移除；
- 8 卡累计 GPU kernel 时间减少 `1309.03 ms`，折算每卡 `163.63 ms`。

对应验证轮中，同一 case 的端到端 TTFT 减少 `163.71 ms`，和 Nsight 折算结果
只差 `0.08 ms`，说明收益确实来自 Prompt Logprobs 路径，而不是 backbone 变化。

### 负载均衡

所有 rank 分配相同的 local tensor 形状，核心算子增量峰值显存卡间差为 0 MiB。
1024 行、K=20 的 CUDA-event 时间卡间差 `(max-min)/max` 为：

| TP | Native | H20 |
| ---: | ---: | ---: |
| 2 | 0.016% | 0.001% |
| 4 | 0.115% | 0.010% |
| 8 | 0.393% | 0.041% |

## 核心算子为什么快，端到端为什么只快几个点

1024 行、K=20 的核心 LM Head + Prompt Logprobs 阶段：

| TP | Native | PR #53267 | H20 | H20 相对 PR 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 14.84 ms | 17.17 ms | 11.28 ms | 1.52x |
| 4 | 10.53 ms | 9.35 ms | 5.79 ms | 1.61x |
| 8 | 8.33 ms | 5.34 ms | 3.01 ms | 1.77x |

核心算子能比 PR 快 `1.52x–1.77x`，但完整 Prefill 还包含 64 层 Transformer 的
Attention、GEMM、归一化，以及返回 Prompt Logprobs 的 Python 对象整理。因此摊到
端到端后是约 `2%–3%` 的典型收益；TP8 的 TTFT 中位收益为 `3.3%`，最高 `7.6%`。
这不是算子收益消失，而是优化部分只占完整请求的一部分。

## TP8 编译缓存问题及修复

早期 TP8 对比曾出现目标分数异常。原因不是 Prompt Logprobs kernel，而是功能开关
被错误放进 `torch.compile` 缓存键，导致 native、PR 和 H20 各自重新编译同一个
backbone；Qwen3.5 TP8 的独立 BF16 编译产物会产生额外数值扰动。

现在已经把 Prompt Logprobs 功能开关和三方 benchmark 后端开关都从
`envs.compile_factors()` 排除，并增加回归测试。三个实现会复用同一 AOT artifact，
只比较 Prompt Logprobs 后端。修复后最大误差回到 `1.907e-6–3.815e-6`。

旧目录 `prompt-logprobs-qwen35-e2e-tp8-b1-mrv2-20260824` 只保留作诊断证据，
不是正式结果；本文只使用 2026-08-25 的三方矩阵。

## 代码检查和基准保护

- 三方选择和 E2E 基准相关的 92 个 CPU 测试通过；
- H20 kernel 的 45 个 GPU 测试通过；
- 完整 staged pre-commit 通过；
- Python 3.10 和 3.12 的 mypy 检查通过；
- 基准会校验模型配置、Prompt、KV Cache、Model Runner 和实际后端；
- 默认任何不等价 case 都会失败。三方采集时显式使用
  `--allow-correctness-mismatch`，只是为了保留 PR 的结果；native 与 H20 的
  30 个 case 仍全部通过严格等价检查。

## 当前限制和下一步

- 真实模型 E2E 目前只测了 Batch 1，不能据此声称 Batch 2/4 或高并发收益；
- 使用离线 `LLM.generate`，尚未测试 OpenAI server、网络层和连续请求调度；
- 只验证了 Qwen3.5-27B 的 5120×248320 LM Head 形状；
- E2E 覆盖 K=0/20；K=32 已有 kernel/单元测试，但没有跑完整三方矩阵；
- 当前支持 raw logprobs、未量化、无 Bias、无新增词表的标准 LM Head；
- 非 H20 会走 PR fused 路径，本文不提供其他 GPU 的真实模型结论；
- Transformers 的 `min_frames/max_frames` 文档提示和 NumPy 2.5/Numba
  兼容提示是当前环境的非阻断告警，不影响本次执行路径。

如果业务验收配置就是 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8，本轮已经完成。
后续只按真实业务补矩阵：线上有并发就补 Batch 2/4 与 P50/P95；业务使用 K=32
就补 K=32；换模型或 GPU 就重新标定自动选择阈值。

## 复现示例

以下命令复现 TP8 三方正式矩阵；TP2/TP4 只需修改 GPU 数和
`--tensor-parallel-size`：

```bash
cd /lustre/raplab/client/binc/workspace/vllm-logprobs

export HF_HOME=/tmp/binc-hf-qwen35-cache
export XDG_CACHE_HOME=/tmp/binc-qwen35-xdg-cache
export VLLM_CACHE_ROOT=/tmp/binc-qwen35-vllm-cache
export TRITON_CACHE_DIR=/tmp/binc-qwen35-triton-cache
export VLLM_NO_USAGE_STATS=1

.venv/bin/python benchmarks/benchmark_prompt_logprobs_e2e.py \
  --model /tmp/binc-qwen3.5-27b-fc05daec \
  --tensor-parallel-size 8 \
  --prompt-lengths 1024 4096 8192 16384 32768 \
  --batch-sizes 1 \
  --top-k 0 20 \
  --num-warmups 1 \
  --num-repeats 3 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory-gib 32 \
  --variants native fused h20 \
  --allow-correctness-mismatch \
  --output-dir /path/to/results
```

`--allow-correctness-mismatch` 只用于让三方任务在记录完 PR 的 FP32 数值语义后继续；
单独跑 `--variants native h20` 时不需要该参数。
