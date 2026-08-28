# H20 Prompt Logprobs 对比报告

H20 的正式测试已经跑完。以下数据都以相同模型、相同硬件和相同 TP 配置下的
vLLM 原生 Prompt Logprobs 为基线。

## Kernel 实现

H20 使用针对 SM90 的 WGMMA CuTeDSL Kernel。它在计算 TP-local LM Head 的同时，
完成 LSE、目标 Token 的 Rank 和 local Top-K 统计，不再单独生成完整的 FP32
Logprobs。对于更适合平台 GEMM 的形状，materialized 路径先用 BF16 GEMM 生成
TP-local Logits，再由一个 Triton Kernel 完成 LSE 和 Rank 归约。

各 TP Rank 算完后，最后一个 Triton Kernel 负责合并 LSE、Rank 和 Top-K 候选，
并计算归一化后的 Logprob。

## vLLM 接入

Prompt Logprobs 流程现在直接接收 Hidden States 和目标 Token ID，绕过原生路径的
完整词表 Logits AllGather。TP Rank 之间只交换目标值、LSE、Rank 和 Top-K 候选。
目标 Token Logit 仍使用 BF16 batched GEMM，和 vLLM 原生 LM Head 的数值语义保持
一致。

每个 Chunk 会根据 GPU、TP、行数、K 和 256 MiB 临时显存上限，在 native、fused
和 materialized 三条路径中选择合适的一条。启动时会先做正确性检查和 Kernel 预热，
避免把首次编译算进正式请求。

## 测试配置

- 模型：Qwen3.5-27B，BF16，`H=5120`，`V=248320`
- GPU：NVIDIA H20-3e
- TP：2、4、8
- Batch size：1
- Prompt：1K、4K、8K、16K、32K
- Prompt Logprobs K：0、20；本文主表使用 K=20
- 每个测试点：预热 1 次，正式测试 3 次，取中位数
- 基线：相同模型、Prompt、硬件和 TP 配置下的 vLLM 原生实现

## Prompt Logprobs 子流程

下表覆盖 LM Head、归约、TP 通信和 Chunk 拼接，不包含 Transformer backbone：

| TP | 1K–32K 加速 | 32K 耗时（原生→优化） | 32K 吞吐（原生→优化） | 32K 临时峰值（原生→优化） | 32K 单卡通信（原生→优化） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.317x–1.320x | 475.39→360.40 ms | 68,929→90,921 tok/s | 1708.4→253.1 MiB | 7760→5.375 MiB |
| 4 | 1.817x–1.824x | 335.59→184.19 ms | 97,642→177,901 tok/s | 1587.9→131.1 MiB | 3880→5.375 MiB |
| 8 | 2.758x–2.770x | 266.05→96.05 ms | 123,164→341,140 tok/s | 1526.5→68.1 MiB | 1940→5.375 MiB |

1024 行、K=20 时，原生路径每卡送入 collective 的数据量为
`242.500/121.250/60.625 MiB`，优化后为 `0.168 MiB`。TP2/4/8 通信耗时
分别由 `1.956/2.198/2.342 ms` 降到 `0.062/0.062/0.063 ms`，加速
`31.69x/35.60x/37.27x`。

## 完整模型结果

完整 Qwen3.5-27B 与 vLLM 原生路径的对比如下：

| TP | 总耗时加速范围 | 总耗时加速中位数 | TTFT 加速范围 | TTFT 加速中位数 | 每卡峰值显存降低 | 结果一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1.009x–1.020x | 1.011x | 1.009x–1.017x | 1.013x | 972–1496 MiB | 10/10 |
| 4 | 1.022x–1.039x | 1.028x | 1.022x–1.044x | 1.033x | 972–1646 MiB | 10/10 |
| 8 | 1.023x–1.062x | 1.042x | 1.023x–1.082x | 1.051x | 972–1750 MiB | 10/10 |

32K、K=20 的结果如下：

| TP | 总耗时（原生→优化） | Prompt 吞吐（原生→优化） | TTFT（原生→优化） | 每卡峰值显存降低 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8798.5→8657.9 ms | 3724→3785 tok/s | 7428.3→7316.7 ms | 1222 MiB |
| 4 | 5397.3→5217.4 ms | 6071→6281 tok/s | 4020.7→3874.9 ms | 1348 MiB |
| 8 | 3705.1→3502.8 ms | 8844→9355 tok/s | 2312.6→2147.4 ms | 1280 MiB |

## 正确性和负载均衡

同一输入下，受控矩阵的最大目标/分数误差为 `1.907e-6`，Rank 全部一致。完整模型
30/30 个测试点结果一致，最大目标 Logprob 误差为 `3.815e-6`。所有 Rank 的增量
峰值显存差为 `0 MiB`。

| TP | 原生计算时间卡间差 | 优化路径卡间差 |
| ---: | ---: | ---: |
| 2 | 0.011% | 0.005% |
| 4 | 0.023% | 0.014% |
| 8 | 0.101% | 0.016% |

## 测试范围

这份结论只覆盖 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8 和 H20。Batch 2/4、在线
服务、其他模型形状和其他 GPU 尚未测试，不能直接套用这里的结果。

原始数据在 `benchmark_artifacts/prompt-logprobs-adaptive-prompt-sweep-h20-20260825/`
和 `benchmark_artifacts/prompt-logprobs-fourway-h20-k20-20260826/`。
