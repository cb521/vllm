# H20 Prompt Logprobs 对比报告

## 这个 PR 做了什么

### Kernel 层

- H20 使用 SM90 WGMMA CuTeDSL Kernel，把 TP-local LM Head、LSE、目标 Rank 统计和
  local Top-K 融合到一个核心 Kernel 中。
- materialized 路径使用平台 BF16 GEMM 生成 TP-local Logits，再用 Triton 一次完成
  LSE 和 Rank 归约；不生成完整 FP32 Logprobs。
- 最后的 Triton 合并 Kernel 一次完成跨 Rank LSE 合并、Rank 求和、Top-K 合并和
  Logprob 归一化。

### 功能和框架层

- TP 间只传目标值、LSE、Rank 和 Top-K 候选，不聚合完整词表 Logits。
- 在 vLLM Prompt Logprobs 流程中直接使用 Hidden States 和目标 Token ID，绕过原生
  的完整词表 Logits AllGather。
- 目标 Token Logit 使用 BF16 batched GEMM，对齐 vLLM 原生 LM Head 数值语义。
- 按 GPU、TP、行数、K 和 256 MiB 临时显存上限逐 Chunk 选择 native、fused 或
  materialized，并在启动阶段完成校验和 Kernel 预热。

## 结论

H20 路径达到长上下文 Prompt Logprobs 的验收目标：相对 vLLM 原生实现，数值结果
一致，临时显存和 TP 通信显著降低，卡间负载更均衡，Prompt Logprobs 子流程和完整
模型吞吐均有提升。

## 测试配置

- 模型：Qwen3.5-27B，BF16，`H=5120`，`V=248320`
- GPU：NVIDIA H20-3e
- TP：2、4、8
- Batch Size：1
- Prompt：1K、4K、8K、16K、32K
- Prompt Logprobs K：0、20；本文主表使用 K=20
- 每个 case：1 次 warmup，3 次正式重复，取中位数
- 基线：相同模型、Prompt、硬件和 TP 配置下的 vLLM 原生实现

## Prompt Logprobs 子流程

下表覆盖 LM Head、归约、TP 通信和 Chunk 拼接，不包含 Transformer backbone：

| TP | 1K–32K 加速 | 32K 耗时 Native→当前 | 32K 吞吐 Native→当前 | 32K 临时峰值 Native→当前 | 32K 单卡通信 Native→当前 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.317x–1.320x | 475.39→360.40 ms | 68,929→90,921 tok/s | 1708.4→253.1 MiB | 7760→5.375 MiB |
| 4 | 1.817x–1.824x | 335.59→184.19 ms | 97,642→177,901 tok/s | 1587.9→131.1 MiB | 3880→5.375 MiB |
| 8 | 2.758x–2.770x | 266.05→96.05 ms | 123,164→341,140 tok/s | 1526.5→68.1 MiB | 1940→5.375 MiB |

1024 行、K=20 时，原生路径每卡 collective 输入为
`242.500/121.250/60.625 MiB`，当前路径为 `0.168 MiB`。TP2/4/8 通信耗时
分别由 `1.956/2.198/2.342 ms` 降到 `0.062/0.062/0.063 ms`，加速
`31.69x/35.60x/37.27x`。

## 完整 Qwen3.5-27B 对比 vLLM 原生

| TP | Wall 加速范围 | Wall 中位数 | TTFT 加速范围 | TTFT 中位数 | 每卡峰值显存降低 | 语义等价 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1.009x–1.020x | 1.011x | 1.009x–1.017x | 1.013x | 972–1496 MiB | 10/10 |
| 4 | 1.022x–1.039x | 1.028x | 1.022x–1.044x | 1.033x | 972–1646 MiB | 10/10 |
| 8 | 1.023x–1.062x | 1.042x | 1.023x–1.082x | 1.051x | 972–1750 MiB | 10/10 |

最长的 32K/K=20 case 原始值如下：

| TP | Wall Native→当前 | Prompt 吞吐 Native→当前 | TTFT Native→当前 | 每卡峰值显存降低 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8798.5→8657.9 ms | 3724→3785 tok/s | 7428.3→7316.7 ms | 1222 MiB |
| 4 | 5397.3→5217.4 ms | 6071→6281 tok/s | 4020.7→3874.9 ms | 1348 MiB |
| 8 | 3705.1→3502.8 ms | 8844→9355 tok/s | 2312.6→2147.4 ms | 1280 MiB |

## 正确性和负载均衡

- 受控矩阵最大目标/分数误差：`1.907e-6`；Rank 全部一致。
- 完整模型 30/30 case 语义等价，最大目标 Logprob 误差：`3.815e-6`。
- 所有 Rank 的增量峰值显存差为 `0 MiB`。

| TP | 原生计算时间卡间差 | 当前路径卡间差 |
| ---: | ---: | ---: |
| 2 | 0.011% | 0.005% |
| 4 | 0.023% | 0.014% |
| 8 | 0.101% | 0.016% |

## 范围

结论适用于已测的 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8 和 H20。Batch 2/4、
在线服务、其他模型形状和其他 GPU 需要单独测试。原 PR #53267 仅作为竞争实现，
正式验收基线始终是 vLLM 原生实现。

原始结果位于 `benchmark_artifacts/prompt-logprobs-adaptive-prompt-sweep-h20-20260825/`
和 `benchmark_artifacts/prompt-logprobs-fourway-h20-k20-20260826/`。
