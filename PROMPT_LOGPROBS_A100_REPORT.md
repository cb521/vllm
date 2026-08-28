# A100 Prompt Logprobs 对比报告

## 这个 PR 做了什么

### Kernel 层

- A100 使用 SM80 `mma.sync` CuTeDSL Kernel 和词表优先 CTA 调度，把 TP-local
  LM Head、LSE、目标 Rank 统计和 local Top-K 融合到一个核心 Kernel 中。
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

- ✅ 数值正确性：通过。
- ✅ 峰值显存：通过。
- ✅ 通信数据量：通过。
- ⚠️ 完整模型吞吐：观察到 `1.191x` 几何平均加速，但只有 TP4 是同卡对比。
- ❌ 通信耗时：未通过最终验收。测试节点的 NCCL 直连路径故障，只能使用 socket
  回退，因此当前通信时间不代表正常 A100 集群。

## 测试配置

- 模型：Qwen3.5-27B，BF16，`H=5120`，`V=248320`
- GPU：8 × NVIDIA A100 80GB PCIe
- TP：2、4、8
- Batch Size：1
- Prompt：1K、4K、8K、16K、32K
- Prompt Logprobs K：0、20；本文主表使用 K=20
- 每个 case：1 次 warmup，3 次正式重复，取中位数
- 基线：相同模型、Prompt 和 TP 配置下的 vLLM 原生实现

## 完整 Qwen3.5-27B 对比 vLLM 原生（K=20）

`加速 = Native 时间 / 当前时间`：

| TP | 自动路径几何平均加速 | 单点范围 | 胜出 | 每卡峰值显存降低 | 同一物理 GPU |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1.145x | 1.140x–1.153x | 5/5 | 972–1692 MiB | 0/5 |
| 4 | 1.190x | 1.178x–1.199x | 5/5 | 972–1728 MiB | 5/5 |
| 8 | 1.240x | 1.175x–1.414x | 5/5 | 972–1750 MiB | 0/5 |
| 全部 | 1.191x | 1.140x–1.414x | 15/15 | 972–1750 MiB | 5/15 |

TP4 是完整的同卡定量结果。TP2 和 TP8 保留为延迟证据，但不能用于精确归因 Kernel
收益；其中 TP8 native 32K/K=20 的三次重复波动达到 `16.54%`。

最长的 32K/K=20 case 原始值如下（耗时含完整模型和结果整理）：

| TP | Wall Native→当前 | Prompt 吞吐 Native→当前 | 每卡峰值显存降低 | 同一物理 GPU |
| ---: | ---: | ---: | ---: | :---: |
| 2 | 120713→105890 ms | 271.5→309.5 tok/s | 1178 MiB | 否 |
| 4 | 191118→162306 ms | 171.5→201.9 tok/s | 1350 MiB | 是 |
| 8 | 293787→207737 ms | 111.5→157.7 tok/s | 1280 MiB | 否 |

## 正确性

正确性使用共享同一输入张量的受控矩阵判断：

| TP | case | 当前最大误差 | 当前 Rank 一致 |
| ---: | ---: | ---: | ---: |
| 2 | 5 | 1.907e-6 | 5/5 |
| 4 | 5 | 1.907e-6 | 5/5 |
| 8 | 5 | 1.907e-6 | 5/5 |
| 全部 | 15 | 1.907e-6 | 15/15 |

独立 vLLM 进程之间的 native-vs-native 重放虽然 Prompt 和生成 Token 6/6 相同，
目标分数仍不能复现。因此跨进程 NPZ 不用于判断 Kernel 正确性。

## 显存、通信和负载

1024 行、K=20 的受控核心测试结果：

| TP | Native | Fused | Materialized | Native→Materialized 临时峰值 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 584.962 ms | 8.627 ms | 7.195 ms | 2668.7→243.2 MiB |
| 4 | 920.648 ms | 5.937 ms | 4.674 ms | 2668.7→123.0 MiB |
| 8 | 1150.270 ms | 6.203 ms | 5.229 ms | 2668.7→62.3 MiB |

Native 时间被故障节点的 socket collective 主导，只能说明本轮回退环境，不能当作
健康 A100 集群的 Kernel 加速倍率。

- 1024 行、K=20 时，原生 collective 每卡输入为
  `242.500/121.250/60.625 MiB`，当前路径为 `0.168 MiB`。
- 受控矩阵所有 Rank 的增量峰值显存差为 `0 MiB`。
- 当前核心计算时间卡间差不超过 `0.77%`，但没有稳定低于原生基线。
- A100 节点必须禁用 NCCL P2P/CUMEM/SHM 并使用 socket 回退，因此通信时间只用于
  同轮实现对比，不代表健康集群。

## 对原 PR 的补充对比

这不是正式验收基线，只用于确认新实现是否优于已有 fused 方案：

| TP | 新 fused 相对原 PR | materialized 相对原 PR |
| ---: | ---: | ---: |
| 2 | +8.35%，5/5 胜出 | +34.18%，5/5 胜出 |
| 4 | +4.42%，5/5 胜出 | +27.73%，5/5 胜出 |
| 8 | +0.25%，3/5 胜出 | +11.77%，5/5 胜出 |
| 全部 | +4.29%，13/15 胜出 | +24.19%，15/15 胜出 |

## 范围和原始结果

结论适用于已测的 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8 和 A100 80GB PCIe。
正式汇总和 353 项 SHA256 manifest 位于：

```text
/home/scratch.binc_gpu_1/prompt-logprobs-a100-20260825/
artifacts/a100-formal-fourway-20260827-3949280
```
