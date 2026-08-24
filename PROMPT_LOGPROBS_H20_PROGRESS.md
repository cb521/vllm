# H20 Prompt Logprobs 优化：真实模型端到端结果

更新时间：2026-08-24

## 一句话结论

H20 Prompt Logprobs 优化已经完成真实 27B 模型端到端验证。使用
[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)、Batch 1、BF16、
K=0/20、Prompt 1K 至 32K、TP2/TP4/TP8 共 30 个正式 case：

- 30/30 个 case 输出语义等价，目标 Logprob 最大绝对误差 `3.815e-6`；
- 完整 `LLM.generate` wall time 提升 `1.007x–1.079x`；
- 引擎 TTFT 提升 `1.008x–1.083x`；
- 在固定 32 GiB KV Cache 的整进程口径下，峰值显存降低
  `972–1750 MiB/GPU`；
- TP8、32K、K=20 的 TTFT 从 `2310.90 ms` 降到 `2147.19 ms`
  （`1.076x`），Nsight 观测到的每卡 kernel time 降幅为 `163.63 ms`，
  与端到端 TTFT 的 `163.71 ms` 差值一致。

因此，“核心算子有效但真实模型收益未知”这一项已经关闭。当前仍未覆盖的是
Batch 大于 1、在线服务并发、其他模型形状和其他 GPU。

## 代码与数据位置

- 仓库：<https://github.com/cb521/vllm>
- 分支：`optimize-prompt-logprobs`
- 本轮开始提交：`b3a200b15fe87b8d1d6ccfa251a3487cd312345a`
- 真实模型 E2E 基准：
  [`benchmarks/benchmark_prompt_logprobs_e2e.py`](benchmarks/benchmark_prompt_logprobs_e2e.py)
- 核心算子基准：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.py`](benchmarks/kernels/benchmark_lm_head_logprobs.py)
- 核心算子详细报告：
  [`benchmarks/kernels/benchmark_lm_head_logprobs.md`](benchmarks/kernels/benchmark_lm_head_logprobs.md)

正式结果保存在工作区外的 `benchmark_artifacts/`，没有提交大体积 NPZ/Nsight
文件：

| 配置 | 结果目录 | `comparison.json` SHA256 |
| --- | --- | --- |
| TP2 | `prompt-logprobs-qwen35-e2e-tp2-b1-mrv2-20260824` | `184b323b1a81857909198159dd044902721ae07260c5896ea621431108cafe47` |
| TP4 | `prompt-logprobs-qwen35-e2e-tp4-b1-mrv2-20260824` | `20c3955e9fc685a12e225a3e68160fae1c58fc40d940f9fc9bd086e8f3c8e9f7` |
| TP8 | `prompt-logprobs-qwen35-e2e-tp8-b1-mrv2-cachefix-20260824` | `fb9042d76c313098d048a759a5bfe58becf62ddce840b1c56ebdf153fc6d71f3` |

Nsight Systems 原始报告在
`benchmark_artifacts/prompt-logprobs-qwen35-nsys-tp8-20260824/`：

- `baseline.nsys-rep`：`ce5b713e895565f16937f5c4e501f02fe3d1e401aa34e68dc34eee429215d053`
- `optimized.nsys-rep`：`7d72d8399c37fad1bdd3be7e683c81de50924dd5f9dac2005ef7dff76f47c71a`

## 实现摘要

原始路径会物化并跨 TP 处理完整词表 Logits。新路径改为：

1. 每张卡只物化自己负责的 BF16 词表分片；
2. 在本卡计算目标 Token、LSE、排名和 Top-K 候选；
3. 多卡之间只合并紧凑统计量和候选，不再 AllGather 完整词表；
4. 不再生成完整词表大小的 FP32 Logprobs；
5. H20 使用实测更快的平台 GEMM + materialized local-logits 路径，其他 GPU
   保留已有 fused 路径。

当前 H20 自适应选择规则为：

- TP1：保留原始路径；
- TP2：当前 Chunk 至少 64 Token 时使用 H20 路径；
- TP4/TP8：当前 Chunk 至少 256 Token 时使用 H20 路径；
- 更小的尾部 Chunk 自动回退，避免小任务回退；
- 功能开关为 `VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS=1`。

自动启用范围保持保守：Model Runner V2、raw logprobs、未量化且无 Bias 的标准
LM Head、无新增词表、K 不大于 32；不满足条件时回退。

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
- vLLM 工作树起点：`b3a200b15`，Model Runner V2 已强制开启并在运行时校验

权重为固定 revision 的节点本地副本：
`/tmp/binc-qwen3.5-27b-fc05daec`。H20-GPU-27 和 H20-GPU-29 均已准备，
避免 52 GiB 权重在每次启动时经过 Lustre。

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
- baseline 与 optimized 使用完全相同的随机 Token ID Prompt

每个 variant 只加载一次模型并跑完整矩阵。baseline 和 optimized 必须是独立进程，
因为功能开关在 worker 初始化时读取。基准会同时保存：

- `LLM.generate` wall time 和 Prompt tokens/s；
- vLLM 请求指标中的 first-token latency（本文称 TTFT）；
- 10 ms 周期的 NVML 各卡显存；
- 目标 Token 分数/排名、Top-K、生成 Token 的压缩 NPZ；
- baseline/optimized 的逐位置数值与语义对拍结果。

Wall time 还包含大体积 Prompt Logprobs Python 对象的前端整理与返回；TTFT 更接近
模型执行路径。K=20、长 Prompt 时前端输出成本约为 0.4–1.35 秒，因此两种口径都保留。

## 真实模型端到端结果

### TP2

| Prompt | K | Wall baseline→opt (ms) | Wall 加速 | TTFT baseline→opt (ms) | TTFT 加速 | 显存降低 (MiB/GPU) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 229.09→227.45 | 1.007x | 227.48→225.74 | 1.008x | 972 |
| 1024 | 20 | 242.93→239.60 | 1.014x | 231.11→227.57 | 1.016x | 972 |
| 4096 | 0 | 867.30→858.71 | 1.010x | 861.03→852.31 | 1.010x | 1496 |
| 4096 | 20 | 920.59→907.64 | 1.014x | 873.68→859.98 | 1.016x | 1496 |
| 8192 | 0 | 1708.78→1692.45 | 1.010x | 1696.65→1679.81 | 1.010x | 1224 |
| 8192 | 20 | 2103.58→2074.92 | 1.014x | 1723.03→1694.95 | 1.017x | 1224 |
| 16384 | 0 | 3505.04→3472.21 | 1.009x | 3481.87→3448.15 | 1.010x | 1224 |
| 16384 | 20 | 4009.44→3958.41 | 1.013x | 3531.34→3475.01 | 1.016x | 1222 |
| 32768 | 0 | 7375.48→7307.82 | 1.009x | 7329.80→7260.73 | 1.010x | 1222 |
| 32768 | 20 | 8771.02→8675.06 | 1.011x | 7426.96→7314.45 | 1.015x | 1222 |

### TP4

| Prompt | K | Wall baseline→opt (ms) | Wall 加速 | TTFT baseline→opt (ms) | TTFT 加速 | 显存降低 (MiB/GPU) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 143.13→132.68 | 1.079x | 141.56→130.99 | 1.081x | 972 |
| 1024 | 20 | 149.12→145.65 | 1.024x | 137.22→133.30 | 1.029x | 972 |
| 4096 | 0 | 477.56→467.35 | 1.022x | 471.47→460.90 | 1.023x | 1646 |
| 4096 | 20 | 531.86→514.27 | 1.034x | 484.87→466.15 | 1.040x | 1646 |
| 8192 | 0 | 933.51→912.40 | 1.023x | 921.39→899.92 | 1.024x | 1350 |
| 8192 | 20 | 1326.55→1301.93 | 1.019x | 944.51→908.78 | 1.039x | 1350 |
| 16384 | 0 | 1896.50→1856.55 | 1.022x | 1873.09→1832.24 | 1.022x | 1350 |
| 16384 | 20 | 2407.22→2349.74 | 1.024x | 1922.70→1850.71 | 1.039x | 1350 |
| 32768 | 0 | 3973.61→3893.37 | 1.021x | 3927.98→3846.01 | 1.021x | 1350 |
| 32768 | 20 | 5368.32→5264.71 | 1.020x | 4023.35→3878.74 | 1.037x | 1348 |

### TP8

| Prompt | K | Wall baseline→opt (ms) | Wall 加速 | TTFT baseline→opt (ms) | TTFT 加速 | 显存降低 (MiB/GPU) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 104.87→101.31 | 1.035x | 103.27→99.70 | 1.036x | 972 |
| 1024 | 20 | 118.22→112.00 | 1.056x | 106.26→99.73 | 1.065x | 972 |
| 4096 | 0 | 276.21→265.25 | 1.041x | 270.06→259.06 | 1.042x | 1750 |
| 4096 | 20 | 330.76→309.73 | 1.068x | 284.36→262.63 | 1.083x | 1750 |
| 8192 | 0 | 529.16→505.60 | 1.047x | 517.14→493.47 | 1.048x | 1282 |
| 8192 | 20 | 927.12→885.35 | 1.047x | 542.30→500.57 | 1.083x | 1282 |
| 16384 | 0 | 1079.40→1033.38 | 1.045x | 1056.29→1009.79 | 1.046x | 1282 |
| 16384 | 20 | 1590.87→1504.31 | 1.058x | 1106.34→1022.55 | 1.082x | 1280 |
| 32768 | 0 | 2261.85→2171.81 | 1.041x | 2215.80→2125.71 | 1.042x | 1280 |
| 32768 | 20 | 3662.36→3489.81 | 1.049x | 2310.90→2147.19 | 1.076x | 1280 |

这里的显存是“模型权重 + 32 GiB KV Cache + CUDA Graph + 临时张量”的整进程
NVML 峰值，所以降低比例不像算子微基准的 85%–95% 那么大；绝对值仍稳定减少约
1.0–1.7 GiB/GPU。

## 正确性结果与并列 Token 口径

30 个正式 case 的结果为：

- 目标 Token Logprob 在绝对误差 `2e-5` 下 100% 一致；
- 最大绝对误差：`3.815e-6`；
- tie-aware 目标排名：100% 等价；
- Top-K 集合/同分替换后的语义：100% 等价；
- 生成 Token：100% 一致。

K=0 的公开 rank 全部逐位完全一致。K=20 有极少量公开 rank 数字不同，但不是
数值错误：BF16 中存在完全相等的分数，`torch.topk` 与 compact Top-K 对同分 Token
可返回不同顺序；同时 vLLM 的公开字典先写目标 Token、再写 Top-K，目标 Token 在
Top-K 内时会被 Top-K ordinal 覆盖。基准因此同时保存同分 rank 区间，并允许同分
Token 在 Top-K 边界等价替换。

为排除“容差掩盖实现错误”，还在同一进程、同一份 hidden states/local logits 上直接
对拍过原始和 compact 内部结果：目标分数与严格大于计数得到的 rank 完全一致。

## TP8 AOT 编译缓存干扰及修复

第一次 TP8 正式运行中，baseline/optimized 的生成 Token 相同，但目标分数出现远大于
算子误差的差异。调查结果不是 Prompt Logprobs kernel 错误，而是功能开关被纳入
`torch.compile` 缓存键：

- 两边 `config_hash`、`code_hash`、`compiler_hash` 完全相同；
- baseline cache：`00812839ff`；
- optimized cache：`5a3bb855b4`；
- 唯一相关差异是
  `VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS=false/true`，导致同一 backbone 独立编译；
- Qwen3.5 TP8 对独立编译产物的 BF16 数值扰动比 TP2/TP4 敏感。

该开关只在模型前向之后选择 Prompt Logprobs worker，不改变编译的 backbone 图。
现在已将它从 `envs.compile_factors()` 排除并增加回归测试。修复后 optimized 直接加载
baseline 的同一 AOT artifact，smoke 的编译耗时由 `21.15 s` 降到 `0.26 s`，TP8
最大误差回到 `1.907e-6`；完整 32K 矩阵最大误差为 `3.815e-6`。

旧目录 `prompt-logprobs-qwen35-e2e-tp8-b1-mrv2-20260824` 仅保留为诊断证据，
**不是正式性能/正确性结果**。正式 TP8 数据只使用带 `cachefix` 的目录。

另有一个误用 Model Runner V1 的 no-op 负对照目录
`prompt-logprobs-qwen35-e2e-tp2-b1-mrv1-control-20260824`，同样不计入正式结果。
基准程序现在强制设置并运行时校验 Model Runner V2。

## Nsight Systems 结论

配置：TP8、Batch 1、32K、K=20；只用 `cudaProfilerApi` 捕获额外一次正式请求。
下表时间均为 8 张 GPU 的 kernel duration 累计，不是单卡 wall time：

| 项目 | Baseline | Optimized | 变化 |
| --- | ---: | ---: | ---: |
| 全部 GPU kernel | 18256.37 ms | 16947.34 ms | -1309.03 ms（每卡 -163.63 ms） |
| NCCL AllGather | 342.86 ms | 10.88 ms | -96.8% |
| Top-K `computeBlockDigitCounts` | 363.14 ms | 54.20 ms | -85.1% |
| Top-K `gatherTopK` | 321.87 ms | 54.51 ms | -83.1% |
| BF16 direct copy | 309.21 ms | 47.11 ms | -84.8% |
| 原始 Top-K + LogSoftmax kernel | 112.95 ms | 0 | 消除 |
| 原始 rank kernel | 33.86 ms | 0 | 消除 |
| 新 compact LSE + rank | 0 | 22.56 ms | 新增 |
| 新 compact reduce + TP merge | 0 | 8.30 ms | 新增 |

Backbone 的主要 GEMM、Attention 和归一化 kernel 实例数及耗时基本不变。例如最大
三个 NVJet GEMM 的累计耗时分别为 `5320.22/3694.50/2572.90 ms`（baseline）和
`5320.45/3694.78/2572.94 ms`（optimized）。因此收益确实来自词表分片、Top-K、
拷贝和通信缩减，不是 backbone 编译差异或测量噪声。

正式三轮矩阵在同一 case 的 TTFT 降低 `163.71 ms`，与 Nsight 折算每卡的
`163.63 ms` 高度一致。

## 合成核心算子结果（保留作分层解释）

真实模型结果不能替代算子微基准；后者仍用于解释显存和通信上限。

### 1024 Token

| TP | vLLM 原始路径 | 上游全融合路径 | H20 路径 | 相对原始路径加速 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 14.84 ms | 17.17 ms | 11.28 ms | 1.32x |
| 4 | 10.53 ms | 9.35 ms | 5.79 ms | 1.82x |
| 8 | 8.33 ms | 5.34 ms | 3.01 ms | 2.76x |

### 32K Token

| TP | 原始路径 | H20 路径 | 加速 | 算子峰值显存降低 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 475.36 ms | 360.26 ms | 1.32x | 85.2% |
| 4 | 335.83 ms | 184.27 ms | 1.82x | 91.8% |
| 8 | 266.20 ms | 136.05 ms | 1.96x | 95.5% |

1024 Token、Top-20 时，compact 通信每卡约 `0.168 MiB`，通信耗时相对原始路径
降低约 13–16 倍。真实模型 Nsight 的 AllGather 累计耗时降低 96.8%，与该结论一致。

## 基准程序的验收保护

新的 E2E 基准程序会：

- 强制 Model Runner V2，并校验 compact 开关与 worker variant 一致；
- 分进程跑 baseline/optimized，固定 Prompt、KV Cache 和模型配置；
- 保存逐 case 原始重复值、TTFT、NVML 显存和 NPZ 数值证据；
- 校验两边所有可比配置字段和 Prompt 完全一致；
- 处理 BF16 同分 Token 的 rank/Top-K 语义；
- 默认在任何 case 非语义等价时以非零状态失败；诊断时可显式传
  `--allow-correctness-mismatch`；
- 支持 `--profile-case`，配合 Nsight 只采指定 case。

## 当前限制

- 真实模型 E2E 目前只测了 Batch 1；没有据此推断 Batch 2/4 或高并发吞吐。
- 使用离线 `LLM.generate`，尚未测 OpenAI server、网络层和连续请求调度。
- 只验证了 Qwen3.5-27B 这一组 5120×248320 形状。
- E2E 覆盖 K=0/20；K=32 已有 kernel/单元测试，但未跑这次完整 30-case 矩阵。
- 当前支持范围仍是 raw logprobs、未量化/无 Bias/无新增词表的标准 LM Head。
- 非 H20 使用 fused 路径；本文不提供其他 GPU 的真实模型性能结论。
- Transformers 的 `min_frames/max_frames` 文档提示和 NumPy 2.5/Numba 兼容提示是
  当前环境的非阻断告警，不影响本次执行路径和结果。

## 下一步

如果业务验收配置就是 Qwen3.5-27B、Batch 1、K≤20、TP2/4/8，本轮真实模型验证
已经完成。后续按实际需求选择，不再无目的扩展矩阵：

1. 若线上有并发，补 Batch 2/4 和 OpenAI server 的持续负载、P50/P95 TTFT、吞吐；
2. 若业务 K=32，补同一真实模型矩阵；
3. 若要覆盖其他模型/GPU，再用该 E2E 基准重新标定自适应阈值；
4. 上游提交时拆分为功能实现、H20 自适应、编译缓存键修复和 benchmark/report，
   并附本文件中的正确性与 Nsight 证据。

## 复现示例

以下命令复现 TP8 正式矩阵；TP2/TP4 只需修改 GPU 数和
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
  --output-dir /path/to/results
```
