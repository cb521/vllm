# H20 Prompt Logprobs Benchmark

## Summary

This change adds an H20-specific prompt-logprobs backend that materializes only
the TP-local BF16 logits, reduces them to compact statistics, and communicates
only the target logit, local LSE/rank, and local Top-K candidates. It never
gathers full-vocabulary logits across TP ranks and never materializes a full
FP32 logprobs tensor.

For large H20 prompt chunks, using the platform GEMM plus a one-pass Triton
LSE/rank reduction is faster than both the native gathered-logits path and the
fully fused CuTe path. An adaptive dispatcher keeps the native path for TP1 and
small tail chunks:

- TP1: native path.
- TP2: H20 path for at least 64 rows.
- TP4 and above: H20 path for at least 256 rows.
- Other GPUs: existing fully fused compact path.

## Test setup

- GPU: NVIDIA H20-3e, SM90, 143771 MiB per GPU.
- Software: CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, NCCL 2.29.7.
- vLLM base: `0ecc284790e5403f74b899524ef82ecb69f83cb3`.
- LM head: BF16, hidden size 5120, vocabulary size 248320.
- Prompt chunk size: 1024 rows.
- Top-K cases: 0 and 20; tables below focus on Top-20.
- Timing: three warmups followed by the median of five CUDA-event samples.
- Memory: incremental peak allocated memory during logprobs computation. It is
  not total model memory.
- Collective payload: bytes contributed to collectives per rank. It is not
  ring-algorithm wire traffic.
- FlashInfer was unavailable, so both native and H20 paths used `torch.topk`.

Run one process per TP rank:

```bash
torchrun --standalone --nproc-per-node=TP \
  benchmarks/kernels/benchmark_lm_head_logprobs.py \
  --num-rows 1024 \
  --prompt-lengths 4096 8192 16384 32768 \
  --top-k 0 20 --warmup 3 --repeats 5 --output result.json
```

## Numerical correctness

The reference uses the native BF16 projection followed by PyTorch FP32
`logsumexp`. At 1024 rows and Top-20:

| TP | Maximum absolute error | Mean absolute error | Target ranks |
| ---: | ---: | ---: | :---: |
| 2 | 1.91e-6 | 2.68e-7 | Exact |
| 8 | 1.91e-6 | 1.68e-7 | Exact |

Kernel tests additionally cover BF16, FP16, FP32, vocabulary padding, rank
boundaries, K=0/5/20/32, empty inputs, TP1, and TP2. The complete H20 kernel
test file passes 45 tests.

BF16 projections can contain exactly tied logits. `torch.topk` does not promise
a stable token order for ties, whereas the compact merge uses token ID as a
deterministic tie-break. Tied token IDs can therefore be ordered differently;
their returned logprobs are identical and target ranks remain exact.

## One 1024-row chunk, Top-20

| TP | Native (ms) | Existing fused (ms) | H20 path (ms) | Speedup vs native | Speedup vs fused |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 14.84 | 17.17 | 11.28 | 1.32x | 1.52x |
| 4 | 10.53 | 9.35 | 5.79 | 1.82x | 1.61x |
| 8 | 8.33 | 5.34 | 3.01 | 2.76x | 1.77x |

The fully fused path remains the lowest-memory option. The H20 path chooses a
different tradeoff: it retains local BF16 logits to use the much faster
platform GEMM, while still removing global logits, FP32 full-vocabulary state,
and full-vocabulary TP communication.

| TP | Native peak (MiB) | Fused peak (MiB) | H20 peak (MiB) | H20 reduction vs native |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1214.5 | 33.4 | 247.8 | 79.6% |
| 4 | 1094.0 | 16.8 | 125.3 | 88.5% |
| 8 | 1032.6 | 8.6 | 62.9 | 93.9% |

## Tensor-parallel communication

For 1024 rows and Top-20, the native collective carries local BF16 vocabulary
logits. The compact paths carry one FP32 target value plus `2K + 2` FP32 words
of compact state per row.

| TP | Native payload/rank (MiB) | Compact payload/rank (MiB) | Native collective (ms) | Compact collectives (ms) | Communication speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 242.500 | 0.168 | 1.97 | 0.150 | 13.1x |
| 4 | 121.250 | 0.168 | 2.20 | 0.136 | 16.2x |
| 8 | 60.625 | 0.168 | 2.35 | 0.147 | 16.0x |

The compact payload is independent of vocabulary size and decreases the
per-rank input payload by 99.9%, 99.9%, and 99.7% for TP2, TP4, and TP8,
respectively.

## Long prompts, Top-20

The reported throughput covers the LM-head and prompt-logprobs stage, not the
full transformer prefill.

| TP | Prompt | Native (ms) | H20 adaptive (ms) | Speedup | Native tokens/s | H20 tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4K | 59.43 | 45.07 | 1.32x | 68,923 | 90,872 |
| 2 | 8K | 118.91 | 90.12 | 1.32x | 68,893 | 90,902 |
| 2 | 16K | 237.73 | 180.24 | 1.32x | 68,919 | 90,902 |
| 2 | 32K | 475.36 | 360.26 | 1.32x | 68,933 | 90,956 |
| 4 | 4K | 41.97 | 23.01 | 1.82x | 97,602 | 177,991 |
| 4 | 8K | 83.97 | 46.14 | 1.82x | 97,562 | 177,534 |
| 4 | 16K | 167.93 | 92.11 | 1.82x | 97,566 | 177,882 |
| 4 | 32K | 335.83 | 184.27 | 1.82x | 97,572 | 177,829 |
| 8 | 4K | 33.31 | 17.01 | 1.96x | 122,970 | 240,799 |
| 8 | 8K | 66.62 | 35.34 | 1.89x | 122,966 | 231,806 |
| 8 | 16K | 133.18 | 72.08 | 1.85x | 123,025 | 227,308 |
| 8 | 32K | 266.20 | 136.05 | 1.96x | 123,094 | 240,861 |

For 32K Top-20, incremental peak memory changes as follows:

| TP | Native peak (MiB) | H20 adaptive peak (MiB) | Reduction |
| ---: | ---: | ---: | ---: |
| 2 | 1708.4 | 253.0 | 85.2% |
| 4 | 1587.1 | 129.7 | 91.8% |
| 8 | 1526.5 | 68.2 | 95.5% |

K=0 also improves consistently across 4K through 32K: approximately 1.22x
for TP2, 1.52x for TP4, and 1.47x to 1.73x for TP8.

## Device balance

All ranks allocate the same local tensor shapes; the measured incremental peak
memory spread is 0 MiB. For a 1024-row Top-20 core chunk, the CUDA-event time
spread `(max - min) / max` was:

| TP | Native spread | H20 path spread |
| ---: | ---: | ---: |
| 2 | 0.016% | 0.001% |
| 4 | 0.115% | 0.010% |
| 8 | 0.393% | 0.041% |

Long multi-chunk Python loops can show process-launch skew when compact
collectives are very short. The core GPU-chunk measurement above is used for
device balance because it isolates GPU work from host scheduling.

## Scope and remaining validation

- Supported configuration remains raw logprobs, unquantized BF16 LM head, no
  LM-head bias, no added vocabulary, and K at most 32.
- H20 routing was tuned for hidden size 5120 and vocabulary size 248320. The
  conservative row thresholds avoid regressions on short tail chunks, but
  other model dimensions should be benchmarked before widening automatic use.
- TP1 intentionally remains native because it showed no material speedup and
  has no cross-rank communication to remove.
- These are synthetic LM-head/logprobs benchmarks. Qwen3.5-27B full-model
  Batch-1 validation is now complete; see
  [`PROMPT_LOGPROBS_H20_PROGRESS.md`](../../PROMPT_LOGPROBS_H20_PROGRESS.md).
  Batch 2/4 and online serving workloads remain to be measured if required by
  the deployment configuration.
