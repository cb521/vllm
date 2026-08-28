# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark compact prompt logprobs against the gathered-logits path.

Launch with one process per tensor-parallel rank, for example::

    torchrun --standalone --nproc-per-node=2 \
        benchmarks/kernels/benchmark_lm_head_logprobs.py \
        --num-rows 16 256 1024 --top-k 0 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    CompactPromptLogprobs,
    compute_prompt_logprobs_with_chunking,
)


@dataclass
class Result:
    tp_size: int
    num_rows: int
    top_k: int
    baseline_ms: float
    compact_ms: float
    materialized_ms: float
    compact_speedup: float
    materialized_speedup: float
    baseline_peak_mib: float
    compact_peak_mib: float
    materialized_peak_mib: float
    compact_target_max_abs_error: float
    compact_target_mean_abs_error: float
    compact_score_max_abs_error: float
    compact_token_ids_match: bool
    compact_ranks_match: bool
    materialized_target_max_abs_error: float
    materialized_target_mean_abs_error: float
    materialized_score_max_abs_error: float
    materialized_token_ids_match: bool
    materialized_ranks_match: bool
    baseline_collective_payload_mib_per_rank: float
    compact_collective_payload_mib_per_rank: float
    baseline_time_spread_percent: float
    compact_time_spread_percent: float
    materialized_time_spread_percent: float
    baseline_memory_spread_mib: float
    compact_memory_spread_mib: float
    materialized_memory_spread_mib: float
    baseline_communication_ms: float
    compact_communication_ms: float
    communication_speedup: float


@dataclass
class PromptResult:
    tp_size: int
    prompt_length: int
    top_k: int
    baseline_ms: float
    adaptive_ms: float
    speedup: float
    baseline_tokens_per_second: float
    adaptive_tokens_per_second: float
    baseline_peak_mib: float
    adaptive_peak_mib: float
    target_max_abs_error: float
    target_mean_abs_error: float
    ranks_match: bool
    baseline_collective_payload_mib_per_rank: float
    adaptive_collective_payload_mib_per_rank: float
    baseline_time_spread_percent: float
    adaptive_time_spread_percent: float
    baseline_memory_spread_mib: float
    adaptive_memory_spread_mib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rows", type=int, nargs="+", default=[16, 256, 1024])
    parser.add_argument("--prompt-lengths", type=int, nargs="*", default=[])
    parser.add_argument("--top-k", type=int, nargs="+", default=[0, 20])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--baseline-warmup", type=int)
    parser.add_argument("--baseline-repeats", type=int)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Use materialized compact results as the reference and skip full logits",
    )
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return world_size, rank, local_rank


def make_lm_head(
    weight: torch.Tensor,
    vocab_size: int,
    world_size: int,
    rank: int,
) -> Any:
    local_vocab_size = weight.shape[0]
    vocab_start = rank * local_vocab_size
    vocab_end = min(vocab_start + local_vocab_size, vocab_size)
    shard_indices = SimpleNamespace(
        org_vocab_start_index=vocab_start,
        org_vocab_end_index=vocab_end,
        num_org_elements=vocab_end - vocab_start,
    )
    return SimpleNamespace(
        weight=weight,
        tp_size=world_size,
        shard_indices=shard_indices,
        quant_method=UnquantizedEmbeddingMethod(),
    )


def median_cuda_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    starts = [torch.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.accelerator.synchronize()
    elapsed = (start.elapsed_time(end) for start, end in zip(starts, ends))
    return statistics.median(elapsed)


def incremental_peak_mib(fn) -> float:
    gc.collect()
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    output = fn()
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - before
    del output
    return peak / (1024**2)


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def reduce_min(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def spread_percent(minimum: float, maximum: float) -> float:
    return 0.0 if maximum == 0.0 else (maximum - minimum) / maximum * 100


def collect_environment(local_rank: int, world_size: int) -> dict[str, Any]:
    """Collect enough per-rank metadata to audit cross-GPU comparisons."""
    properties = torch.cuda.get_device_properties(local_rank)
    local_device = {
        "rank": dist.get_rank(),
        "name": properties.name,
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_mib": properties.total_memory / (1024**2),
    }
    devices: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(devices, local_device)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "world_size": world_size,
        "devices": devices,
    }


def run_case(
    processor: LogitsProcessor,
    lm_head: Any,
    hidden_states: torch.Tensor,
    target_token_ids: torch.Tensor,
    top_k: int,
    warmup: int,
    repeats: int,
    baseline_warmup: int,
    baseline_repeats: int,
    skip_baseline: bool,
    world_size: int,
) -> Result:
    def baseline():
        logits = processor(lm_head, hidden_states)
        return compute_topk_scores(logits, top_k, target_token_ids)

    def compact():
        return processor.get_prompt_logprobs(
            lm_head, hidden_states, target_token_ids, top_k
        )

    def materialized():
        return processor.get_materialized_prompt_logprobs(
            lm_head, hidden_states, target_token_ids, top_k
        )

    materialized_ids, materialized_scores, materialized_ranks = materialized()
    if skip_baseline:
        expected_ids = materialized_ids
        expected_scores = materialized_scores
        expected_ranks = materialized_ranks
        reference_target_scores = materialized_scores[:, 0]
    else:
        reference_logits = processor(lm_head, hidden_states)
        expected = compute_topk_scores(reference_logits, top_k, target_token_ids)
        reference_lse = torch.logsumexp(reference_logits.float(), dim=1)
        reference_target_scores = (
            reference_logits.gather(1, target_token_ids[:, None]).squeeze(1).float()
            - reference_lse
        )
        expected_ids = expected.logprob_token_ids
        expected_scores = expected.logprobs
        expected_ranks = expected.selected_token_ranks
    actual_ids, actual_scores, actual_ranks = compact()
    torch.accelerator.synchronize()
    compact_target_error = (actual_scores[:, 0] - reference_target_scores).abs()
    compact_score_error = (actual_scores - expected_scores).abs()
    materialized_target_error = (
        materialized_scores[:, 0] - reference_target_scores
    ).abs()
    materialized_score_error = (materialized_scores - expected_scores).abs()
    if not skip_baseline:
        del reference_logits, reference_lse
    del reference_target_scores

    baseline_ms = (
        float("nan")
        if skip_baseline
        else median_cuda_ms(baseline, baseline_warmup, baseline_repeats)
    )
    compact_ms = median_cuda_ms(compact, warmup, repeats)
    materialized_ms = median_cuda_ms(materialized, warmup, repeats)
    baseline_peak_mib = (
        float("nan") if skip_baseline else incremental_peak_mib(baseline)
    )
    compact_peak_mib = incremental_peak_mib(compact)
    materialized_peak_mib = incremental_peak_mib(materialized)

    device = hidden_states.device
    baseline_min_ms = baseline_ms if skip_baseline else reduce_min(baseline_ms, device)
    compact_min_ms = reduce_min(compact_ms, device)
    materialized_min_ms = reduce_min(materialized_ms, device)
    if not skip_baseline:
        baseline_ms = reduce_max(baseline_ms, device)
    compact_ms = reduce_max(compact_ms, device)
    materialized_ms = reduce_max(materialized_ms, device)
    baseline_min_peak_mib = (
        baseline_peak_mib if skip_baseline else reduce_min(baseline_peak_mib, device)
    )
    compact_min_peak_mib = reduce_min(compact_peak_mib, device)
    materialized_min_peak_mib = reduce_min(materialized_peak_mib, device)
    if not skip_baseline:
        baseline_peak_mib = reduce_max(baseline_peak_mib, device)
    compact_peak_mib = reduce_max(compact_peak_mib, device)
    materialized_peak_mib = reduce_max(materialized_peak_mib, device)

    compact_ids_match = torch.equal(actual_ids, expected_ids)
    compact_ranks_match = torch.equal(
        actual_ranks.to(torch.int64), expected_ranks.to(torch.int64)
    )
    materialized_ids_match = torch.equal(materialized_ids, expected_ids)
    materialized_ranks_match = torch.equal(
        materialized_ranks.to(torch.int64), expected_ranks.to(torch.int64)
    )
    matches = torch.tensor(
        [
            compact_ids_match,
            compact_ranks_match,
            materialized_ids_match,
            materialized_ranks_match,
        ],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(matches, op=dist.ReduceOp.MIN)
    compact_max_error = reduce_max(float(compact_target_error.max().item()), device)
    compact_mean_error = reduce_max(float(compact_target_error.mean().item()), device)
    compact_score_max_error = reduce_max(
        float(compact_score_error.max().item()), device
    )
    materialized_max_error = reduce_max(
        float(materialized_target_error.max().item()), device
    )
    materialized_mean_error = reduce_max(
        float(materialized_target_error.mean().item()), device
    )
    materialized_score_max_error = reduce_max(
        float(materialized_score_error.max().item()), device
    )
    if world_size == 1:
        baseline_payload_mib = 0.0
        compact_payload_mib = 0.0
        baseline_communication_ms = 0.0
        compact_communication_ms = 0.0
    else:
        baseline_payload_mib = (
            hidden_states.shape[0]
            * lm_head.weight.shape[0]
            * lm_head.weight.element_size()
            / (1024**2)
        )
        compact_payload_mib = (
            hidden_states.shape[0] * (1 + 2 * top_k + 2) * 4 / (1024**2)
        )
        target_state = torch.zeros(
            hidden_states.shape[0], dtype=torch.float32, device=device
        )
        compact_state = torch.zeros(
            (hidden_states.shape[0], 2 * top_k + 2),
            dtype=torch.float32,
            device=device,
        )

        def compact_communication():
            target = tensor_model_parallel_all_reduce(target_state)
            state = tensor_model_parallel_all_gather(compact_state, dim=0)
            return target, state

        if skip_baseline:
            baseline_communication_ms = float("nan")
        else:
            local_logits = processor._apply_head(lm_head, hidden_states, None)

            def baseline_communication():
                return tensor_model_parallel_all_gather(local_logits)

            baseline_communication_ms = reduce_max(
                median_cuda_ms(
                    baseline_communication,
                    baseline_warmup,
                    baseline_repeats,
                ),
                device,
            )
        compact_communication_ms = reduce_max(
            median_cuda_ms(compact_communication, warmup, repeats), device
        )

    return Result(
        tp_size=world_size,
        num_rows=hidden_states.shape[0],
        top_k=top_k,
        baseline_ms=baseline_ms,
        compact_ms=compact_ms,
        materialized_ms=materialized_ms,
        compact_speedup=baseline_ms / compact_ms,
        materialized_speedup=baseline_ms / materialized_ms,
        baseline_peak_mib=baseline_peak_mib,
        compact_peak_mib=compact_peak_mib,
        materialized_peak_mib=materialized_peak_mib,
        compact_target_max_abs_error=compact_max_error,
        compact_target_mean_abs_error=compact_mean_error,
        compact_score_max_abs_error=compact_score_max_error,
        compact_token_ids_match=bool(matches[0].item()),
        compact_ranks_match=bool(matches[1].item()),
        materialized_target_max_abs_error=materialized_max_error,
        materialized_target_mean_abs_error=materialized_mean_error,
        materialized_score_max_abs_error=materialized_score_max_error,
        materialized_token_ids_match=bool(matches[2].item()),
        materialized_ranks_match=bool(matches[3].item()),
        baseline_collective_payload_mib_per_rank=baseline_payload_mib,
        compact_collective_payload_mib_per_rank=compact_payload_mib,
        baseline_time_spread_percent=spread_percent(baseline_min_ms, baseline_ms),
        compact_time_spread_percent=spread_percent(compact_min_ms, compact_ms),
        materialized_time_spread_percent=spread_percent(
            materialized_min_ms, materialized_ms
        ),
        baseline_memory_spread_mib=baseline_peak_mib - baseline_min_peak_mib,
        compact_memory_spread_mib=compact_peak_mib - compact_min_peak_mib,
        materialized_memory_spread_mib=(
            materialized_peak_mib - materialized_min_peak_mib
        ),
        baseline_communication_ms=baseline_communication_ms,
        compact_communication_ms=compact_communication_ms,
        communication_speedup=(
            baseline_communication_ms / compact_communication_ms
            if compact_communication_ms > 0
            else 0.0
        ),
    )


def run_prompt_case(
    processor: LogitsProcessor,
    lm_head: Any,
    hidden_states: torch.Tensor,
    target_token_ids: torch.Tensor,
    top_k: int,
    warmup: int,
    repeats: int,
    world_size: int,
) -> PromptResult:
    compact = CompactPromptLogprobs(
        processor,
        lm_head,
        backend="auto",
    )

    def baseline():
        return compute_prompt_logprobs_with_chunking(
            target_token_ids,
            hidden_states,
            lambda chunk: processor(lm_head, chunk),
            top_k,
        )

    def adaptive():
        return compute_prompt_logprobs_with_chunking(
            target_token_ids,
            hidden_states,
            lambda chunk: processor(lm_head, chunk),
            top_k,
            compact_prompt_logprobs_fn=compact.compute,
            compact_prompt_logprobs_predicate=compact.supports_chunk,
        )

    expected_ids, expected_scores, expected_ranks = baseline()
    actual_ids, actual_scores, actual_ranks = adaptive()
    torch.accelerator.synchronize()
    del expected_ids, actual_ids
    target_error = (actual_scores[:, 0] - expected_scores[:, 0]).abs()

    baseline_ms = median_cuda_ms(baseline, warmup, repeats)
    adaptive_ms = median_cuda_ms(adaptive, warmup, repeats)
    baseline_peak_mib = incremental_peak_mib(baseline)
    adaptive_peak_mib = incremental_peak_mib(adaptive)

    device = hidden_states.device
    baseline_min_ms = reduce_min(baseline_ms, device)
    adaptive_min_ms = reduce_min(adaptive_ms, device)
    baseline_ms = reduce_max(baseline_ms, device)
    adaptive_ms = reduce_max(adaptive_ms, device)
    baseline_min_peak_mib = reduce_min(baseline_peak_mib, device)
    adaptive_min_peak_mib = reduce_min(adaptive_peak_mib, device)
    baseline_peak_mib = reduce_max(baseline_peak_mib, device)
    adaptive_peak_mib = reduce_max(adaptive_peak_mib, device)
    max_error = reduce_max(float(target_error.max().item()), device)
    mean_error = reduce_max(float(target_error.mean().item()), device)
    ranks_match = torch.tensor(
        torch.equal(actual_ranks.to(torch.int64), expected_ranks),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(ranks_match, op=dist.ReduceOp.MIN)
    prompt_length = hidden_states.shape[0]
    if world_size == 1:
        baseline_payload_mib = 0.0
        adaptive_payload_mib = 0.0
    else:
        local_vocab_bytes_per_row = (
            lm_head.weight.shape[0] * lm_head.weight.element_size()
        )
        baseline_payload_mib = prompt_length * local_vocab_bytes_per_row / (1024**2)
        adaptive_payload_bytes = 0
        for start in range(0, prompt_length, 1024):
            chunk_rows = min(1024, prompt_length - start)
            if compact.supports_chunk(chunk_rows, top_k):
                adaptive_payload_bytes += chunk_rows * (1 + 2 * top_k + 2) * 4
            else:
                adaptive_payload_bytes += chunk_rows * local_vocab_bytes_per_row
        adaptive_payload_mib = adaptive_payload_bytes / (1024**2)
    return PromptResult(
        tp_size=world_size,
        prompt_length=prompt_length,
        top_k=top_k,
        baseline_ms=baseline_ms,
        adaptive_ms=adaptive_ms,
        speedup=baseline_ms / adaptive_ms,
        baseline_tokens_per_second=prompt_length * 1000 / baseline_ms,
        adaptive_tokens_per_second=prompt_length * 1000 / adaptive_ms,
        baseline_peak_mib=baseline_peak_mib,
        adaptive_peak_mib=adaptive_peak_mib,
        target_max_abs_error=max_error,
        target_mean_abs_error=mean_error,
        ranks_match=bool(ranks_match.item()),
        baseline_collective_payload_mib_per_rank=baseline_payload_mib,
        adaptive_collective_payload_mib_per_rank=adaptive_payload_mib,
        baseline_time_spread_percent=spread_percent(baseline_min_ms, baseline_ms),
        adaptive_time_spread_percent=spread_percent(adaptive_min_ms, adaptive_ms),
        baseline_memory_spread_mib=baseline_peak_mib - baseline_min_peak_mib,
        adaptive_memory_spread_mib=adaptive_peak_mib - adaptive_min_peak_mib,
    )


def main() -> None:
    args = parse_args()
    baseline_warmup = (
        args.warmup if args.baseline_warmup is None else args.baseline_warmup
    )
    baseline_repeats = (
        args.repeats if args.baseline_repeats is None else args.baseline_repeats
    )
    if baseline_warmup < 0:
        raise ValueError("baseline warmup must be non-negative")
    if baseline_repeats < 1:
        raise ValueError("baseline repeats must be positive")
    if args.skip_baseline and args.prompt_lengths:
        raise ValueError("--skip-baseline does not support --prompt-lengths")
    config = VllmConfig()
    config.parallel_config.disable_custom_all_reduce = args.disable_custom_all_reduce
    if args.disable_custom_all_reduce:
        set_custom_all_reduce(False)
    with set_current_vllm_config(config):
        world_size, rank, local_rank = init_distributed()
        if args.vocab_size % world_size:
            raise ValueError("vocab size must be divisible by the TP size")
        environment = collect_environment(local_rank, world_size)

        device = torch.device("cuda", local_rank)
        local_vocab_size = args.vocab_size // world_size
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        weight = torch.randn(
            (local_vocab_size, args.hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        weight.mul_(args.weight_scale)
        lm_head = make_lm_head(weight, args.vocab_size, world_size, rank)
        processor = LogitsProcessor(args.vocab_size)

        results: list[Result] = []
        for num_rows in args.num_rows:
            hidden_generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000 + num_rows
            )
            hidden_states = torch.randn(
                (num_rows, args.hidden_size),
                dtype=torch.bfloat16,
                device=device,
                generator=hidden_generator,
            )
            target_token_ids = (
                torch.arange(num_rows, device=device, dtype=torch.int64) * 997 + 13
            ) % args.vocab_size
            for top_k in args.top_k:
                result = run_case(
                    processor,
                    lm_head,
                    hidden_states,
                    target_token_ids,
                    top_k,
                    args.warmup,
                    args.repeats,
                    baseline_warmup,
                    baseline_repeats,
                    args.skip_baseline,
                    world_size,
                )
                results.append(result)
                if rank == 0:
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)

        prompt_results: list[PromptResult] = []
        for prompt_length in args.prompt_lengths:
            prompt_generator = torch.Generator(device=device).manual_seed(
                args.seed + 2000 + prompt_length
            )
            prompt_hidden_states = torch.randn(
                (prompt_length, args.hidden_size),
                dtype=torch.bfloat16,
                device=device,
                generator=prompt_generator,
            )
            prompt_target_token_ids = (
                torch.arange(prompt_length, device=device, dtype=torch.int64) * 997 + 13
            ) % args.vocab_size
            for top_k in args.top_k:
                prompt_result = run_prompt_case(
                    processor,
                    lm_head,
                    prompt_hidden_states,
                    prompt_target_token_ids,
                    top_k,
                    args.warmup,
                    args.repeats,
                    world_size,
                )
                prompt_results.append(prompt_result)
                if rank == 0:
                    print(
                        json.dumps(asdict(prompt_result), sort_keys=True),
                        flush=True,
                    )

        if rank == 0 and args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                json.dump(
                    {
                        "environment": environment,
                        "config": vars(args),
                        "kernel_results": [asdict(result) for result in results],
                        "prompt_results": [asdict(result) for result in prompt_results],
                    },
                    output_file,
                    indent=2,
                )

        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
