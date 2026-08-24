# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark prompt logprobs with a real model, end to end.

The compact prompt-logprobs switch is read when vLLM workers start, so the
baseline and optimized variants run in separate subprocesses. Each worker
loads the model once, measures every requested case, and writes compact NumPy
artifacts for numerical comparison.

For example, on one eight-GPU node::

    python benchmarks/benchmark_prompt_logprobs_e2e.py \
        --model /models/Qwen3.5-27B \
        --tensor-parallel-size 2 \
        --prompt-lengths 1024 4096 8192 16384 32768 \
        --batch-sizes 1 --top-k 0 20 \
        --output-dir /tmp/prompt-logprobs-tp2

Use ``--profile-case 32768,1,20`` together with Nsight Systems and
``--capture-range=cudaProfilerApi`` to capture only that case.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_COMPACT_ENV = "VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS"
_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_GIB = 1024**3
_MIB = 1024**2
_SCORE_ATOL = 2e-5


@dataclass(frozen=True)
class Case:
    prompt_length: int
    batch_size: int
    top_k: int

    @property
    def name(self) -> str:
        return f"prompt-{self.prompt_length}-batch-{self.batch_size}-topk-{self.top_k}"


def parse_case(value: str) -> Case:
    try:
        prompt_length, batch_size, top_k = (int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "profile case must have the form PROMPT_LENGTH,BATCH_SIZE,TOP_K"
        ) from exc
    return Case(prompt_length, batch_size, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--top-k", type=int, nargs="+", default=[0, 20])
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--kv-cache-memory-gib",
        type=float,
        default=32.0,
        help=(
            "Fixed KV-cache memory per GPU. A fixed value keeps total-memory "
            "comparisons meaningful; increase it for larger batches, or use "
            "0 to let vLLM size the cache."
        ),
    )
    parser.add_argument("--memory-sample-interval", type=float, default=0.01)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--flat-logprobs", action="store_true")
    parser.add_argument("--profile-case", type=parse_case)
    parser.add_argument(
        "--allow-correctness-mismatch",
        action="store_true",
        help=(
            "Write and print comparisons without failing when baseline and "
            "optimized outputs are not semantically equivalent."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "optimized"),
        default=("baseline", "optimized"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--worker-variant",
        choices=("baseline", "optimized"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    positive_values = {
        "tensor_parallel_size": [args.tensor_parallel_size],
        "batch_sizes": args.batch_sizes,
        "num_repeats": [args.num_repeats],
        "max_num_batched_tokens": [args.max_num_batched_tokens],
    }
    for name, values in positive_values.items():
        if any(value <= 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} values must be positive")
    if any(value < 2 for value in args.prompt_lengths):
        parser.error("--prompt-lengths values must be at least 2")
    if args.num_warmups < 0:
        parser.error("--num-warmups must be non-negative")
    if any(not 0 <= value <= 32 for value in args.top_k):
        parser.error("--top-k values must be between 0 and 32")
    if args.kv_cache_memory_gib < 0:
        parser.error("--kv-cache-memory-gib must be non-negative")
    if args.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be positive")
    if args.profile_case is not None and args.profile_case not in make_cases(args):
        parser.error("--profile-case must also be present in the requested matrix")
    return args


def make_cases(args: argparse.Namespace) -> list[Case]:
    return [
        Case(prompt_length, batch_size, top_k)
        for prompt_length in args.prompt_lengths
        for batch_size in args.batch_sizes
        for top_k in args.top_k
    ]


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


class NvmlMemorySampler:
    def __init__(self, interval: float) -> None:
        from vllm.utils.import_utils import import_pynvml

        self._pynvml = import_pynvml()
        self._pynvml.nvmlInit()
        self._handles = self._get_visible_handles()
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[list[int]] = []

    def _get_visible_handles(self) -> list[Any]:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            identifiers = [item.strip() for item in visible_devices.split(",")]
            handles = []
            for identifier in identifiers:
                if identifier.isdigit():
                    handle = self._pynvml.nvmlDeviceGetHandleByIndex(int(identifier))
                elif identifier.startswith("GPU-"):
                    handle = self._pynvml.nvmlDeviceGetHandleByUUID(identifier)
                else:
                    raise ValueError(
                        f"unsupported CUDA_VISIBLE_DEVICES entry: {identifier!r}"
                    )
                handles.append(handle)
            return handles

        device_count = self._pynvml.nvmlDeviceGetCount()
        return [
            self._pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(device_count)
        ]

    @property
    def devices(self) -> list[dict[str, Any]]:
        devices = []
        for handle in self._handles:
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                {
                    "name": self._pynvml.nvmlDeviceGetName(handle),
                    "uuid": self._pynvml.nvmlDeviceGetUUID(handle),
                    "total_mib": memory.total / _MIB,
                }
            )
        return devices

    def _sample(self) -> None:
        self._samples.append(
            [
                self._pynvml.nvmlDeviceGetMemoryInfo(handle).used
                for handle in self._handles
            ]
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory sampler is already running")
        self._samples = []
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, list[float]]:
        if self._thread is None:
            raise RuntimeError("memory sampler is not running")
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._sample()
        samples = np.asarray(self._samples, dtype=np.float64) / _MIB
        return {
            "start_used_mib": samples[0].tolist(),
            "peak_used_mib": samples.max(axis=0).tolist(),
            "incremental_peak_mib": (samples.max(axis=0) - samples[0]).tolist(),
        }


def make_prompts(
    case: Case,
    seed: int,
    vocab_size: int,
) -> tuple[list[Any], np.ndarray]:
    from vllm.inputs import TokensPrompt

    seed_sequence = np.random.SeedSequence([seed, case.prompt_length, case.batch_size])
    rng = np.random.default_rng(seed_sequence)
    upper_bound = min(vocab_size, 10_000)
    if upper_bound <= 100:
        raise ValueError(f"vocabulary is unexpectedly small: {vocab_size}")
    token_ids = rng.integers(
        100,
        upper_bound,
        size=(case.batch_size, case.prompt_length),
        dtype=np.int64,
    )
    prompts = [
        TokensPrompt(prompt_token_ids=request_token_ids.tolist())
        for request_token_ids in token_ids
    ]
    return prompts, token_ids


def extract_artifact(
    outputs: list[Any],
    prompt_token_ids: np.ndarray,
    top_k: int,
    artifact_path: Path,
) -> None:
    batch_size, prompt_length = prompt_token_ids.shape
    target_scores = np.empty((batch_size, prompt_length - 1), dtype=np.float32)
    target_ranks = np.empty((batch_size, prompt_length - 1), dtype=np.int32)
    target_tie_min_ranks = np.empty_like(target_ranks)
    target_tie_max_ranks = np.empty_like(target_ranks)
    top_token_ids = np.empty((batch_size, prompt_length - 1, top_k), dtype=np.int32)
    top_scores = np.empty((batch_size, prompt_length - 1, top_k), dtype=np.float32)
    generated_token_ids = np.empty(batch_size, dtype=np.int32)

    for batch_index, output in enumerate(outputs):
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != prompt_length:
            raise RuntimeError("vLLM returned incomplete prompt logprobs")
        generated_token_ids[batch_index] = output.outputs[0].token_ids[0]
        for position in range(1, prompt_length):
            entries = prompt_logprobs[position]
            if entries is None:
                raise RuntimeError(f"missing prompt logprobs at position {position}")
            target_token_id = int(prompt_token_ids[batch_index, position])
            target = entries[target_token_id]
            if target.rank is None:
                raise RuntimeError("target token rank is missing")
            target_scores[batch_index, position - 1] = target.logprob
            target_ranks[batch_index, position - 1] = target.rank
            # The public dict writes the target first and Top-K entries after
            # it. If the target is itself in Top-K, its ordinal can therefore
            # replace the selected-token rank. Preserve the full equal-score
            # interval so differing, unspecified tie orders compare fairly.
            tied_ranks = [
                entry.rank
                for entry in entries.values()
                if entry.rank is not None and entry.logprob == target.logprob
            ]
            target_tie_min_ranks[batch_index, position - 1] = min(tied_ranks)
            target_tie_max_ranks[batch_index, position - 1] = max(tied_ranks)

            candidates = sorted(
                (
                    entry.rank,
                    token_id,
                    entry.logprob,
                )
                for token_id, entry in entries.items()
                if entry.rank is not None and entry.rank <= top_k
            )
            if len(candidates) != top_k:
                raise RuntimeError(
                    f"expected {top_k} Top-K entries, received {len(candidates)}"
                )
            for candidate_index, (_, token_id, score) in enumerate(candidates):
                top_token_ids[batch_index, position - 1, candidate_index] = token_id
                top_scores[batch_index, position - 1, candidate_index] = score

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        prompt_token_ids=prompt_token_ids.astype(np.int32),
        target_scores=target_scores,
        target_ranks=target_ranks,
        target_tie_min_ranks=target_tie_min_ranks,
        target_tie_max_ranks=target_tie_max_ranks,
        top_token_ids=top_token_ids,
        top_scores=top_scores,
        generated_token_ids=generated_token_ids,
    )


def first_token_latencies(outputs: list[Any]) -> list[float]:
    latencies = []
    for output in outputs:
        if output.metrics is None:
            raise RuntimeError("request metrics are disabled")
        latencies.append(output.metrics.first_token_latency)
    return latencies


def run_case(
    llm: Any,
    case: Case,
    args: argparse.Namespace,
    variant_dir: Path,
    memory_sampler: NvmlMemorySampler,
    vocab_size: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompts, prompt_token_ids = make_prompts(case, args.seed, vocab_size)
    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=1,
        prompt_logprobs=case.top_k,
        flat_logprobs=args.flat_logprobs,
        detokenize=False,
        seed=args.seed,
    )

    for _ in range(args.num_warmups):
        warmup_outputs = llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        del warmup_outputs
    gc.collect()

    elapsed_seconds = []
    ttft_seconds = []
    memory_samples = []
    artifact_path = variant_dir / f"{case.name}.npz"
    for repeat in range(args.num_repeats):
        gc.collect()
        memory_sampler.start()
        try:
            start = time.perf_counter()
            outputs = llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
            elapsed = time.perf_counter() - start
        finally:
            memory = memory_sampler.stop()
        elapsed_seconds.append(elapsed)
        ttft_seconds.append(first_token_latencies(outputs))
        memory_samples.append(memory)
        if repeat == 0:
            extract_artifact(outputs, prompt_token_ids, case.top_k, artifact_path)
        del outputs

    if args.profile_case == case:
        profile_outputs = None
        llm.start_profile()
        try:
            profile_outputs = llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
        finally:
            llm.stop_profile()
        if profile_outputs is not None:
            del profile_outputs

    median_elapsed = statistics.median(elapsed_seconds)
    flattened_ttft = [value for repeat in ttft_seconds for value in repeat]
    total_prompt_tokens = case.prompt_length * case.batch_size
    peak_used_mib = max(max(sample["peak_used_mib"]) for sample in memory_samples)
    print(
        f"[{args.worker_variant}] {case.name}: {median_elapsed * 1000:.2f} ms, "
        f"{total_prompt_tokens / median_elapsed:.1f} prompt tokens/s, "
        f"TTFT {statistics.median(flattened_ttft) * 1000:.2f} ms, "
        f"peak {peak_used_mib:.1f} MiB/GPU",
        flush=True,
    )
    return {
        **asdict(case),
        "name": case.name,
        "elapsed_seconds": elapsed_seconds,
        "median_elapsed_seconds": median_elapsed,
        "prompt_tokens_per_second": total_prompt_tokens / median_elapsed,
        "first_token_latencies_seconds": ttft_seconds,
        "median_first_token_latency_seconds": statistics.median(flattened_ttft),
        "memory": memory_samples,
        "max_peak_used_mib_per_gpu": peak_used_mib,
        "artifact": artifact_path.name,
    }


def model_dimensions(model_config: Any) -> dict[str, Any]:
    text_config = model_config.hf_text_config
    architectures = getattr(model_config, "architectures", ())
    return {
        "architecture": architectures[0] if architectures else None,
        "hidden_size": getattr(text_config, "hidden_size", None),
        "vocab_size": model_config.get_vocab_size(),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "dtype": str(model_config.dtype),
        "max_model_len": model_config.max_model_len,
    }


def run_worker(args: argparse.Namespace) -> None:
    os.environ[_MODEL_RUNNER_ENV] = "1"
    expected_enabled = args.worker_variant == "optimized"
    actual_enabled = bool(int(os.environ.get(_COMPACT_ENV, "0")))
    if actual_enabled != expected_enabled:
        raise RuntimeError(
            f"{_COMPACT_ENV}={int(actual_enabled)} does not match worker variant "
            f"{args.worker_variant!r}"
        )

    import torch

    import vllm
    from vllm import LLM

    cases = make_cases(args)
    max_model_len = max(case.prompt_length for case in cases) + 1
    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "runner": "generate",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "seed": args.seed,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": max(args.batch_sizes),
        "max_logprobs": max(args.top_k),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
        "language_model_only": args.language_model_only,
        "trust_remote_code": args.trust_remote_code,
        "logprobs_mode": "raw_logprobs",
    }
    if args.kv_cache_memory_gib:
        engine_kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_memory_gib * _GIB)
    if args.profile_case is not None:
        engine_kwargs["profiler_config"] = {"profiler": "cuda"}

    llm = LLM(**engine_kwargs)
    if not llm.llm_engine.vllm_config.use_v2_model_runner:
        raise RuntimeError("this benchmark requires Model Runner V2")
    memory_sampler = NvmlMemorySampler(args.memory_sample_interval)
    if len(memory_sampler.devices) != args.tensor_parallel_size:
        raise RuntimeError(
            f"expected {args.tensor_parallel_size} visible GPUs, found "
            f"{len(memory_sampler.devices)}"
        )
    model_config = llm.llm_engine.model_config
    dimensions = model_dimensions(model_config)
    vocab_size = dimensions["vocab_size"]
    variant_dir = args.output_dir / args.worker_variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            llm,
            case,
            args,
            variant_dir,
            memory_sampler,
            vocab_size,
        )
        for case in cases
    ]
    output = {
        "variant": args.worker_variant,
        "compact_prompt_logprobs": actual_enabled,
        "model_runner_v2": True,
        "git_revision": git_revision(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": args.model,
        "model_revision": args.revision,
        "model_dimensions": dimensions,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_gib": args.kv_cache_memory_gib,
        "flat_logprobs": args.flat_logprobs,
        "enforce_eager": args.enforce_eager,
        "language_model_only": args.language_model_only,
        "num_warmups": args.num_warmups,
        "num_repeats": args.num_repeats,
        "devices": memory_sampler.devices,
        "cases": results,
    }
    result_path = args.output_dir / f"{args.worker_variant}.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")


def load_artifact(result_dir: Path, case: dict[str, Any]) -> Any:
    return np.load(result_dir / case["artifact"])


def target_rank_range(artifact: Any) -> tuple[np.ndarray, np.ndarray]:
    if "target_tie_min_ranks" not in artifact:
        return artifact["target_ranks"], artifact["target_ranks"]
    return artifact["target_tie_min_ranks"], artifact["target_tie_max_ranks"]


def top_k_equivalent_rows(
    baseline_ids: np.ndarray,
    baseline_scores: np.ndarray,
    optimized_ids: np.ndarray,
    optimized_scores: np.ndarray,
) -> np.ndarray:
    """Compare Top-K rows while allowing equal-score token substitutions."""
    row_shape = baseline_ids.shape[:-1]
    width = baseline_ids.shape[-1]
    if width == 0:
        return np.ones(row_shape, dtype=np.bool_)

    equivalent = np.ones(int(np.prod(row_shape)), dtype=np.bool_)
    baseline_ids = baseline_ids.reshape(-1, width)
    baseline_scores = baseline_scores.reshape(-1, width)
    optimized_ids = optimized_ids.reshape(-1, width)
    optimized_scores = optimized_scores.reshape(-1, width)
    for row in range(equivalent.size):
        baseline_by_id = dict(
            zip(baseline_ids[row].tolist(), baseline_scores[row].tolist())
        )
        optimized_by_id = dict(
            zip(optimized_ids[row].tolist(), optimized_scores[row].tolist())
        )
        common_ids = baseline_by_id.keys() & optimized_by_id.keys()
        if any(
            abs(baseline_by_id[token_id] - optimized_by_id[token_id]) > _SCORE_ATOL
            for token_id in common_ids
        ):
            equivalent[row] = False
            continue

        baseline_only = sorted(
            baseline_by_id[token_id] for token_id in baseline_by_id.keys() - common_ids
        )
        optimized_only = sorted(
            optimized_by_id[token_id]
            for token_id in optimized_by_id.keys() - common_ids
        )
        equivalent[row] = len(baseline_only) == len(optimized_only) and np.allclose(
            baseline_only,
            optimized_only,
            rtol=0.0,
            atol=_SCORE_ATOL,
        )
    return equivalent.reshape(row_shape)


def compare_artifacts(baseline: Any, optimized: Any) -> dict[str, Any]:
    if not np.array_equal(baseline["prompt_token_ids"], optimized["prompt_token_ids"]):
        raise RuntimeError("baseline and optimized prompts differ")

    target_errors = np.abs(
        baseline["target_scores"].astype(np.float64)
        - optimized["target_scores"].astype(np.float64)
    )
    target_scores_close = target_errors <= _SCORE_ATOL
    target_rank_matches = baseline["target_ranks"] == optimized["target_ranks"]
    baseline_rank_min, baseline_rank_max = target_rank_range(baseline)
    optimized_rank_min, optimized_rank_max = target_rank_range(optimized)
    target_rank_equivalent = np.maximum(
        baseline_rank_min, optimized_rank_min
    ) <= np.minimum(baseline_rank_max, optimized_rank_max)
    generated_matches = (
        baseline["generated_token_ids"] == optimized["generated_token_ids"]
    )
    top_id_matches = baseline["top_token_ids"] == optimized["top_token_ids"]
    top_set_matches = np.all(
        np.sort(baseline["top_token_ids"], axis=-1)
        == np.sort(optimized["top_token_ids"], axis=-1),
        axis=-1,
    )
    top_k_equivalent = top_k_equivalent_rows(
        baseline["top_token_ids"],
        baseline["top_scores"],
        optimized["top_token_ids"],
        optimized["top_scores"],
    )
    top_score_errors = np.abs(
        baseline["top_scores"].astype(np.float64)
        - optimized["top_scores"].astype(np.float64)
    )
    outputs_equivalent = (
        target_scores_close.all()
        and target_rank_equivalent.all()
        and generated_matches.all()
        and top_k_equivalent.all()
    )
    return {
        "target_max_abs_error": float(target_errors.max(initial=0.0)),
        "target_mean_abs_error": float(target_errors.mean()),
        "target_score_close_fraction": float(target_scores_close.mean()),
        "target_scores_close": bool(target_scores_close.all()),
        "target_rank_match_fraction": float(target_rank_matches.mean()),
        "target_ranks_match": bool(target_rank_matches.all()),
        "target_rank_equivalent_fraction": float(target_rank_equivalent.mean()),
        "target_ranks_equivalent": bool(target_rank_equivalent.all()),
        "generated_token_match_fraction": float(generated_matches.mean()),
        "generated_tokens_match": bool(generated_matches.all()),
        "top_token_id_match_fraction": float(top_id_matches.mean())
        if top_id_matches.size
        else 1.0,
        "top_token_ids_match": bool(top_id_matches.all()),
        "top_token_set_match_fraction": float(top_set_matches.mean()),
        "top_token_sets_match": bool(top_set_matches.all()),
        "top_k_equivalent_fraction": float(top_k_equivalent.mean()),
        "top_k_equivalent": bool(top_k_equivalent.all()),
        "top_rank_score_max_abs_error": float(top_score_errors.max(initial=0.0)),
        "top_rank_score_mean_abs_error": float(top_score_errors.mean())
        if top_score_errors.size
        else 0.0,
        "score_equivalence_atol": _SCORE_ATOL,
        "outputs_semantically_equivalent": bool(outputs_equivalent),
    }


def compare_variants(output_dir: Path) -> dict[str, Any]:
    baseline = json.loads((output_dir / "baseline.json").read_text())
    optimized = json.loads((output_dir / "optimized.json").read_text())
    comparable_fields = (
        "git_revision",
        "model",
        "model_revision",
        "model_dimensions",
        "tensor_parallel_size",
        "max_num_batched_tokens",
        "kv_cache_memory_gib",
        "flat_logprobs",
        "enforce_eager",
        "language_model_only",
        "num_warmups",
        "num_repeats",
    )
    for field in comparable_fields:
        if baseline[field] != optimized[field]:
            raise RuntimeError(f"baseline and optimized {field} differ")
    baseline_cases = {case["name"]: case for case in baseline["cases"]}
    optimized_cases = {case["name"]: case for case in optimized["cases"]}
    if baseline_cases.keys() != optimized_cases.keys():
        raise RuntimeError("baseline and optimized case matrices differ")

    comparisons = []
    for name, baseline_case in baseline_cases.items():
        optimized_case = optimized_cases[name]
        with (
            load_artifact(output_dir / "baseline", baseline_case) as baseline_data,
            load_artifact(output_dir / "optimized", optimized_case) as optimized_data,
        ):
            correctness = compare_artifacts(baseline_data, optimized_data)
        baseline_latency = baseline_case["median_elapsed_seconds"]
        optimized_latency = optimized_case["median_elapsed_seconds"]
        baseline_ttft = baseline_case["median_first_token_latency_seconds"]
        optimized_ttft = optimized_case["median_first_token_latency_seconds"]
        total_prompt_tokens = (
            baseline_case["prompt_length"] * baseline_case["batch_size"]
        )
        baseline_peak = baseline_case["max_peak_used_mib_per_gpu"]
        optimized_peak = optimized_case["max_peak_used_mib_per_gpu"]
        comparisons.append(
            {
                "name": name,
                "prompt_length": baseline_case["prompt_length"],
                "batch_size": baseline_case["batch_size"],
                "top_k": baseline_case["top_k"],
                "baseline_median_ms": baseline_latency * 1000,
                "optimized_median_ms": optimized_latency * 1000,
                "speedup": baseline_latency / optimized_latency,
                "baseline_prompt_tokens_per_second": baseline_case[
                    "prompt_tokens_per_second"
                ],
                "optimized_prompt_tokens_per_second": optimized_case[
                    "prompt_tokens_per_second"
                ],
                "baseline_median_ttft_ms": baseline_ttft * 1000,
                "optimized_median_ttft_ms": optimized_ttft * 1000,
                "ttft_speedup": baseline_ttft / optimized_ttft,
                "baseline_engine_prompt_tokens_per_second": (
                    total_prompt_tokens / baseline_ttft
                ),
                "optimized_engine_prompt_tokens_per_second": (
                    total_prompt_tokens / optimized_ttft
                ),
                "baseline_frontend_output_ms": (baseline_latency - baseline_ttft)
                * 1000,
                "optimized_frontend_output_ms": (optimized_latency - optimized_ttft)
                * 1000,
                "baseline_peak_used_mib_per_gpu": baseline_peak,
                "optimized_peak_used_mib_per_gpu": optimized_peak,
                "peak_used_memory_reduction_percent": (
                    (baseline_peak - optimized_peak) / baseline_peak * 100
                ),
                **correctness,
            }
        )

    comparison = {
        "git_revision": baseline["git_revision"],
        "model": baseline["model"],
        "model_revision": baseline["model_revision"],
        "model_dimensions": baseline["model_dimensions"],
        "tensor_parallel_size": baseline["tensor_parallel_size"],
        "cases": comparisons,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def print_comparison(comparison: dict[str, Any]) -> None:
    print()
    print(
        "| Prompt | Batch | K | Baseline (ms) | Optimized (ms) | "
        "Speedup | TTFT speedup | Rank exact | Equivalent | Max target error |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |")
    for case in comparison["cases"]:
        rank_match = "yes" if case["target_ranks_match"] else "no"
        equivalent = "yes" if case["outputs_semantically_equivalent"] else "no"
        print(
            f"| {case['prompt_length']} | {case['batch_size']} | "
            f"{case['top_k']} | {case['baseline_median_ms']:.2f} | "
            f"{case['optimized_median_ms']:.2f} | {case['speedup']:.3f}x | "
            f"{case['ttft_speedup']:.3f}x | {rank_match} | {equivalent} | "
            f"{case['target_max_abs_error']:.3e} |"
        )


def run_parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_paths = [args.output_dir / f"{variant}.json" for variant in args.variants]
    existing = [path for path in result_paths if path.exists()]
    if existing and not args.overwrite:
        formatted_paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing results: {formatted_paths}; "
            "pass --overwrite to replace them"
        )

    for variant in args.variants:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-variant",
            variant,
        ]
        environment = os.environ.copy()
        environment[_MODEL_RUNNER_ENV] = "1"
        environment[_COMPACT_ENV] = "1" if variant == "optimized" else "0"
        subprocess.run(command, check=True, env=environment)

    if set(args.variants) == {"baseline", "optimized"}:
        comparison = compare_variants(args.output_dir)
        print_comparison(comparison)
        mismatches = [
            case["name"]
            for case in comparison["cases"]
            if not case["outputs_semantically_equivalent"]
        ]
        if mismatches and not args.allow_correctness_mismatch:
            formatted_cases = ", ".join(mismatches)
            raise RuntimeError(
                "baseline and optimized outputs are not semantically equivalent for: "
                f"{formatted_cases}"
            )


def main() -> None:
    args = parse_args()
    if args.worker_variant is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
