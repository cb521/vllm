# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np

from benchmarks.benchmark_prompt_logprobs_e2e import (
    compare_artifacts,
    compare_variant_pair,
    top_k_equivalent_rows,
)


def test_top_k_equivalence_allows_equal_score_token_substitution() -> None:
    equivalent = top_k_equivalent_rows(
        baseline_ids=np.array([[[1, 2]]]),
        baseline_scores=np.array([[[-0.1, -0.2]]]),
        optimized_ids=np.array([[[1, 3]]]),
        optimized_scores=np.array([[[-0.1, -0.2]]]),
    )

    assert equivalent.tolist() == [[True]]


def test_top_k_equivalence_rejects_score_change() -> None:
    equivalent = top_k_equivalent_rows(
        baseline_ids=np.array([[[1, 2]]]),
        baseline_scores=np.array([[[-0.1, -0.2]]]),
        optimized_ids=np.array([[[1, 3]]]),
        optimized_scores=np.array([[[-0.1, -0.3]]]),
    )

    assert equivalent.tolist() == [[False]]


def test_compare_artifacts_accepts_overlapping_tie_rank_ranges() -> None:
    baseline = {
        "prompt_token_ids": np.array([[7, 8]]),
        "target_scores": np.array([[-0.1]]),
        "target_ranks": np.array([[2]]),
        "target_tie_min_ranks": np.array([[2]]),
        "target_tie_max_ranks": np.array([[3]]),
        "top_token_ids": np.empty((1, 1, 0), dtype=np.int32),
        "top_scores": np.empty((1, 1, 0), dtype=np.float32),
        "generated_token_ids": np.array([9]),
    }
    optimized = {
        **baseline,
        "target_ranks": np.array([[3]]),
        "target_tie_min_ranks": np.array([[3]]),
        "target_tie_max_ranks": np.array([[4]]),
    }

    comparison = compare_artifacts(baseline, optimized)

    assert not comparison["target_ranks_match"]
    assert comparison["target_ranks_equivalent"]
    assert comparison["outputs_semantically_equivalent"]


def test_compare_variant_pair_reports_candidate_speedup(tmp_path: Path) -> None:
    case_name = "prompt-1024-batch-1-topk-0"
    artifact = {
        "prompt_token_ids": np.array([[7, 8]]),
        "target_scores": np.array([[-0.1]]),
        "target_ranks": np.array([[2]]),
        "top_token_ids": np.empty((1, 1, 0), dtype=np.int32),
        "top_scores": np.empty((1, 1, 0), dtype=np.float32),
        "generated_token_ids": np.array([9]),
    }
    common = {
        "git_revision": "revision",
        "model": "model",
        "model_revision": None,
        "model_dimensions": {"hidden_size": 4, "vocab_size": 8},
        "tensor_parallel_size": 2,
        "max_num_batched_tokens": 1024,
        "kv_cache_memory_gib": 1.0,
        "flat_logprobs": False,
        "enforce_eager": False,
        "language_model_only": False,
        "num_warmups": 1,
        "num_repeats": 3,
    }
    for variant, latency, peak in (
        ("native", 2.0, 100.0),
        ("h20", 1.0, 75.0),
    ):
        variant_dir = tmp_path / variant
        variant_dir.mkdir()
        np.savez_compressed(variant_dir / f"{case_name}.npz", **artifact)
        result = {
            **common,
            "variant": variant,
            "cases": [
                {
                    "name": case_name,
                    "prompt_length": 1024,
                    "batch_size": 1,
                    "top_k": 0,
                    "median_elapsed_seconds": latency,
                    "prompt_tokens_per_second": 1024 / latency,
                    "median_first_token_latency_seconds": latency,
                    "max_peak_used_mib_per_gpu": peak,
                    "artifact": f"{case_name}.npz",
                }
            ],
        }
        (tmp_path / f"{variant}.json").write_text(json.dumps(result))

    comparison = compare_variant_pair(tmp_path, "native", "h20")

    assert comparison["reference_variant"] == "native"
    assert comparison["candidate_variant"] == "h20"
    assert comparison["cases"][0]["speedup"] == 2.0
    assert comparison["cases"][0]["peak_used_memory_reduction_percent"] == 25.0
    assert comparison["cases"][0]["outputs_semantically_equivalent"]
