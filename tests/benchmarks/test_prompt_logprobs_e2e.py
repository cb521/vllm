# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from benchmarks.benchmark_prompt_logprobs_e2e import (
    compare_artifacts,
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
