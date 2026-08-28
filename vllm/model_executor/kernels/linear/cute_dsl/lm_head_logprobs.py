# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact prompt-logprob kernels for unquantized BF16 LM heads."""

from __future__ import annotations

from typing import NamedTuple

import cuda.bindings.driver as cuda
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.cute_dsl._lm_head_logprobs_mainloop import (
    _K0_GROUP_N,
    _TILE_N,
    _TOPK_GROUP_N,
    _TOPK_PARTIAL_WIDTH,
    _compile_lm_head_logprobs,
    _validate_device_environment,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream


class LMHeadLogprobsOutput(NamedTuple):
    """Compact LM-head results for one tensor-parallel rank."""

    topk_values: torch.Tensor
    topk_ids: torch.Tensor
    lse: torch.Tensor
    rank_count: torch.Tensor


_TARGET_GEMM_BLOCK_M = 128
_SM80_LARGE_TARGET_GEMM_BLOCK_M = 1024
_SM80_LARGE_TARGET_GEMM_MIN_ROWS = 513


@triton.jit
def _pack_logit_token_keys(logits, token_ids, valid_mask):
    float_bits = logits.to(tl.uint32, bitcast=True)
    sign_bit = float_bits & 0x80000000
    ordered_bits = tl.where(
        sign_bit != 0,
        float_bits ^ 0xFFFFFFFF,
        float_bits ^ 0x80000000,
    )
    token_tie_break = token_ids.to(tl.uint32) ^ 0xFFFFFFFF
    keys = (ordered_bits.to(tl.uint64) << 32) | token_tie_break.to(tl.uint64)
    return tl.where(valid_mask, keys, 0)


@triton.jit
def _unpack_logit_token_keys(keys):
    ordered_bits = (keys >> 32).to(tl.uint32)
    was_nonnegative = (ordered_bits & 0x80000000) != 0
    float_bits = tl.where(
        was_nonnegative,
        ordered_bits ^ 0x80000000,
        ordered_bits ^ 0xFFFFFFFF,
    )
    values = float_bits.to(tl.float32, bitcast=True)
    token_ids = ((keys & 0xFFFFFFFF).to(tl.uint32) ^ 0xFFFFFFFF).to(tl.int32)
    valid = keys != 0
    return (
        tl.where(valid, values, -float("inf")),
        tl.where(valid, token_ids, -1),
    )


@triton.jit
def _materialized_lse_rank_kernel(
    logits_ptr,
    target_logits_ptr,
    local_lse_ptr,
    local_rank_count_ptr,
    logits_stride,
    valid_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    target_logit = tl.load(target_logits_ptr + row)

    running_max = float("-inf")
    running_sum_exp = 0.0
    rank_count = 0
    for start in range(0, valid_vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < valid_vocab_size
        values = tl.load(
            row_ptr + offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        block_max = tl.max(values)
        next_max = tl.maximum(running_max, block_max)
        running_sum_exp = running_sum_exp * tl.exp(running_max - next_max) + tl.sum(
            tl.exp(values - next_max)
        )
        running_max = next_max
        rank_count += tl.sum(tl.where(mask & (values >= target_logit), 1, 0))

    tl.store(local_lse_ptr + row, running_max + tl.log(running_sum_exp))
    tl.store(local_rank_count_ptr + row, rank_count)


@triton.jit
def _merge_lse_rank_partials_kernel(
    partial_max_ptr,
    partial_sum_exp_ptr,
    partial_rank_count_ptr,
    local_lse_ptr,
    local_rank_count_ptr,
    num_rows,
    num_partials,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_S)
    mask = offsets < num_partials
    state_offsets = row * num_partials + offsets
    partial_max = tl.load(
        partial_max_ptr + state_offsets,
        mask=mask,
        other=-float("inf"),
    )
    partial_sum_exp = tl.load(
        partial_sum_exp_ptr + state_offsets,
        mask=mask,
        other=0.0,
    )
    partial_rank_count = tl.load(
        partial_rank_count_ptr + state_offsets,
        mask=mask,
        other=0,
    )
    local_max = tl.max(partial_max, axis=0)
    local_sum_exp = tl.sum(partial_sum_exp * tl.exp(partial_max - local_max), axis=0)
    tl.store(local_lse_ptr + row, local_max + tl.log(local_sum_exp))
    tl.store(local_rank_count_ptr + row, tl.sum(partial_rank_count, axis=0))


@triton.jit
def _merge_topk_partials_kernel(
    input_values_ptr,
    input_ids_ptr,
    output_values_ptr,
    output_ids_ptr,
    num_rows,
    num_input_groups,
    num_output_groups,
    INPUTS_PER_OUTPUT: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    TOPK_BUCKET: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    output_group = tl.program_id(1)
    candidate_offsets = tl.arange(0, BLOCK_CANDIDATES)
    group_in_output = candidate_offsets // INPUT_WIDTH
    candidate_in_group = candidate_offsets % INPUT_WIDTH
    input_group = output_group * INPUTS_PER_OUTPUT + group_in_output
    mask = (group_in_output < INPUTS_PER_OUTPUT) & (input_group < num_input_groups)
    input_offsets = (
        row * num_input_groups + input_group
    ) * INPUT_WIDTH + candidate_in_group
    values = tl.load(input_values_ptr + input_offsets, mask=mask, other=-float("inf"))
    ids = tl.load(input_ids_ptr + input_offsets, mask=mask, other=0)
    keys = _pack_logit_token_keys(values, ids, mask)
    topk_keys = tl.max(keys, axis=0) if TOPK_BUCKET == 1 else tl.topk(keys, TOPK_BUCKET)
    topk_values, topk_ids = _unpack_logit_token_keys(topk_keys)
    output_indices = tl.arange(0, TOPK_BUCKET)
    output_offsets = (
        row * num_output_groups + output_group
    ) * OUTPUT_WIDTH + output_indices
    output_mask = output_indices < OUTPUT_WIDTH
    tl.store(output_values_ptr + output_offsets, topk_values, mask=output_mask)
    tl.store(output_ids_ptr + output_offsets, topk_ids, mask=output_mask)


@triton.jit
def _merge_tp_prompt_logprobs_kernel(
    tp_topk_values_ptr,  # [TP, M, K] FP32 logits
    tp_topk_ids_ptr,  # [TP, M, K] INT32 global token IDs
    tp_local_lse_ptr,  # [TP, M] FP32
    tp_rank_count_ptr,  # [TP, M] INT32
    target_token_ids_ptr,  # [M]
    target_logits_ptr,  # [M] FP32
    output_token_ids_ptr,  # [M, K + 1] INT32
    output_logprobs_ptr,  # [M, K + 1] FP32
    output_ranks_ptr,  # [M] INT32
    num_rows,
    topk_value_stride_tp,
    topk_value_stride_m,
    topk_value_stride_k,
    topk_id_stride_tp,
    topk_id_stride_m,
    topk_id_stride_k,
    lse_stride_tp,
    lse_stride_m,
    rank_stride_tp,
    rank_stride_m,
    target_id_stride_m,
    target_logit_stride_m,
    BLOCK_M: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    TOPK_BUCKET: tl.constexpr,
    NUM_TOPK: tl.constexpr,
    TP_SIZE: tl.constexpr,
):
    # Each program independently merges a row block across all TP ranks.
    row_offsets = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    row_mask = row_offsets < num_rows
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum_exp = tl.zeros((BLOCK_M,), tl.float32)
    running_rank_count = tl.zeros((BLOCK_M,), tl.int32)
    if TOPK_BUCKET > 0:
        running_topk_keys = tl.full((BLOCK_M, TOPK_BUCKET), 0, tl.uint64)
        topk_offsets = tl.arange(0, TOPK_BUCKET).to(tl.int64)

    # Local LSE values merge with logaddexp; rank counts are additive because
    # TP shards partition the vocabulary.
    for tp_rank in range(TP_SIZE):
        local_lse = tl.load(
            tp_local_lse_ptr + tp_rank * lse_stride_tp + row_offsets * lse_stride_m,
            mask=row_mask,
            other=-float("inf"),
        ).to(tl.float32)
        local_rank_count = tl.load(
            tp_rank_count_ptr + tp_rank * rank_stride_tp + row_offsets * rank_stride_m,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        merged_max = tl.maximum(running_max, local_lse)
        running_sum_exp = running_sum_exp * tl.exp(running_max - merged_max) + tl.exp(
            local_lse - merged_max
        )
        running_max = merged_max
        running_rank_count += local_rank_count

        if TOPK_BUCKET > 0:
            # A shard's local top-K contains every candidate that can enter
            # the global top-K, so the full vocabulary is never gathered.
            value_ptrs = (
                tp_topk_values_ptr
                + tp_rank * topk_value_stride_tp
                + row_offsets[:, None] * topk_value_stride_m
                + topk_offsets[None, :] * topk_value_stride_k
            )
            id_ptrs = (
                tp_topk_ids_ptr
                + tp_rank * topk_id_stride_tp
                + row_offsets[:, None] * topk_id_stride_m
                + topk_offsets[None, :] * topk_id_stride_k
            )
            topk_mask = row_mask[:, None] & (topk_offsets[None, :] < TOPK_WIDTH)
            local_values = tl.load(
                value_ptrs,
                mask=topk_mask,
                other=-float("inf"),
            ).to(tl.float32)
            local_ids = tl.load(
                id_ptrs,
                mask=topk_mask,
                other=-1,
            ).to(tl.int32)
            local_keys = _pack_logit_token_keys(
                local_values,
                local_ids,
                topk_mask & (local_ids >= 0),
            )
            running_topk_keys = tl.topk(
                tl.cat(running_topk_keys, local_keys, dim=1),
                TOPK_BUCKET,
                dim=1,
            )

    global_lse = running_max + tl.log(running_sum_exp)

    # Column zero always contains the target, even when it is not in top-K.
    target_token_ids = tl.load(
        target_token_ids_ptr + row_offsets * target_id_stride_m,
        mask=row_mask,
        other=-1,
    ).to(tl.int32)
    target_logits = tl.load(
        target_logits_ptr + row_offsets * target_logit_stride_m,
        mask=row_mask,
        other=-float("inf"),
    ).to(tl.float32)
    output_width = NUM_TOPK + 1
    output_row_offsets = row_offsets * output_width
    tl.store(
        output_token_ids_ptr + output_row_offsets,
        target_token_ids,
        mask=row_mask,
    )
    tl.store(
        output_logprobs_ptr + output_row_offsets,
        target_logits - global_lse,
        mask=row_mask,
    )
    tl.store(
        output_ranks_ptr + row_offsets,
        running_rank_count,
        mask=row_mask,
    )

    if NUM_TOPK > 0:
        # Remaining columns contain global top-K normalized by the same LSE.
        topk_values, topk_ids = _unpack_logit_token_keys(running_topk_keys)
        output_offsets = output_row_offsets[:, None] + 1 + topk_offsets[None, :]
        output_mask = row_mask[:, None] & (topk_offsets[None, :] < NUM_TOPK)
        tl.store(
            output_token_ids_ptr + output_offsets,
            topk_ids,
            mask=output_mask,
        )
        tl.store(
            output_logprobs_ptr + output_offsets,
            topk_values - global_lse[:, None],
            mask=output_mask,
        )


def prompt_target_logits(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    local_target_ids: torch.Tensor,
    *,
    block_m: int | None = None,
) -> torch.Tensor:
    """Compute BF16 target logits with the platform GEMM accumulation path.

    The fused LM-head paths consume BF16 logits, while a standalone FP32 dot
    product can round differently from the platform BF16 LM-head projection.
    Grouping 128 rows into small square GEMMs keeps the target-only work compact
    and follows the same tensor-core accumulation contract as the materialized
    projection. ``block_m`` can match an architecture's preferred platform-GEMM
    row shape. Non-local target IDs contribute zero before the TP SUM all-reduce.
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [M, H]")
    if lm_head_weight.ndim != 2:
        raise ValueError("lm_head_weight must have shape [V_local, H]")
    if hidden_states.shape[1] != lm_head_weight.shape[1]:
        raise ValueError(
            "hidden_states and lm_head_weight hidden dimensions must match"
        )
    if local_target_ids.shape != (hidden_states.shape[0],):
        raise ValueError("local_target_ids must have shape [M]")
    if local_target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("local_target_ids must use torch.int32 or torch.int64")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must use bfloat16")
    if lm_head_weight.dtype != torch.bfloat16:
        raise TypeError("lm_head_weight must use bfloat16")
    if block_m is not None and block_m <= 0:
        raise ValueError("block_m must be positive")
    if not hidden_states.is_cuda or not lm_head_weight.is_cuda:
        raise ValueError("hidden_states and lm_head_weight must be CUDA tensors")
    if hidden_states.device != lm_head_weight.device:
        raise ValueError("hidden_states and lm_head_weight must share a device")
    if local_target_ids.device != hidden_states.device:
        raise ValueError("local_target_ids must share the input device")

    num_rows, hidden_size = hidden_states.shape
    if num_rows == 0:
        return torch.empty(0, dtype=torch.float32, device=hidden_states.device)
    local_vocab_size = lm_head_weight.shape[0]
    if hidden_size == 0 or local_vocab_size == 0:
        return torch.zeros(num_rows, dtype=torch.float32, device=hidden_states.device)

    if block_m is None:
        block_m = _TARGET_GEMM_BLOCK_M
        device_index = hidden_states.device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        capability = current_platform.get_device_capability(device_index)
        if (
            capability is not None
            and capability.to_int() == 80
            and num_rows >= _SM80_LARGE_TARGET_GEMM_MIN_ROWS
        ):
            block_m = _SM80_LARGE_TARGET_GEMM_BLOCK_M

    is_local = (local_target_ids >= 0) & (local_target_ids < local_vocab_size)
    safe_target_ids = torch.where(is_local, local_target_ids, 0).to(torch.int64)
    padded_rows = ((num_rows + block_m - 1) // block_m) * block_m
    if padded_rows != num_rows:
        row_padding = padded_rows - num_rows
        hidden_states = F.pad(hidden_states, (0, 0, 0, row_padding))
        safe_target_ids = F.pad(safe_target_ids, (0, row_padding))

    num_blocks = padded_rows // block_m
    selected_weight = lm_head_weight.index_select(0, safe_target_ids).view(
        num_blocks,
        block_m,
        hidden_size,
    )
    block_logits = torch.bmm(
        hidden_states.view(num_blocks, block_m, hidden_size),
        selected_weight.transpose(1, 2),
    )
    target_logits = block_logits.diagonal(dim1=1, dim2=2).reshape(-1)[:num_rows]
    return torch.where(is_local, target_logits.float(), 0.0)


def materialized_lse_and_rank(
    local_logits: torch.Tensor,
    target_logits: torch.Tensor,
    valid_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce materialized local logits to LSE and target-rank statistics."""
    if local_logits.ndim != 2:
        raise ValueError("local_logits must have shape [M, V_local]")
    if target_logits.shape != (local_logits.shape[0],):
        raise ValueError("target_logits must have shape [M]")
    if local_logits.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("local_logits must use float16, bfloat16, or float32")
    if target_logits.dtype != torch.float32:
        raise TypeError("target_logits must use float32")
    if not local_logits.is_cuda or target_logits.device != local_logits.device:
        raise ValueError("inputs must be CUDA tensors on the same device")
    if local_logits.stride(-1) != 1:
        raise ValueError("the local vocabulary dimension must be contiguous")
    if not 0 <= valid_vocab_size <= local_logits.shape[1]:
        raise ValueError("valid_vocab_size must be in [0, V_local]")

    num_rows = local_logits.shape[0]
    local_lse = torch.empty(num_rows, dtype=torch.float32, device=local_logits.device)
    local_rank_count = torch.empty(
        num_rows, dtype=torch.int32, device=local_logits.device
    )
    if num_rows == 0:
        return local_lse, local_rank_count
    if valid_vocab_size == 0:
        local_lse.fill_(-float("inf"))
        local_rank_count.zero_()
        return local_lse, local_rank_count

    _materialized_lse_rank_kernel[(num_rows,)](
        local_logits,
        target_logits,
        local_lse,
        local_rank_count,
        local_logits.stride(0),
        valid_vocab_size,
        BLOCK_SIZE=1024,
        num_warps=8,
    )
    return local_lse, local_rank_count


def validate_lm_head_logprobs_environment(lm_head_weight: torch.Tensor) -> None:
    """Validate requirements that are fixed after the LM head is loaded."""
    if lm_head_weight.ndim != 2:
        raise ValueError("the LM-head weight must have shape [V_local, H]")
    if lm_head_weight.dtype != torch.bfloat16:
        raise TypeError("the LM-head weight must use BF16")
    if not lm_head_weight.is_cuda:
        raise RuntimeError("the compact prompt-logprobs path requires CUDA")
    if not lm_head_weight.is_contiguous():
        raise ValueError("the LM-head weight must be contiguous")
    if lm_head_weight.shape[0] == 0:
        raise ValueError("the local vocabulary must not be empty")
    if lm_head_weight.shape[1] % 8:
        raise ValueError("the hidden size must be divisible by 8")
    if lm_head_weight.data_ptr() % 16:
        raise ValueError("the LM-head weight must be 16-byte aligned")

    device = lm_head_weight.device
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    _validate_device_environment(device_index)


def uses_sm90_lm_head_logprobs(hidden_states: torch.Tensor) -> bool:
    """Return whether this input uses the Hopper WGMMA specialization."""
    if not hidden_states.is_cuda:
        return False
    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    capability = current_platform.get_device_capability(device_index)
    return (
        capability is not None
        and capability.to_int() == 90
        and hidden_states.shape[1] >= 128
    )


def uses_sm80_lm_head_logprobs(hidden_states: torch.Tensor) -> bool:
    """Return whether this input uses the A100 warp-MMA specialization."""
    if not hidden_states.is_cuda:
        return False
    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    capability = current_platform.get_device_capability(device_index)
    return capability is not None and capability.to_int() == 80


def lm_head_logprobs(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    local_target_ids: torch.Tensor,
    global_target_logits: torch.Tensor,
    num_topk: int,
    *,
    valid_vocab_size: int | None = None,
    global_vocab_start: int = 0,
) -> LMHeadLogprobsOutput:
    """Compute TP-local logit statistics without materializing full logits.

    Args:
        hidden_states: Prompt hidden states with shape ``[M, H]`` in BF16.
        lm_head_weight: Local LM-head shard with shape ``[V_local, H]`` in
            BF16.
        local_target_ids: Shard-local target IDs with shape ``[M]``. A negative
            ID denotes a target owned by another TP rank.
        global_target_logits: TP-reduced target logits with shape ``[M]`` in
            FP32.
        num_topk: Number of local top tokens to return, in ``[0, 32]``.
        valid_vocab_size: Number of non-padding rows in the local LM-head shard.
        global_vocab_start: Global token ID corresponding to local row zero.

    Returns:
        Compact local top-K values and IDs, LSE, and target-rank counts.

    Notes:
        For positive K, the CuTe kernel always emits 32 candidates per partial
        and the local Triton merge immediately reduces them to the requested K.
        With ``P = ceil(ceil(V_local / 256) / group_n)``, the principal
        workspace buffers occupy approximately
        ``M * (12P + 256P + 8K * ceil(P / 16) + 8K + 8)`` bytes before allocator
        and communication temporaries. Large prompt chunks or vocabularies can
        therefore still cause OOM even though this path does not materialize
        the full ``[M, V]`` logits tensor.
    """
    if not 0 <= num_topk <= 32:
        raise ValueError("num_topk must be in [0, 32]")
    topk_bucket = 0 if num_topk == 0 else 1 << (num_topk - 1).bit_length()
    if hidden_states.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("hidden_states and lm_head_weight must be matrices")
    if hidden_states.shape[1] != lm_head_weight.shape[1]:
        raise ValueError("hidden dimensions must match")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must use torch.bfloat16")
    if lm_head_weight.dtype != torch.bfloat16:
        raise TypeError("lm_head_weight must use torch.bfloat16")
    if not hidden_states.is_cuda or not lm_head_weight.is_cuda:
        raise ValueError("hidden_states and lm_head_weight must be CUDA tensors")
    if hidden_states.device != lm_head_weight.device:
        raise ValueError("hidden_states and lm_head_weight must share a device")
    if not hidden_states.is_contiguous() or not lm_head_weight.is_contiguous():
        raise ValueError("hidden_states and lm_head_weight must be contiguous")
    if hidden_states.shape[1] % 8:
        raise ValueError("hidden size must be divisible by 8 for 128-bit copies")
    if local_target_ids.shape != (hidden_states.shape[0],):
        raise ValueError("local_target_ids must have shape [M]")
    if global_target_logits.shape != (hidden_states.shape[0],):
        raise ValueError("global_target_logits must have shape [M]")
    if global_target_logits.dtype != torch.float32:
        raise TypeError("global_target_logits must use torch.float32")
    if local_target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("local_target_ids must use torch.int32 or torch.int64")
    if local_target_ids.device != hidden_states.device:
        raise ValueError("local_target_ids must share the input device")
    if global_target_logits.device != hidden_states.device:
        raise ValueError("global_target_logits must share the input device")
    if hidden_states.data_ptr() % 16:
        raise ValueError("hidden_states must be 16-byte aligned")
    if global_vocab_start < 0:
        raise ValueError("global_vocab_start must be non-negative")

    local_vocab_size = lm_head_weight.shape[0]
    if valid_vocab_size is None:
        valid_vocab_size = local_vocab_size
    if not 0 < valid_vocab_size <= local_vocab_size:
        raise ValueError("valid_vocab_size must be in (0, V_local]")
    num_rows = hidden_states.shape[0]
    if num_rows == 0:
        empty_values = torch.empty(
            (0, num_topk), dtype=torch.float32, device=hidden_states.device
        )
        empty_ids = torch.empty(
            (0, num_topk), dtype=torch.int32, device=hidden_states.device
        )
        return LMHeadLogprobsOutput(
            empty_values,
            empty_ids,
            torch.empty(0, dtype=torch.float32, device=hidden_states.device),
            torch.empty(0, dtype=torch.int32, device=hidden_states.device),
        )

    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    # The three-stage Hopper pipeline needs at least two 64-wide K tiles.
    # Tiny synthetic heads use the original warp-MMA specialization instead
    # of carrying a dynamic short-pipeline branch in the production hot path.
    use_sm90 = uses_sm90_lm_head_logprobs(hidden_states)
    use_sm80 = not use_sm90 and uses_sm80_lm_head_logprobs(hidden_states)
    if num_topk == 0:
        partial_topk_width = 0
        group_n = _K0_GROUP_N
    else:
        partial_topk_width = num_topk if use_sm80 else _TOPK_PARTIAL_WIDTH
        group_n = _TOPK_GROUP_N

    target_ids = local_target_ids.to(dtype=torch.int32).contiguous()
    target_logits = global_target_logits.contiguous()
    # A CTA emits one partial for group_n adjacent vocabulary blocks.
    num_vocab_blocks = triton.cdiv(valid_vocab_size, _TILE_N)
    num_partials = triton.cdiv(num_vocab_blocks, group_n)
    partial_max = torch.empty(
        (num_rows, num_partials), dtype=torch.float32, device=hidden_states.device
    )
    partial_sum_exp = torch.empty_like(partial_max)
    partial_rank_count = torch.empty(
        (num_rows, num_partials), dtype=torch.int32, device=hidden_states.device
    )
    partial_topk_values = torch.empty(
        (num_rows, num_partials, partial_topk_width),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    partial_topk_ids = torch.empty(
        (num_rows, num_partials, partial_topk_width),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    local_lse = torch.empty(num_rows, dtype=torch.float32, device=hidden_states.device)
    local_rank_count = torch.empty(
        num_rows, dtype=torch.int32, device=hidden_states.device
    )

    # Static H/V/K dimensions select a cached CuTe specialization; M remains
    # symbolic so chunk lengths do not trigger recompilation.
    with torch.accelerator.device_index(device_index):
        compiled = _compile_lm_head_logprobs(
            hidden_states.shape[1],
            lm_head_weight.shape[0],
            num_vocab_blocks,
            device_index,
            partial_topk_width,
            group_n,
            use_sm80,
            use_sm90,
        )
        stream = cuda.CUstream(current_stream().cuda_stream)
        compiled(
            hidden_states,
            lm_head_weight,
            target_ids,
            target_logits,
            partial_max,
            partial_sum_exp,
            partial_rank_count,
            partial_max if partial_topk_width == 0 else partial_topk_values,
            partial_rank_count if partial_topk_width == 0 else partial_topk_ids,
            valid_vocab_size,
            global_vocab_start,
            0,
            num_vocab_blocks,
            stream,
        )

    # The CuTe kernel leaves only compact per-group states in GMEM.
    _merge_lse_rank_partials_kernel[(num_rows,)](
        partial_max,
        partial_sum_exp,
        partial_rank_count,
        local_lse,
        local_rank_count,
        num_rows,
        num_partials,
        BLOCK_S=triton.next_power_of_2(num_partials),
        num_warps=4,
    )
    if partial_topk_width == 0:
        local_topk_values = partial_topk_values.reshape(num_rows, 0)
        local_topk_ids = partial_topk_ids.reshape(num_rows, 0)
    else:
        # Bound each Triton selection to 16 input groups, then merge the
        # reduced groups once more to produce the final local top-K.
        merge_width = 16
        merged_groups = triton.cdiv(num_partials, merge_width)
        merged_values = torch.empty(
            (num_rows, merged_groups, num_topk),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        merged_ids = torch.empty(
            (num_rows, merged_groups, num_topk),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        _merge_topk_partials_kernel[(num_rows, merged_groups)](
            partial_topk_values,
            partial_topk_ids,
            merged_values,
            merged_ids,
            num_rows,
            num_partials,
            merged_groups,
            INPUTS_PER_OUTPUT=merge_width,
            INPUT_WIDTH=partial_topk_width,
            OUTPUT_WIDTH=num_topk,
            TOPK_BUCKET=topk_bucket,
            BLOCK_CANDIDATES=triton.next_power_of_2(merge_width * partial_topk_width),
            num_warps=8,
        )
        local_topk_values = torch.empty(
            (num_rows, num_topk),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        local_topk_ids = torch.empty(
            (num_rows, num_topk),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        final_width = triton.next_power_of_2(merged_groups)
        _merge_topk_partials_kernel[(num_rows, 1)](
            merged_values,
            merged_ids,
            local_topk_values[:, None, :],
            local_topk_ids[:, None, :],
            num_rows,
            merged_groups,
            1,
            INPUTS_PER_OUTPUT=final_width,
            INPUT_WIDTH=num_topk,
            OUTPUT_WIDTH=num_topk,
            TOPK_BUCKET=topk_bucket,
            BLOCK_CANDIDATES=triton.next_power_of_2(final_width * num_topk),
            num_warps=8,
        )
    return LMHeadLogprobsOutput(
        local_topk_values,
        local_topk_ids,
        local_lse,
        local_rank_count,
    )


def merge_tp_prompt_logprobs(
    tp_topk_values: torch.Tensor,
    tp_topk_ids: torch.Tensor,
    tp_local_lse: torch.Tensor,
    tp_rank_count: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    num_topk: int,
    *,
    block_m: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge gathered TP-local statistics into prompt log probabilities.

    Args:
        tp_topk_values: Rank-major local top-K logits with shape
            ``[TP, M, K]`` in FP32.
        tp_topk_ids: Global token IDs corresponding to ``tp_topk_values`` with
            shape ``[TP, M, K]`` in INT32.
        tp_local_lse: Local log-sum-exp values with shape ``[TP, M]`` in FP32.
        tp_rank_count: Local counts of logits greater than or equal to the
            target logit with shape ``[TP, M]`` in INT32.
        target_token_ids: Prompt target token IDs with shape ``[M]``.
        target_logits: TP-reduced target logits with shape ``[M]`` in FP32.
        num_topk: Number of global top tokens to place after the target column.
        block_m: Optional number of rows processed by one Triton program.

    Returns:
        Prompt log probabilities with the target in column zero and global
        top-K tokens in the remaining columns.
    """
    if tp_topk_values.ndim != 3:
        raise ValueError("tp_topk_values must have shape [TP, M, K]")
    if tp_topk_ids.shape != tp_topk_values.shape:
        raise ValueError("tp_topk_ids must match tp_topk_values shape")
    tp_size, num_rows, topk_width = tp_topk_values.shape
    if tp_size == 0:
        raise ValueError("TP size must be positive")
    if tp_local_lse.shape != (tp_size, num_rows):
        raise ValueError("tp_local_lse must have shape [TP, M]")
    if tp_rank_count.shape != (tp_size, num_rows):
        raise ValueError("tp_rank_count must have shape [TP, M]")
    if target_token_ids.shape != (num_rows,):
        raise ValueError("target_token_ids must have shape [M]")
    if target_logits.shape != (num_rows,):
        raise ValueError("target_logits must have shape [M]")
    if not 0 <= num_topk <= topk_width:
        raise ValueError("num_topk must be in [0, local top-K width]")
    if topk_width > 32:
        raise ValueError("local top-K width must not exceed 32")

    if tp_topk_values.dtype != torch.float32:
        raise TypeError("tp_topk_values must use torch.float32")
    if tp_topk_ids.dtype != torch.int32:
        raise TypeError("tp_topk_ids must use torch.int32")
    if tp_local_lse.dtype != torch.float32:
        raise TypeError("tp_local_lse must use torch.float32")
    if tp_rank_count.dtype != torch.int32:
        raise TypeError("tp_rank_count must use torch.int32")
    if target_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_token_ids must use torch.int32 or torch.int64")
    if target_logits.dtype != torch.float32:
        raise TypeError("target_logits must use torch.float32")
    if not tp_topk_values.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    device = tp_topk_values.device
    for name, tensor in (
        ("tp_topk_ids", tp_topk_ids),
        ("tp_local_lse", tp_local_lse),
        ("tp_rank_count", tp_rank_count),
        ("target_token_ids", target_token_ids),
        ("target_logits", target_logits),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as tp_topk_values")

    output_token_ids = torch.empty(
        (num_rows, num_topk + 1), dtype=torch.int32, device=device
    )
    output_logprobs = torch.empty(
        (num_rows, num_topk + 1), dtype=torch.float32, device=device
    )
    output_ranks = torch.empty(num_rows, dtype=torch.int32, device=device)
    if num_rows == 0:
        return output_token_ids, output_logprobs, output_ranks

    # Bound the per-program TP x K candidate matrix as either dimension grows.
    if block_m is None:
        block_m = 16 if num_topk == 0 else 8 if tp_size == 1 else 2
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("block_m must be a positive power of two")
    topk_bucket = 0 if topk_width == 0 else triton.next_power_of_2(topk_width)

    # Compile-time TP/K dimensions let Triton unroll the rank merge and top-K.
    _merge_tp_prompt_logprobs_kernel[(triton.cdiv(num_rows, block_m),)](
        tp_topk_values,
        tp_topk_ids,
        tp_local_lse,
        tp_rank_count,
        target_token_ids,
        target_logits,
        output_token_ids,
        output_logprobs,
        output_ranks,
        num_rows,
        tp_topk_values.stride(0),
        tp_topk_values.stride(1),
        tp_topk_values.stride(2),
        tp_topk_ids.stride(0),
        tp_topk_ids.stride(1),
        tp_topk_ids.stride(2),
        tp_local_lse.stride(0),
        tp_local_lse.stride(1),
        tp_rank_count.stride(0),
        tp_rank_count.stride(1),
        target_token_ids.stride(0),
        target_logits.stride(0),
        BLOCK_M=block_m,
        TOPK_WIDTH=topk_width,
        TOPK_BUCKET=topk_bucket,
        NUM_TOPK=num_topk,
        TP_SIZE=tp_size,
        num_warps=4,
    )
    return output_token_ids, output_logprobs, output_ranks


__all__ = [
    "LMHeadLogprobsOutput",
    "lm_head_logprobs",
    "materialized_lse_and_rank",
    "merge_tp_prompt_logprobs",
    "prompt_target_logits",
    "validate_lm_head_logprobs_environment",
]
