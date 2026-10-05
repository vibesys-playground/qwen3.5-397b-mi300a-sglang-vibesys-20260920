# SPDX-License-Identifier: Apache-2.0
"""Grouped-GEMM fused MXFP4 (w4a16) MoE forward pass, HIP path.

Same external contract as sglang's Triton ``fused_experts`` for this quant
path (see ``sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe``):
takes ``hidden_states``/``w13``/``w2`` (+ MXFP4 scales) and a top-k routing
decision, returns the combined ``[num_tokens, hidden_size]`` bf16 output.

Orchestration mirrors ``_fused_moe_kernel_sequence``'s two-kernel structure
(``invoke_fused_moe_kernel`` for gate_up, an activation step, then
``invoke_fused_moe_kernel`` again for down) but fuses the activation into
stage1's own epilogue (see the .cu file's docstring) and reuses sglang's own
``moe_align_block_size`` for the sorted-token/expert dispatch so this path
stays bit-for-bit consistent with production's block-padding scheme.

CUDA-graph safety: every buffer here is sized from host-visible tensor
shapes only (``num_tokens``, ``top_k``, ``E`` -- the same quantities a CUDA
graph capture is keyed on), never from a value read back from the device.
``moe_align_block_size`` itself makes no host sync (see its own
implementation); ``num_tokens_post_padded`` stays on-device and is read
inside the kernels, exactly as production does.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from sglang.srt.layers.hip_extension import load_hip_extension
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCE = os.path.join(_CSRC_DIR, "mxfp4_fused_moe.cu")

# This path is only selectable on gfx942 (see quark_w4a4_mxfp4_moe.py's
# SGLANG_MXFP4_MOE_HIP gate); pin the offload arch explicitly rather than
# relying on autodetection, since JIT compilation may run before the caller
# has touched a CUDA context.
_OFFLOAD_ARCH = "gfx942"


def get_extension() -> Any:
    """Return the compiled MXFP4 fused-MoE HIP extension, building it (via
    the shared ``load_hip_extension`` JIT loader) on first call. The sole
    extension-building seam in this module: tests that exercise
    ``fused_experts_mxfp4_hip`` without a real GPU monkeypatch this one
    function rather than needing a HIP toolchain."""
    return load_hip_extension(
        name="sglang_mxfp4_fused_moe",
        sources=[_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={_OFFLOAD_ARCH}"],
    )


BLOCK_M = 16


def fused_experts_mxfp4_hip(
    hidden_states: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """MXFP4 w4a16 grouped-GEMM MoE forward, HIP kernels.

    Args:
        hidden_states: [num_tokens, hidden_size] bf16.
        w13_fp4: [E, 2*inter_size, hidden_size/2] uint8 (packed e2m1).
        w13_scale: [E, 2*inter_size, hidden_size/32] uint8 (e8m0).
        w2_fp4: [E, hidden_size, inter_size/2] uint8.
        w2_scale: [E, hidden_size, inter_size/32] uint8.
        topk_weights: [num_tokens, top_k] routing weights (any float dtype).
        topk_ids: [num_tokens, top_k] int, expert ids.
        activation: only "silu" (gated silu(gate)*up) is implemented.
        apply_router_weight_on_input: not yet implemented on this path (the
            HIP kernels apply the routed weight on the down-GEMM output,
            matching production's default; pre-multiplying the input is a
            separate, currently unsupported, code path in production too).

    Returns:
        [num_tokens, hidden_size] bf16.
    """
    if activation != "silu":
        raise NotImplementedError(
            f"fused_experts_mxfp4_hip only implements activation='silu', got {activation!r}"
        )
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "fused_experts_mxfp4_hip does not yet support apply_router_weight_on_input"
        )

    assert hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16
    num_tokens, hidden_size = hidden_states.shape
    E = w13_fp4.shape[0]
    inter_size = w13_fp4.shape[1] // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, E
    )

    num_valid_tokens = topk_ids.numel()
    ext = get_extension()

    intermediate = torch.empty(
        (num_valid_tokens, inter_size), dtype=torch.bfloat16, device=device
    )
    ext.moe_gemm_w4a16_stage1(
        hidden_states.contiguous(),
        w13_fp4,
        w13_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        intermediate,
        top_k,
        num_valid_tokens,
    )

    down_out = torch.empty(
        (num_valid_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    # The kernel's routed-weight epilogue expects a flat fp32 view addressed
    # by sorted_token_ids' value directly (matching production's
    # topk_weights_ptr + offs_token addressing) -- cast/flatten here rather
    # than constraining the caller's topk_weights dtype/shape.
    topk_weights_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()
    ext.moe_gemm_w4a16_stage2(
        intermediate,
        w2_fp4,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_flat,
        down_out,
        num_valid_tokens,
    )

    return down_out.view(num_tokens, top_k, hidden_size).sum(dim=1)
