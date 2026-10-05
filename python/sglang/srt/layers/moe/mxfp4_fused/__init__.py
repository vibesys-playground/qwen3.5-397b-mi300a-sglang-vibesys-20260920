# SPDX-License-Identifier: Apache-2.0
"""HIP grouped-GEMM fused MoE for MXFP4 (w4a16) weights on gfx942/MI300A.

Scaffold: the inner K-loop (``produce_b_fragments_32k`` /
``gemm16x16_fp4_accumulate`` in ``csrc/mxfp4_fused_moe.cu``) is copied from
the campaign's single-tile HIP microkernel
(``scratchpad/loop/moe-fused-tile/hip/tile_gemm.cu``), unoptimized. The
grouped-GEMM wrapper around it (``sorted_token_ids``/``expert_ids`` dispatch,
top_k-strided A gather, the two-stage gate_up/down pipeline, and the final
top_k reduction) mirrors sglang's Triton production path
(``sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py``) so a later
agent can drop in a faster inner loop without touching the surrounding
dispatch/epilogue code. See ``scratchpad/loop/moe-fused-grouped-design.md``
for the full design.

See ``fused_moe.py`` for the JIT loader (built on the shared
``sglang.srt.layers.hip_extension.load_hip_extension``) and
``fused_experts_mxfp4_hip``, the entry point ``quark_w4a4_mxfp4_moe.py``
calls when ``SGLANG_MXFP4_MOE_HIP`` is set.
"""

from sglang.srt.layers.moe.mxfp4_fused.fused_moe import fused_experts_mxfp4_hip

__all__ = ["fused_experts_mxfp4_hip"]
