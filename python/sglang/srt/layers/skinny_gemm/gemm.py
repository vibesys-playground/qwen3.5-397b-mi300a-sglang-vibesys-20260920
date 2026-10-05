# SPDX-License-Identifier: Apache-2.0
"""Skinny bf16 GEMM for small-M dense (unquantized) linear layers on HIP.

y = x @ w.T (+ bias), where x is [M, K] bf16 and w is [N, K] bf16
(torch.nn.Linear weight layout). Targets the decode-time small-M shapes
(qkv/o/gate dense projections at M <= ~16-32) where aiter's tgemm.mm picks a
compute-oriented (MFMA) kernel that leaves most of HBM bandwidth idle; see
csrc/skinny_gemm_bf16.cu for the kernel design and the campaign results this
was ported from.

``supports()`` is the dispatch gate ``unquant.py`` calls before routing a
dense-linear ``apply()`` here; it takes only static Python ints and a dtype
(no tensor access), so it never syncs and is safe to call under CUDA graph
capture, where M is fixed per captured shape.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.hip_extension import load_hip_extension

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCE = os.path.join(_CSRC_DIR, "skinny_gemm_bf16.cu")

# gfx942 (MI300A) is the only target this kernel has been designed and
# measured against; pin the offload arch explicitly rather than relying on
# autodetection, since JIT compilation may run before the caller has touched
# a CUDA context.
_OFFLOAD_ARCH = "gfx942"


def get_extension() -> Any:
    """Return the compiled skinny-GEMM HIP extension, building it (via the
    shared ``load_hip_extension`` JIT loader) on first call. The sole
    extension-building seam in this module: tests that exercise
    ``skinny_gemm()`` without a real GPU monkeypatch this one function
    rather than needing a HIP toolchain."""
    return load_hip_extension(
        name="sglang_skinny_gemm_bf16",
        sources=[_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={_OFFLOAD_ARCH}"],
    )


# Kernel's own hard limit (see skinny_gemm_bf16.cu's TORCH_CHECK on M);
# SGLANG_SKINNY_GEMM_MAX_M is clamped to this regardless of its configured
# value.
_MAX_M_HARD_CAP = 32
_VALID_ROWS_PER_WG = (4, 8)
# Below this N, a single row-block's worth of workgroups (N / rows_per_wg)
# is too small to fill an MI300A's CUs; split_k fans a row-block's work out
# over K instead. Matches the campaign prototype's finding for N=32, N=512.
_SPLIT_K_N_THRESHOLD = 1024
# Candidate split_k values, smallest first: choose_split_k prefers the
# smallest split_k (least atomic-reduction overhead) that reaches the
# occupancy target.
_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
# rows_per_wg the campaign prototype's own choose_config heuristic mostly
# picked, and thus the rows_per_wg the "4x cu_num total workgroups" occupancy
# target below was implicitly calibrated against (see _choose_split_k).
_TARGET_REFERENCE_ROWS_PER_WG = 4


@functools.lru_cache(maxsize=None)
def _cu_num_for_index(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _cu_num_for(device: torch.device) -> int:
    """Compute-unit count for ``device``. The sole device-querying seam in
    this module: tests that exercise ``skinny_gemm()`` without a real GPU
    (a stubbed extension, CPU tensors) monkeypatch this one function rather
    than needing a CUDA context."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cu_num_for_index(index)


def _rows_per_wg() -> int:
    rpw = envs.SGLANG_SKINNY_GEMM_ROWS_PER_WG.get()
    if rpw not in _VALID_ROWS_PER_WG:
        raise ValueError(
            "SGLANG_SKINNY_GEMM_ROWS_PER_WG must be one of "
            f"{_VALID_ROWS_PER_WG}, got {rpw}"
        )
    return rpw


def _choose_split_k(N: int, K: int, rows_per_wg: int, cu_num: int) -> int:
    """Pick split_k for an (N, K, rows_per_wg) shape on a device with
    ``cu_num`` compute units.

    split_k=1 once N >= _SPLIT_K_N_THRESHOLD (a single row-block already
    gives enough workgroups). Below that, scans _SPLIT_K_CANDIDATES for the
    smallest split_k whose (row-blocks x split_k) total reaches an occupancy
    target, falling back to the largest valid split_k tried if none reaches
    the target. Depends only on static shape ints and cu_num (fixed per
    device), so this is deterministic per captured graph and never syncs or
    touches device memory.

    The occupancy target is 4 x cu_num total workgroups at
    _TARGET_REFERENCE_ROWS_PER_WG (4), scaled down proportionally as
    rows_per_wg grows: each split_k division re-reads the same output
    element (an atomicAdd contention point shared across every split of
    that row), so at a fixed target block count, doubling rows_per_wg (half
    as many distinct row-blocks) forces split_k to double too just to hit
    the same raw block count -- doubling the number of atomic writers
    contending on each output element for no added distinct-row
    parallelism. Measured on gfx942 (see the cluster validation task's
    stage-1 report): at the unscaled target, rows_per_wg=8 picked split_k=16
    for N=512 (K=4096) and ran 2x slower than the campaign prototype's own
    heuristic (rows_per_wg=4, split_k=8, same K); scaling the target down by
    _TARGET_REFERENCE_ROWS_PER_WG / rows_per_wg recovers split_k=1 for the
    four large-N shapes (matching the prototype's exhaustive-sweep best) and
    split_k=8 for N=512, leaving the already-target-unreachable N=32 case
    (which maxes out _SPLIT_K_CANDIDATES either way) unaffected.
    """
    if N >= _SPLIT_K_N_THRESHOLD:
        return 1
    target = 4 * cu_num * _TARGET_REFERENCE_ROWS_PER_WG // rows_per_wg
    base_blocks = N // rows_per_wg
    best = 1
    for split_k in _SPLIT_K_CANDIDATES:
        if K % split_k != 0:
            continue
        k_len = K // split_k
        if k_len % 8 != 0:
            continue
        best = split_k
        if base_blocks * split_k >= target:
            break
    return best


def supports(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Static shape/dtype gate: bf16 only, M within the configured cap, K a
    multiple of 64, N a multiple of 32 (the kernel's 16-byte-load and
    rows_per_wg tiling constraints).

    Callers must separately confirm x and the weight are contiguous 2-D
    row-major tensors (e.g. ``x.is_contiguous()``) -- a property of the
    specific tensors passed at the call site, not of shape and dtype alone,
    so it is intentionally not part of this predicate.
    """
    if dtype != torch.bfloat16:
        return False
    max_m = min(envs.SGLANG_SKINNY_GEMM_MAX_M.get(), _MAX_M_HARD_CAP)
    if not (1 <= M <= max_m):
        return False
    if K % 64 != 0:
        return False
    if N % 32 != 0:
        return False
    return True


# One fp32 [M, N] accumulator per (M, N, device) shape, reused across calls
# rather than allocated fresh each time: split_k > 1 zeros and then
# atomically accumulates into it every call (a graph-capturable memset, see
# skinny_gemm_bf16.cu), and split_k == 1 overwrites every element via plain
# stores, so correctness does not depend on reuse -- this cache exists to
# avoid a fresh cudaMalloc-class allocation on every decode step.
_scratch_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _get_scratch(M: int, N: int, device: torch.device) -> torch.Tensor:
    key = (M, N, device)
    scratch = _scratch_cache.get(key)
    if scratch is None:
        scratch = torch.empty((M, N), dtype=torch.float32, device=device)
        _scratch_cache[key] = scratch
    return scratch


def skinny_gemm(
    x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """y = x @ w.T (+ bias) via the skinny bf16 GEMM HIP kernel.

    x: [M, K] bf16, contiguous. w: [N, K] bf16, contiguous (torch.nn.Linear
    weight layout). bias: optional 1-D tensor of length N, broadcast over
    the M rows.

    Callers should already have checked ``supports(M, N, K, x.dtype)`` and
    tensor contiguity before calling; this function re-validates shapes and
    dtypes (cheap, matches the extension's own checks) but does not
    re-derive the supports() decision, so calling it outside supports()'s
    contract raises rather than silently falling back.
    """
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError(
            f"x and w must be 2-D, got x.shape={tuple(x.shape)} w.shape={tuple(w.shape)}"
        )
    if not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("x and w must be contiguous row-major tensors")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(
            f"x and w must be bf16, got x.dtype={x.dtype} w.dtype={w.dtype}"
        )

    M, K = x.shape
    N, K_w = w.shape
    if K_w != K:
        raise ValueError(f"K mismatch: x has K={K}, w has K={K_w}")

    rows_per_wg = _rows_per_wg()
    cu_num = _cu_num_for(x.device)
    split_k = _choose_split_k(N, K, rows_per_wg, cu_num)

    scratch = _get_scratch(M, N, x.device)
    ext = get_extension()
    ext.skinny_gemm_bf16(x, w, scratch, rows_per_wg, split_k)

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if bias is None:
        y.copy_(scratch)
    else:
        # Fuse the bias add and the fp32 -> output-dtype downcast into one
        # elementwise kernel for the common case (a 1-D [N] bias broadcasts
        # trivially against [M, N]); fall back to an explicit two-step add
        # if the shapes do not broadcast that way.
        try:
            torch.add(scratch, bias, out=y)
        except RuntimeError:
            y.copy_(scratch)
            y.add_(bias)
    return y
