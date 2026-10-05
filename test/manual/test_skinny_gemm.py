"""GPU functional check for the skinny bf16 GEMM (manual, not registered
with any CI suite -- run by hand on real MI300A/gfx942 hardware).

For each of the six decode-step dense-projection shapes on the
Qwen3.5-397B-A17B-MXFP4 @ TP=4 critical path (see
skinny_gemm/csrc/skinny_gemm_bf16.cu and the campaign prototype it was
ported from) and M in {1, 8, 15, 16}:

  1. Runs ``sglang.srt.layers.skinny_gemm.skinny_gemm`` and checks it
     against ``F.linear(x.float(), w.float()).to(bf16)`` (the same
     relative/absolute tolerance gate the campaign prototype used).
  2. Times it against ``aiter.tuned_gemm.tgemm.mm`` (the path it replaces
     in ``UnquantizedLinearMethod.apply``) and prints the speedup.

Requires: a ROCm/HIP gfx942 device and aiter installed.

Usage:
    python3 test/manual/test_skinny_gemm.py
"""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F

NK_SHAPES = [
    (4096, 2048),
    (4608, 4096),
    (512, 4096),
    (5120, 4096),
    (32, 4096),
    (4096, 256),
]

M_VALUES = [1, 8, 15, 16]


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms)


def main():
    from sglang.srt.environ import envs
    from sglang.srt.layers.skinny_gemm import skinny_gemm, supports

    device = "cuda:0"
    torch.manual_seed(0)

    try:
        from aiter.tuned_gemm import tgemm

        have_aiter = True
    except Exception as exc:  # noqa: BLE001
        print("import aiter FAILED:", repr(exc))
        have_aiter = False

    max_m = envs.SGLANG_SKINNY_GEMM_MAX_M.get()
    print(f"SGLANG_SKINNY_GEMM_MAX_M={max_m}")

    failures = []
    for n, k in NK_SHAPES:
        for m in M_VALUES:
            if not supports(m, n, k, torch.bfloat16):
                print(f"M={m:3d} N={n:5d} K={k:5d}  supports()=False, skipped")
                continue

            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

            y = skinny_gemm(x, w, bias=None)
            ref = F.linear(x.float(), w.float()).to(torch.bfloat16)
            diff = (y.float() - ref.float()).abs()
            denom = ref.float().abs().clamp_min(1e-6)
            max_abs = diff.max().item()
            max_rel = (diff / denom).max().item()
            out_absmax = ref.float().abs().max().item()
            ok = (max_rel <= 1e-2) or (max_abs <= 1e-2 * max(out_absmax, 1.0))
            status = "OK" if ok else "FAIL"
            if not ok:
                failures.append((m, n, k, max_abs, max_rel))

            us_new = time_call(lambda: skinny_gemm(x, w, bias=None)) * 1000.0

            if have_aiter:
                try:
                    us_tgemm = (
                        time_call(lambda: tgemm.mm(x, w, None, otype=x.dtype)) * 1000.0
                    )
                    speedup = us_tgemm / us_new
                    tgemm_str = f"{us_tgemm:8.2f}us  speedup={speedup:5.2f}x"
                except Exception as exc:  # noqa: BLE001
                    tgemm_str = f"ERR({exc!r})"
            else:
                tgemm_str = "aiter not importable"

            print(
                f"M={m:3d} N={n:5d} K={k:5d}  max_abs_diff={max_abs:.6f} "
                f"max_rel_diff={max_rel:.6f}  new_us={us_new:8.2f}  "
                f"tgemm={tgemm_str}  [{status}]"
            )

    print(f"CORRECTNESS_FAILURES={len(failures)}")
    for f in failures:
        print("  FAIL:", f)

    graph_failures = graph_capture_check(device)
    print(f"GRAPH_CAPTURE_FAILURES={len(graph_failures)}")
    for f in graph_failures:
        print("  FAIL:", f)

    print("DONE")


def graph_capture_check(device: str) -> list[tuple[int, int, int, float, float]]:
    """CUDA-graph capture/replay check: for each of the six shapes, capture a
    graph of exactly one ``skinny_gemm`` call at M=16, replay it 3 times, and
    compare each replay's output against the pre-capture eager result.

    ``supports()`` is a pure function of static ints/dtype (no device access
    or sync), and ``skinny_gemm``'s scratch buffer is cached per (M, N,
    device) and reused across calls (zeroed in-kernel when split_k > 1,
    fully overwritten otherwise) -- both properties the PR body argues make
    this path graph-safe. This check exercises that claim directly instead
    of only by source inspection.
    """
    from sglang.srt.layers.skinny_gemm import skinny_gemm, supports

    print("=== CUDA graph capture/replay check (M=16) ===")
    m = 16
    failures: list[tuple[int, int, int, float, float]] = []
    for n, k in NK_SHAPES:
        if not supports(m, n, k, torch.bfloat16):
            print(f"N={n:5d} K={k:5d} M={m}: supports()=False, skipped")
            continue

        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

        eager_out = skinny_gemm(x, w, bias=None).clone()

        # Warm up the extension/allocator on a side stream before capture,
        # as torch.cuda.graph requires.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                skinny_gemm(x, w, bias=None)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = skinny_gemm(x, w, bias=None)

        denom = eager_out.float().abs().clamp_min(1e-6)
        out_absmax = eager_out.float().abs().max().item()
        for replay_i in range(3):
            graph.replay()
            torch.cuda.synchronize()
            diff = (graph_out.float() - eager_out.float()).abs()
            max_abs = diff.max().item()
            max_rel = (diff / denom).max().item()
            ok = (max_rel <= 1e-2) or (max_abs <= 1e-2 * max(out_absmax, 1.0))
            status = "OK" if ok else "FAIL"
            if not ok:
                failures.append((n, k, replay_i, max_abs, max_rel))
            print(
                f"N={n:5d} K={k:5d} replay={replay_i}  "
                f"max_abs_diff_vs_eager={max_abs:.6f} max_rel_diff_vs_eager={max_rel:.6f}  [{status}]"
            )

    return failures


if __name__ == "__main__":
    main()
