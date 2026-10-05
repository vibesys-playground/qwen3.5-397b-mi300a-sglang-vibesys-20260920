"""CPU-only unit tests for the aiter dense-GEMM M-padding helpers in
``sglang.srt.layers.quantization.unquant``
(``aiter_gemm_padded_m``/``run_aiter_dense_gemm_padded``), used by
``UnquantizedLinearMethod.apply``'s aiter path to avoid the untuned
hipBLASLt fallback PyTorch TunableOp cannot cover above M=192 (see
manual/stall-diag/results.md and manual/gemm-pad/results.md).

No aiter, ROCm, or GPU is required: these helpers are plain torch/env-var
logic with no aiter import at module scope (aiter is only imported when
SGLANG_USE_AITER is set, guarded separately in unquant.py), and the tests
below use a deterministic CPU matmul in place of tgemm.mm.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@unittest.skipUnless(torch is not None, "requires torch")
class TestAiterGemmPaddedM(unittest.TestCase):
    """aiter_gemm_padded_m: pure shape arithmetic, no tensors involved."""

    def test_at_or_below_threshold_unchanged(self):
        from sglang.srt.layers.quantization.unquant import aiter_gemm_padded_m

        for m in (1, 16, 64, 191, 192):
            self.assertEqual(aiter_gemm_padded_m(m, pad_to=64, threshold=192), m)

    def test_already_multiple_of_pad_to_unchanged(self):
        from sglang.srt.layers.quantization.unquant import aiter_gemm_padded_m

        for m in (256, 320, 1024):
            self.assertEqual(aiter_gemm_padded_m(m, pad_to=64, threshold=192), m)

    def test_pads_up_to_next_multiple_of_64(self):
        from sglang.srt.layers.quantization.unquant import aiter_gemm_padded_m

        cases = {
            193: 256,
            200: 256,
            255: 256,
            256: 256,  # already on-grid
            1000: 1024,
            1024: 1024,  # already on-grid
        }
        for m, expected in cases.items():
            self.assertEqual(
                aiter_gemm_padded_m(m, pad_to=64, threshold=192),
                expected,
                f"m={m}",
            )

    def test_pad_to_zero_disables_padding(self):
        from sglang.srt.layers.quantization.unquant import aiter_gemm_padded_m

        for m in (193, 1000, 5000):
            self.assertEqual(aiter_gemm_padded_m(m, pad_to=0, threshold=192), m)


@unittest.skipUnless(torch is not None, "requires torch")
class TestRunAiterDenseGemmPadded(unittest.TestCase):
    """run_aiter_dense_gemm_padded: actually pads/slices tensors, using a
    deterministic CPU matmul as the stand-in for tgemm.mm. On CPU,
    torch.matmul computes each output row independently of the others, so
    the padded rows (all zero, contributing nothing to any other row's
    result) cannot perturb the kept rows: the sliced output should equal
    the unpadded call exactly. We still use assertTrue(torch.allclose(...))
    rather than a bitwise equality assert, in case a given CPU BLAS backend
    does not guarantee bit-identical results across differently-shaped
    calls; allclose is called with a tight (default) tolerance and every
    case in this test is in fact bit-exact when checked manually.
    """

    K, N = 40, 24

    def _gemm_fn(self, weight, bias):
        def gemm_fn(xx):
            return torch.matmul(xx, weight.t()) + bias

        return gemm_fn

    def _run(self, m: int, pad_to: int = 64, threshold: int = 192):
        from sglang.srt.layers.quantization.unquant import (
            aiter_gemm_padded_m,
            run_aiter_dense_gemm_padded,
        )

        torch.manual_seed(0)
        x = torch.randn(m, self.K, dtype=torch.float32)
        weight = torch.randn(self.N, self.K, dtype=torch.float32)
        bias = torch.randn(self.N, dtype=torch.float32)
        gemm_fn = self._gemm_fn(weight, bias)

        out = run_aiter_dense_gemm_padded(
            x, gemm_fn, pad_to=pad_to, threshold=threshold
        )
        expected = gemm_fn(x)
        expected_padded_m = aiter_gemm_padded_m(m, pad_to=pad_to, threshold=threshold)
        return out, expected, expected_padded_m

    def test_output_has_m_rows_and_matches_unpadded(self):
        for m in (193, 200, 255, 256, 1000, 1024):
            with self.subTest(m=m):
                out, expected, padded_m = self._run(m)
                self.assertEqual(out.shape, (m, self.N))
                self.assertTrue(torch.allclose(out, expected, atol=0.0, rtol=0.0))

    def test_padded_shape_matches_helper(self):
        # For an M that actually needs padding, confirm the internal padded
        # call really ran at the next multiple of 64 (not just that the
        # final sliced output has the right shape).
        from sglang.srt.layers.quantization.unquant import aiter_gemm_padded_m

        seen_shapes = []

        def gemm_fn(xx):
            seen_shapes.append(tuple(xx.shape))
            return torch.zeros(xx.shape[0], self.N, dtype=xx.dtype)

        from sglang.srt.layers.quantization.unquant import run_aiter_dense_gemm_padded

        for m, expected_padded in ((193, 256), (1000, 1024)):
            seen_shapes.clear()
            x = torch.zeros(m, self.K, dtype=torch.float32)
            run_aiter_dense_gemm_padded(x, gemm_fn, pad_to=64, threshold=192)
            self.assertEqual(seen_shapes, [(expected_padded, self.K)])
            self.assertEqual(
                aiter_gemm_padded_m(m, pad_to=64, threshold=192), expected_padded
            )

    def test_no_padding_when_at_or_below_threshold_or_on_grid(self):
        seen_shapes = []

        def gemm_fn(xx):
            seen_shapes.append(tuple(xx.shape))
            return torch.zeros(xx.shape[0], self.N, dtype=xx.dtype)

        from sglang.srt.layers.quantization.unquant import run_aiter_dense_gemm_padded

        for m in (16, 192, 256, 1024):
            seen_shapes.clear()
            x = torch.zeros(m, self.K, dtype=torch.float32)
            run_aiter_dense_gemm_padded(x, gemm_fn, pad_to=64, threshold=192)
            self.assertEqual(seen_shapes, [(m, self.K)])

    def test_pad_to_zero_disables_padding_end_to_end(self):
        seen_shapes = []

        def gemm_fn(xx):
            seen_shapes.append(tuple(xx.shape))
            return torch.zeros(xx.shape[0], self.N, dtype=xx.dtype)

        from sglang.srt.layers.quantization.unquant import run_aiter_dense_gemm_padded

        x = torch.zeros(1000, self.K, dtype=torch.float32)
        run_aiter_dense_gemm_padded(x, gemm_fn, pad_to=0, threshold=192)
        self.assertEqual(seen_shapes, [(1000, self.K)])


if __name__ == "__main__":
    unittest.main()
