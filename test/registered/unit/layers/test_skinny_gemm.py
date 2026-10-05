"""CPU-only unit tests for the skinny bf16 GEMM's shape/dtype gate
(``supports()``), the split_k occupancy heuristic, and the routing decision
in ``UnquantizedLinearMethod.apply``, exercised against a stub kernel.

No GPU, ROCm, or aiter is required: skinny_gemm/gemm.py only touches the HIP
extension inside ``get_extension()`` (called lazily from ``skinny_gemm()``),
so these tests import ``sglang.srt.layers.skinny_gemm`` directly and stub
``gemm.get_extension`` rather than building the real kernel.
"""

import unittest

try:
    import torch
except ImportError:
    torch = None

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@unittest.skipUnless(torch is not None, "requires torch")
class TestSupports(unittest.TestCase):
    def test_rejects_non_bf16(self):
        from sglang.srt.layers.skinny_gemm import supports

        self.assertFalse(supports(8, 4096, 2048, torch.float16))
        self.assertFalse(supports(8, 4096, 2048, torch.float32))

    def test_accepts_bf16_within_default_max_m(self):
        from sglang.srt.layers.skinny_gemm import supports

        self.assertTrue(supports(1, 4096, 2048, torch.bfloat16))
        self.assertTrue(supports(16, 4096, 2048, torch.bfloat16))

    def test_rejects_m_above_max_m_env(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.skinny_gemm import supports

        with envs.SGLANG_SKINNY_GEMM_MAX_M.override("8"):
            self.assertTrue(supports(8, 4096, 2048, torch.bfloat16))
            self.assertFalse(supports(9, 4096, 2048, torch.bfloat16))

    def test_max_m_env_clamped_to_kernel_hard_cap(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.skinny_gemm import supports

        with envs.SGLANG_SKINNY_GEMM_MAX_M.override("64"):
            self.assertTrue(supports(32, 4096, 2048, torch.bfloat16))
            self.assertFalse(supports(33, 4096, 2048, torch.bfloat16))

    def test_rejects_m_zero_or_negative(self):
        from sglang.srt.layers.skinny_gemm import supports

        self.assertFalse(supports(0, 4096, 2048, torch.bfloat16))

    def test_rejects_k_not_multiple_of_64(self):
        from sglang.srt.layers.skinny_gemm import supports

        self.assertFalse(supports(8, 4096, 2000, torch.bfloat16))
        self.assertTrue(supports(8, 4096, 2048, torch.bfloat16))

    def test_rejects_n_not_multiple_of_32(self):
        from sglang.srt.layers.skinny_gemm import supports

        self.assertFalse(supports(8, 4095, 2048, torch.bfloat16))
        self.assertTrue(supports(8, 4096, 2048, torch.bfloat16))

    def test_does_not_touch_device_memory(self):
        # Graph-safety requirement: supports() must be a pure function of
        # its (M, N, K, dtype) arguments, since it runs every forward pass
        # including inside CUDA graph capture/replay where a host-device
        # sync or a data-dependent read would break replay.
        import inspect

        from sglang.srt.layers.skinny_gemm import gemm

        source = inspect.getsource(gemm.supports)
        self.assertNotIn(".item()", source)
        self.assertNotIn("synchronize", source)


@unittest.skipUnless(torch is not None, "requires torch")
class TestChooseSplitK(unittest.TestCase):
    def test_large_n_never_splits(self):
        from sglang.srt.layers.skinny_gemm.gemm import _choose_split_k

        self.assertEqual(_choose_split_k(4096, 2048, 8, cu_num=228), 1)

    def test_small_n_splits_to_reach_occupancy_target(self):
        from sglang.srt.layers.skinny_gemm.gemm import _choose_split_k

        # N=32, rows_per_wg=8 -> 4 row-blocks; needs split_k>=1 to reach
        # 4*228=912 total workgroups, so this must split.
        split_k = _choose_split_k(32, 4096, 8, cu_num=228)
        self.assertGreater(split_k, 1)
        self.assertEqual(4096 % split_k, 0)
        self.assertEqual((4096 // split_k) % 8, 0)

    def test_split_k_divides_k_and_leaves_multiple_of_8(self):
        from sglang.srt.layers.skinny_gemm.gemm import _choose_split_k

        for n, k in [(32, 4096), (512, 4096), (32, 256)]:
            split_k = _choose_split_k(n, k, 8, cu_num=228)
            self.assertEqual(k % split_k, 0)
            self.assertEqual((k // split_k) % 8, 0)


@unittest.skipUnless(torch is not None, "requires torch")
class TestRowsPerWg(unittest.TestCase):
    def test_default_is_eight(self):
        from sglang.srt.layers.skinny_gemm.gemm import _rows_per_wg

        self.assertEqual(_rows_per_wg(), 8)

    def test_rejects_unsupported_value(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.skinny_gemm.gemm import _rows_per_wg

        with envs.SGLANG_SKINNY_GEMM_ROWS_PER_WG.override("16"):
            with self.assertRaises(ValueError):
                _rows_per_wg()


@unittest.skipUnless(torch is not None, "requires torch")
class TestSkinnyGemmCallsExtension(unittest.TestCase):
    """The real extension is HIP-only (built lazily by gemm.get_extension())
    and the split_k heuristic queries the device's compute-unit count (which
    requires a real CUDA context); stub both out to test the Python-side
    call shape and bias handling on CPU tensors without a GPU."""

    def setUp(self):
        from sglang.srt.layers.skinny_gemm import gemm

        self._orig_cu_num_for = gemm._cu_num_for
        gemm._cu_num_for = lambda device: 228

    def tearDown(self):
        from sglang.srt.layers.skinny_gemm import gemm

        gemm._cu_num_for = self._orig_cu_num_for

    def _stub(self, fill_value=1.0):
        calls = []

        class StubExt:
            def skinny_gemm_bf16(self, x, w, scratch, rows_per_wg, split_k):
                calls.append((x, w, rows_per_wg, split_k))
                scratch.fill_(fill_value)

        return StubExt(), calls

    def test_forwards_shapes_and_config_no_bias(self):
        from sglang.srt.layers.skinny_gemm import gemm

        stub, calls = self._stub(fill_value=2.0)
        original = gemm.get_extension
        gemm.get_extension = lambda: stub
        try:
            x = torch.zeros(8, 4096, dtype=torch.bfloat16)
            w = torch.zeros(2048, 4096, dtype=torch.bfloat16)
            y = gemm.skinny_gemm(x, w, bias=None)
        finally:
            gemm.get_extension = original

        self.assertEqual(len(calls), 1)
        got_x, got_w, rows_per_wg, split_k = calls[0]
        self.assertIs(got_x, x)
        self.assertIs(got_w, w)
        self.assertEqual(rows_per_wg, 8)
        self.assertEqual(split_k, 1)
        self.assertEqual(y.shape, (8, 2048))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.all(y == 2.0))

    def test_adds_bias(self):
        from sglang.srt.layers.skinny_gemm import gemm

        stub, _ = self._stub(fill_value=1.0)
        original = gemm.get_extension
        gemm.get_extension = lambda: stub
        try:
            x = torch.zeros(4, 64, dtype=torch.bfloat16)
            w = torch.zeros(32, 64, dtype=torch.bfloat16)
            bias = torch.full((32,), 3.0, dtype=torch.bfloat16)
            y = gemm.skinny_gemm(x, w, bias=bias)
        finally:
            gemm.get_extension = original

        self.assertTrue(torch.all(y == 4.0))

    def test_rejects_non_contiguous_input(self):
        from sglang.srt.layers.skinny_gemm import gemm

        x = torch.zeros(8, 128, dtype=torch.bfloat16).t()  # non-contiguous
        w = torch.zeros(64, 8, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            gemm.skinny_gemm(x, w)

    def test_reuses_scratch_across_calls_same_shape(self):
        from sglang.srt.layers.skinny_gemm import gemm

        stub, _ = self._stub()
        original = gemm.get_extension
        gemm.get_extension = lambda: stub
        try:
            x = torch.zeros(8, 64, dtype=torch.bfloat16)
            w = torch.zeros(32, 64, dtype=torch.bfloat16)
            scratch_before = gemm._get_scratch(8, 32, x.device)
            gemm.skinny_gemm(x, w)
            scratch_after = gemm._get_scratch(8, 32, x.device)
        finally:
            gemm.get_extension = original

        self.assertIs(scratch_before, scratch_after)


@unittest.skipUnless(torch is not None, "requires torch")
class TestUnquantRoutingDecision(unittest.TestCase):
    """``UnquantizedLinearMethod.apply``'s aiter branch: route to the
    skinny GEMM when the gate is on and supports() holds, else fall through
    to tgemm.mm. unquant.py's ``if _use_aiter:`` import block only runs
    when the real aiter package is present, so on a CPU-only box (no
    ROCm/aiter) it imports cleanly with ``_use_aiter is False`` and never
    defines ``tgemm``/``skinny_gemm`` in its namespace; these tests
    monkeypatch those module globals directly to exercise the branch
    without requiring aiter or a GPU.
    """

    def _make_layer(self, n=64, k=128):
        import types

        return types.SimpleNamespace(weight=torch.zeros(n, k, dtype=torch.bfloat16))

    def _patch_aiter_branch(self, unquant, supports_return, calls):
        class FakeTgemm:
            def mm(self, x, w, bias, otype):
                calls.append(("tgemm", x, w, bias))
                return torch.empty(x.shape[0], w.shape[0], dtype=otype)

        def fake_skinny_gemm(x, w, bias=None):
            calls.append(("skinny", x, w, bias))
            return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype)

        def fake_supports(m, n, k, dtype):
            return supports_return

        unquant._use_aiter = True
        unquant.tgemm = FakeTgemm()
        unquant.skinny_gemm = fake_skinny_gemm
        unquant._skinny_gemm_supports = fake_supports

    def test_routes_to_skinny_gemm_when_gate_on_and_supported(self):
        import sglang.srt.layers.quantization.unquant as unquant
        from sglang.srt.environ import envs

        calls = []
        orig_use_aiter = unquant._use_aiter
        orig_tgemm = getattr(unquant, "tgemm", None)
        orig_skinny = getattr(unquant, "skinny_gemm", None)
        orig_supports = getattr(unquant, "_skinny_gemm_supports", None)
        self._patch_aiter_branch(unquant, supports_return=True, calls=calls)
        try:
            with envs.SGLANG_SKINNY_GEMM.override("1"):
                method = unquant.UnquantizedLinearMethod()
                layer = self._make_layer()
                x = torch.zeros(8, 128, dtype=torch.bfloat16)
                method.apply(layer, x, bias=None)
        finally:
            unquant._use_aiter = orig_use_aiter
            unquant.tgemm = orig_tgemm
            unquant.skinny_gemm = orig_skinny
            unquant._skinny_gemm_supports = orig_supports

        self.assertEqual([c[0] for c in calls], ["skinny"])

    def test_falls_through_to_tgemm_when_gate_off(self):
        import sglang.srt.layers.quantization.unquant as unquant
        from sglang.srt.environ import envs

        calls = []
        orig_use_aiter = unquant._use_aiter
        orig_tgemm = getattr(unquant, "tgemm", None)
        orig_skinny = getattr(unquant, "skinny_gemm", None)
        orig_supports = getattr(unquant, "_skinny_gemm_supports", None)
        self._patch_aiter_branch(unquant, supports_return=True, calls=calls)
        try:
            with envs.SGLANG_SKINNY_GEMM.override("0"):
                method = unquant.UnquantizedLinearMethod()
                layer = self._make_layer()
                x = torch.zeros(8, 128, dtype=torch.bfloat16)
                method.apply(layer, x, bias=None)
        finally:
            unquant._use_aiter = orig_use_aiter
            unquant.tgemm = orig_tgemm
            unquant.skinny_gemm = orig_skinny
            unquant._skinny_gemm_supports = orig_supports

        self.assertEqual([c[0] for c in calls], ["tgemm"])

    def test_falls_through_to_tgemm_when_supports_rejects_shape(self):
        import sglang.srt.layers.quantization.unquant as unquant
        from sglang.srt.environ import envs

        calls = []
        orig_use_aiter = unquant._use_aiter
        orig_tgemm = getattr(unquant, "tgemm", None)
        orig_skinny = getattr(unquant, "skinny_gemm", None)
        orig_supports = getattr(unquant, "_skinny_gemm_supports", None)
        self._patch_aiter_branch(unquant, supports_return=False, calls=calls)
        try:
            with envs.SGLANG_SKINNY_GEMM.override("1"):
                method = unquant.UnquantizedLinearMethod()
                layer = self._make_layer()
                x = torch.zeros(8, 128, dtype=torch.bfloat16)
                method.apply(layer, x, bias=None)
        finally:
            unquant._use_aiter = orig_use_aiter
            unquant.tgemm = orig_tgemm
            unquant.skinny_gemm = orig_skinny
            unquant._skinny_gemm_supports = orig_supports

        self.assertEqual([c[0] for c in calls], ["tgemm"])


if __name__ == "__main__":
    unittest.main()
