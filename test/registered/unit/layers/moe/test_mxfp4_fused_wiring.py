"""CPU-only unit tests for the HIP grouped-GEMM fused MXFP4 MoE scaffold:
``sglang.srt.layers.moe.mxfp4_fused.fused_moe.fused_experts_mxfp4_hip``'s
buffer sizing/dispatch, and ``quark_w4a4_mxfp4_moe.py``'s
``SGLANG_MXFP4_MOE_HIP`` routing gate.

No GPU, ROCm, or real HIP toolchain is required: the JIT extension
(``sglang.srt.layers.moe.mxfp4_fused.fused_moe.get_extension``) is stubbed
out with a fake module rather than built, and CPU tensors stand in for CUDA
ones (the wrapper never touches ``.is_cuda()`` itself -- that check lives in
the C++ TORCH_CHECKs, which the stub never reaches). These tests exercise
Python-side plumbing only; the kernel math is untested here (see
test/manual/test_mxfp4_fused_moe.py for the GPU correctness check).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@unittest.skipUnless(torch is not None, "requires torch")
class TestFusedExpertsMxfp4HipWiring(unittest.TestCase):
    """Exercises fused_experts_mxfp4_hip's buffer sizing and call sequence
    with moe_align_block_size and the JIT extension both stubbed out."""

    def _fake_align(self):
        """Returns a moe_align_block_size stand-in whose outputs have the
        real function's shapes/dtypes (int32 sorted_token_ids/expert_ids,
        int32 num_tokens_post_padded), so the wrapper's downstream shape
        arithmetic is exercised the same way it would be against the real
        thing, without needing a GPU kernel."""

        def fake(topk_ids, block_size, num_experts):
            numel = topk_ids.numel()
            max_padded = numel + (num_experts + 1) * (block_size - 1)
            num_m_blocks = (max_padded + block_size - 1) // block_size
            sorted_token_ids = torch.arange(max_padded, dtype=torch.int32) % max(
                numel, 1
            )
            expert_ids = torch.zeros(num_m_blocks, dtype=torch.int32)
            num_tokens_post_padded = torch.tensor([max_padded], dtype=torch.int32)
            return sorted_token_ids, expert_ids, num_tokens_post_padded

        return fake

    def _fake_extension(self, calls):
        ext = mock.Mock()

        def stage1(
            hidden_states,
            w13_fp4,
            w13_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            intermediate,
            top_k,
            num_valid_tokens,
        ):
            calls["stage1_args"] = dict(
                top_k=top_k,
                num_valid_tokens=num_valid_tokens,
                intermediate_shape=tuple(intermediate.shape),
                intermediate_dtype=intermediate.dtype,
            )
            intermediate.fill_(1.0)

        def stage2(
            intermediate,
            w2_fp4,
            w2_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            down_out,
            num_valid_tokens,
        ):
            calls["stage2_args"] = dict(
                num_valid_tokens=num_valid_tokens,
                down_out_shape=tuple(down_out.shape),
                down_out_dtype=down_out.dtype,
                topk_weights_dtype=topk_weights.dtype,
                topk_weights_shape=tuple(topk_weights.shape),
            )
            down_out.fill_(2.0)

        ext.moe_gemm_w4a16_stage1 = mock.Mock(side_effect=stage1)
        ext.moe_gemm_w4a16_stage2 = mock.Mock(side_effect=stage2)
        return ext

    def _call(
        self, num_tokens=4, top_k=2, num_experts=3, hidden_size=32, inter_size=16
    ):
        from sglang.srt.layers.moe.mxfp4_fused import fused_moe as mod

        calls = {}
        hidden_states = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16)
        w13_fp4 = torch.zeros(
            num_experts, 2 * inter_size, hidden_size // 2, dtype=torch.uint8
        )
        w13_scale = torch.zeros(
            num_experts, 2 * inter_size, hidden_size // 32, dtype=torch.uint8
        )
        w2_fp4 = torch.zeros(
            num_experts, hidden_size, inter_size // 2, dtype=torch.uint8
        )
        w2_scale = torch.zeros(
            num_experts, hidden_size, inter_size // 32, dtype=torch.uint8
        )
        topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)
        topk_weights = torch.ones(num_tokens, top_k, dtype=torch.bfloat16)

        with mock.patch.object(
            mod, "moe_align_block_size", self._fake_align()
        ), mock.patch.object(
            mod, "get_extension", return_value=self._fake_extension(calls)
        ):
            out = mod.fused_experts_mxfp4_hip(
                hidden_states,
                w13_fp4,
                w13_scale,
                w2_fp4,
                w2_scale,
                topk_weights,
                topk_ids,
            )
        return out, calls, num_tokens, top_k, hidden_size

    def test_output_shape_and_dtype(self):
        out, _, num_tokens, top_k, hidden_size = self._call()
        self.assertEqual(tuple(out.shape), (num_tokens, hidden_size))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_reduction_sums_over_top_k(self):
        # stage2 fills down_out with 2.0 everywhere; the wrapper's final
        # reduction sums top_k of those per token, so every output element
        # must equal 2.0 * top_k regardless of routing content.
        out, _, _, top_k, _ = self._call(top_k=3)
        expected = torch.full_like(out, 2.0 * 3, dtype=torch.float32)
        self.assertTrue(torch.allclose(out.float(), expected))

    def test_stage1_buffer_sizing(self):
        num_tokens, top_k, num_experts = 5, 4, 6
        _, calls, _, _, _ = self._call(
            num_tokens=num_tokens, top_k=top_k, num_experts=num_experts, inter_size=16
        )
        num_valid_tokens = num_tokens * top_k
        self.assertEqual(calls["stage1_args"]["top_k"], top_k)
        self.assertEqual(calls["stage1_args"]["num_valid_tokens"], num_valid_tokens)
        self.assertEqual(
            calls["stage1_args"]["intermediate_shape"], (num_valid_tokens, 16)
        )
        self.assertEqual(calls["stage1_args"]["intermediate_dtype"], torch.bfloat16)

    def test_stage2_buffer_sizing_and_routed_weight_flattening(self):
        num_tokens, top_k, hidden_size = 5, 4, 32
        _, calls, _, _, _ = self._call(
            num_tokens=num_tokens, top_k=top_k, hidden_size=hidden_size
        )
        num_valid_tokens = num_tokens * top_k
        self.assertEqual(calls["stage2_args"]["num_valid_tokens"], num_valid_tokens)
        self.assertEqual(
            calls["stage2_args"]["down_out_shape"], (num_valid_tokens, hidden_size)
        )
        self.assertEqual(calls["stage2_args"]["down_out_dtype"], torch.bfloat16)
        # topk_weights is passed to stage2 as a flat fp32 view (matching
        # production's topk_weights_ptr + offs_token addressing), regardless
        # of the caller's dtype/shape -- the wrapper's test input above is
        # bf16 [num_tokens, top_k].
        self.assertEqual(calls["stage2_args"]["topk_weights_dtype"], torch.float32)
        self.assertEqual(
            calls["stage2_args"]["topk_weights_shape"], (num_valid_tokens,)
        )

    def test_rejects_non_silu_activation(self):
        from sglang.srt.layers.moe.mxfp4_fused.fused_moe import fused_experts_mxfp4_hip

        hidden_states = torch.zeros(2, 32, dtype=torch.bfloat16)
        w = torch.zeros(1, 1, 1, dtype=torch.uint8)
        topk_ids = torch.zeros(2, 1, dtype=torch.int64)
        topk_weights = torch.ones(2, 1, dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            fused_experts_mxfp4_hip(
                hidden_states, w, w, w, w, topk_weights, topk_ids, activation="gelu"
            )

    def test_rejects_apply_router_weight_on_input(self):
        from sglang.srt.layers.moe.mxfp4_fused.fused_moe import fused_experts_mxfp4_hip

        hidden_states = torch.zeros(2, 32, dtype=torch.bfloat16)
        w = torch.zeros(1, 1, 1, dtype=torch.uint8)
        topk_ids = torch.zeros(2, 1, dtype=torch.int64)
        topk_weights = torch.ones(2, 1, dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            fused_experts_mxfp4_hip(
                hidden_states,
                w,
                w,
                w,
                w,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input=True,
            )


@unittest.skipUnless(torch is not None, "requires torch")
class TestHipMxfp4RoutingGate(unittest.TestCase):
    """Exercises QuarkW4A4MXFp4MoE's SGLANG_MXFP4_MOE_HIP module-level gate
    by reloading the module under mocked platform-detection functions and
    env vars. is_hip=True also triggers a pre-existing (not new to this
    change) unconditional ``import aiter...`` at module scope, so those
    cases stub sys.modules with fake aiter submodules rather than requiring
    aiter to actually be installed in the CPU CI environment."""

    MODULE = "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe"

    def setUp(self):
        # Undo whatever this test leaves imported so later tests (and other
        # test files importing the real module) see the normal, un-reloaded
        # behavior: reload it once more with every patch already torn down.
        self.addCleanup(self._reload_clean)

    def _reload_clean(self):
        if self.MODULE in sys.modules:
            importlib.reload(sys.modules[self.MODULE])

    @staticmethod
    def _fake_aiter_modules():
        """A minimal fake aiter package tree covering every submodule
        quark_w4a4_mxfp4_moe.py imports at module scope when is_hip() is
        True, so reloading it under is_hip=True does not require aiter to
        be installed."""
        mods = {}

        def add(name, **attrs):
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            mods[name] = m

        add("aiter")
        add("aiter.ops")
        add("aiter.ops.triton")
        add("aiter.ops.triton.quant", dynamic_mxfp4_quant=mock.Mock())
        add("aiter.ops.shuffle", shuffle_weight=mock.Mock())
        add("aiter.utility")
        add("aiter.utility.fp4_utils", e8m0_shuffle=mock.Mock())
        return mods

    def _reload_with(self, *, is_hip, is_gfx95, is_gfx942, hip_env):
        import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe as mod

        env = {
            "SGLANG_MXFP4_MOE_HIP": hip_env,
            "SGLANG_USE_AITER": "false",
            # Pinned explicitly so this test's result cannot depend on
            # whatever the ambient test environment happens to have set;
            # "true" is the flag's own default.
            "SGLANG_MXFP4_MOE_TRITON_FALLBACK": "true",
        }
        patches = [
            mock.patch.dict(os.environ, env),
            mock.patch("sglang.srt.utils.is_hip", return_value=is_hip),
            mock.patch(
                "sglang.srt.utils.common.is_gfx95_supported", return_value=is_gfx95
            ),
            mock.patch(
                "sglang.srt.utils.common.is_gfx942_supported", return_value=is_gfx942
            ),
        ]
        if is_hip:
            # is_hip=True makes the reload hit a pre-existing (not new to
            # this change) unconditional `from aiter... import ...` at
            # module scope; stub sys.modules so that succeeds without aiter
            # actually being installed.
            patches.append(mock.patch.dict(sys.modules, self._fake_aiter_modules()))
        with _apply_all(patches):
            importlib.reload(mod)
        return mod

    def test_gate_on_when_hip_gfx942_and_env_set(self):
        mod = self._reload_with(
            is_hip=True, is_gfx95=False, is_gfx942=True, hip_env="true"
        )
        self.assertTrue(mod._use_hip_moe_mxfp4)

    def test_gate_off_by_default(self):
        mod = self._reload_with(
            is_hip=True, is_gfx95=False, is_gfx942=True, hip_env="false"
        )
        self.assertFalse(mod._use_hip_moe_mxfp4)

    def test_gate_off_on_non_gfx942(self):
        mod = self._reload_with(
            is_hip=True, is_gfx95=False, is_gfx942=False, hip_env="true"
        )
        self.assertFalse(mod._use_hip_moe_mxfp4)

    def test_gate_off_on_gfx95(self):
        # gfx95 has native MX hardware support (_use_triton_moe_mxfp4 itself
        # is False there), so the HIP fused-kernel gate must also be off.
        mod = self._reload_with(
            is_hip=True, is_gfx95=True, is_gfx942=False, hip_env="true"
        )
        self.assertFalse(mod._use_hip_moe_mxfp4)

    def test_gate_off_without_hip(self):
        mod = self._reload_with(
            is_hip=False, is_gfx95=False, is_gfx942=False, hip_env="true"
        )
        self.assertFalse(mod._use_hip_moe_mxfp4)


class _apply_all:
    """Applies a list of mock.patch(...) context managers together."""

    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False


if __name__ == "__main__":
    unittest.main()
