"""GPU functional/perf check for the HIP grouped-GEMM fused MXFP4 MoE path
(manual, not registered with any CI suite -- run by hand on real MI300A
hardware with real weights).

Loads one MoE layer's real w13/w2 weights from a ``save_sharded_model``-style
safetensors shard (path given on the CLI), then for M in
{16, 32, 48, 64, 256, 1024}:

  1. Runs SGLang's production Triton MXFP4 w4a16 MoE path (the same
     ``MoeRunner(MoeRunnerBackend.TRITON, ...)`` call
     ``quark_w4a4_mxfp4_moe.py``'s ``_use_triton_moe_mxfp4`` branch makes).
  2. Runs ``sglang.srt.layers.moe.mxfp4_fused.fused_experts_mxfp4_hip``
     directly (bypassing the env gate/MoeRunner plumbing -- this script
     calls the function the same way ``apply_weights`` does when
     ``SGLANG_MXFP4_MOE_HIP`` is set), plus its two kernel launches
     (``moe_gemm_w4a16_stage1``/``stage2``) individually, for per-stage
     timing. stage1 is timed twice: once under its shipped per-launch
     dispatch ("new": auto-picks scaffold/big by sorted-block count, see
     the .cu file's "Stage1 per-launch dispatch" section) and once forced
     onto the "big" templated kernel unconditionally ("old": this model's
     dispatch before that change, always big for K=4096).

Both are compared against a from-scratch fp32 MoE reference computed from an
exact (no lossy conversion) dequant of the same real weights, and against
each other. Prints max/mean/L2 relative error (a PASS/FAIL line vs. a
1e-2 relative-L2 gate) and median-of-N per-call timing (whole call, plus
stage1/stage2 broken out, old vs. new stage1 dispatch).

At M in {16, 32, 48, 64} (the sizes small enough for the dispatch choice to
matter) also sweeps stage1 alone across all three kernel candidates
(scaffold, small templated, big templated) via the .cu file's
``SGLANG_MXFP4_FUSED_FORCE_{SCAFFOLD,SMALL,BIG}_STAGE1`` env overrides, and
prints a decision table -- the raw data the shipped
``STAGE1_SCAFFOLD_BLOCK_THRESHOLD`` constant was picked from.

Also captures the whole HIP call (moe_align_block_size + both kernel
launches + the final view+sum reduction) into a CUDA graph at M=16 and
replays it 3 times, checking the output tensor is bit-identical across
replays (see fused_moe.py's CUDA-graph-safety docstring: every buffer is
sized from host-visible shapes only, and the one device-scalar read
(``num_tokens_post_padded``) happens in-kernel, never host-synced).

Mirrors test/manual/test_mxfp4_hybrid_layer.py's structure (same weight
loader, exact-dequant reference, and production-Triton call) and
scratchpad/loop/moe-hybrid/bench_hybrid2.py's weight reading and reference
before that.

Requires: a ROCm/HIP gfx942 device and a checkpoint directory containing
``model-rank-{rank}-part-*.safetensors`` shards (e.g. produced by
``.vibesys/tasks/multiturn/tools/save_sharded.py``). Does NOT require aiter
(this path is a from-scratch HIP kernel, no aiter dependency).

Usage:
    python3 test/manual/test_mxfp4_fused_moe.py \\
        --model-dir /path/to/Qwen3.5-397B-A17B-MXFP4-sharded-tp4 \\
        --layer-idx 30 --rank 0 --top-k 10 \\
        --results-json /path/to/results.json
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

OCP_MX_BLOCK_SIZE = 32
REL_L2_GATE = 1e-2

# M values small enough for stage1's per-launch dispatch choice to matter
# (see the .cu file's STAGE1_SCAFFOLD_BLOCK_THRESHOLD): the decision-table
# sweep (scaffold vs. small vs. big) runs at these.
STAGE1_SWEEP_M = (16, 32, 48, 64)
# Full M sweep for correctness/timing/graph-replay.
FULL_M = (16, 32, 48, 64, 256, 1024)

# Names must match the .cu file's env_flag_set() calls in
# moe_gemm_w4a16_stage1.
_FORCE_SCAFFOLD_ENV = "SGLANG_MXFP4_FUSED_FORCE_SCAFFOLD_STAGE1"
_FORCE_SMALL_ENV = "SGLANG_MXFP4_FUSED_FORCE_SMALL_STAGE1"
_FORCE_BIG_ENV = "SGLANG_MXFP4_FUSED_FORCE_BIG_STAGE1"


@contextlib.contextmanager
def _env_override(name: str | None):
    """Sets ``os.environ[name] = "1"`` for the block, restoring the prior
    value (or absence) on exit. ``name=None`` is a no-op (the "auto"/shipped
    dispatch case -- no env var set)."""
    if name is None:
        yield
        return
    had_prev = name in os.environ
    prev = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if had_prev:
            os.environ[name] = prev
        else:
            del os.environ[name]


def load_real_layer(model_dir: str, layer_idx: int, rank: int, device: torch.device):
    from safetensors import safe_open

    prefix = f"model.layers.{layer_idx}.mlp.experts."
    names = {
        "w13_weight": prefix + "w13_weight",
        "w13_weight_scale": prefix + "w13_weight_scale",
        "w2_weight": prefix + "w2_weight",
        "w2_weight_scale": prefix + "w2_weight_scale",
    }
    found = {}
    files = sorted(
        glob.glob(os.path.join(model_dir, f"model-rank-{rank}-part-*.safetensors"))
    )
    if not files:
        raise FileNotFoundError(
            f"no shard files matching model-rank-{rank}-part-*.safetensors under {model_dir}"
        )
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for short, full in names.items():
                if short not in found and full in keys:
                    found[short] = sf.get_tensor(full).to(device)
        if len(found) == len(names):
            break
    missing = set(names) - set(found)
    if missing:
        raise KeyError(
            f"missing tensors for layer {layer_idx} rank {rank}: {missing} (searched {files})"
        )
    return (
        found["w13_weight"],
        found["w13_weight_scale"],
        found["w2_weight"],
        found["w2_weight_scale"],
    )


def exact_dequant_fp32(
    w_packed: torch.Tensor,
    w_scale_e8m0: torch.Tensor,
    group_size: int = OCP_MX_BLOCK_SIZE,
) -> torch.Tensor:
    """Reference MXFP4 dequant, no chan_exp rescale -- the numerics oracle
    for both the production Triton path and the HIP fused kernel (both
    consume the raw checkpoint value, unlike the aiter-shuffled hybrid
    path's fp8 target)."""
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=w_packed.device,
    )
    lo = (w_packed & 0xF).long()
    hi = ((w_packed >> 4) & 0xF).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).flatten(-2)  # even=lo, odd=hi
    block_exp = (w_scale_e8m0.to(torch.int32) - 127).float()
    block_exp = block_exp.repeat_interleave(group_size, dim=-1)
    return vals * torch.exp2(block_exp)


def rel_err_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def torch_naive_moe_ref(hidden_states, w13_exact, w2_exact, topk_ids, topk_weights):
    M, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(M, hidden_size, dtype=torch.float32, device=hidden_states.device)
    for m in range(M):
        acc = torch.zeros(hidden_size, dtype=torch.float32, device=hidden_states.device)
        h = hidden_states[m].float()
        for j in range(top_k):
            e = int(topk_ids[m, j].item())
            gate_up = h @ w13_exact[e].T
            gate, up = gate_up.chunk(2)
            inter = F.silu(gate) * up
            down = inter @ w2_exact[e].T
            acc = acc + down * topk_weights[m, j].float()
        out[m] = acc
    return out


def build_routing(M, num_experts, hidden_size, top_k, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(M, hidden_size, generator=g).to(
        device=device, dtype=torch.bfloat16
    )
    random_gate = torch.randn(num_experts, hidden_size, generator=g).to(
        device=device, dtype=torch.bfloat16
    )
    router_logits = hidden_states.float() @ random_gate.float().T
    topk_vals, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    return hidden_states, topk_ids.to(torch.int64), topk_weights


def median_time_ms(fn, reps: int = 50, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def time_hip_stages(
    hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, reps: int
):
    """Time moe_gemm_w4a16_stage1/stage2 individually, plus the whole
    fused_experts_mxfp4_hip call, all median-of-``reps``. Stage1 is timed
    twice: once under the shipped auto dispatch ("new") and once forced
    onto the "big" templated kernel ("old", this model's pre-per-launch-
    dispatch behavior -- see moe_gemm_w4a16_stage1's env override).
    Stage timings replicate fused_experts_mxfp4_hip's own buffer setup so
    the isolated stage calls see the exact same shapes/strides as the real
    call."""
    from sglang.srt.layers.moe.mxfp4_fused.fused_moe import BLOCK_M, get_extension
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    num_tokens, hidden_size = hidden_states.shape
    E = w13.shape[0]
    inter_size = w13.shape[1] // 2
    device = hidden_states.device
    ext = get_extension()

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, E
    )
    num_valid_tokens = topk_ids.numel()
    intermediate = torch.empty(
        (num_valid_tokens, inter_size), dtype=torch.bfloat16, device=device
    )
    down_out = torch.empty(
        (num_valid_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    topk_weights_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()
    hidden_states_c = hidden_states.contiguous()

    def run_stage1():
        ext.moe_gemm_w4a16_stage1(
            hidden_states_c,
            w13,
            w13_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            intermediate,
            topk_ids.shape[1],
            num_valid_tokens,
        )

    def run_stage2():
        ext.moe_gemm_w4a16_stage2(
            intermediate,
            w2,
            w2_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights_flat,
            down_out,
            num_valid_tokens,
        )

    # Populate `intermediate` for a realistic stage2 timing (not all-zero
    # input), then time each stage independently.
    run_stage1()
    stage1_new_ms = median_time_ms(run_stage1, reps=reps)
    with _env_override(_FORCE_BIG_ENV):
        stage1_old_ms = median_time_ms(run_stage1, reps=reps)
    stage2_ms = median_time_ms(run_stage2, reps=reps)

    from sglang.srt.layers.moe.mxfp4_fused import fused_experts_mxfp4_hip

    def run_whole():
        return fused_experts_mxfp4_hip(
            hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids
        )

    total_new_ms = median_time_ms(run_whole, reps=reps)
    with _env_override(_FORCE_BIG_ENV):
        total_old_ms = median_time_ms(run_whole, reps=reps)
    return {
        "stage1_new_ms": stage1_new_ms,
        "stage1_old_ms": stage1_old_ms,
        "stage2_ms": stage2_ms,
        "total_new_ms": total_new_ms,
        "total_old_ms": total_old_ms,
    }


def sweep_stage1_variants(hidden_states, w13, w13_scale, topk_ids, reps: int):
    """Times stage1 alone under each of the three kernel candidates
    (scaffold, small templated, big templated) via the .cu file's force
    envs, at the caller's current M. Returns {mode: ms} plus the sorted
    16-row block count (grid.x) this M produced -- the raw decision-table
    data the shipped STAGE1_SCAFFOLD_BLOCK_THRESHOLD was picked from."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.mxfp4_fused.fused_moe import BLOCK_M, get_extension

    E = w13.shape[0]
    device = hidden_states.device
    inter_size = w13.shape[1] // 2
    ext = get_extension()

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, E
    )
    num_valid_tokens = topk_ids.numel()
    intermediate = torch.empty(
        (num_valid_tokens, inter_size), dtype=torch.bfloat16, device=device
    )
    hidden_states_c = hidden_states.contiguous()

    def run_stage1():
        ext.moe_gemm_w4a16_stage1(
            hidden_states_c,
            w13,
            w13_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            intermediate,
            topk_ids.shape[1],
            num_valid_tokens,
        )

    results = {}
    for mode, env_name in (
        ("scaffold", _FORCE_SCAFFOLD_ENV),
        ("small", _FORCE_SMALL_ENV),
        ("big", _FORCE_BIG_ENV),
        ("auto", None),
    ):
        with _env_override(env_name):
            run_stage1()  # warm this variant's kernel/JIT path
            results[mode] = median_time_ms(run_stage1, reps=reps)
    return results, int(expert_ids.shape[0])


def cuda_graph_replay_check(run_hip, hidden_states, topk_ids, topk_weights):
    """Capture the whole HIP call in a CUDA graph at the caller's current M
    (expected: 16) and replay 3 times, checking the output is bit-identical
    across replays. Uses static input tensors (copy-in before each replay)
    per the standard torch CUDA-graph capture pattern."""
    static_hidden = hidden_states.clone()
    static_topk_ids = topk_ids.clone()
    static_topk_weights = topk_weights.clone()

    # Warm up outside the graph (JIT/allocator warmup) on a side stream, per
    # torch.cuda.graph's documented capture prerequisites.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_hip(static_hidden, static_topk_ids, static_topk_weights)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = run_hip(static_hidden, static_topk_ids, static_topk_weights)

    outputs = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        outputs.append(static_out.clone())

    identical = all(torch.equal(outputs[0], o) for o in outputs[1:])
    max_diff = max(
        (outputs[0].float() - o.float()).abs().max().item() for o in outputs[1:]
    )
    return identical, max_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, help="save_sharded_model checkpoint directory"
    )
    parser.add_argument("--layer-idx", type=int, default=30)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--top-k", type=int, default=10, help="Must match the checkpoint's router top_k"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reps",
        type=int,
        default=50,
        help="Timed repetitions per (M, path), median taken",
    )
    parser.add_argument("--results-json", default=None)
    args = parser.parse_args()

    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.mxfp4_fused import fused_experts_mxfp4_hip
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import is_gfx95_supported, is_gfx942_supported, is_hip
    from sglang.test.layer_ut_utils import init_single_process_dist

    if not (is_hip() and is_gfx942_supported() and not is_gfx95_supported()):
        raise SystemExit(
            "This test targets gfx942 (MI300A) specifically -- the HIP kernel is "
            f"gfx942-only: is_hip={is_hip()} is_gfx942_supported={is_gfx942_supported()} "
            f"is_gfx95_supported={is_gfx95_supported()}."
        )

    # See test_mxfp4_hybrid_layer.py's identical setup: the production
    # Triton path reads process-global config via sglang.srt.runtime_context,
    # normally installed by full server bootstrap; install a minimal one by
    # hand, plus a world=1 TP group (srt layers require it even at tp=1).
    publish(ServerArgs(model_path="dummy"), role="test")
    init_single_process_dist(master_port=29877)

    device = torch.device("cuda")
    w13, w13_scale, w2, w2_scale = load_real_layer(
        args.model_dir, args.layer_idx, args.rank, device
    )
    num_experts, w13_up_dim, packed_hidden = w13.shape
    hidden_size = packed_hidden * 2
    inter_dim = w2.shape[2] * 2

    exact_w13 = exact_dequant_fp32(w13, w13_scale)
    exact_w2 = exact_dequant_fp32(w2, w2_scale)

    moe_runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=inter_dim,
        top_k=args.top_k,
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        gate_up_interleaved=True,
    )
    triton_quant_info = TritonMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_int4_w4a16=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[0, OCP_MX_BLOCK_SIZE],
    )
    triton_runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def run_production(hidden_states, topk_ids, topk_weights):
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
        )
        return triton_runner.run(dispatch_output, triton_quant_info).hidden_states

    def run_hip(hidden_states, topk_ids, topk_weights):
        return fused_experts_mxfp4_hip(
            hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids
        )

    results = {
        "correctness": {},
        "timing": {},
        "graph_replay": {},
        "stage1_dispatch_sweep": {},
    }
    any_fail = False

    for M in FULL_M:
        hidden_states, topk_ids, topk_weights = build_routing(
            M, num_experts, hidden_size, args.top_k, device, args.seed
        )
        ref = torch_naive_moe_ref(
            hidden_states, exact_w13, exact_w2, topk_ids, topk_weights
        )

        out_prod = run_production(hidden_states, topk_ids, topk_weights)
        out_hip = run_hip(hidden_states, topk_ids, topk_weights)

        def report(out, name):
            diff = (out.float() - ref.float()).abs()
            rel = diff / ref.float().abs().clamp_min(1e-6)
            l2 = rel_err_l2(out, ref)
            print(
                f"M={M:5d} {name:12s} rel_l2={l2:10.6f} "
                f"max_rel={rel.max().item():8.4%} mean_rel={rel.mean().item():8.4%}"
            )
            return l2

        prod_l2 = report(out_prod, "production")
        hip_l2 = report(out_hip, "hip-fused")
        prod_vs_hip = rel_err_l2(out_hip, out_prod)
        gate_pass = hip_l2 < REL_L2_GATE
        status = "PASS" if gate_pass else "FAIL"
        print(
            f"M={M:5d} {'prod-vs-hip':12s} rel_l2={prod_vs_hip:10.6f} "
            f"gate(<{REL_L2_GATE})={status}"
        )
        any_fail = any_fail or not gate_pass
        results["correctness"][str(M)] = {
            "production_rel_l2": prod_l2,
            "hip_rel_l2": hip_l2,
            "prod_vs_hip_rel_l2": prod_vs_hip,
            "gate_pass": gate_pass,
        }

        prod_total_ms = median_time_ms(
            lambda: run_production(hidden_states, topk_ids, topk_weights),
            reps=args.reps,
        )
        timing = time_hip_stages(
            hidden_states,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            args.reps,
        )
        print(
            f"M={M:5d} timing(median of {args.reps}): production={prod_total_ms:8.4f} ms "
            f"hip_total_new={timing['total_new_ms']:8.4f} ms "
            f"hip_total_old={timing['total_old_ms']:8.4f} ms "
            f"hip_stage1_new={timing['stage1_new_ms']:8.4f} ms "
            f"hip_stage1_old={timing['stage1_old_ms']:8.4f} ms "
            f"hip_stage2={timing['stage2_ms']:8.4f} ms"
        )
        results["timing"][str(M)] = {"production_total_ms": prod_total_ms, **timing}

        if M in STAGE1_SWEEP_M:
            sweep_ms, num_blocks = sweep_stage1_variants(
                hidden_states, w13, w13_scale, topk_ids, args.reps
            )
            print(
                f"M={M:5d} stage1_dispatch_sweep(median of {args.reps}, "
                f"num_blocks={num_blocks}): "
                f"scaffold={sweep_ms['scaffold']:8.4f} ms "
                f"small={sweep_ms['small']:8.4f} ms "
                f"big={sweep_ms['big']:8.4f} ms "
                f"auto={sweep_ms['auto']:8.4f} ms"
            )
            results["stage1_dispatch_sweep"][str(M)] = {
                "num_blocks": num_blocks,
                **sweep_ms,
            }

        if M == 16:
            identical, max_diff = cuda_graph_replay_check(
                run_hip, hidden_states, topk_ids, topk_weights
            )
            print(
                f"M={M:5d} cuda_graph_replay: identical_across_3_replays={identical} "
                f"max_diff={max_diff}"
            )
            results["graph_replay"] = {"identical": identical, "max_diff": max_diff}
            any_fail = any_fail or not identical

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)

    if any_fail:
        raise SystemExit("FAIL: one or more correctness/graph-replay gates failed")
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
