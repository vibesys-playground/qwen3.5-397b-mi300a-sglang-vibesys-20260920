"""Generate a pre-sharded TP=4 ``sharded_state`` artifact for
amd/Qwen3.5-397B-A17B-MXFP4 on the test cluster (MI300A).

Loads the full HF checkpoint once with the validated harness recipe (see
``_server.py`` in this task directory: ``build_launch_argv``/
``build_launch_env``), then writes each rank's already-sharded state dict
with ``Engine.save_sharded_model``.

Key settings and why:

- ``mem_fraction_static=0.85``: this pool must cover model weights plus the
  mandatory Mamba/linear-attention state cache this hybrid model always
  allocates (see ``kv_cache_configurator.py``), not just serving KV. The
  AITER backend also multiplies the input fraction by 0.85 internally
  whenever ``attention_backend == "aiter"`` and context length exceeds 8192
  (``server_args.py``), so the effective fraction here is ~0.7225. That
  leaves enough headroom (~17 GB/rank) for the mamba-cache minimum without
  the rank OOM crashes lower fractions hit during scheduler init.
- ``max_running_requests=8``: the mamba-cache minimum scales with this
  value. No serving happens in this job, so shrinking it keeps that
  mandatory allocation negligible against the memory margin above.
- ``disable_cuda_graph=True``: no serving happens in this job, so graph
  capture is pure wasted time and memory.
- 4 GiB parts (``MAX_PART_BYTES``, not 8 GiB): ``ShardedStateLoader.save_model``
  calls ``safetensors.torch.save_file`` on each accumulated part while the
  model's state_dict tensors are still resident; ``save_file`` copies each
  tensor to host before serializing. On this APU, host and device draw from
  the same unified HBM pool the memory math above is already margin-thin
  on, so keep the transient per-part staging copy small.
- Non-weight files (config.json, tokenizer*, chat_template.jinja,
  generation_config.json, etc.) are not written by ``save_model`` and must
  be copied separately; this script copies everything from the source
  checkpoint whose extension is not ``.bin``/``.pt``/``.safetensors``,
  matching ``examples/runtime/engine/save_sharded_state.py``.

Save mechanism: ``Engine.save_sharded_model`` (``engine.py``) does a
synchronous ``collective_rpc("save_sharded_model")`` to every TP rank's
scheduler process; each rank's ``WeightExporter``
(``model_executor/model_runner_components/weight_exporter.py``) calls
``ShardedStateLoader.save_model(model, path, pattern, max_size)``
(``model_loader/loader.py``), which writes only that rank's
tensor-parallel-sharded state dict to
``<path>/model-rank-{rank}-part-{part}.safetensors``, splitting into a new
part whenever accumulated tensor bytes would exceed ``max_size``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR))
import _server  # noqa: E402

# Source/output/tp default to the site/platform config (see
# config/sites/<site>.toml, config/platforms/<platform>.toml, selected via
# $VIBESYS_SITE) instead of literals here.
SOURCE_MODEL_PATH = _server.DEFAULT_MODEL_PATH
OUTPUT_PATH = _server.DEFAULT_SHARDED_MODEL_PATH
TP_SIZE = _server.DEFAULT_TP
# 4 GiB: see the module docstring above for why the part size stays small.
# Save-specific, not part of the platform config.
MAX_PART_BYTES = 4 * 1024**3

# _server.py's build_launch_env() is the source of truth for the launch env
# overlay; reuse it instead of duplicating the var list here. It returns a
# copy of os.environ with the overlay applied, so fold that back into this
# process's actual environment (the Engine() call below reads os.environ
# directly, not a passed-in dict). The config module _server.py loads is
# always present in this checkout, so no import-failure fallback is needed.
os.environ.update(_server.build_launch_env())


def log(msg: str) -> None:
    print(f"[save_sharded] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def log_free_memory(tag: str) -> None:
    """Best-effort node-level free-HBM readout (driver process is not itself a
    GPU worker -- Engine spawns per-rank subprocesses -- so query via rocm-smi
    rather than torch.cuda, which would open an unrelated context here)."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        log(f"[{tag}] rocm-smi --showmeminfo vram:\n{out.stdout.strip()}")
        if out.stderr.strip():
            log(f"[{tag}] rocm-smi stderr: {out.stderr.strip()}")
    except Exception as e:  # noqa: BLE001
        log(f"[{tag}] rocm-smi query failed: {e!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=SOURCE_MODEL_PATH, help="HF checkpoint to load and reshard (default: site config)."
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Directory to write the sharded_state artifact to (default: site config).",
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, help="Tensor-parallel degree (default: platform config)."
    )
    parser.add_argument(
        "--max-part-bytes",
        type=int,
        default=MAX_PART_BYTES,
        help="Max bytes per saved part file (default 4 GiB; see module docstring).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sglang import Engine

    Path(args.output).mkdir(parents=True, exist_ok=True)

    log_free_memory("before Engine construction")

    log(f"constructing Engine over {args.source} (this loads+shards the full checkpoint)")
    t0 = time.perf_counter()
    # attention_backend/page_size/weight-loader recipe come from the
    # platform config (config/platforms/<platform>.toml) via _server.py, so
    # this save job loads the checkpoint the same way the served launch
    # does. mem_fraction_static/max_running_requests/disable_cuda_graph
    # below are save-specific (see module docstring) and stay literal.
    platform = _server._PLATFORM  # noqa: SLF001
    llm = Engine(
        model_path=args.source,
        tp_size=args.tp,
        trust_remote_code=True,
        attention_backend=platform.attention_backend,
        page_size=platform.page_size,
        mem_fraction_static=0.85,
        max_running_requests=8,
        disable_cuda_graph=True,
        weight_loader_disable_mmap=platform.hf_loader.disable_mmap,
        model_loader_extra_config=platform.hf_loader.extra_config,
    )
    t1 = time.perf_counter()
    log(f"Engine construction (load+shard) done in {t1 - t0:.2f}s")

    sa = llm.server_args
    log(
        "effective ServerArgs memory/schedule settings: "
        f"mem_fraction_static={sa.mem_fraction_static} "
        f"max_running_requests={sa.max_running_requests} "
        f"disable_cuda_graph={sa.disable_cuda_graph} "
        f"context_length={sa.context_length} "
        f"attention_backend={sa.attention_backend}"
    )
    log_free_memory("after Engine construction")

    log(f"saving sharded state to {args.output} (max_size={args.max_part_bytes} bytes/part)")
    t2 = time.perf_counter()
    llm.save_sharded_model(path=args.output, pattern=None, max_size=args.max_part_bytes)
    t3 = time.perf_counter()
    log(f"save_sharded_model done in {t3 - t2:.2f}s")

    log("copying non-weight files from source checkpoint")
    copied = []
    for name in sorted(os.listdir(args.source)):
        ext = os.path.splitext(name)[1]
        src = os.path.join(args.source, name)
        dst = os.path.join(args.output, name)
        if ext in (".bin", ".pt", ".safetensors"):
            continue
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
        copied.append(name)
    log(f"copied {len(copied)} non-weight entries: {copied}")

    log("done; exiting cleanly (Engine.shutdown is atexit-registered)")


if __name__ == "__main__":
    main()
