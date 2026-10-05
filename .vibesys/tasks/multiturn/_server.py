"""Shared SGLang server lifecycle helpers for the multiturn task.

Used by both ``benchmark/run.py`` and ``accuracy_checker/check.py``. Neither
of those entry points imports ``sglang`` at module load time; this module
only shells out to ``python3 -m sglang.launch_server`` as a subprocess and
polls its HTTP surface, so it is safe to import before sglang is installed.

Machine-specific values (model/artifact paths, launch env and argv, the KV
pool pin, startup timeout) are loaded from ``config/sites/<site>.toml`` and
``config/platforms/<platform>.toml`` at import time (see
``config/loader.py`` and ``README.md``'s Configuration section) rather than
hardcoded here. Site selection is via ``$VIBESYS_SITE`` (default
``"example"``).

The platform config's own ``extra_args`` list (see
``config/platforms/<platform>.toml``) is appended to the launch argv after
the other config-derived flags, followed by the platform's scheduler flags
(see ``scheduler_args()``) when the platform config declares a
``[scheduler]`` table with ``disable_overlap_schedule = true`` and
``VIBESYS_OVERLAP_SCHEDULE`` is not set to a truthy value, then the
platform's speculative-decoding flags (see ``speculative_decode_args()``)
when the platform config declares a ``[speculative]`` table and
``VIBESYS_SPEC_DECODE`` is not set to a falsy value, then the platform's
prefill CUDA-graph flag (see ``prefill_cuda_graph_args()``) when the
platform config declares a ``[prefill_cuda_graph]`` table and
``VIBESYS_PREFILL_GRAPH`` is not set to a falsy value; the default is on
wherever a platform declares any of these three tables.
``$VIBESYS_EXTRA_SERVER_ARGS``, if set, is shell-word-split and appended
after that; see ``extra_server_args()`` and README.md's Configuration
section.

``build_launch_env()`` additionally hard-sets four ``PYTORCH_TUNABLEOP_*``
env vars when the platform config declares ``[tunableop]`` with ``enabled =
true`` and ``VIBESYS_TUNABLEOP`` is not set to a falsy value; see
``tunableop_enabled()`` and README.md's "Tuned dense GEMM tiles (TunableOp)"
section. In that case ``start_server()`` also calls ``stage_tunableop_files()``
before launching, to fan the one committed CSV out into one read-only,
per-device-ordinal copy per TP rank; see that function's docstring for why a
single shared path does not work.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from config.loader import load_platform, load_site  # noqa: E402

_SITE = load_site()
_PLATFORM = load_platform(_SITE.platform)

DEFAULT_MODEL_PATH = _SITE.paths.model
# Used when present via resolve_model_path() below; MODEL_PATH still
# overrides both. See config/sites/<site>.toml for what these paths are and
# why the striped one is preferred.
DEFAULT_SHARDED_MODEL_PATH = _SITE.paths.sharded_artifact
DEFAULT_SHARDED_MODEL_PATH_STRIPED = _SITE.paths.sharded_artifact_striped
SHARDED_STATE_MARKER = "model-rank-0-part-0.safetensors"
# See config/platforms/<platform>.toml for why this is pinned rather than
# left to SGLang's post-load free-memory sizing.
MAX_TOTAL_TOKENS = _PLATFORM.max_total_tokens
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
DEFAULT_TP = _PLATFORM.tp
# See config/sites/<site>.toml: dominated by Lustre-contended load time.
STARTUP_TIMEOUT_SECONDS = _SITE.startup_timeout_s
POLL_INTERVAL_SECONDS = 5.0
SHUTDOWN_GRACE_SECONDS = 30.0


def is_sharded_state_dir(model_path: str) -> bool:
    """True if ``model_path`` holds a ``save_sharded_model`` artifact."""
    return (Path(model_path) / SHARDED_STATE_MARKER).is_file()


def resolve_model_path() -> str:
    """Return the model checkpoint path, honoring the MODEL_PATH override.

    Without an override, prefer the pre-sharded artifact when it exists.
    """
    override = os.environ.get("MODEL_PATH")
    if override:
        return override
    for candidate in (DEFAULT_SHARDED_MODEL_PATH_STRIPED, DEFAULT_SHARDED_MODEL_PATH):
        if is_sharded_state_dir(candidate):
            return candidate
    return DEFAULT_MODEL_PATH


def ensure_sglang_importable(workspace: Path) -> None:
    """Make ``sglang`` importable, installing it from the checkout if needed.

    Tries a dependency-free editable install first (the run container is
    expected to already carry sglang's runtime dependencies); falls back to
    the full ``[all]`` extra only if that does not make the package
    importable.
    """
    if _is_sglang_importable(workspace):
        return
    python_dir = workspace / "python"
    attempts = (
        ["-e", str(python_dir), "--no-deps"],
        ["-e", f"{python_dir}[all]"],
    )
    last_output = ""
    for pip_args in attempts:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", *pip_args],
            capture_output=True,
            text=True,
            check=False,
        )
        last_output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and _is_sglang_importable(workspace):
            return
    raise RuntimeError(
        "sglang is not importable after install attempts; last pip output:\n"
        f"{last_output[-4000:]}"
    )


def _is_sglang_importable(workspace: Path) -> bool:
    """Check whether ``sglang`` would import, without paying for a full
    import.

    A real ``import sglang`` measured ~7s on the test cluster MI300A (sglang's own
    module chain does real work at import time), paid serially in this
    process before the actual server subprocess -- which does its own,
    unavoidable ``import sglang`` regardless -- is even launched.
    ``importlib.util.find_spec`` locates the module by walking ``sys.path``
    and reading just enough of its ``__init__`` to build a spec, without
    executing the module body, which is what actually costs the seconds.

    The common case is this checkout's own ``sglang -> python/sglang``
    symlink at the workspace root: a subprocess launched with
    ``cwd=workspace`` (``start_server``'s ``python3 -m sglang.launch_server``)
    gets that on ``sys.path`` for free via ``-m``'s cwd rule, but this
    call runs in a plain script (``serve.py``) whose ``sys.path[0]`` is its
    own directory, not the workspace, so add it explicitly here.
    """
    import importlib.util

    workspace_str = str(workspace)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
    importlib.invalidate_caches()
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


def build_launch_env(*, tp: int = DEFAULT_TP, run_dir: Path | None = None) -> dict[str, str]:
    """Return the environment overlay for the AMD/ROCm + MXFP4 launch.

    The hard-set and setdefault vars come from the platform config
    (``config/platforms/<platform>.toml``); AITER_JIT_DIR's and
    SGLANG_HIP_EXT_DIR's paths (also setdefault) come from the site config's
    ``paths.aiter_jit_dir`` and ``paths.hip_ext_dir``; see that file for why
    those JIT/build cache dirs must be fixed and pre-warmed rather than a
    fresh per-launch directory.

    When the platform config declares ``[tunableop]`` with ``enabled =
    true`` and ``tunableop_enabled()`` is true (the default), also hard-sets
    four PyTorch TunableOp env vars: ``PYTORCH_TUNABLEOP_ENABLED=1``,
    ``PYTORCH_TUNABLEOP_TUNING=0`` (tuning stays off in serving; the CSV is
    read-only input, never written by the harness), ``PYTORCH_TUNABLEOP_FILENAME``
    set by ``stage_tunableop_files(tp, run_dir)`` to a ``%d``-templated path
    over ``tp`` freshly staged, read-only per-device-ordinal copies of the
    platform config's committed ``tunableop.results_file`` -- one copy per
    TP rank, since each rank reads a different file (see that function's
    docstring) -- and ``PYTORCH_TUNABLEOP_VERBOSE=1``, so PyTorch logs the
    validator check and the "reading tuning results from ..." line per rank
    at init; this is log-only and does not affect dispatch. ``tp`` and
    ``run_dir`` are only consulted in the file-staging branch. See
    ``tunableop_enabled()`` and README.md's "Tuned dense GEMM tiles
    (TunableOp)" section.
    """
    env = dict(os.environ)
    env.update(_PLATFORM.env)
    env.setdefault("AITER_JIT_DIR", _SITE.paths.aiter_jit_dir)
    env.setdefault("SGLANG_HIP_EXT_DIR", _SITE.paths.hip_ext_dir)
    for key, value in _PLATFORM.env_defaults.items():
        env.setdefault(key, value)
    tunableop = _PLATFORM.tunableop
    if tunableop is not None and tunableop.enabled and tunableop_enabled():
        env["PYTORCH_TUNABLEOP_ENABLED"] = "1"
        env["PYTORCH_TUNABLEOP_TUNING"] = "0"
        env["PYTORCH_TUNABLEOP_FILENAME"] = stage_tunableop_files(tp=tp, run_dir=run_dir)
        env["PYTORCH_TUNABLEOP_VERBOSE"] = "1"
    return env


def stage_tunableop_files(*, tp: int, run_dir: Path | None) -> str:
    """Fan the committed TunableOp CSV out into one read-only copy per
    device ordinal, and return the ``%d``-templated
    ``PYTORCH_TUNABLEOP_FILENAME`` value for it.

    PyTorch's TunableOp (``torch/cuda/tunable.py``, torch 2.9.0a0) resolves
    ``PYTORCH_TUNABLEOP_FILENAME`` per process by substituting the device
    ordinal for a literal ``%d`` when the value contains one, or otherwise
    inserting the ordinal before the extension -- either way, each TP rank
    reads a *different* file (``<stem><k>.csv`` for device ``k``). Pointing
    every rank at one shared path (the platform config's committed
    ``tunableop.results_file`` verbatim, the pre-fix behavior) left every
    rank but one with no per-device file to load: job 633800 had only rank
    0 find a file, ranks 1-3 ran untuned, and since a TP step waits for the
    slowest rank, the run measured nothing. Staging ``tp`` copies up front
    (named ``tunableop<k>.csv`` for ``k`` in ``range(tp)``) and pointing
    ``PYTORCH_TUNABLEOP_FILENAME`` at ``.../tunableop%d.csv`` (the literal
    ``%d``) makes every rank find its own copy of the same tuned table.

    The copies land in a fresh directory from ``tempfile.mkdtemp``, created
    under ``run_dir`` (the caller's log/output directory for this launch)
    when given, else in the system temp dir; a fresh directory per launch
    keeps concurrent launches from colliding on the same files. Each copy
    is ``chmod``'d ``0o444``: PyTorch rewrites its TunableOp results file at
    process exit even with ``PYTORCH_TUNABLEOP_TUNING=0`` (writing an empty
    file when nothing was loaded), and no env var suppresses that -- only
    the Python API ``torch.cuda.tunable.write_file_on_exit(False)``, which
    is out of reach from this subprocess-launching harness. Read-only turns
    that exit-time rewrite into a harmless warning instead of silently
    replacing the staged copy.

    Before staging, the committed CSV is rejected outright if it contains
    any carriage return (``\\r``) byte. PyTorch's TunableOp validates a
    loaded file by comparing its five ``Validator`` header lines
    (PT_VERSION/ROCM_VERSION/HIPBLASLT_VERSION/GCN_ARCH_NAME/ROCBLAS_VERSION)
    against the running process with an exact string compare
    (``torch/cuda/tunable.py``); a CRLF-terminated line never matches even
    when both sides print identically, and a failed validator silently
    discards the *entire* results table, not just the affected row, with
    only a single easy-to-miss ``Failed validator`` warning in the server
    log and no boot failure. This previously went undetected for every
    shape and every batch size in a merged CSV (see
    ``manual/dense-keys/results.md`` on the test cluster in the campaign's
    scratch tree): the file loaded and TunableOp-load-verified clean on
    every rank, but zero of the tuned rows were ever actually usable.
    Failing fast here, at staging time, turns that silent full-table loss
    into a loud, immediate error instead of a warning nobody greps for.
    """
    tunableop = _PLATFORM.tunableop
    assert tunableop is not None and tunableop.enabled, "stage_tunableop_files() called with TunableOp not enabled"
    committed_csv = Path(tunableop.results_file)
    committed_bytes = committed_csv.read_bytes()
    if b"\r" in committed_bytes:
        bad_line = next(
            (i + 1 for i, line in enumerate(committed_bytes.split(b"\n")) if b"\r" in line),
            None,
        )
        raise ValueError(
            f"TunableOp results file {committed_csv} contains carriage-return "
            f"(\\r) byte(s), first on line {bad_line}. PyTorch's TunableOp "
            "validator does an exact string compare against its five "
            "Validator header lines; a CRLF-terminated line never matches, "
            "which silently discards the entire results table (not just "
            "that row) with only a warning, no boot failure. Re-save this "
            "file with plain LF line endings (e.g. `tr -d '\\r' < FILE > "
            "FILE.lf && mv FILE.lf FILE`) before committing it."
        )
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix="tunableop-", dir=str(run_dir)))
    else:
        staging_dir = Path(tempfile.mkdtemp(prefix="tunableop-"))
    for device in range(tp):
        dest = staging_dir / f"tunableop{device}.csv"
        shutil.copyfile(committed_csv, dest)
        dest.chmod(0o444)
    print(
        f"[tunableop] staged {tp} read-only per-device file(s) in {staging_dir}",
        file=sys.stderr,
        flush=True,
    )
    return str(staging_dir / "tunableop%d.csv")


def tunableop_enabled() -> bool:
    """True unless ``$VIBESYS_TUNABLEOP`` is set to a falsy value.

    Kill switch for the ``PYTORCH_TUNABLEOP_*`` vars in
    ``build_launch_env()``, from the same ``VIBESYS_*`` family as
    ``VIBESYS_SPEC_DECODE`` and ``VIBESYS_OVERLAP_SCHEDULE``. Falsy values
    are ``"0"``, ``"false"``, and ``"no"`` (case-insensitive); unset or
    anything else is truthy. Only has an effect when the selected platform
    declares ``[tunableop]`` with ``enabled = true`` (see
    ``config/platforms/<platform>.toml``); a platform with no
    ``[tunableop]`` table, or ``enabled = false``, never sets those vars
    regardless of this switch.
    """
    raw = os.environ.get("VIBESYS_TUNABLEOP", "1").strip().lower()
    return raw not in ("0", "false", "no")


def extra_server_args() -> list[str]:
    """Return extra argv tokens from ``$VIBESYS_EXTRA_SERVER_ARGS``, if set.

    The value is parsed with shell-word splitting (``shlex.split``), e.g.
    ``VIBESYS_EXTRA_SERVER_ARGS="--enable-mixed-chunk --chunked-prefill-size
    1024"``. Intended for one-off overrides (A/B comparisons, ad hoc
    debugging) that don't warrant a config/platform change; the tokens are
    appended after the config-derived args, including the platform config's
    own ``extra_args``, so an override earlier in argv takes precedence only
    if SGLang's own arg parser prefers the first occurrence of a flag. Note
    that this can only add flags, not remove one baked into a platform's
    ``extra_args``; see README.md's Configuration section for how to
    disable one of those for an A/B.
    """
    raw = os.environ.get("VIBESYS_EXTRA_SERVER_ARGS", "")
    if not raw.strip():
        return []
    return shlex.split(raw)


def spec_decode_enabled() -> bool:
    """True unless ``$VIBESYS_SPEC_DECODE`` is set to a falsy value.

    Single documented switch for turning off a platform's default
    speculative-decoding config (``[speculative]`` in
    ``config/platforms/<platform>.toml``) without editing a committed file,
    from the same ``VIBESYS_*`` env-var family as ``VIBESYS_SITE``/
    ``VIBESYS_PLATFORM``/``VIBESYS_EXTRA_SERVER_ARGS``. Falsy values are
    ``"0"``, ``"false"``, and ``"no"`` (case-insensitive); unset or anything
    else is truthy. To keep a permanent side without speculative decoding
    instead (e.g. a staged platform copy for a paired A/B), use a platform
    file with no ``[speculative]`` table and select it via
    ``VIBESYS_PLATFORM``; see README.md's Configuration section.
    """
    raw = os.environ.get("VIBESYS_SPEC_DECODE", "1").strip().lower()
    return raw not in ("0", "false", "no")


def speculative_decode_args() -> list[str]:
    """Return ``--speculative-*`` argv tokens, or ``[]`` if not applicable.

    Empty when the selected platform has no ``[speculative]`` table, or
    when ``spec_decode_enabled()`` is false. The backend-independent flags
    (algorithm, step count, topk, draft-token count, the linear-replay
    verify path) come from the platform config; the draft model path comes
    from the site config, never hardcoded here or in a platform TOML, since
    a platform file is shared across every site that runs it.

    Two draft sources, chosen by whether the site config sets
    ``paths.draft_model``:

    - Set: the draft loads from ``paths.draft_model`` with
      ``--speculative-draft-load-format sharded_state``. This is a
      draft-only ``sharded_state`` artifact (see
      ``manual/draft-shard/save_draft_shard_v5.py``); the implicit rule is
      "draft_model set implies sharded_state", since nothing else is ever
      written to that kind of path. Boot cost: the draft's own weight load
      drops from several hundred seconds (see below) to under 10s, because
      ``ShardedStateLoader`` copies tensors into ``model.state_dict()`` by
      key match and never calls the model's own ``load_weights()``.
    - Unset (the default absent a produced artifact): the draft loads from
      ``paths.model`` -- the *original*, unsharded checkpoint -- with the
      platform's own ``draft_load_format`` (normally ``"auto"``).
      ``paths.model`` is the only copy carrying the mtp.* tensors the
      sharded fast-boot target artifact lacks, and this checkpoint's
      architecture is not on the fork's ``speculative_hook.py``
      auto-default allowlist, so the draft path must be set explicitly
      rather than inferred. This path pays ``Qwen3_5ForCausalLMMTP
      .load_weights``'s unthreaded per-expert dispatch (~455-498s
      measured), see config/platforms/<platform>.toml's ``[speculative]``
      comment.
    """
    spec = _PLATFORM.speculative
    if spec is None or not spec_decode_enabled():
        return []
    argv = [
        "--speculative-algorithm",
        spec.algorithm,
        "--speculative-num-steps",
        str(spec.num_steps),
        "--speculative-eagle-topk",
        str(spec.eagle_topk),
        "--speculative-num-draft-tokens",
        str(spec.num_draft_tokens),
    ]
    if spec.enable_linear_replayssm_spec:
        argv.append("--enable-linear-replayssm-spec")
    if _SITE.paths.draft_model:
        draft_model_path = _SITE.paths.draft_model
        draft_load_format = "sharded_state"
    else:
        draft_model_path = _SITE.paths.model
        draft_load_format = spec.draft_load_format
    argv += [
        "--speculative-draft-model-path",
        draft_model_path,
        "--speculative-draft-load-format",
        draft_load_format,
    ]
    return argv


def prefill_graph_enabled() -> bool:
    """True unless ``$VIBESYS_PREFILL_GRAPH`` is set to a falsy value.

    Kill switch for ``prefill_cuda_graph_args()``, from the same
    ``VIBESYS_*`` family as ``VIBESYS_TUNABLEOP``/``VIBESYS_SPEC_DECODE``/
    ``VIBESYS_OVERLAP_SCHEDULE``. Falsy values are ``"0"``, ``"false"``, and
    ``"no"`` (case-insensitive); unset or anything else is truthy. Only has
    an effect when the selected platform declares ``[prefill_cuda_graph]``
    (see ``config/platforms/<platform>.toml``); a platform with no
    ``[prefill_cuda_graph]`` table never emits the flag regardless of this
    switch.
    """
    raw = os.environ.get("VIBESYS_PREFILL_GRAPH", "1").strip().lower()
    return raw not in ("0", "false", "no")


def prefill_cuda_graph_args() -> list[str]:
    """Return ``["--cuda-graph-config", <json>]``, or ``[]`` if not applicable.

    Empty when the selected platform has no ``[prefill_cuda_graph]`` table,
    or when ``prefill_graph_enabled()`` is false. The JSON payload locks the
    prefill phase to the platform config's ``backend`` (e.g. ``"breakable"``)
    over the same 13-value capture list the gemm-pad-m warmup already
    targets (``GEMM_PAD_WARMUP_TOKEN_COUNTS``, defined below in this
    module): the warmup and the graph capture need to cover the identical
    set of padded prefill token counts, so this reuses that list rather
    than a second copy in the platform TOML that could drift out of sync
    with it. Because this argv token is built directly (not shell-word-split
    from an environment variable the way ``$VIBESYS_EXTRA_SERVER_ARGS`` is),
    the JSON string needs no shell quoting here: ``build_launch_argv``'s
    caller (``start_server``) passes argv straight to ``subprocess.Popen``
    as a list, not through a shell. See
    ``config/platforms/<platform>.toml``'s ``[prefill_cuda_graph]`` comment
    for the measured effect and why ``breakable`` is the accepted backend.
    """
    cfg = _PLATFORM.prefill_cuda_graph
    if cfg is None or not prefill_graph_enabled():
        return []
    payload = {"prefill": {"backend": cfg.backend, "bs": GEMM_PAD_WARMUP_TOKEN_COUNTS}}
    return ["--cuda-graph-config", json.dumps(payload)]


def overlap_schedule_enabled() -> bool:
    """True if ``$VIBESYS_OVERLAP_SCHEDULE`` is set to a truthy value.

    Same falsy-value parsing as ``spec_decode_enabled()`` (``"0"``,
    ``"false"``, ``"no"``, case-insensitive; unset or anything else is
    truthy), but the default polarity is reversed: a platform's
    ``[scheduler] disable_overlap_schedule = true`` (see
    ``config/platforms/<platform>.toml``) leaves the overlap scheduler off
    by default, and setting this variable is what turns it back on for one
    run -- it does not turn a default-off scheduler further off. See
    ``scheduler_args()``.
    """
    raw = os.environ.get("VIBESYS_OVERLAP_SCHEDULE", "0").strip().lower()
    return raw not in ("0", "false", "no")


def scheduler_args() -> list[str]:
    """Return ``["--disable-overlap-schedule"]``, or ``[]`` if not applicable.

    Empty when the selected platform has no ``[scheduler]`` table, when
    that table's ``disable_overlap_schedule`` is false, or when
    ``overlap_schedule_enabled()`` is true (``VIBESYS_OVERLAP_SCHEDULE`` set
    to a truthy value overrides the platform default and leaves the
    overlap scheduler on).
    """
    scheduler = _PLATFORM.scheduler
    if scheduler is None or not scheduler.disable_overlap_schedule:
        return []
    if overlap_schedule_enabled():
        return []
    return ["--disable-overlap-schedule"]


def build_launch_argv(
    *,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tp: int = DEFAULT_TP,
) -> list[str]:
    """Return the argv for ``python3 -m sglang.launch_server``.

    Attention backend, page size, mem-fraction, KV pool pin, and the
    HF-loader recipe come from the platform config
    (``config/platforms/<platform>.toml``); see that file for the rationale
    behind each value. The platform config's own ``extra_args`` list is
    appended next, after the config-derived and loader-specific args, then
    ``scheduler_args()`` (a no-op unless the platform declares
    ``[scheduler]`` and it isn't overridden), then
    ``speculative_decode_args()`` (see that function; a no-op unless the
    platform declares ``[speculative]`` and it isn't disabled), then
    ``prefill_cuda_graph_args()`` (see that function; a no-op unless the
    platform declares ``[prefill_cuda_graph]`` and it isn't disabled). Any
    tokens from ``$VIBESYS_EXTRA_SERVER_ARGS`` (see ``extra_server_args()``)
    are appended last, after all four.
    """
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(tp),
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--attention-backend",
        _PLATFORM.attention_backend,
        "--page-size",
        str(_PLATFORM.page_size),
        "--mem-fraction-static",
        _PLATFORM.mem_fraction_static,
        "--max-total-tokens",
        str(_PLATFORM.max_total_tokens),
    ]
    if is_sharded_state_dir(model_path):
        # ShardedStateLoader rejects --model-loader-extra-config keys and
        # ignores --weight-loader-disable-mmap; the checkout's loader reads
        # the per-rank files without mmap on its own (mmap over Lustre
        # page-faulted for >10 min, job 631026).
        argv += ["--load-format", "sharded_state"]
    else:
        if _PLATFORM.hf_loader.disable_mmap:
            argv.append("--weight-loader-disable-mmap")
        argv += ["--model-loader-extra-config", _PLATFORM.hf_loader.extra_config]
    argv += _PLATFORM.extra_args
    argv += scheduler_args()
    argv += speculative_decode_args()
    argv += prefill_cuda_graph_args()
    argv += extra_server_args()
    return argv


def start_server(
    *,
    workspace: Path,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tp: int = DEFAULT_TP,
    log_path: Path,
) -> subprocess.Popen:
    """Launch the SGLang server in its own process group and return the handle.

    ``log_path.parent`` is passed to ``build_launch_env()`` as the TunableOp
    per-launch staging directory's parent (see ``stage_tunableop_files()``);
    it is a no-op when TunableOp is disabled.
    """
    argv = build_launch_argv(model_path=model_path, host=host, port=port, tp=tp)
    env = build_launch_env(tp=tp, run_dir=log_path.parent)
    log_file = log_path.open("w")
    return subprocess.Popen(  # noqa: S603
        argv,
        cwd=str(workspace),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # noqa: PLW1509
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the server's whole process group, escalating to SIGKILL."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


async def wait_until_ready(
    base_url: str,
    *,
    proc: subprocess.Popen,
    timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
    log_path: Path,
) -> None:
    """Poll ``/health`` until the server answers, the process dies, or we time out."""
    import aiohttp

    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                tail = tail_server_log(log_path)
                raise RuntimeError(
                    f"sglang server process exited early (code {exit_code}) before "
                    f"becoming ready; last log output:\n{tail}"
                )
            try:
                async with session.get(
                    f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return
                    last_error = f"/health returned status {resp.status}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    tail = tail_server_log(log_path)
    raise RuntimeError(
        f"sglang server did not become ready within {timeout_seconds:.0f}s "
        f"(last probe error: {last_error}); last log output:\n{tail}"
    )


def tail_server_log(log_path: Path, *, max_bytes: int = 4000) -> str:
    try:
        data = log_path.read_bytes()
    except OSError:
        return "(no log available)"
    return data[-max_bytes:].decode("utf-8", errors="replace")


@contextlib.asynccontextmanager
async def running_server(
    *,
    workspace: Path,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tp: int = DEFAULT_TP,
    log_path: Path,
    startup_timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
):
    """Async context manager: start the server, wait for readiness, always tear down."""
    overall_start = time.monotonic()
    ensure_sglang_importable(workspace)
    proc = start_server(
        workspace=workspace,
        model_path=model_path,
        host=host,
        port=port,
        tp=tp,
        log_path=log_path,
    )
    base_url = f"http://{host}:{port}"
    try:
        await wait_until_ready(
            base_url,
            proc=proc,
            timeout_seconds=startup_timeout_seconds,
            log_path=log_path,
        )
        print(
            f"[timing] server ready {time.monotonic() - overall_start:.1f}s "
            "after running_server() entry (includes sglang import + launch)",
            file=sys.stderr,
            flush=True,
        )
        yield base_url
    finally:
        stop_server(proc)


@contextlib.asynccontextmanager
async def server_endpoint(
    *,
    base_url: str | None,
    workspace: Path,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tp: int = DEFAULT_TP,
    log_path: Path,
    startup_timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
):
    """Yield a base_url, either by reusing one or by booting a server for it.

    When ``base_url`` is given, no server is started or torn down: callers
    run against the already-running server at that URL (e.g. one booted by
    ``serve.py``). Otherwise this behaves exactly like ``running_server``.
    """
    if base_url is not None:
        yield base_url
        return
    async with running_server(
        workspace=workspace,
        model_path=model_path,
        host=host,
        port=port,
        tp=tp,
        log_path=log_path,
        startup_timeout_seconds=startup_timeout_seconds,
    ) as booted_url:
        yield booted_url


# Every multiple of 64 from 256 (the first bucket above
# AITER_GEMM_TUNABLEOP_M_THRESHOLD=192, unquant.py) to 1024 (this platform's
# chunked-prefill-size, config/platforms/<platform>.toml, the largest M a
# single chunked prefill can ever present). 256 itself covers every M from
# 193 to 256, the only padded bucket below 256, so no separate bucket is
# needed there. 13 buckets total.
GEMM_PAD_WARMUP_TOKEN_COUNTS = list(range(256, 1024 + 1, 64))


def _gemm_pad_warmup_m(default: int = 64) -> int:
    """Mirrors ``SGLANG_AITER_GEMM_PAD_M``'s own read in
    ``sglang.srt.layers.quantization.unquant`` (default 64; ``0`` disables):
    this warmup only matters when that padding is actually on, so it shares
    the exact same env var and default rather than introducing a second
    switch."""
    value = os.environ.get("SGLANG_AITER_GEMM_PAD_M")
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


async def _tokenized_len(session, base_url: str, model_path: str, content: str) -> int:
    """Prompt token count for ``content`` as a lone chat message, via the
    server's own ``/v1/tokenize`` (OpenAI-compatible, applies the real chat
    template server-side). Used instead of a client-side
    ``AutoTokenizer.from_pretrained`` so building warmup prompts costs a
    handful of small HTTP round trips against an already-loaded model,
    not a second, independent tokenizer-files load from Lustre (observed
    ~37s cold, job 634677 -- long enough by itself to blow this warmup's
    time budget before a single request fires)."""
    import aiohttp

    async with session.post(
        f"{base_url}/v1/tokenize",
        json={"model": model_path, "messages": [{"role": "user", "content": content}]},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json()
    return int(data["count"])


# Generous upper bound on words ever needed: even a pathological tokenizer
# that spends close to 1 token per word would need <=2000 words to reach
# the largest warmup target (1024) many times over. Pre-generated once and
# cached; callers always take a *prefix* of this fixed list, which makes
# content(n) monotonic in n (a longer prefix never re-randomizes the
# shorter prefix it contains, so token count does not decrease as n
# grows) -- unlike drawing fresh random words on every call, which does
# not guarantee that and would break the bracketing search below.
_WARMUP_FILLER_WORDS: list[str] = []


def _warmup_filler_words(n: int) -> list[str]:
    global _WARMUP_FILLER_WORDS
    if not _WARMUP_FILLER_WORDS:
        import random
        import string

        rng = random.Random(20260913)
        _WARMUP_FILLER_WORDS = [
            "".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(2000)
        ]
    return _WARMUP_FILLER_WORDS[:n]


async def _build_warmup_prompt(
    session, base_url: str, model_path: str, target_tokens: int
) -> tuple[str, int]:
    """Fresh filler content sized, via ``_tokenized_len`` (the real chat
    template, server-side), to land as close as possible to
    ``target_tokens`` prompt tokens *without exceeding it*: every warmup
    target here is itself a padded-bucket boundary (a multiple of 64), so
    overshooting by even one token pads to the *next* bucket instead of the
    intended one, silently leaving the intended bucket unwarmed.

    Uses a plain bracketing search (exponential growth to find a word count
    that overshoots, then binary search down to the largest word count that
    does not) rather than a calibrated linear estimate: a prior version
    calibrated tokens-per-word from a *different-shaped* probe string
    (digit-suffixed words) than the filler words it actually built prompts
    from, and this tokenizer costs digit-heavy text very differently from
    plain lowercase text, so the calibration was systematically wrong and
    a fixed-iteration linear correction could not close the gap for larger
    targets (job 634688's warmup.out: target 1024 landed at actual 796,
    missing the 896/960/1024 buckets entirely). A bracketing search needs
    no estimate of tokens-per-word to be correct, only that token count is
    non-decreasing as the filler content grows (see
    ``_warmup_filler_words``), which holds for any real tokenizer.
    """

    def make(n: int) -> str:
        return " ".join(_warmup_filler_words(n))

    overhead = await _tokenized_len(session, base_url, model_path, "")
    if overhead >= target_tokens:
        return "", overhead

    lo, lo_len = 0, overhead
    hi = 1
    while True:
        content = make(hi)
        length = await _tokenized_len(session, base_url, model_path, content)
        if length > target_tokens:
            hi_len = length
            break
        lo, lo_len = hi, length
        hi *= 2

    while hi - lo > 1:
        mid = (lo + hi) // 2
        content = make(mid)
        length = await _tokenized_len(session, base_url, model_path, content)
        if length > target_tokens:
            hi, hi_len = mid, length
        else:
            lo, lo_len = mid, length

    return (make(lo) if lo > 0 else ""), lo_len


async def run_gemm_pad_warmup(
    base_url: str,
    model_path: str,
    *,
    log_path: Path | None = None,
    budget_s: float = 25.0,
) -> list[dict]:
    """First-touch every M the aiter dense-GEMM padding fix
    (``SGLANG_AITER_GEMM_PAD_M`` / ``unquant.py``) can pad prefill up to,
    before the server is marked ready: without this, the first real request
    landing on each never-before-seen padded M pays aiter's untuned
    hipBLASLt fallback (roughly 180ms extra per shape, see
    ``manual/stall-diag/results.md``) live, during real traffic. No-op when
    padding is disabled (``SGLANG_AITER_GEMM_PAD_M=0``). Bounded to
    ``budget_s`` wall-clock seconds total. Prompts are sized via the
    server's own ``/v1/tokenize`` (see ``_build_warmup_prompt``) rather than
    a client-side ``AutoTokenizer.from_pretrained``, specifically so this
    does not pay a second, independent tokenizer-files load from Lustre
    (observed ~37s cold, job 634677 -- long enough by itself to blow this
    entire budget before a single warmup request fired, the bug that first
    version of this function shipped with). A stalled request past the
    deadline logs and skips the remaining buckets (they still get warmed
    live, just later, same as before this fix existed).

    Sizing each prompt to hit an exact token target went through two more
    broken designs before landing on the current one. A fixed
    tokens-per-word estimate (job 634682) overshot every bucket, because
    the real ratio for the filler words used was higher than assumed and
    the loop could only grow, never shrink, once past target. A one-shot
    calibration probe plus a small bounded correction loop (job 634688)
    replaced the fixed estimate but measured the ratio on digit-suffixed
    probe words that tokenize very differently from the plain-lowercase
    words the warmup itself sends, so the calibrated rate was wrong in the
    other direction (undershoot growing with target size, e.g. target 1024
    landed at actual 796) and the correction loop's bound was too small to
    close the gap; buckets 896, 960 and 1024 were never actually warmed.
    ``_build_warmup_prompt`` now uses a fixed, pre-generated word list
    (``_warmup_filler_words``) whose token count is monotonic non-decreasing
    in the number of words taken from it, and finds the exact target via
    exponential-growth-then-binary-search over word count. This needs no
    per-word cost estimate at all, so it is insensitive to how any given
    tokenizer costs a word.

    Returns one dict per bucket actually attempted, and also writes the
    same information, one line per request, to ``log_path`` if given (else
    to stderr) -- this is "the warmup output" other tooling (this repo's
    validation job) greps for.
    """
    import json

    import aiohttp

    rows: list[dict] = []
    pad_m = _gemm_pad_warmup_m()
    log_fh = log_path.open("a") if log_path is not None else sys.stderr
    try:
        if pad_m <= 0:
            print(
                "[gemm-pad-warmup] SGLANG_AITER_GEMM_PAD_M=0, padding disabled, skipping",
                file=log_fh,
                flush=True,
            )
            return rows

        deadline = time.monotonic() + budget_s

        async with aiohttp.ClientSession() as session:
            for target in GEMM_PAD_WARMUP_TOKEN_COUNTS:
                if time.monotonic() > deadline:
                    print(
                        f"[gemm-pad-warmup] budget_s={budget_s} exceeded, "
                        f"skipping remaining buckets from {target}",
                        file=log_fh,
                        flush=True,
                    )
                    break
                content, actual = await _build_warmup_prompt(
                    session, base_url, model_path, target
                )
                payload = {
                    "model": model_path,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                t0 = time.monotonic()
                error = None
                try:
                    async with session.post(
                        f"{base_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        await resp.read()
                        status = resp.status
                except Exception as exc:  # noqa: BLE001
                    status = None
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.monotonic() - t0) * 1000
                row = {
                    "target_tokens": target,
                    "actual_tokens": actual,
                    "status": status,
                    "latency_ms": round(latency_ms, 2),
                    "error": error,
                }
                rows.append(row)
                print(f"[gemm-pad-warmup] {json.dumps(row)}", file=log_fh, flush=True)
    finally:
        if log_path is not None:
            log_fh.close()
    return rows
