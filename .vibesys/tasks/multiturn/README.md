# multiturn task harness

Operator notes for the harness that boots and evaluates SGLang for the
multiturn task. This is infrastructure the optimizing agent does not touch
(see `OBJECTIVE.md`); it is documented here for whoever runs or maintains it.

## 1. What is here

- `_server.py`: shared server lifecycle (launch argv/env, readiness polling,
  model-path resolution). Imported by both evaluator steps and by `serve.py`.
  Loads its machine-specific values from `config/` at import time, see
  "Configuration" below.
- `serve.py`: boots one server and holds it until SIGTERM/SIGINT, so
  `checker.py` and `run.py` can both run against it with `--base-url` instead
  of each booting their own.
- `evaluate_once.sh`: boots a server via `serve.py`, runs the accuracy
  checker and the benchmark against it, and always tears the server down.
- `benchmark/run.py`, `accuracy_checker/checker.py`, `reference/pins.json`,
  `objectives.toml`: the load-ramp benchmark (`benchmark_version` 6), the
  accuracy checker with greedy-token pins, and the goodput objectives, shared
  verbatim with the bespoke-engine task for this model and hardware (see
  `OBJECTIVE.md`). `benchmark/launcher.py` adapts them to boot SGLang through
  `_server.py`.
- `stage_workspace.sh`: extracts a checkout tarball into node-local tmpfs so
  imports skip Lustre's cold metadata cost.
- `config/`: committed site and platform TOML files, plus the loader that
  reads them.
- `tools/`: scripts to (re)generate the pre-sharded model artifact this
  harness boots from, and `submit.sh` to submit them through the site
  config.

## 2. Configuration

Machine-specific values live in two kinds of committed TOML file under
`config/`, loaded by `config/loader.py`:

- `config/sites/<site>.toml`: everything true of one place that runs the
  harness; filesystem paths (model, sharded artifacts, JIT cache, tmpfs
  root, log dir, operator checkout), Slurm identity (account, partition,
  EDF), Lustre striping, and timeouts dominated by I/O (the server startup
  budget). A site names the platform it runs.
- `config/platforms/<platform>.toml`: everything true of any node with a
  given GPU and container image; the launch env overlay and argv
  (attention backend, page size, mem fraction), the weight-loader recipe,
  the KV pool pin, and timeouts dominated by JIT compilation.

`_server.py` and the `tools/` scripts load these at import/run time instead
of hardcoding values. Selection is via the `VIBESYS_SITE` environment
variable (default `"example"`); `VIBESYS_PLATFORM` overrides the platform a
site names, e.g. to test a site against a platform config that isn't
(yet) its committed default.

For shell scripts, `config/loader.py --shell` prints `SITE_*`/`PLATFORM_*`
assignments suitable for `eval`:

```
eval "$(python3 config/loader.py --shell)"
echo "$SITE_MODEL_PATH $SITE_SLURM_ACCOUNT $PLATFORM_TP"
```

`tools/submit.sh` uses this shim to submit the `.sbatch` scripts in `tools/`
(see "Regenerating the artifact" below). `config/loader.py --json` prints
the full merged config; `config/loader.py KEY` prints one dotted value, e.g.
`config/loader.py paths.tmpfs_root`.

To add a second site, copy `config/sites/example.toml`, fill in that site's
paths/Slurm/Lustre values, and point `platform` at an existing platform file
(if the new site's GPU and image match one already committed) or add a new
`config/platforms/<platform>.toml`.

### Platform-config server args

`config/platforms/<platform>.toml`'s optional `extra_args` is a list of
argv token strings, appended to the launch argv after the other
config-derived flags (attention backend, page size, mem fraction, KV pool
pin, loader flags) and before `$VIBESYS_EXTRA_SERVER_ARGS` (see
`build_launch_argv`, `_server.py`). Use it for flag choices that are meant
to stick, as opposed to the one-off overrides `VIBESYS_EXTRA_SERVER_ARGS`
is for (see below). The loader rejects a non-list or a list with a
non-string element. `config/loader.py --shell` exposes it as
`PLATFORM_EXTRA_ARGS` (space-joined); `--json` as a list under
`platform.extra_args`.

`mi300a-rocm700` sets `extra_args = ["--enable-mixed-chunk",
"--chunked-prefill-size", "1024"]`: mixed chunked prefill at chunk size
1024 is on by default. See the comment on `extra_args` in that file for the
measured effect (job 632584, ledger row H8b) and why it also guards
against a multi-second stall seen with unchunked prefill.

To disable it for an A/B: `VIBESYS_EXTRA_SERVER_ARGS` cannot remove an
argv token already baked into `extra_args`, only append more (SGLang's
argument parser would have to prefer the first occurrence of a repeated
flag for an override to win, and nothing here relies on that). A site
config cannot override individual platform fields either; `sites/<site>.toml`
only names which platform file it uses (`platform = "..."`), a choice that
can itself be overridden wholesale via `VIBESYS_PLATFORM`, but there is no
per-field override. So to disable mixed chunk for a comparison run,
either:

- edit `extra_args` in `config/platforms/mi300a-rocm700.toml` directly
  (drop the flags, run, revert), or
- add a second platform file (e.g.
  `config/platforms/mi300a-rocm700-no-mixed-chunk.toml`, a copy with
  `extra_args` removed or set to `[]`) and point the run at it with
  `VIBESYS_PLATFORM=mi300a-rocm700-no-mixed-chunk`.

### Ad hoc server arg overrides

`$VIBESYS_EXTRA_SERVER_ARGS`, if set, is split with shell-word rules
(`shlex.split`) and appended to the launch argv after all config-derived
args, including the platform config's own `extra_args`, e.g.:

```
VIBESYS_EXTRA_SERVER_ARGS="--stream-interval 4" \
    evaluate_once.sh $WORKSPACE OUTDIR
```

This is for one-off overrides such as a paired A/B comparison of a single
scheduler flag; it is not a substitute for a config/platform change once a
flag choice is validated and meant to stick. Because the tokens are
appended last, they only win over a config-derived flag if SGLang's own
argument parser prefers the last occurrence of a repeated flag.

### HIP extension build cache

`sglang.srt.layers.hip_extension.load_hip_extension` JIT-compiles ad hoc
HIP source extensions with `torch.utils.cpp_extension.load` on first use.
`SGLANG_HIP_EXT_DIR` picks the build directory, the same way `AITER_JIT_DIR`
picks aiter's; it is `setdefault`'d from the site config's
`paths.hip_ext_dir` (see `config/sites/example.toml`) for the same reason
`aiter_jit_dir` is: a fresh per-launch build dir pays the HIP compile every
boot, and a killed build can leave a stale lock behind, so point it at a
persistent, pre-warmed cache instead.

`SGLANG_MXFP4_MOE_HIP` and `SGLANG_SKINNY_GEMM` are both on (`"1"`) by
default via `[env_defaults]` on `mi300a-rocm700`, per job 632503 (ledger rows
H7f and H5g): the fused MoE kernel alone took p95 TTFT turn 2+ from 784.7 to
537.5 ms and TPOT from 106.8 to 61.0 ms, and the skinny GEMM on top of it took
TPOT to 35.6 ms with p95 458.8 ms, both gates 13/13 under benchmark version 2.
`SGLANG_MXFP4_MOE_HIP` (`"0"`/`"1"`) routes MoE through the HIP grouped-GEMM
fused kernel (`sglang.srt.layers.moe.mxfp4_fused`) instead of the Triton
w4a16 fallback. `SGLANG_SKINNY_GEMM` (`"0"`/`"1"`) routes small-M dense linear
projections through the skinny bf16 GEMM HIP kernel in
`python/sglang/srt/layers/skinny_gemm/` instead of aiter's `tgemm.mm`, when
the call's shape qualifies; see that module's `gemm.supports()` for the gate.

Because these are `[env_defaults]`, a launch's own environment overrides
them (`os.environ.setdefault` in `build_launch_env`, `_server.py`): to turn
one off for an A/B, export it as `"0"` before launching, e.g.
`SGLANG_SKINNY_GEMM=0 evaluate_once.sh $WORKSPACE OUTDIR` to isolate the
fused MoE's effect, or `SGLANG_MXFP4_MOE_HIP=0 evaluate_once.sh $WORKSPACE
OUTDIR` to fall back to the pre-fused baseline entirely (this also disables
the skinny GEMM's precondition in practice, since it stacks on the fused
MoE path).

Both kernels are HIP extensions built through the same JIT cache described
above: a cold first boot with an unwarmed `hip_ext_dir` pays roughly 110 s of
compile time per extension (about 220 s total for both) before the server
becomes healthy. Make sure the site's `paths.hip_ext_dir` cache is warm
before relying on boot time budgets.

### Speculative decoding (NEXTN)

On by default via `[speculative]` in `config/platforms/mi300a-rocm700.toml`:
NEXTN (single-linear-draft-chain MTP) speculative decoding, 3 drafted
tokens (`--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`), verified through the fast linear-replay
path (`--enable-linear-replayssm-spec`). Acceptance jobs 633511/633512 (48
sessions uncapped, 5 reps, one node) measured median mean TPOT 71 -> 38 ms
(-47%), p95 TTFT turn 2+ about 5% better, gates 13/13 every rep, mean
accept length 2.9 of 4 drafted tokens; the probe that picked k=3 over k=2
(job 633510) measured defaults 65.6 ms, k=2 36.0 ms (accept 2.52 of 3), k=3
33.6 ms (accept 2.92 of 4). The 16-session-cap acceptance result is still
pending.

The draft model's source is picked by `_server.py`'s
`speculative_decode_args()`, based on whether the site config sets
`paths.draft_model`:

- **Set (example's default):** the draft loads from `paths.draft_model`, a
  draft-only `sharded_state` artifact holding just the MTP module's
  tensors, with `--speculative-draft-load-format sharded_state` forced
  regardless of the platform's `draft_load_format` -- the implicit rule is
  "draft_model set implies sharded_state", since nothing else is ever
  written to that kind of path. `ShardedStateLoader` copies tensors into
  `model.state_dict()` by key match and never calls the model's own
  `load_weights()`, which is what makes this fast: see "Boot cost" below.
- **Unset:** the draft falls back to `paths.model` (the *original*,
  unsharded checkpoint) with the platform's own `draft_load_format`
  (normally `"auto"`). `paths.model` is the only copy carrying the `mtp.*`
  tensors the sharded fast-boot target artifact lacks, and this
  checkpoint's architecture is not on the fork's `speculative_hook.py`
  auto-default allowlist, so the draft path has to be set explicitly
  rather than inferred either way.

**Boot cost:** with `paths.draft_model` set, the draft's own weight load
takes about 10s and total boot is about 6 minutes (job 633763, the test cluster:
365s), close to the ~200s no-speculative-decoding boot. Without it (the HF
checkpoint fallback), the draft's per-expert weight load
(`Qwen3_5ForCausalLMMTP.load_weights`, unthreaded, one Python call per
expert per projection) takes ~455-498s and adds about 12.5 minutes to
boot versus under 5 minutes without speculative decoding at all (job
633763's `spec_hf` side: 866s total). `config/sites/example.toml`'s
`startup_timeout_s` is 1200s (reduced from a prior 1800s that was set for
the HF-fallback draft load, back when no `sharded_state` draft artifact
existed) to keep real margin over that slower fallback path without
carrying the old ceiling. This is a ceiling, not an expected wait.

**Producing the artifact:** `manual/draft-shard/save_draft_shard_v5.py`
(fork commit 1443e6bfdc, job 633762) stages the fork checkout the same way
`stage_workspace.sh` does (a tmpfs-extracted checkout with
`PYTHONPATH=$WS/python` exported and `sglang.__file__` asserted against
that root before Engine construction -- catches a stock, pre-fix
container-image `sglang` silently shadowing the staged one), constructs an
`Engine` with the target and draft both loaded, then calls
`llm.save_sharded_model(pattern=None, max_size=..., skip_target=True,
draft_path=<destination>)` so only the draft's tensors are written (the
target artifact, already produced separately, is untouched). `config.json`
and the checkpoint's other non-weight files are copied in afterward. See
PR #55 (`uw-syfi/sglang`) for the `save_sharded_model` draft-worker
plumbing this depends on.

**Off switch:** export `VIBESYS_SPEC_DECODE=0` before launching to disable
speculative decoding entirely for one run (checked in `_server.py`'s
`spec_decode_enabled()`), e.g. `VIBESYS_SPEC_DECODE=0 evaluate_once.sh
$WORKSPACE OUTDIR`. For a permanent side without it instead (the pattern
the acceptance jobs' paired defaults/spec comparisons use, staging one
platform copy per side), copy `mi300a-rocm700.toml` with the
`[speculative]` table removed and select it with `VIBESYS_PLATFORM`. To
fall back to the slow HF-checkpoint draft load specifically (keeping
speculative decoding on), comment out or remove `paths.draft_model` in the
site config instead; nothing else needs to change.

### Overlap scheduler

Off by default via `[scheduler]` in `config/platforms/mi300a-rocm700.toml`
(`disable_overlap_schedule = true`, applied as `--disable-overlap-schedule`).
With the overlap scheduler on, the CPU-side scheduler runs one batch ahead
of the GPU, so each turn's first token waits for an extra scheduler
iteration before it is even forwarded. Accepted at both concurrencies
(acceptance jobs 633754, 48 sessions uncapped, and 633755, 16-session cap;
see `manual/noovl/acceptance.md`): pooled p95 TTFT turn 2+ fell 24.1% at 48
sessions (734.4 ms to 557.5 ms) and 28.4% at the 16-session cap (438.5 ms
to 313.8 ms) with the overlap scheduler off, against a median TPOT
regression of +3.3% and +6.8% respectively, both well inside the 10%
budget. Gates 13/13 on every rep of both sides in both jobs.

**Off switch (i.e. turn the overlap scheduler back on):** export
`VIBESYS_OVERLAP_SCHEDULE=1` before launching (checked in `_server.py`'s
`overlap_schedule_enabled()`), e.g. `VIBESYS_OVERLAP_SCHEDULE=1
evaluate_once.sh $WORKSPACE OUTDIR`. For a permanent side with it on
instead, copy `mi300a-rocm700.toml` with the `[scheduler]` table removed
(or `disable_overlap_schedule` set to `false`) and select it with
`VIBESYS_PLATFORM`, the same pattern used above for speculative decoding.

### Tuned dense GEMM tiles (TunableOp)

On by default via `[tunableop]` in `config/platforms/mi300a-rocm700.toml`
(`enabled = true`). A GEMM kernel library like hipBLASLt normally picks one
fixed tile shape (block sizes, split-K, etc.) for a given problem regardless
of the actual batch size `M`; PyTorch's TunableOp instead benchmarks a set
of candidate tiles per `(N, M, K)` ahead of time and records the fastest one
for each `M` in a CSV, so the server can pick a per-`M`-optimal tile at
runtime instead of hipBLASLt's one-size-fits-all choice.

Measured on the six per-rank dense projections and the LM head, at the
36 `M` values the CUDA-graph capture set dispatches (job 633793,
`manual/dense-bench/bench_tunable_full.py`; see that job's write-up for the
full per-shape table): dense GEMM GPU time per verify forward drops from
about 23 ms to about 3.7 ms at the modal batch, output exact to one bf16
ULP against the untuned baseline.

PyTorch resolves `PYTORCH_TUNABLEOP_FILENAME` per process
(`torch/cuda/tunable.py`, torch 2.9.0a0): a literal `%d` is replaced with
the device ordinal, or else the ordinal is inserted before the extension,
so each TP rank reads a *different* file, and `_server.py`'s
`stage_tunableop_files()` stages one read-only copy of the committed CSV
per device ordinal before launch and points the env var at the
`%d`-templated path so every rank finds its own copy. PyTorch also
rewrites this file at process exit even with `PYTORCH_TUNABLEOP_TUNING=0`
(an empty file when nothing was loaded), and no env var disables that,
only the Python API `torch.cuda.tunable.write_file_on_exit(False)`, so the
staged copies are `chmod`'d `0o444` to turn that rewrite into a harmless
warning instead of clobbering the file.

**Accepted** at both 48-session and 16-session-cap concurrency (jobs
633839, 633841; hard gates 13/13 every rep at both concurrencies). At 48
sessions: median TPOT 38.05 to 22.86 ms (-39.9%), pooled p95 TTFT turn2+
499.0 to 493.3 ms (unchanged, within rep spread). At 16 sessions: median
TPOT 14.31 to 12.44 ms (-13.1%), pooled p95 TTFT turn2+ 331.5 to 334.4 ms
(unchanged, within rep spread). Both concurrencies pass the acceptance
rule (gates clean, TPOT gain exceeds rep-to-rep spread, p95 TTFT within
+-10%); the smaller TPOT gain at 16 sessions reflects the smaller tuned
GEMM shapes at that concurrency (M=64/M=16 vs. M=192/M=48), not a
weaker effect. See `manual/tunable/acceptance.md` on the test cluster for the
full pooled analysis, stall-gap checks, and boot-time investigation. The
prior pair (633800/633801) measured nothing because
`PYTORCH_TUNABLEOP_FILENAME` pointed all four TP ranks at one shared
path, so only rank 0 found a file, the other three ran untuned, and a TP
step waits for the slowest rank.

**Off switch:** export `VIBESYS_TUNABLEOP=0` before launching (checked in
`_server.py`'s `tunableop_enabled()`), e.g. `VIBESYS_TUNABLEOP=0
evaluate_once.sh $WORKSPACE OUTDIR`, to fall back to hipBLASLt's own
heuristic for one run without editing the platform config. For a permanent
side without it instead, copy `mi300a-rocm700.toml` with the `[tunableop]`
table removed (or `enabled` set to `false`) and select it with
`VIBESYS_PLATFORM`, the same pattern used above for speculative decoding and
the overlap scheduler.

**Detecting a stale table:** the CSV
(`config/platforms/tunableop/mi300a-rocm700.csv`) opens with a validator
header (PyTorch, ROCm, hipBLASLt, and GPU-arch versions) that PyTorch checks
against the running process before using the file. On a mismatch PyTorch
silently ignores the whole file and falls back to hipBLASLt's untuned
heuristic; it only logs a warning, it does not raise or fail the boot.
`build_launch_env()` sets `PYTORCH_TUNABLEOP_VERBOSE=1` by default whenever
TunableOp is enabled (log-only; it does not affect dispatch), so the load
result for each rank is always in the server log without an override. Grep
for `reading tuning results from` (expect one line per TP rank, naming that
rank's staged `tunableop<k>.csv`) to confirm the table was accepted; a
`could not open ... for reading` line instead means the validator rejected
it (stale table) or the staged file is missing, and the run fell back to
hipBLASLt's untuned heuristic. A *third* failure mode looks identical from
the outside but prints neither of those two strings: `Failed validator: ...`
/ `results validator check failed` in the server log, PyTorch's own message
when a validator line's exact string compare fails even though both sides
print identically (see the CRLF paragraph below) -- grep for
`Failed validator` too if tuned-tile perf does not show up despite a clean
`reading tuning results from` line.

**CRLF line endings silently discard the whole table:** a dense-keys
investigation (`manual/dense-keys/` on the test cluster in the campaign's scratch
tree) found the merged CSV that became this file's current content had its
five `Validator` header lines terminated with CRLF (`\r\n`) instead of
plain LF, from a merge step that ran through a tool that rewrote line
endings. PyTorch's validator compare is exact-string, so
`"5.0.0.976b9c4a87\r" != "5.0.0.976b9c4a87"` failed on every load even
though both sides printed identically, which discarded all 356 tuned rows
uniformly, on every shape and every `M`, while the server log still showed
a normal-looking `reading tuning results from` line and the existing hard
gate (which only checked for that string and the absence of
`could not open`) never caught it. `stage_tunableop_files()`
(`_server.py`) now rejects the committed CSV outright, with a `ValueError`
naming the first offending line, if it contains any `\r` byte at all,
before staging it to any TP rank; this turns a silent full-table loss back
into a loud failure at launch time. Regenerate a CSV with `tr -d '\r' <
FILE > FILE.lf && mv FILE.lf FILE` (byte-preserving otherwise) if this
check ever fires.

Paired end-to-end acceptance of this content against the CSV it replaced
(jobs 635328/635329/635330 on the test cluster, `manual/tunableop-lf-accept/`):
pooled p95 TTFT turn2+ improved 39.7% at 48-session uncapped concurrency
(342.1 to 206.1 ms) and 10.8-16.4% at the 16-session cap (raw /
collapse-excluded), median TPOT improved at both concurrencies, gate
13/13 on every rep both sides both jobs, and GSM8K accuracy (94.6%, band
91.9-96.3%) plus level-2 agreement against the prior CSV's own recorded
run both landed inside tolerance. `schedule_bound_fraction` was 1.000 on
every rep, meaning this benchmark was fully queue-bound at both
concurrencies, which is why the tail (p95/p99) improved by more than the
median: a smaller per-forward GPU time compounds through the queue rather
than passing straight through.

**Regenerating the table:** re-run
`manual/dense-bench/bench_tunable_full.py` (one process per dense/LM-head
shape, tuning all `M` values the capture set dispatches) whenever the
container image, PyTorch, hipBLASLt, or the model's per-rank GEMM shapes
change, and replace `config/platforms/tunableop/mi300a-rocm700.csv` with the
new output. Tuning must stay off in serving (`PYTORCH_TUNABLEOP_TUNING` is
always `"0"` in the launch env, never set to `"1"` by the harness); the CSV
is read-only input to the server process.

### Prefill CUDA graph (breakable)

On by default via `[prefill_cuda_graph]` in `config/platforms/mi300a-rocm700.toml`
(`backend = "breakable"`). Prefill has no CUDA graph unless
`cuda_graph_config[prefill].backend` is set (decode already captures one);
`_server.py`'s `prefill_cuda_graph_args()` renders `--cuda-graph-config` with
that backend over the same 13-value token-count list the gemm-pad-m warmup
already targets (`GEMM_PAD_WARMUP_TOKEN_COUNTS`, `_server.py`: every
multiple of 64 from 256 to 1024), reused rather than duplicated in the
platform TOML.

**Accepted** at both 48-session and 16-session-cap concurrency (jobs
634901/634902, same gemm-pad-m bundle both sides, draft PR #99). At 48
sessions: pooled p95 TTFT turn2+ 363.6 to 321.1 ms (-11.7%), pooled p50
195.6 to 150.1 ms (-23.3%), median TPOT 12.94 to 11.34 ms (-12.4%). At 16
sessions: pooled p95 TTFT turn2+ 292.9 to 206.4 ms (-29.5%), median TPOT
10.27 to 9.34 ms (-9.1%). Gates 13/13 on every rep, both sides, both jobs.
The c48 p95 gain reads smaller than the noisy 5-rep range-based spread
statistic alone, but the paired sign test (candidate faster in all 5 reps),
the standard-deviation framing, and the pooled n=830 percentile comparison
all agree the effect is real; see `manual/pcg-accept/acceptance.md` on
the test cluster for the full analysis.

`breakable`, not `tc_piecewise`: this platform's NEXTN speculative decoding
(see "Speculative decoding (NEXTN)" above) resolves to the EAGLE algorithm
family, and the fork's prefill-graph setup routes the target model's
prefill forward to the eager path under EAGLE-family speculative decode for
every prefill CUDA-graph backend except `breakable` (its runner already
captures a full graph for the EAGLE target on its own). `tc_piecewise`
would run eager here regardless of the flag, so it was not part of this
acceptance.

Exactness: token-level divergence between a captured-graph replay and an
eager forward sits at the stack's own bf16/reduction-order nondeterminism
floor, this track's accepted exactness tier; correctness is judged by this
task's own accuracy gate (`checker.py`) and the GSM8K evaluation of record,
not by byte-exact output-token replay.

**Off switch:** export `VIBESYS_PREFILL_GRAPH=0` before launching (checked
in `_server.py`'s `prefill_graph_enabled()`) to drop the
`--cuda-graph-config` flag for one run without editing the platform config.
For a permanent side without it instead, copy `mi300a-rocm700.toml` with
the `[prefill_cuda_graph]` table removed and select it with
`VIBESYS_PLATFORM`, the same pattern used above for speculative decoding,
the overlap scheduler, and TunableOp.

### Tokenizer fast path (send-ids, incremental tokenize)

On by default via `[env_defaults]` in `config/platforms/mi300a-rocm700.toml`:
`SGLANG_TEXT_ONLY_SEND_IDS = "1"` and `SGLANG_INCREMENTAL_TOKENIZE = "1"`.
Both are bit-exact, request-scoped tokenizer optimizations in `serving_chat.py`,
independent of each other and safe to run together.

`SGLANG_TEXT_ONLY_SEND_IDS` skips the decode-then-re-tokenize round trip
`serving_chat.py` otherwise does for every chat turn on a multimodal-capable
model: Qwen3.5's own `config.json` carries a `vision_config` field
unconditionally, so `model_config.is_multimodal` is `True` for this
checkpoint even though this deployment never sends image/audio/video
input. When a turn's own content is text-only, the request already has its
own `prompt_ids`; this sends those directly as `input_ids` instead of
decoding them back to a string and re-tokenizing. Evidence: 0/253 real
turns mismatched against the decode-then-re-tokenize path, bit-exact
(`kwargs_differ=False` on this checkpoint).

`SGLANG_INCREMENTAL_TOKENIZE` tokenizes only the new suffix of a growing
multi-turn prompt instead of re-tokenizing the whole conversation from
scratch each turn, reusing a cache keyed at special-token boundaries.
Evidence: 0/253 real turns mismatched against full re-tokenization, 75.9%
cache hit rate (misses concentrated at each session's own turn 1 plus a
small number of LRU evictions across 48 concurrently-growing sessions
sharing one 256-entry cache).

Paired acceptance of the combined path (both switches on together, three
counterbalanced c48/c16 jobs on this platform): pooled p50 TTFT turn2+
improved consistently across every job and both boot orders (-4.4%/-4.5%
at c48, -3.5% at c16); pooled p95 TTFT moved -0.8% to -17.7% depending on
boot order, inside this cluster's own rep-to-rep spread, so not claimed as
a clean win; median TPOT never regressed. Gates 13/13 on every rep, both
sides, every job.

**Off switches:** export `SGLANG_TEXT_ONLY_SEND_IDS=0` and/or
`SGLANG_INCREMENTAL_TOKENIZE=0` before launching (each checked via
`os.environ.setdefault()` in `_server.py`'s `build_launch_env()`, so an
explicit value already in the caller's environment wins over the platform
default) to turn either off independently for one run without editing the
platform config. For a permanent side without one or both instead, copy
`mi300a-rocm700.toml` with the corresponding `[env_defaults]` line removed
and select it with `VIBESYS_PLATFORM`, the same pattern used above for
speculative decoding, the overlap scheduler, TunableOp, and the prefill
CUDA graph.

## 3. Boot recipe and artifacts

`resolve_model_path()` in `_server.py` picks the model path in this order
(current values on the the test cluster site; see `config/sites/example.toml` for
the live paths):

1. `$MODEL_PATH`, if set.
2. `/path/to/models/Qwen3.5-397B-A17B-MXFP4-sharded-tp4-striped`
   (the striped sharded artifact), if it exists.
3. `/path/to/models/Qwen3.5-397B-A17B-MXFP4-sharded-tp4`
   (the unstriped sharded artifact), if it exists.
4. `/path/to/models/Qwen3.5-397B-A17B-MXFP4` (the HF
   checkpoint), otherwise.

The sharded artifacts are TP=4 `sharded_state` dumps (56 parts, 212 GB)
produced by `tools/save_sharded.py`, one directory of
`model-rank-{rank}-part-{part}.safetensors` files plus the non-weight files
copied from the HF checkpoint. `_server.py` selects `--load-format
sharded_state` for either sharded path. The striped copy is the same files
written into a directory striped per the site's `[lustre]` config (stripe
count 8, stripe size 4M on the test cluster).

## 4. Why the KV pool is pinned

`build_launch_argv()` always passes `--max-total-tokens`, pinned to the
platform config's `max_total_tokens` (787936 on `mi300a-rocm700`). Without
this pin, SGLang sizes the KV pool from free memory after weight load, which
is not stable: it floated between about 780k and 900k tokens across
otherwise-identical boots on the HF checkpoint path, and reached 1.76M on
the sharded path (page cache no longer occupies HBM at profile time there).
The larger pool lengthened CUDA graph capture to 136 s and left only 9 GB
free. Pinning the pool to the validated HF-path baseline keeps serving
performance comparable across loaders instead of confounding it with
whatever pool size a given boot happened to profile.

## 5. Measured boot cost

Single node, from a tmpfs-staged checkout (`stage_workspace.sh`), booting
from the striped sharded artifact:

| phase | seconds |
|---|---|
| stage + imports | 19 |
| spawn | 48 |
| weight load | 67 |
| pool alloc | 6 |
| graph capture | 34 |
| warmup | 21 |
| **total** | **~200** |

Versus 430-500 s booting the same way from the HF checkpoint, and about
9 minutes for the original (pre-sharded, pre-tmpfs) recipe. A boot that
overlapped another node's weight load on the same Lustre filesystem measured
324 s, still well under the HF-checkpoint baseline.

These numbers are all without the speculative-decoding draft model. With it
(the default; see "Speculative decoding (NEXTN)" above) and
`paths.draft_model` set to a draft-only sharded artifact, add only about
10s for the draft's own weight load (job 633763: total boot 365s). Without
`paths.draft_model` (the HF-checkpoint draft fallback), add about 12.5
minutes instead, for the draft model's own load from the original
checkpoint plus its aiter backend's first JIT build.

## 6. Regenerating the artifact

Submit through `tools/submit.sh`, which exports the site's `SITE_*`/
`PLATFORM_*` variables (account, partition, log dir, checkout, EDF, AITER
JIT dir) and supplies `--account`/`--partition`/`--output` so the `.sbatch`
scripts don't have to hardcode them:

```
tools/submit.sh tools/save_sharded.sbatch
```

Then stripe and copy it. The striped destination must be created with the
site's own stripe count/size before `restripe.sbatch` runs; read them from
the site config first:

```
eval "$(python3 config/loader.py --shell)"
mkdir "$SITE_SHARDED_ARTIFACT_STRIPED"
lfs setstripe -c "$SITE_LUSTRE_STRIPE_COUNT" -S "$SITE_LUSTRE_STRIPE_SIZE" "$SITE_SHARDED_ARTIFACT_STRIPED"
tools/submit.sh tools/restripe.sbatch
```

See the header comments in each `.sbatch` file for the site variables they
read.

## 7. Running one evaluation by hand

```
WORKSPACE=$(stage_workspace.sh TARBALL)
evaluate_once.sh $WORKSPACE OUTDIR
```

`TARBALL` is produced at ship time with:

```
git archive --format=tar HEAD | gzip -1 > TARBALL
```

Set `VIBESYS_SITE` first if evaluating anywhere other than the default site.

## 8. Known limits

- Per-node Lustre read cap is about 3.5 GB/s, so weight load has a floor of
  about 60 s for the 212 GB sharded artifact regardless of striping or
  thread count.
- CUDA graph capture is process-local and cannot be cached across boots.
- On the HF checkpoint path, loader thread counts above 2
  (`--model-loader-extra-config {"num_threads": N}`) crash or OOM on the
  APU's unified host+device memory.

## 9. Benchmark: per-turn records and pacing modes

> Historical: this section and the `LEDGER.md` rows describe the earlier
> `benchmark_version` 1-4 benchmark (fixed session set, `--pacing`,
> `--concurrency`, metric `p95_ttft_turn2plus_ms`). The shared load-ramp
> benchmark in `benchmark/run.py` replaces it; see `OBJECTIVE.md`.

Besides its result-protocol-v2 aggregate row (the `"result"` record on
`--vs-output`), `benchmark/run.py` writes a `<vs-output>.turns.jsonl` sibling
file with one JSON record per attempted turn: `session_id`, `turn_index`,
`ok`, `send_ts_monotonic`, `send_ts_wall`, `ttft_ms`, `first_token_ts_wall`,
`completion_ts_wall`, `output_tokens`, `prompt_tokens`, `cached_tokens`,
`schedule_bound`, `scheduled_send_ts`, `scheduled_admission_delay_s`, and
`error`. It is not part of result-protocol-v2 (the protocol rejects any
record whose `kind` it does not recognize) and is never read by the
evaluator; it exists for post-hoc p95-tail and server-event correlation
analysis. `schedule_bound` and `scheduled_send_ts` are populated only under
`--pacing scheduled` (see below); both are `None` for turn 1 and for every
turn under `--pacing closed`. `scheduled_admission_delay_s` is populated only
on turn 1 under `--pacing scheduled`: the wait `compute_schedule`'s
admission-queue simulation assigned this session before its schedule could
start (see below), i.e. `None` for every turn 2+ and for every turn under
`--pacing closed`.

`benchmark/run.py --pacing {closed,scheduled}` controls how each session
decides when to send its next turn:

- `closed` (the original behavior): a session sends turn `k` as soon as turn
  `k-1`'s response completes plus think time. Offered load is thus coupled to
  the server's own speed: a server that answers faster makes every session
  loop faster, which raises the prefill arrival rate and can inflate
  latency percentiles even though nothing about the server got slower. This
  is what made `p95_ttft_turn2plus_ms` sensitive to decode speed on an
  otherwise unrelated code change.
- `scheduled` (the default): before the run, every turn's send time is
  precomputed from a fixed reference server speed (`REF_TTFT_MS`,
  `REF_TPOT_MS` in `run.py`, overridable with `--ref-ttft-ms`/
  `--ref-tpot-ms`), independent of how the server under test actually
  performs. At run time a turn is sent at the later of its scheduled time and
  its closed-loop-ready time: a server faster than the reference waits for
  the schedule, and a server slower than the reference degrades to
  closed-loop pacing for that turn instead of falling further and further
  behind. This decouples offered load from the server's speed, which is what
  the metric is supposed to measure in the first place.

A turn's schedule chains off its own session's turn-1 send time, so that
time has to be right too. `run_session`'s `CONCURRENCY = 16` semaphore is
held for a whole session, so with 48 sessions the 17th through 48th sessions
only get admitted once an earlier session finishes; a run against a real
server measures that admission wait directly. `compute_schedule` predicts
each session's turn-1 send time (`T[s][0]`) with a deterministic
discrete-event simulation of that same 16-slot admission queue, run at the
reference speed: sessions arrive at their `session_start_delays` stagger, at
most `CONCURRENCY` are ever active, and a session's simulated occupancy is
its reference-speed round trip summed over every turn (`REF_TTFT_MS +
REF_TPOT_MS * max_tokens`, plus think times) -- the same quantities the
turn-to-turn chain already uses. Earlier versions of this function set
`T[s][0]` to the raw stagger with no admission wait at all, so sessions 17-48
were scheduled as if admitted instantly while the real semaphore held them
back tens to over a hundred seconds; every later turn in those sessions was
then chained from a starting point the real run never reached, and
`schedule_bound_fraction` for the run collapsed toward closed-loop pacing
regardless of how fast the server under test was -- exactly the coupling
`scheduled` pacing exists to remove. Job 632584 measured
`schedule_bound_fraction` of 0.35-0.39 and an offered turn rate that still
rose with server speed (0.78-0.97 turns/s) even though the server was up to
1.8x faster than the reference.

The fix keeps the run-time semaphore as the real concurrency bound (nothing
about the actual send logic changes); it only makes the schedule the
semaphore is measured against account for admission queueing too. With it,
a server at or above the reference speed is schedule-bound throughout (its
own queueing resolves faster than the reference-speed simulation assumed, so
it always still has to wait for `T[s][k]`), and a server slower than the
reference degrades to closed-loop pacing per turn exactly as before.

The result record's `schedule_bound_fraction` reports what share of turn-2+
sends actually waited for the schedule (as opposed to firing immediately in
the closed-loop fallback); a run where this stays near 1.0 is a real
constant-load measurement, while a run where it drops toward 0.0 is
degrading toward closed-loop pacing because the server is slower than the
assumed reference. `offered_turn_rate_per_s` and `wall_duration_s` are
reported for the same reason: to make the offered load and run length
visible alongside the latency metrics they produced. The per-turn
`schedule_bound` field in `turns.jsonl` is the same signal at per-turn
granularity, letting you locate exactly which turns degraded to closed-loop
pacing rather than only seeing the aggregate fraction.

**Baselines are not comparable across pacing modes or across
`benchmark_version`.** `--pacing closed` reproduces today's exact send
timing bit-for-bit, so historical `benchmark_version` 1 numbers stay valid
for it. `scheduled` is a materially different workload (constant offered
load instead of closed-loop load), so switching the default from `closed` to
`scheduled` (`benchmark_version` 2) required a fresh baseline: do not compare
a `scheduled`-mode `p95_ttft_turn2plus_ms` against a `closed`-mode one.

`benchmark_version` 2 rows are also not comparable to `benchmark_version` 3:
version 2's schedule ignored admission queueing (see above), so its offered
load still rose with server speed instead of staying constant -- the numbers
it measured are the closed-loop-degraded ones, not the constant-load ones
`scheduled` pacing is meant to produce. Under version 3, `offered_turn_rate_per_s`
should come out at essentially the same value regardless of which server is
under test, since the schedule (turn-1 admission plus every later turn) is
now derived entirely from the reference speed: for this workload (48
sessions, 3-6 turns each, `REF_TTFT_MS = 700`, `REF_TPOT_MS = 110`,
`CONCURRENCY = 16`) that value is about 0.52 turns/s, computed by running
the same admission-queue simulation to completion at exactly the reference
speed and dividing total turns by the simulated wall time. Do not compare a
`benchmark_version` 2 ledger row against a `benchmark_version` 3 row.

`benchmark_version` 4: concurrency cap configurable via `--concurrency`,
default unlimited (was fixed 16). `--concurrency N` sets the admission-queue
size for both the real `asyncio.Semaphore` and the `--pacing scheduled`
simulation; `N = 0` (the new default) means unlimited, all 48 sessions may
run at once. Pass `--concurrency 16` to reproduce the `benchmark_version` 3
workload exactly; `evaluate_once.sh` forwards `$VIBESYS_BENCH_CONCURRENCY` as
`--concurrency` when set. Do not compare a `benchmark_version` 3 ledger row
against a `benchmark_version` 4 row unless both used the same
`--concurrency`.
