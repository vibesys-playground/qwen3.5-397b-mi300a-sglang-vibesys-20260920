# Objective — SGLang multi-turn chat serving on 4x AMD MI300A

Serve `amd/Qwen3.5-397B-A17B-MXFP4` with this SGLang checkout on a single node
with 4x AMD MI300A (ROCm), tensor-parallel degree 4. Model weights are on disk
at the path given by the `MODEL_PATH` environment variable (default
`/path/to/models/Qwen3.5-397B-A17B-MXFP4`). The server must
expose the standard OpenAI-compatible `/v1/chat/completions` endpoint with
streaming responses; the harness talks to nothing else.

The benchmark (`benchmark/run.py`), the accuracy checker
(`accuracy_checker/checker.py`), the greedy-token pins (`reference/pins.json`),
and the objectives (`objectives.toml`) are the same as in the bespoke-engine
task for this model and hardware. Only `benchmark/launcher.py` differs: it boots
SGLang through `_server.py` instead of a from-scratch `server.py`.

## Workload

The benchmark drives many concurrent multi-turn chat sessions against the
server. Each session is a short back-and-forth: a user message, the server's
reply, another user message appended after the server's own prior reply, and so
on for 3 to 6 turns. Every request carries the session's full accumulated
history (the real text the server returned earlier), so later turns send
strictly more prompt tokens than earlier ones. Between a session's turns there
is a think-time pause of 1 to 8 seconds.

Decoding is greedy (`temperature` 0) with `ignore_eos`: each turn generates
exactly its `max_tokens` budget (80 to 300 tokens).

Load is a concurrency ramp: for each level (default 1, 2, 4, 8, 16, 32, 48, 64,
96, 128, 192, 256) an in-flight controller holds exactly that many requests
concurrent, runs a warmup, then a measurement window. The ramp stops on a
throughput plateau, a latency-guardrail breach, or an invalid level. The
workload is fixed and deterministic (`benchmark/run.py`); it is not yours to
change.

## Metric

Two objectives (`objectives.toml`):

- `peak_goodput_tok_s` (maximize, primary): the best window throughput among
  ramp levels that stay within the latency guardrails, p95 time-per-output-token
  at most 250 ms and p95 turn-2+ time-to-first-token at most 10 s.
- `p95_ttft_turn2plus_ms_at_ref` (minimize): the 95th percentile of
  time-to-first-token over turns with index 2 or later, at the reference
  concurrency level (16).

## Correctness

Any failing check invalidates the round.

1. Accuracy gate (`accuracy_checker/checker.py`), thinking disabled: 13 probes
   (6 held-out multi-turn sessions replayed greedily, 4 history-recall probes,
   3 arithmetic probes) plus a greedy-token pin check. The pins in
   `reference/pins.json` were produced by a reference transformers forward; at
   least 90 percent of the pins must match the first 32 generated tokens
   exactly, and the rest must not diverge before token 8. See
   `accuracy_checker/README.md`.
2. Benchmark integrity: a level is invalid if any request is dropped, errored,
   returns an empty reply, or generates fewer or more completion tokens than
   its fixed budget; an invalid first level or reference level fails the run.

## Scope

You may edit any SGLang source in this checkout. You may not modify anything
under `.vibesys/` (including the benchmark and accuracy-checker programs
themselves and this objective), and you may not modify the model weight
files at `MODEL_PATH`. The server must keep speaking the standard
OpenAI-compatible `/v1/chat/completions` streaming API; the harness has no
other way to reach it.
