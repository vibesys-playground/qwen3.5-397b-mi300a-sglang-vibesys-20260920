#!/usr/bin/env bash
#
# Boot the multiturn task's SGLang server once via serve.py, run the
# accuracy checker and the benchmark against it with --base-url (so neither
# boots its own server), and always tear the server down afterward.
#
# Usage:
#   evaluate_once.sh WORKSPACE OUTDIR
#
# Env:
#   STARTUP_TIMEOUT             Seconds to wait for the server to become
#                               ready. Default 3600.
#   VIBESYS_BENCH_RAMP          Forwarded to benchmark/run.py as --ramp
#                               (comma-separated concurrency levels) when
#                               set. Unset (default) runs the full default
#                               ramp.
#
# Writes OUTDIR/accuracy.json, OUTDIR/bench.jsonl, OUTDIR/server.log, and
# OUTDIR/timestamps.txt (UTC timestamps for start/ready/each step/end, plus
# each step's exit code). Exits non-zero if the server never became ready,
# or if either the accuracy check or the benchmark failed.
#
# Deliberately does not `set -e`: a failing accuracy-check or benchmark step
# must not skip the teardown below it, and the script's own exit code is
# computed explicitly from the two step results afterward. Cleanup runs from
# an EXIT trap, so serve.py is stopped regardless of how this script exits
# (normal return, an early `exit`, or a signal) -- including when a caller
# with `set -e` invokes it as a plain subprocess, which is unaffected by the
# caller's shell options in any case.
set -uo pipefail
set -m # give serve.py its own process group (pgid == pid) so it can be
       # signaled as a group below, without relying on `setsid`.

if [[ $# -lt 2 ]]; then
  echo "usage: $0 WORKSPACE OUTDIR" >&2
  exit 2
fi

WORKSPACE="$1"
OUTDIR="$2"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -- "$OUTDIR"

SERVER_LOG="$OUTDIR/server.log"
ACCURACY_JSON="$OUTDIR/accuracy.json"
BENCH_JSONL="$OUTDIR/bench.jsonl"
TIMESTAMPS="$OUTDIR/timestamps.txt"
READY_FILE="$OUTDIR/.serve-ready"

now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

: > "$TIMESTAMPS"
rm -f -- "$READY_FILE"

# Read _server.py's own host/port defaults instead of hardcoding a copy, so
# this script always talks to the same address serve.py actually binds.
read -r SERVER_HOST SERVER_PORT < <(
  python3 -c "
import sys
sys.path.insert(0, '$TASK_DIR')
import _server
print(_server.DEFAULT_HOST, _server.DEFAULT_PORT)
"
)
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"

SERVE_PID=""

stop_serve() {
  if [[ -n "$SERVE_PID" ]]; then
    kill -TERM -- "-$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap stop_serve EXIT

echo "start $(now)" >> "$TIMESTAMPS"

# The server's own log goes to OUTDIR (not the shared checkout) so that
# concurrent runs against one checkout do not interleave their logs; the
# holder's own stderr goes to a separate file.
python3 "$TASK_DIR/serve.py" \
  --workspace "$WORKSPACE" \
  --startup-timeout-seconds "$STARTUP_TIMEOUT" \
  --ready-file "$READY_FILE" \
  --log-path "$SERVER_LOG" \
  > "$OUTDIR/serve.log" 2>&1 &
SERVE_PID=$!

waited=0
while [[ ! -e "$READY_FILE" ]]; do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "evaluate_once.sh: serve.py exited before the server became ready; server.log tail:" >&2
    tail -c 4000 -- "$SERVER_LOG" >&2
    echo "ready_failed $(now)" >> "$TIMESTAMPS"
    exit 1
  fi
  if (( waited >= STARTUP_TIMEOUT )); then
    echo "evaluate_once.sh: server did not become ready within ${STARTUP_TIMEOUT}s" >&2
    echo "ready_timeout $(now)" >> "$TIMESTAMPS"
    exit 1
  fi
  sleep 2
  waited=$((waited + 2))
done
echo "ready $(now)" >> "$TIMESTAMPS"

python3 "$TASK_DIR/accuracy_checker/checker.py" \
  --workspace "$WORKSPACE" \
  --base-url "$BASE_URL" \
  --output-json "$ACCURACY_JSON"
ACCURACY_EXIT=$?
echo "accuracy_exit=$ACCURACY_EXIT $(now)" >> "$TIMESTAMPS"

BENCH_ARGS=()
if [[ -n "${VIBESYS_BENCH_RAMP:-}" ]]; then
  BENCH_ARGS+=(--ramp "$VIBESYS_BENCH_RAMP")
fi

python3 "$TASK_DIR/benchmark/run.py" \
  --workspace "$WORKSPACE" \
  --base-url "$BASE_URL" \
  --vs-output "$BENCH_JSONL" \
  "${BENCH_ARGS[@]}"
BENCH_EXIT=$?
echo "bench_exit=$BENCH_EXIT $(now)" >> "$TIMESTAMPS"

echo "end $(now)" >> "$TIMESTAMPS"

if [[ $ACCURACY_EXIT -ne 0 || $BENCH_EXIT -ne 0 ]]; then
  exit 1
fi
exit 0
