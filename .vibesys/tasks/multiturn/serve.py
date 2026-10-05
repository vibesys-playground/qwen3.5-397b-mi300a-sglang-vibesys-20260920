#!/usr/bin/env python3
"""Long-lived SGLang server holder for the multiturn task.

Boots the server the same way ``benchmark/run.py`` and
``accuracy_checker/checker.py`` do (see ``_server.py``), waits for readiness,
then blocks until it receives SIGTERM/SIGINT, tearing the server down on
exit either way. Meant to run in the background of a job (or a long-lived
"hold" allocation) so that ``checker.py`` and ``run.py`` can be invoked
repeatedly against it with ``--base-url``, instead of each booting its own
server.

If the server process dies on its own while this script is waiting, it exits
non-zero and prints the server log tail to stderr instead of blocking
forever. Does not import ``sglang`` at module load time; the server always
runs as a subprocess.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _server  # noqa: E402

POLL_INTERVAL_SECONDS = 1.0


async def main_async(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    model_path = args.model_path or _server.resolve_model_path()
    # Concurrent runs sharing one checkout must not share one server log.
    log_path = Path(args.log_path) if args.log_path else workspace / ".vibesys-serve-server.log"
    base_url = f"http://{args.host}:{args.port}"

    _server.ensure_sglang_importable(workspace)
    proc = _server.start_server(
        workspace=workspace,
        model_path=model_path,
        host=args.host,
        port=args.port,
        tp=args.tp,
        log_path=log_path,
    )
    try:
        try:
            await _server.wait_until_ready(
                base_url,
                proc=proc,
                timeout_seconds=args.startup_timeout_seconds,
                log_path=log_path,
            )
        except RuntimeError as exc:
            print(f"serve.py: server failed to start: {exc}", file=sys.stderr)
            return 1

        print(f"[timing] server ready at {base_url}", file=sys.stderr, flush=True)

        warmup_log = (
            Path(args.gemm_pad_warmup_log) if args.gemm_pad_warmup_log else None
        )
        t_warmup0 = time.monotonic()
        warmup_rows = await _server.run_gemm_pad_warmup(
            base_url, model_path, log_path=warmup_log
        )
        if warmup_rows:
            print(
                f"[timing] gemm-pad warmup ran {len(warmup_rows)} requests in "
                f"{time.monotonic() - t_warmup0:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        if args.ready_file:
            Path(args.ready_file).touch()

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)

        while not stop_event.is_set():
            exit_code = proc.poll()
            if exit_code is not None:
                tail = _server.tail_server_log(log_path)
                print(
                    f"serve.py: server process exited on its own (code {exit_code}); "
                    f"last log output:\n{tail}",
                    file=sys.stderr,
                )
                return 1
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        return 0
    finally:
        _server.stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Path to the SGLang checkout (project root).")
    parser.add_argument("--model-path", default=None, help="Overrides MODEL_PATH / the built-in default.")
    parser.add_argument("--host", default=_server.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_server.DEFAULT_PORT)
    parser.add_argument("--tp", type=int, default=_server.DEFAULT_TP)
    parser.add_argument(
        "--startup-timeout-seconds", type=float, default=_server.STARTUP_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="Path to touch once the server responds healthy, for callers polling readiness.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Where to write the server's stdout/stderr (default: <workspace>/.vibesys-serve-server.log).",
    )
    parser.add_argument(
        "--gemm-pad-warmup-log",
        default=None,
        help=(
            "Where to write the gemm-pad-M boot warmup's per-request token "
            "count and latency (see _server.run_gemm_pad_warmup); default "
            "stderr. No-op when SGLANG_AITER_GEMM_PAD_M=0."
        ),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
