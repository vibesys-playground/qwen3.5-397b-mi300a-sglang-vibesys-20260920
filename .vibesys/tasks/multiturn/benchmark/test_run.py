#!/usr/bin/env python3
"""Unit tests for the multiturn benchmark: the concurrency ramp, its metrics, and the protocol.

Covers ``write_metrics``, ``write_turn_records``, ``validate_attempt``,
``parse_ramp_levels``, the window-attribution pure functions
(``tokens_attributed_to_window`` / ``window_throughput``), the session pool and
worker-loop in-flight controller (``_SessionPool``, ``_worker_loop``, virtual
clock), ``compute_level_metrics`` and ``decide_level_outcome`` (the stop
rule), ``aggregate_ramp_metrics`` and ``resolve_peak_goodput`` (including the
guardrail-eligibility fallback), and the evaluator result protocol this
benchmark emits under ``--vs-output`` (``vs_protocol``, the declared metric
schema, and ``main_async``'s record stream on the success, hard-error, and
partial-run paths). No real server, no GPU. Run with:

    uv run pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/benchmark/test_run.py -q --no-cov -p no:tach

Skips cleanly if ``run.py`` cannot be imported (e.g. ``aiohttp`` missing).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import heapq
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vs_protocol  # noqa: E402

try:
    import run
except Exception as exc:  # noqa: BLE001 -- env-dependent import; skip, don't fail.
    run = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class WriteMetricsTests(unittest.TestCase):
    def test_writes_the_payload_as_flat_json(self) -> None:
        payload = {"peak_goodput_tok_s": 123.4, "partial": False, "ramp_levels_run": [{"a": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            run.write_metrics(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_metrics_names_include_the_objectives(self) -> None:
        self.assertIn("peak_goodput_tok_s", run.METRICS)
        self.assertIn("p95_ttft_turn2plus_ms_at_ref", run.METRICS)


def _attempt(text: str = "hi", completion_tokens: int | None = 10) -> run.TurnAttempt:
    return run.TurnAttempt(text, 0.1, completion_tokens, 5, None, 1.0, None)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ValidateAttemptTests(unittest.TestCase):
    def test_exact_budget_passes(self) -> None:
        self.assertIsNone(run.validate_attempt(_attempt(completion_tokens=10), 10))

    def test_truncated_fails(self) -> None:
        self.assertIn("truncated", run.validate_attempt(_attempt(completion_tokens=9), 10))

    def test_over_budget_fails(self) -> None:
        self.assertIn("truncated", run.validate_attempt(_attempt(completion_tokens=11), 10))

    def test_missing_usage_fails(self) -> None:
        self.assertIn("usage", run.validate_attempt(_attempt(completion_tokens=None), 10))

    def test_empty_text_fails(self) -> None:
        self.assertIn("empty", run.validate_attempt(_attempt(text=""), 10))


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class TurnsOutputPathTests(unittest.TestCase):
    def test_appends_turns_jsonl_suffix(self) -> None:
        self.assertEqual(
            run.turns_output_path(Path("/tmp/out.json")),
            Path("/tmp/out.json.turns.jsonl"),
        )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class WriteTurnRecordsTests(unittest.TestCase):
    def _fake_results(self) -> list[run.TurnResult]:
        return [
            run.TurnResult(
                session_id=0,
                turn_index=1,
                ok=True,
                ttft_s=0.123,
                completion_tokens=42,
                latency_s=1.5,
                error=None,
                send_ts_monotonic=100.0,
                send_ts_wall=1_700_000_000.0,
                first_token_ts_wall=1_700_000_000.123,
                completion_ts_wall=1_700_000_001.5,
                prompt_tokens=17,
                cached_tokens=5,
                first_token_ts_perf=100.123,
                completion_ts_perf=101.5,
            ),
            run.TurnResult(
                session_id=0,
                turn_index=2,
                ok=False,
                ttft_s=None,
                completion_tokens=None,
                latency_s=0.05,
                error="HTTP 500: boom",
                send_ts_monotonic=200.0,
                send_ts_wall=1_700_000_010.0,
                first_token_ts_wall=None,
                completion_ts_wall=1_700_000_010.05,
                prompt_tokens=None,
                cached_tokens=None,
            ),
        ]

    def test_writes_one_record_per_turn_with_expected_fields(self) -> None:
        results = self._fake_results()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl.turns.jsonl"
            run.write_turn_records(path, results)
            lines = path.read_text().splitlines()

        self.assertEqual(len(lines), len(results))
        first = json.loads(lines[0])
        self.assertEqual(first["session_id"], 0)
        self.assertEqual(first["turn_index"], 1)
        self.assertIs(first["ok"], True)
        self.assertEqual(first["send_ts_monotonic"], 100.0)
        self.assertEqual(first["send_ts_wall"], 1_700_000_000.0)
        self.assertAlmostEqual(first["ttft_ms"], 123.0)
        self.assertEqual(first["first_token_ts_wall"], 1_700_000_000.123)
        self.assertEqual(first["completion_ts_wall"], 1_700_000_001.5)
        self.assertEqual(first["output_tokens"], 42)
        self.assertEqual(first["prompt_tokens"], 17)
        self.assertEqual(first["cached_tokens"], 5)
        self.assertIsNone(first["error"])

        second = json.loads(lines[1])
        self.assertIs(second["ok"], False)
        self.assertIsNone(second["ttft_ms"])
        self.assertIsNone(second["output_tokens"])
        self.assertEqual(second["error"], "HTTP 500: boom")

    def test_empty_results_writes_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl.turns.jsonl"
            run.write_turn_records(path, [])
            self.assertEqual(path.read_text(), "")


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class MetricsOutputPathTests(unittest.TestCase):
    def test_explicit_output_json_wins(self) -> None:
        self.assertEqual(
            run.metrics_output_path("/tmp/out.json", "/tmp/stream.json"), Path("/tmp/out.json")
        )

    def test_defaults_beside_the_protocol_stream(self) -> None:
        self.assertEqual(
            run.metrics_output_path(None, "/tmp/vibesys-framework-benchmark-r1-abc.json"),
            Path("/tmp/vibesys-framework-benchmark-r1-abc.json.metrics.json"),
        )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ParseRampLevelsTests(unittest.TestCase):
    def test_none_returns_the_default_schedule(self) -> None:
        self.assertEqual(run.parse_ramp_levels(None), run.DEFAULT_RAMP_LEVELS)

    def test_explicit_parse(self) -> None:
        self.assertEqual(run.parse_ramp_levels("1,4,8"), (1, 4, 8))

    def test_tolerates_whitespace(self) -> None:
        self.assertEqual(run.parse_ramp_levels(" 1, 4 ,8 "), (1, 4, 8))

    def test_rejects_empty_string(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_ramp_levels("")

    def test_rejects_non_increasing(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_ramp_levels("4,2,8")

    def test_rejects_non_positive(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_ramp_levels("0,4,8")

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_ramp_levels("4,4,8")

    def test_rejects_non_integer_entries(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run.parse_ramp_levels("1,four,8")


def _config_args(**overrides: object) -> argparse.Namespace:
    """A minimal Namespace covering exactly what resolve_ramp_levels/ramp_config_from_args read."""
    defaults: dict[str, object] = dict(
        ramp=None,
        quick=False,
        warmup_s=None,
        window_s=None,
        window_extension_cap_s=None,
        ref_concurrency=16,
        guardrail_tpot_p95_ms=250.0,
        guardrail_ttft_p95_ms=10_000.0,
        no_early_stop=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ResolveRampLevelsTests(unittest.TestCase):
    """--quick's single-level default, and --ramp overriding it (see resolve_ramp_levels)."""

    def test_quick_defaults_to_a_single_level_at_c48(self) -> None:
        levels, used_default_schedule = run.resolve_ramp_levels(_config_args(quick=True))
        self.assertEqual(levels, (48,))
        self.assertEqual(levels, run.QUICK_RAMP_LEVELS)
        self.assertFalse(used_default_schedule)

    def test_explicit_ramp_overrides_quick(self) -> None:
        levels, used_default_schedule = run.resolve_ramp_levels(
            _config_args(quick=True, ramp=(1, 4, 8))
        )
        self.assertEqual(levels, (1, 4, 8))
        self.assertFalse(used_default_schedule)

    def test_neither_quick_nor_ramp_uses_the_scored_default_schedule(self) -> None:
        levels, used_default_schedule = run.resolve_ramp_levels(_config_args())
        self.assertEqual(levels, run.DEFAULT_RAMP_LEVELS)
        self.assertTrue(used_default_schedule)

    def test_explicit_ramp_without_quick_is_not_the_default_schedule(self) -> None:
        levels, used_default_schedule = run.resolve_ramp_levels(_config_args(ramp=(1, 4)))
        self.assertEqual(levels, (1, 4))
        self.assertFalse(used_default_schedule)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class RampConfigFromArgsQuickTests(unittest.TestCase):
    """--quick's warmup/window/extension-cap defaults, and explicit flags still winning."""

    def test_quick_uses_the_short_warmup_and_window_with_no_extension(self) -> None:
        config = run.ramp_config_from_args(_config_args(quick=True))
        self.assertEqual(config.warmup_s, run.QUICK_WARMUP_S)
        self.assertEqual(config.window_s, run.QUICK_WINDOW_S)
        # No extension beyond 1x: the cap equals the window itself.
        self.assertEqual(config.extension_cap_s, config.window_s)
        self.assertTrue(config.early_abort_enabled)

    def test_quick_window_is_within_the_45_to_60s_band(self) -> None:
        self.assertGreaterEqual(run.QUICK_WINDOW_S, 45.0)
        self.assertLessEqual(run.QUICK_WINDOW_S, 60.0)

    def test_non_quick_uses_the_scored_defaults(self) -> None:
        config = run.ramp_config_from_args(_config_args(quick=False))
        self.assertEqual(config.warmup_s, run.DEFAULT_WARMUP_S)
        self.assertEqual(config.window_s, run.DEFAULT_WINDOW_S)
        self.assertEqual(
            config.extension_cap_s, run.DEFAULT_WINDOW_EXTENSION_MULTIPLIER * run.DEFAULT_WINDOW_S
        )
        self.assertFalse(config.early_abort_enabled)

    def test_explicit_warmup_and_window_override_quick(self) -> None:
        config = run.ramp_config_from_args(_config_args(quick=True, warmup_s=3.0, window_s=20.0))
        self.assertEqual(config.warmup_s, 3.0)
        self.assertEqual(config.window_s, 20.0)

    def test_explicit_extension_cap_overrides_quicks_no_extension_default(self) -> None:
        config = run.ramp_config_from_args(_config_args(quick=True, window_extension_cap_s=200.0))
        self.assertEqual(config.extension_cap_s, 200.0)

    def test_no_early_stop_disables_quicks_early_abort(self) -> None:
        config = run.ramp_config_from_args(_config_args(quick=True, no_early_stop=True))
        self.assertFalse(config.early_abort_enabled)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class TokensAttributedToWindowTests(unittest.TestCase):
    def test_turn_fully_inside_window_gets_full_credit(self) -> None:
        self.assertAlmostEqual(run.tokens_attributed_to_window(10.0, 11.0, 100, 0.0, 20.0), 100.0)

    def test_turn_fully_outside_window_gets_no_credit(self) -> None:
        self.assertEqual(run.tokens_attributed_to_window(30.0, 31.0, 100, 0.0, 20.0), 0.0)

    def test_turn_straddling_the_window_start_gets_partial_credit(self) -> None:
        # rate = 100 tokens / 2s = 50 tok/s; overlap with [6, 20) is [6, 7) = 1s.
        result = run.tokens_attributed_to_window(5.0, 7.0, 100, 6.0, 20.0)
        self.assertAlmostEqual(result, 50.0)

    def test_turn_straddling_the_window_end_gets_partial_credit(self) -> None:
        # rate = 80 tokens / 4s = 20 tok/s; overlap with [0, 7) is [5, 7) = 2s.
        result = run.tokens_attributed_to_window(5.0, 9.0, 80, 0.0, 7.0)
        self.assertAlmostEqual(result, 40.0)

    def test_instantaneous_turn_inside_window_gets_full_credit(self) -> None:
        self.assertEqual(run.tokens_attributed_to_window(5.0, 5.0, 1, 0.0, 10.0), 1.0)

    def test_instantaneous_turn_outside_window_gets_no_credit(self) -> None:
        self.assertEqual(run.tokens_attributed_to_window(15.0, 15.0, 1, 0.0, 10.0), 0.0)

    def test_instantaneous_turn_at_the_window_end_is_excluded(self) -> None:
        # The window is half-open: [window_start, window_end).
        self.assertEqual(run.tokens_attributed_to_window(10.0, 10.0, 1, 0.0, 10.0), 0.0)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class WindowThroughputTests(unittest.TestCase):
    def test_sums_multiple_turns_over_the_window_duration(self) -> None:
        turns = [(0.0, 1.0, 100), (0.0, 1.0, 100)]
        self.assertAlmostEqual(run.window_throughput(turns, 0.0, 2.0), 100.0)

    def test_empty_turns_is_zero(self) -> None:
        self.assertEqual(run.window_throughput([], 0.0, 10.0), 0.0)

    def test_zero_duration_window_is_zero(self) -> None:
        self.assertEqual(run.window_throughput([(0.0, 1.0, 10)], 5.0, 5.0), 0.0)


def _turn_result(
    session_id: int = 0,
    turn_index: int = 1,
    ok: bool = True,
    ttft_s: float | None = 0.01,
    completion_tokens: int | None = 10,
    latency_s: float | None = 0.5,
    send_ts_monotonic: float = 0.0,
    error: str | None = None,
) -> run.TurnResult:
    first_token_ts_perf = None if ttft_s is None else send_ts_monotonic + ttft_s
    completion_ts_perf = None if latency_s is None else send_ts_monotonic + latency_s
    return run.TurnResult(
        session_id=session_id,
        turn_index=turn_index,
        ok=ok,
        ttft_s=ttft_s,
        completion_tokens=completion_tokens,
        latency_s=latency_s,
        error=error,
        send_ts_monotonic=send_ts_monotonic,
        first_token_ts_perf=first_token_ts_perf,
        completion_ts_perf=completion_ts_perf,
    )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ComputeLevelMetricsTests(unittest.TestCase):
    def test_valid_level_reports_percentiles_and_throughput(self) -> None:
        results = [
            _turn_result(
                turn_index=1, send_ts_monotonic=5.0, ttft_s=0.1, completion_tokens=50, latency_s=1.0
            ),
            _turn_result(
                turn_index=2, send_ts_monotonic=6.0, ttft_s=0.2, completion_tokens=50, latency_s=1.0
            ),
        ]
        window = run._Window(0.0, 10.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=2, min_turns=2
        )
        self.assertTrue(level.valid)
        self.assertIsNone(level.invalid_reason)
        self.assertFalse(level.low_sample)
        self.assertEqual(level.num_turns_sent_in_window, 2)
        self.assertGreater(level.throughput_tok_s, 0.0)

    def test_failed_turn_in_window_is_invalid(self) -> None:
        results = [
            _turn_result(
                turn_index=1,
                send_ts_monotonic=1.0,
                ok=False,
                ttft_s=None,
                completion_tokens=None,
                latency_s=0.1,
                error="boom",
            ),
            _turn_result(turn_index=1, send_ts_monotonic=2.0),
        ]
        window = run._Window(0.0, 10.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=1, min_turns=1
        )
        self.assertFalse(level.valid)
        self.assertIn("failed", level.invalid_reason)

    def test_failed_turn_during_warmup_still_invalidates_the_level(self) -> None:
        # send_ts_monotonic=1.0 falls before window.start=5.0 (warmup), but is
        # still inside [level_start, window.end).
        results = [
            _turn_result(
                send_ts_monotonic=1.0,
                ok=False,
                ttft_s=None,
                completion_tokens=None,
                latency_s=0.1,
                error="boom",
            )
        ]
        window = run._Window(5.0, 15.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=1, min_turns=1
        )
        self.assertFalse(level.valid)

    def test_insufficient_turns_in_window_is_low_sample_not_invalid(self) -> None:
        # A slow server must score low, not error: a sample shortfall (even
        # after the window's extension cap) flags low_sample rather than
        # invalidating the level.
        results = [_turn_result(send_ts_monotonic=1.0)]
        window = run._Window(0.0, 10.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=1, min_turns=5
        )
        self.assertTrue(level.valid)
        self.assertIsNone(level.invalid_reason)
        self.assertTrue(level.low_sample)
        self.assertEqual(level.num_turns_sent_in_window, 1)

    def test_failed_turn_with_low_sample_is_still_invalid(self) -> None:
        # Invalid stays reserved for failed/dropped/empty/short/errored turns;
        # a low-sample level with an actual failure is still invalid.
        results = [
            _turn_result(
                send_ts_monotonic=1.0,
                ok=False,
                ttft_s=None,
                completion_tokens=None,
                latency_s=0.1,
                error="boom",
            )
        ]
        window = run._Window(0.0, 10.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=1, min_turns=5
        )
        self.assertFalse(level.valid)
        self.assertIn("failed", level.invalid_reason)
        self.assertTrue(level.low_sample)

    def test_turn2plus_ttft_excludes_turn_one(self) -> None:
        results = [
            _turn_result(turn_index=1, send_ts_monotonic=1.0, ttft_s=1.0),
            _turn_result(turn_index=2, send_ts_monotonic=2.0, ttft_s=0.5),
        ]
        window = run._Window(0.0, 10.0)
        level = run.compute_level_metrics(
            results, level_start=0.0, window=window, concurrency=1, min_turns=2
        )
        self.assertAlmostEqual(level.ttft_turn2plus_p50_ms, 500.0)
        self.assertAlmostEqual(level.ttft_p50_ms, 750.0)  # median of [1000, 500]

    def test_finalization_includes_turn_completed_after_window_snapshot(self) -> None:
        completed_at_snapshot = _turn_result(
            session_id=0,
            send_ts_monotonic=1.0,
            ttft_s=0.0,
            completion_tokens=100,
            latency_s=1.0,
        )
        completed_during_drain = _turn_result(
            session_id=1,
            send_ts_monotonic=9.0,
            ttft_s=0.5,
            completion_tokens=100,
            latency_s=1.5,
        )
        window = run._Window(0.0, 10.0)
        provisional = run.compute_level_metrics(
            [completed_at_snapshot],
            level_start=0.0,
            window=window,
            concurrency=1,
            min_turns=20,
        )
        provisional = dataclasses.replace(provisional, early_aborted=True)
        config = run.RampConfig(
            warmup_s=0.0,
            window_s=10.0,
            extension_cap_s=10.0,
            ref_concurrency=1,
            guardrail_tpot_p95_ms=10_000.0,
            guardrail_ttft_p95_ms=10_000.0,
            early_stop_disabled=False,
        )

        finalized, best_goodput = run._finalize_ramp_levels(
            [provisional], [completed_at_snapshot, completed_during_drain], config
        )

        self.assertEqual(finalized[0].num_turns_sent_in_window, 2)
        self.assertGreater(finalized[0].throughput_tok_s, provisional.throughput_tok_s)
        self.assertEqual(best_goodput, finalized[0].throughput_tok_s)
        self.assertTrue(finalized[0].early_aborted)


def _level(
    valid: bool = True,
    throughput: float = 100.0,
    ttft: float = 100.0,
    tpot: float = 50.0,
    concurrency: int = 16,
    invalid_reason: str | None = None,
    low_sample: bool = False,
    early_aborted: bool = False,
) -> run.LevelMetrics:
    return run.LevelMetrics(
        concurrency=concurrency,
        level_start=0.0,
        window_start=0.0,
        window_end=1.0,
        num_turns_sent_in_window=100,
        throughput_tok_s=throughput,
        ttft_p50_ms=ttft,
        ttft_p95_ms=ttft,
        ttft_p99_ms=ttft,
        ttft_turn2plus_p50_ms=ttft,
        ttft_turn2plus_p95_ms=ttft,
        ttft_turn2plus_p99_ms=ttft,
        tpot_p50_ms=tpot,
        tpot_p95_ms=tpot,
        low_sample=low_sample,
        valid=valid,
        invalid_reason=invalid_reason,
        early_aborted=early_aborted,
    )


def _config(
    ref_concurrency: int = 16,
    guardrail_tpot: float = 250.0,
    guardrail_ttft: float = 10_000.0,
    early_stop_disabled: bool = False,
    early_abort_enabled: bool = False,
    warmup_s: float = 1.0,
    window_s: float = 1.0,
    extension_cap_s: float = 4.0,
) -> run.RampConfig:
    return run.RampConfig(
        warmup_s=warmup_s,
        window_s=window_s,
        extension_cap_s=extension_cap_s,
        ref_concurrency=ref_concurrency,
        guardrail_tpot_p95_ms=guardrail_tpot,
        guardrail_ttft_p95_ms=guardrail_ttft,
        early_stop_disabled=early_stop_disabled,
        early_abort_enabled=early_abort_enabled,
    )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class DecideLevelOutcomeTests(unittest.TestCase):
    def test_invalid_first_level_is_hard_error(self) -> None:
        level = _level(valid=False, concurrency=1, invalid_reason="boom")
        decision = run.decide_level_outcome(
            level, is_first_level=True, config=_config(), progress=run._RampProgress(None, 0)
        )
        self.assertEqual(decision.action, "hard_error")

    def test_invalid_ref_level_is_hard_error(self) -> None:
        level = _level(valid=False, concurrency=16, invalid_reason="boom")
        decision = run.decide_level_outcome(
            level,
            is_first_level=False,
            config=_config(ref_concurrency=16),
            progress=run._RampProgress(100.0, 0),
        )
        self.assertEqual(decision.action, "hard_error")

    def test_invalid_non_critical_level_stops_the_ramp(self) -> None:
        level = _level(valid=False, concurrency=32, invalid_reason="boom")
        decision = run.decide_level_outcome(
            level,
            is_first_level=False,
            config=_config(ref_concurrency=16),
            progress=run._RampProgress(100.0, 0),
        )
        self.assertEqual(decision.action, "stop")

    def test_guardrail_breach_stops_by_default(self) -> None:
        level = _level(tpot=300.0)  # over GUARDRAIL_TPOT_P95_MS
        decision = run.decide_level_outcome(
            level, is_first_level=False, config=_config(), progress=run._RampProgress(50.0, 0)
        )
        self.assertEqual(decision.action, "stop")

    def test_ttft_guardrail_breach_stops_by_default(self) -> None:
        level = _level(ttft=20_000.0)  # over GUARDRAIL_TTFT_TURN2PLUS_P95_MS
        decision = run.decide_level_outcome(
            level, is_first_level=False, config=_config(), progress=run._RampProgress(50.0, 0)
        )
        self.assertEqual(decision.action, "stop")

    def test_guardrail_breach_continues_and_excludes_from_goodput_with_no_early_stop(self) -> None:
        level = _level(tpot=300.0, throughput=999.0)
        progress = run._RampProgress(50.0, 0)
        decision = run.decide_level_outcome(
            level, is_first_level=False, config=_config(early_stop_disabled=True), progress=progress
        )
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.progress.best_goodput, 50.0)  # unchanged by the breaching level

    def test_plateau_stops_after_two_consecutive_low_gain_levels(self) -> None:
        config = _config()
        progress = run._RampProgress(100.0, 0)
        d1 = run.decide_level_outcome(
            _level(throughput=101.0), is_first_level=False, config=config, progress=progress
        )
        self.assertEqual(d1.action, "continue")
        self.assertEqual(d1.progress.consecutive_low_gain, 1)
        d2 = run.decide_level_outcome(
            _level(throughput=102.0), is_first_level=False, config=config, progress=d1.progress
        )
        self.assertEqual(d2.action, "stop")

    def test_high_gain_resets_the_plateau_counter(self) -> None:
        config = _config()
        progress = run._RampProgress(100.0, 1)
        decision = run.decide_level_outcome(
            _level(throughput=200.0), is_first_level=False, config=config, progress=progress
        )
        self.assertEqual(decision.progress.consecutive_low_gain, 0)
        self.assertEqual(decision.action, "continue")

    def test_no_early_stop_disables_the_plateau_stop(self) -> None:
        config = _config(early_stop_disabled=True)
        progress = run._RampProgress(100.0, 1)
        decision = run.decide_level_outcome(
            _level(throughput=101.0), is_first_level=False, config=config, progress=progress
        )
        self.assertEqual(decision.action, "continue")

    def test_first_level_ever_measured_seeds_best_goodput(self) -> None:
        decision = run.decide_level_outcome(
            _level(throughput=42.0),
            is_first_level=True,
            config=_config(ref_concurrency=999),
            progress=run._RampProgress(None, 0),
        )
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.progress.best_goodput, 42.0)

    def test_low_sample_valid_level_is_not_hard_error_even_at_first_or_ref_level(self) -> None:
        # Regression for a real run: a bespoke engine at ~22s/turn sent only
        # 11 turns at C=1 (min_turns=20), which the old code treated as
        # invalid and hard-errored the whole run on the first level. A slow
        # server must score low, not error: low_sample must not trip the
        # invalid-level (hard-error) path, at the first level or at C_REF.
        level = _level(valid=True, low_sample=True, concurrency=1, throughput=5.0)
        first_level_decision = run.decide_level_outcome(
            level,
            is_first_level=True,
            config=_config(ref_concurrency=999),
            progress=run._RampProgress(None, 0),
        )
        self.assertEqual(first_level_decision.action, "continue")
        self.assertEqual(first_level_decision.progress.best_goodput, 5.0)

        ref_level = _level(valid=True, low_sample=True, concurrency=16, throughput=5.0)
        ref_level_decision = run.decide_level_outcome(
            ref_level,
            is_first_level=False,
            config=_config(ref_concurrency=16),
            progress=run._RampProgress(4.9, 0),
        )
        self.assertNotEqual(ref_level_decision.action, "hard_error")


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ResolvePeakGoodputTests(unittest.TestCase):
    def test_uses_best_goodput_when_present(self) -> None:
        self.assertEqual(run.resolve_peak_goodput([], 123.0), 123.0)

    def test_falls_back_to_max_valid_level_when_every_level_breached_the_guardrail(self) -> None:
        levels_run = [_level(valid=True, throughput=10.0), _level(valid=True, throughput=30.0)]
        self.assertEqual(run.resolve_peak_goodput(levels_run, None), 30.0)

    def test_none_when_no_valid_level_ran(self) -> None:
        levels_run = [_level(valid=False)]
        self.assertIsNone(run.resolve_peak_goodput(levels_run, None))


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class AggregateRampMetricsTests(unittest.TestCase):
    def _run_result(self, levels_run: list, best_goodput: float | None = 100.0) -> run.RampRun:
        return run.RampRun(
            levels_run=levels_run, all_results=[], best_goodput=best_goodput, wall_s=10.0
        )

    def test_ref_level_present_reports_ttft_at_ref(self) -> None:
        levels_run = [_level(concurrency=1), _level(concurrency=16, ttft=321.0)]
        values = run.aggregate_ramp_metrics(
            self._run_result(levels_run), _config(ref_concurrency=16), used_default_schedule=True
        )
        self.assertEqual(values["p95_ttft_turn2plus_ms_at_ref"], 321.0)
        self.assertEqual(values["ramp_default_schedule_complete"], 1.0)

    def test_ref_level_absent_omits_the_metric(self) -> None:
        levels_run = [_level(concurrency=1), _level(concurrency=32)]
        values = run.aggregate_ramp_metrics(
            self._run_result(levels_run), _config(ref_concurrency=16), used_default_schedule=True
        )
        self.assertNotIn("p95_ttft_turn2plus_ms_at_ref", values)
        self.assertEqual(values["ramp_default_schedule_complete"], 0.0)

    def test_explicit_subset_schedule_is_never_marked_complete_even_with_ref(self) -> None:
        levels_run = [_level(concurrency=16)]
        values = run.aggregate_ramp_metrics(
            self._run_result(levels_run), _config(ref_concurrency=16), used_default_schedule=False
        )
        self.assertEqual(values["ramp_default_schedule_complete"], 0.0)

    def test_peak_goodput_uses_the_resolve_peak_goodput_fallback(self) -> None:
        levels_run = [_level(concurrency=1, throughput=10.0)]
        values = run.aggregate_ramp_metrics(
            self._run_result(levels_run, best_goodput=None),
            _config(ref_concurrency=16),
            used_default_schedule=True,
        )
        self.assertEqual(values["peak_goodput_tok_s"], 10.0)

    def test_reports_completed_and_failed_turn_counts(self) -> None:
        all_results = [
            _turn_result(ok=True),
            _turn_result(ok=True),
            _turn_result(ok=False, ttft_s=None, completion_tokens=None),
        ]
        run_result = run.RampRun(
            levels_run=[_level(concurrency=1)],
            all_results=all_results,
            best_goodput=1.0,
            wall_s=5.0,
        )
        values = run.aggregate_ramp_metrics(run_result, _config(), used_default_schedule=True)
        self.assertEqual(values["num_completed_turns"], 2.0)
        self.assertEqual(values["num_failed_turns"], 1.0)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class DeclaredMetricsTests(unittest.TestCase):
    """The declared schema against objectives.toml."""

    def _objectives(self) -> list[dict]:
        import tomllib

        path = Path(__file__).resolve().parent.parent / "objectives.toml"
        return tomllib.loads(path.read_text())["objective"]

    def test_metrics_tuple_is_derived_from_the_spec_table(self) -> None:
        self.assertEqual(run.METRICS, tuple(run.METRIC_SPECS))

    def test_objectives_are_declared_and_required(self) -> None:
        names = [entry["name"] for entry in self._objectives()]
        self.assertEqual(names, ["peak_goodput_tok_s", "p95_ttft_turn2plus_ms_at_ref"])
        for name in names:
            with self.subTest(objective=name):
                self.assertIn(name, run.METRIC_SPECS)
                self.assertTrue(run.METRIC_SPECS[name].required)

    def test_objective_directions_match_the_declared_spec(self) -> None:
        for entry in self._objectives():
            with self.subTest(objective=entry["name"]):
                self.assertEqual(run.METRIC_SPECS[entry["name"]].direction, entry["direction"])

    def test_ref_metric_required_iff_ref_concurrency_in_levels(self) -> None:
        specs_true = run.build_metric_specs(ref_concurrency_in_levels=True)
        specs_false = run.build_metric_specs(ref_concurrency_in_levels=False)
        self.assertTrue(specs_true["p95_ttft_turn2plus_ms_at_ref"].required)
        self.assertFalse(specs_false["p95_ttft_turn2plus_ms_at_ref"].required)

    def test_peak_goodput_is_always_required(self) -> None:
        for flag in (True, False):
            with self.subTest(ref_concurrency_in_levels=flag):
                specs = run.build_metric_specs(ref_concurrency_in_levels=flag)
                self.assertTrue(specs["peak_goodput_tok_s"].required)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class ScoringFieldTests(unittest.TestCase):
    """--output-json's 'scoring' marker: true for a default run, false under --quick."""

    def _run_result(self) -> run.RampRun:
        return run.RampRun(
            levels_run=[_level(concurrency=16)], all_results=[], best_goodput=1.0, wall_s=1.0
        )

    def test_default_run_is_scoring(self) -> None:
        outcome = run.build_measurement_outcome(
            self._run_result(), (16,), _config(), used_default_schedule=True
        )
        self.assertTrue(outcome.scoring)

    def test_quick_run_is_not_scoring(self) -> None:
        outcome = run.build_measurement_outcome(
            self._run_result(), (48,), _config(), used_default_schedule=False, scoring=False
        )
        self.assertFalse(outcome.scoring)

    def test_payload_carries_scoring_true_for_a_default_run(self) -> None:
        outcome = run.build_measurement_outcome(
            self._run_result(), (16,), _config(), used_default_schedule=True
        )
        payload = run.build_output_json_payload(outcome, _config())
        self.assertTrue(payload["scoring"])

    def test_payload_carries_scoring_false_for_a_quick_run(self) -> None:
        outcome = run.build_measurement_outcome(
            self._run_result(), (48,), _config(), used_default_schedule=False, scoring=False
        )
        payload = run.build_output_json_payload(outcome, _config())
        self.assertFalse(payload["scoring"])


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class MeasureScoringWiringTests(unittest.TestCase):
    """measure()'s own args.quick -> MeasurementOutcome.scoring wiring, server/ramp mocked out."""

    def _args(self, tmp: str, *, quick: bool) -> argparse.Namespace:
        return argparse.Namespace(
            workspace=tmp,
            model_path=None,
            base_url="http://fake",
            host=run.launcher.DEFAULT_HOST,
            port=run.launcher.DEFAULT_PORT,
            startup_timeout_seconds=1.0,
            quick=quick,
        )

    async def _measure(self, quick: bool) -> run.MeasurementOutcome:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, quick=quick)
            levels = (48,) if quick else run.DEFAULT_RAMP_LEVELS
            config = _config()
            run_result = run.RampRun(
                levels_run=[_level(concurrency=48 if quick else 16)],
                all_results=[],
                best_goodput=1.0,
                wall_s=1.0,
            )
            with mock.patch.object(run, "run_ramp", mock.AsyncMock(return_value=run_result)):
                return await run.measure(args, levels, config, used_default_schedule=not quick)

    def test_quick_run_produces_a_non_scoring_outcome(self) -> None:
        outcome = asyncio.run(self._measure(quick=True))
        self.assertFalse(outcome.scoring)

    def test_default_run_produces_a_scoring_outcome(self) -> None:
        outcome = asyncio.run(self._measure(quick=False))
        self.assertTrue(outcome.scoring)


# ---------------------------------------------------------------------------
# In-flight controller: _SessionPool (synchronous, no clock needed) and
# _worker_loop / _run_level_window (async, driven by a virtual clock).
# ---------------------------------------------------------------------------


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class SessionPoolTests(unittest.TestCase):
    """A free worker never idles: it always has a fresh session to admit."""

    def test_admits_fresh_sessions_in_order_when_nothing_is_resting(self) -> None:
        pool = run._SessionPool(seed=run.SEED)
        first = pool.take_work(now=0.0)
        second = pool.take_work(now=0.0)
        self.assertEqual(first.session.session_id, 0)
        self.assertEqual(second.session.session_id, 1)
        self.assertEqual(first.next_turn_index, 0)
        self.assertEqual(first.history, [])

    def test_resting_session_not_yet_ready_is_skipped_for_a_fresh_admission(self) -> None:
        pool = run._SessionPool(seed=run.SEED)
        state = pool.take_work(now=0.0)
        pool.rest(state, ready_at=5.0)
        fresh = pool.take_work(now=1.0)
        self.assertNotEqual(fresh.session.session_id, state.session.session_id)

    def test_resting_session_is_returned_once_ready_and_carries_its_state(self) -> None:
        pool = run._SessionPool(seed=run.SEED)
        state = pool.take_work(now=0.0)
        state.history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
        ]
        state.next_turn_index = 1
        pool.rest(state, ready_at=5.0)
        pool.take_work(now=1.0)  # a fresh admission, not yet ready.
        returned = pool.take_work(now=5.0)
        self.assertEqual(returned.session.session_id, state.session.session_id)
        self.assertEqual(returned.next_turn_index, 1)
        self.assertEqual(returned.history, state.history)

    def test_earliest_ready_resting_session_is_returned_first(self) -> None:
        pool = run._SessionPool(seed=run.SEED)
        a = pool.take_work(now=0.0)
        b = pool.take_work(now=0.0)
        pool.rest(b, ready_at=2.0)
        pool.rest(a, ready_at=1.0)
        returned = pool.take_work(now=10.0)
        self.assertEqual(returned.session.session_id, a.session.session_id)


class _VirtualClock:
    """A deterministic virtual clock for driving many concurrent asyncio tasks.

    Tracks a fixed number of "active" tasks that may block via
    ``asyncio.sleep``; once at least that many sleeps are simultaneously
    pending, "now" advances to the earliest pending wake time and resolves it
    (and any ties). A task that yields via ``asyncio.sleep(0)`` (or any
    ``delay <= 0``) takes a fast, real-yield path and is never counted as
    "active" for this purpose, so driving code may freely poll a condition
    with ``await asyncio.sleep(0)`` without needing to be registered.
    """

    def __init__(self, active_tasks: int) -> None:
        self.now = 0.0
        self._active = active_tasks
        self._heap: list[tuple[float, int, asyncio.Future]] = []
        self._seq = 0

    def perf_counter(self) -> float:
        return self.now

    def add_active(self, n: int) -> None:
        self._active += n
        self._maybe_advance()

    def task_done(self) -> None:
        """One fewer task will ever register a sleep again (it finished, or stopped sleeping)."""
        self._active -= 1
        self._maybe_advance()

    def _maybe_advance(self) -> None:
        if not self._heap or len(self._heap) < self._active:
            return
        self.now = max(self.now, self._heap[0][0])
        while self._heap and self._heap[0][0] <= self.now:
            _, _, future = heapq.heappop(self._heap)
            if not future.done():
                future.set_result(None)

    async def sleep(self, delay: float | None, result: object = None) -> object:
        if not delay or delay <= 0:
            await _REAL_ASYNCIO_SLEEP(0)
            return result
        future = asyncio.get_event_loop().create_future()
        self._seq += 1
        heapq.heappush(self._heap, (self.now + delay, self._seq, future))
        self._maybe_advance()
        await future
        return result


_REAL_ASYNCIO_SLEEP = asyncio.sleep


class _FakeResponse:
    """Minimal aiohttp-response stand-in for one streamed chat completion."""

    def __init__(self, ttft_s: float, total_s: float, n_tokens: int) -> None:
        self.status = 200
        self._ttft_s = ttft_s
        self._total_s = total_s
        self._n_tokens = n_tokens

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def text(self) -> str:
        return ""

    @property
    def content(self):
        return self._iter_lines()

    async def _iter_lines(self):
        await asyncio.sleep(self._ttft_s)
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
        remaining = self._n_tokens - 1
        if remaining > 0:
            per_token_s = (self._total_s - self._ttft_s) / remaining
            for _ in range(remaining):
                await asyncio.sleep(per_token_s)
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
        usage = json.dumps({"usage": {"completion_tokens": self._n_tokens}})
        yield f"data: {usage}\n".encode()
        yield b"data: [DONE]\n"


class _SpeedFakeClient:
    """A fake HTTP client whose response speed is a fixed (ttft_s, tpot_s).

    Each reply is computed from the request's own ``max_tokens``, so it works
    correctly with many concurrent, interleaved requests. ``n_tokens`` is
    exactly ``max_tokens``: decoding is greedy with ``ignore_eos``.
    """

    def __init__(self, ttft_s: float, tpot_s: float) -> None:
        self._ttft_s = ttft_s
        self._tpot_s = tpot_s

    def post(self, url: str, json: dict, timeout: object) -> _FakeResponse:  # noqa: A002
        max_tokens = json["max_tokens"]
        total_s = self._ttft_s + self._tpot_s * (max_tokens - 1)
        return _FakeResponse(self._ttft_s, total_s, max_tokens)


class _ConcurrencyTrackingResponse:
    """Wraps a fake response's context-manager lifetime to track in-flight count."""

    def __init__(self, inner, on_exit) -> None:
        self._inner = inner
        self._on_exit = on_exit

    async def __aenter__(self):
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc_info: object) -> bool:
        result = await self._inner.__aexit__(*exc_info)
        self._on_exit()
        return result


class _ConcurrencyTrackingClient:
    """Wraps a fake client's ``post`` to record concurrent-in-flight and total sent."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.current_in_flight = 0
        self.max_in_flight = 0
        self.total_sent = 0

    def post(self, url: str, json: dict, timeout: object):  # noqa: A002
        self.current_in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.current_in_flight)
        self.total_sent += 1

        def _on_exit() -> None:
            self.current_in_flight -= 1

        return _ConcurrencyTrackingResponse(
            self._inner.post(url, json=json, timeout=timeout), _on_exit
        )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class WorkerLoopConcurrencyTests(unittest.TestCase):
    """_worker_loop against a real _SessionPool: the concurrency cap holds, no idling."""

    def test_exactly_c_requests_in_flight_and_steady_throughput(self) -> None:
        asyncio.run(self._body())

    async def _body(self) -> None:
        concurrency = 3
        target_sent = 30
        clock = _VirtualClock(active_tasks=concurrency)
        tracking_client = _ConcurrencyTrackingClient(_SpeedFakeClient(ttft_s=0.05, tpot_s=0.01))
        all_results: list[run.TurnResult] = []
        stop_event = asyncio.Event()
        ctx = run._WorkerContext(
            pool=run._SessionPool(run.SEED),
            client=tracking_client,
            base_url="http://fake",
            all_results=all_results,
            stop_event=stop_event,
        )

        async def run_and_finish() -> None:
            try:
                await run._worker_loop(ctx)
            finally:
                clock.task_done()

        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
        ):
            workers = [asyncio.create_task(run_and_finish()) for _ in range(concurrency)]
            deadline = time.monotonic() + 20.0
            while tracking_client.total_sent < target_sent:
                await asyncio.sleep(0)
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"only {tracking_client.total_sent}/{target_sent} requests sent before timeout"
                    )
            stop_event.set()
            await asyncio.wait_for(asyncio.gather(*workers), timeout=20.0)

        self.assertEqual(tracking_client.max_in_flight, concurrency)
        self.assertGreaterEqual(tracking_client.total_sent, target_sent)
        self.assertGreater(len(all_results), 0)
        self.assertTrue(all(r.ok for r in all_results))


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class RunLevelWindowExtensionTests(unittest.TestCase):
    """A slow-enough producer forces the window to extend, up to the cap."""

    def test_extends_to_the_cap_and_flags_low_sample_not_invalid(self) -> None:
        asyncio.run(self._body())

    async def _body(self) -> None:
        clock = _VirtualClock(active_tasks=2)  # the producer, plus this coroutine's own sleeps.
        all_results: list[run.TurnResult] = []

        async def producer() -> None:
            i = 0
            while True:
                await asyncio.sleep(5.0)
                all_results.append(
                    _turn_result(session_id=i, send_ts_monotonic=time.perf_counter())
                )
                i += 1

        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
        ):
            producer_task = asyncio.create_task(producer())
            config = run.RampConfig(
                warmup_s=0.0,
                window_s=10.0,
                extension_cap_s=40.0,
                ref_concurrency=16,
                guardrail_tpot_p95_ms=250.0,
                guardrail_ttft_p95_ms=10_000.0,
                early_stop_disabled=False,
            )
            level = await run._run_level_window(
                all_results, level_start=0.0, config=config, concurrency=1
            )
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task

        # min_turns = max(20, 2*1) = 20; the producer emits one turn per 5s of
        # virtual time, so reaching 20 turns would need the window extended to
        # 100s -- past the 40s cap. This is a slow server (production analog:
        # a bespoke engine at ~22s/turn sent 11 turns at C=1, min_turns=20),
        # and a slow server must score low, not error: the level is flagged
        # low_sample, not invalidated.
        self.assertTrue(level.valid)
        self.assertIsNone(level.invalid_reason)
        self.assertTrue(level.low_sample)
        self.assertGreaterEqual(level.window_end - level.window_start, 40.0)

    def test_enough_turns_arrive_before_the_cap_is_hit(self) -> None:
        asyncio.run(self._body_enough())

    async def _body_enough(self) -> None:
        clock = _VirtualClock(active_tasks=2)
        all_results: list[run.TurnResult] = []

        async def producer() -> None:
            i = 0
            while True:
                await asyncio.sleep(0.04)
                all_results.append(
                    _turn_result(session_id=i, send_ts_monotonic=time.perf_counter())
                )
                i += 1

        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
        ):
            producer_task = asyncio.create_task(producer())
            config = run.RampConfig(
                warmup_s=0.0,
                window_s=1.0,
                extension_cap_s=4.0,
                ref_concurrency=16,
                guardrail_tpot_p95_ms=250.0,
                guardrail_ttft_p95_ms=10_000.0,
                early_stop_disabled=False,
            )
            level = await run._run_level_window(
                all_results, level_start=0.0, config=config, concurrency=1
            )
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task

        self.assertTrue(level.valid)
        self.assertAlmostEqual(level.window_end - level.window_start, 1.0)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class EarlyAbortBreachTests(unittest.TestCase):
    """_early_abort_breach: the pure predicate behind --quick's fail-fast check."""

    def _breaching_turns(self, n: int) -> list[run.TurnResult]:
        # ttft=0.01s, completion_tokens=11, latency=1.51s -> tpot=150ms/token.
        return [_turn_result(ttft_s=0.01, completion_tokens=11, latency_s=1.51) for _ in range(n)]

    def test_fewer_than_min_turns_never_triggers_even_with_a_clear_breach(self) -> None:
        turns = self._breaching_turns(run.EARLY_ABORT_MIN_TURNS - 1)
        config = _config(guardrail_tpot=50.0, early_abort_enabled=True)  # 150ms is 3x this.
        self.assertFalse(run._early_abort_breach(turns, config))

    def test_at_least_min_turns_with_a_clear_breach_triggers(self) -> None:
        turns = self._breaching_turns(run.EARLY_ABORT_MIN_TURNS)
        config = _config(guardrail_tpot=50.0, early_abort_enabled=True)
        self.assertTrue(run._early_abort_breach(turns, config))

    def test_within_guardrail_never_triggers_even_with_enough_turns(self) -> None:
        turns = [
            _turn_result(ttft_s=0.01, completion_tokens=11, latency_s=0.51)  # tpot=50ms
            for _ in range(run.EARLY_ABORT_MIN_TURNS)
        ]
        config = _config(guardrail_tpot=50.0, early_abort_enabled=True)
        self.assertFalse(run._early_abort_breach(turns, config))

    def test_just_under_2x_the_guardrail_does_not_trigger(self) -> None:
        # tpot=99ms is under EARLY_ABORT_GUARDRAIL_MULTIPLIER(2.0) * 50ms = 100ms.
        turns = [
            _turn_result(ttft_s=0.01, completion_tokens=11, latency_s=1.0)
            for _ in range(run.EARLY_ABORT_MIN_TURNS)
        ]
        config = _config(guardrail_tpot=50.0, early_abort_enabled=True)
        self.assertFalse(run._early_abort_breach(turns, config))


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class EarlyAbortRunLevelWindowTests(unittest.TestCase):
    """_run_level_window under RampConfig.early_abort_enabled (quick mode only)."""

    def _config_with(self, *, early_abort_enabled: bool, window_s: float) -> run.RampConfig:
        return run.RampConfig(
            warmup_s=0.0,
            window_s=window_s,
            extension_cap_s=window_s,
            ref_concurrency=16,
            guardrail_tpot_p95_ms=50.0,
            guardrail_ttft_p95_ms=10_000.0,
            early_stop_disabled=False,
            early_abort_enabled=early_abort_enabled,
        )

    async def _run_with_breaching_producer(
        self, config: run.RampConfig
    ) -> tuple[run.LevelMetrics, str]:
        clock = _VirtualClock(active_tasks=2)  # the producer, plus this coroutine's own sleeps.
        all_results: list[run.TurnResult] = []

        async def producer() -> None:
            i = 0
            while True:
                await asyncio.sleep(0.1)
                all_results.append(
                    _turn_result(
                        session_id=i,
                        send_ts_monotonic=time.perf_counter(),
                        ttft_s=0.01,
                        completion_tokens=11,
                        latency_s=1.51,  # tpot = 150ms, a clear breach of the 50ms guardrail.
                    )
                )
                i += 1

        stderr = io.StringIO()
        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
            contextlib.redirect_stderr(stderr),
        ):
            producer_task = asyncio.create_task(producer())
            level = await run._run_level_window(
                all_results, level_start=0.0, config=config, concurrency=48
            )
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
        return level, stderr.getvalue()

    def test_breach_partway_through_aborts_the_window_early_and_reports_it(self) -> None:
        asyncio.run(self._body_enabled())

    async def _body_enabled(self) -> None:
        # 30 turns (EARLY_ABORT_MIN_TURNS) need 3.0s of virtual time at one
        # turn per 0.1s; the first poll after that, at the 5.0s
        # EARLY_ABORT_POLL_INTERVAL_S mark, finds the breach and cuts the
        # 100s window down to about 5s instead of running it out.
        config = self._config_with(early_abort_enabled=True, window_s=100.0)
        level, stderr_output = await self._run_with_breaching_producer(config)

        self.assertTrue(level.early_aborted)
        self.assertLess(level.window_end - level.window_start, 100.0)
        self.assertGreater(
            level.tpot_p95_ms, run.EARLY_ABORT_GUARDRAIL_MULTIPLIER * config.guardrail_tpot_p95_ms
        )
        self.assertIn("aborting", stderr_output)
        self.assertIn("C=48", stderr_output)

    def test_disabled_by_default_runs_the_full_window_despite_the_same_breach(self) -> None:
        asyncio.run(self._body_disabled())

    async def _body_disabled(self) -> None:
        config = self._config_with(early_abort_enabled=False, window_s=10.0)
        level, stderr_output = await self._run_with_breaching_producer(config)

        self.assertFalse(level.early_aborted)
        self.assertAlmostEqual(level.window_end - level.window_start, 10.0)
        self.assertEqual(stderr_output, "")
        # The scored path's own guardrail check still sees the breach at the
        # end of the (full) window -- it is just not cut short getting there.
        self.assertGreater(
            level.tpot_p95_ms, run.EARLY_ABORT_GUARDRAIL_MULTIPLIER * config.guardrail_tpot_p95_ms
        )


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class RunRampEndToEndTests(unittest.TestCase):
    """run_ramp's real wiring: spawn, measure, decide, and aggregate across levels."""

    def test_two_level_ramp_completes_and_carries_workers_forward(self) -> None:
        asyncio.run(self._body())

    async def _body(self) -> None:
        levels = (1, 2)
        config = run.RampConfig(
            warmup_s=0.0,
            window_s=2.0,
            extension_cap_s=8.0,
            ref_concurrency=2,
            guardrail_tpot_p95_ms=10_000.0,
            guardrail_ttft_p95_ms=60_000.0,
            early_stop_disabled=True,  # exercise the full schedule regardless of plateau/guardrail.
        )
        # active_tasks starts at 1 (this coordinator, awaiting _run_level_window's
        # own sleeps); spawn_and_track below bumps it by exactly the number of
        # workers spawned at each level transition, and stop_and_track drops it
        # by 1 once the coordinator stops sleeping (it only awaits gather from
        # then on), so the count always matches the real number of concurrently
        # sleeping flows.
        clock = _VirtualClock(active_tasks=1)
        client = _SpeedFakeClient(ttft_s=0.01, tpot_s=0.001)

        class _FakeClientSessionCM:
            async def __aenter__(self) -> _SpeedFakeClient:
                return client

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        real_spawn = run._spawn_additional_workers
        real_stop = run._stop_workers

        def spawn_and_track(workers, current, target, ctx):
            before = len(workers)
            result = real_spawn(workers, current, target, ctx)
            for task in workers[before:]:
                task.add_done_callback(lambda _t: clock.task_done())
            clock.add_active(target - current)
            return result

        async def stop_and_track(stop_event, workers):
            clock.task_done()
            return await real_stop(stop_event, workers)

        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
            mock.patch("aiohttp.ClientSession", return_value=_FakeClientSessionCM()),
            mock.patch.object(run, "_spawn_additional_workers", side_effect=spawn_and_track),
            mock.patch.object(run, "_stop_workers", side_effect=stop_and_track),
        ):
            run_result = await asyncio.wait_for(
                run.run_ramp("http://fake", run.SEED, levels, config), timeout=20.0
            )

        self.assertEqual([lvl.concurrency for lvl in run_result.levels_run], [1, 2])
        self.assertTrue(all(lvl.valid for lvl in run_result.levels_run))
        self.assertGreater(len(run_result.all_results), 0)
        self.assertTrue(all(r.ok for r in run_result.all_results))
        self.assertTrue(
            any(
                result.send_ts_monotonic is not None
                and result.completion_ts_perf is not None
                and result.send_ts_monotonic < level.window_end < result.completion_ts_perf
                for level in run_result.levels_run
                for result in run_result.all_results
            )
        )
        for level in run_result.levels_run:
            recomputed = run.compute_level_metrics(
                run_result.all_results,
                level.level_start,
                run._Window(level.window_start, level.window_end),
                level.concurrency,
                max(20, 2 * level.concurrency),
            )
            self.assertAlmostEqual(level.throughput_tok_s, recomputed.throughput_tok_s)
            self.assertEqual(level.num_turns_sent_in_window, recomputed.num_turns_sent_in_window)

    def test_slow_server_is_low_sample_not_hard_error_and_ramp_does_not_stop_early(
        self,
    ) -> None:
        asyncio.run(self._slow_server_body())

    async def _slow_server_body(self) -> None:
        """Regression for a real run: a bespoke engine at ~22s/turn sent only

        11 of the required 20 turns at C=1 (even after the window's
        extension cap), which the old code treated as an invalid first level
        and hard-errored the whole run. Reproduce that shape (a server too
        slow, at low concurrency, to fill the measurement window) end to end
        through ``run_ramp`` and confirm: no hard error, every requested
        level runs to completion (i.e. neither a spurious plateau nor a
        guardrail breach cuts the ramp short just because the low-
        concurrency levels are slow/low-sample -- ``decide_level_outcome``
        does not even look at ``low_sample``, only at ``valid`` and the
        measured throughput/latency numbers).
        """
        levels = (1, 2, 4)
        config = run.RampConfig(
            warmup_s=0.0,
            window_s=1.0,
            extension_cap_s=4.0,
            ref_concurrency=999,  # not one of `levels`; isolates the first-level path.
            guardrail_tpot_p95_ms=250.0,
            guardrail_ttft_p95_ms=10_000.0,
            early_stop_disabled=False,
        )
        clock = _VirtualClock(active_tasks=1)
        # ttft=0.3s, tpot=0.02s/token: a turn (80-300 tokens) takes roughly
        # 1.9-6.3s, comfortably longer than the 4s window+extension-cap at
        # C=1, so C=1 cannot reach min_turns=20 within it -- the production
        # shape, reproduced with virtual-clock-friendly magnitudes.
        client = _SpeedFakeClient(ttft_s=0.3, tpot_s=0.02)

        class _FakeClientSessionCM:
            async def __aenter__(self) -> _SpeedFakeClient:
                return client

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        real_spawn = run._spawn_additional_workers
        real_stop = run._stop_workers

        def spawn_and_track(workers, current, target, ctx):
            before = len(workers)
            result = real_spawn(workers, current, target, ctx)
            for task in workers[before:]:
                task.add_done_callback(lambda _t: clock.task_done())
            clock.add_active(target - current)
            return result

        async def stop_and_track(stop_event, workers):
            clock.task_done()
            return await real_stop(stop_event, workers)

        with (
            mock.patch("time.perf_counter", new=clock.perf_counter),
            mock.patch("asyncio.sleep", new=clock.sleep),
            mock.patch("aiohttp.ClientSession", return_value=_FakeClientSessionCM()),
            mock.patch.object(run, "_spawn_additional_workers", side_effect=spawn_and_track),
            mock.patch.object(run, "_stop_workers", side_effect=stop_and_track),
        ):
            run_result = await asyncio.wait_for(
                run.run_ramp("http://fake", run.SEED, levels, config), timeout=20.0
            )

        # No hard error: every requested level ran (a hard error would have
        # raised instead of returning, and a spurious plateau/guardrail stop
        # would have truncated levels_run before reaching C=4).
        self.assertEqual([lvl.concurrency for lvl in run_result.levels_run], [1, 2, 4])
        self.assertTrue(all(lvl.valid for lvl in run_result.levels_run))
        self.assertTrue(run_result.levels_run[0].low_sample)


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class MainAsyncProtocolTests(unittest.TestCase):
    """main_async's record stream, with measure() stubbed out."""

    def _args(self, tmp: str, **overrides) -> argparse.Namespace:
        defaults = dict(
            workspace=tmp,
            output_json=None,
            vs_output=str(Path(tmp) / "stream.jsonl"),
            model_path=None,
            host=run.launcher.DEFAULT_HOST,
            port=run.launcher.DEFAULT_PORT,
            startup_timeout_seconds=1.0,
            base_url="http://fake",
            ramp=None,
            quick=False,
            warmup_s=1.0,
            window_s=1.0,
            window_extension_cap_s=None,
            ref_concurrency=16,
            guardrail_tpot_p95_ms=250.0,
            guardrail_ttft_p95_ms=10_000.0,
            no_early_stop=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _records(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _make_outcome(self, **overrides) -> run.MeasurementOutcome:
        values = {name: 1.0 for name in run.METRICS}
        values["peak_goodput_tok_s"] = 275.5
        values["p95_ttft_turn2plus_ms_at_ref"] = 812.0
        defaults = dict(
            values=values,
            all_results=[],
            levels_run=[_level(concurrency=16)],
            ref_required=True,
            ref_reached=True,
            used_default_schedule=True,
        )
        defaults.update(overrides)
        return run.MeasurementOutcome(**defaults)

    def test_success_writes_hello_then_result_with_both_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            outcome = self._make_outcome()
            with mock.patch.object(run, "measure", mock.AsyncMock(return_value=outcome)):
                code = asyncio.run(run.main_async(args))
            self.assertEqual(code, 0)
            records = self._records(Path(args.vs_output))
            self.assertEqual([record["kind"] for record in records], ["hello", "result"])
            self.assertEqual(records[0]["protocol"], vs_protocol.PROTOCOL_VERSION)
            self.assertEqual(set(records[0]["metrics"]), set(run.METRICS))
            self.assertEqual(records[0]["metrics"]["peak_goodput_tok_s"]["direction"], "max")
            self.assertEqual(
                records[0]["metrics"]["p95_ttft_turn2plus_ms_at_ref"]["direction"], "min"
            )
            self.assertEqual(records[1]["values"]["peak_goodput_tok_s"], 275.5)
            self.assertEqual(records[1]["values"]["p95_ttft_turn2plus_ms_at_ref"], 812.0)

    def test_success_also_writes_the_human_readable_json_with_the_level_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            outcome = self._make_outcome()
            with mock.patch.object(run, "measure", mock.AsyncMock(return_value=outcome)):
                asyncio.run(run.main_async(args))
            metrics_path = Path(args.vs_output + ".metrics.json")
            payload = json.loads(metrics_path.read_text())
            self.assertEqual(payload["peak_goodput_tok_s"], 275.5)
            self.assertFalse(payload["partial"])
            self.assertTrue(payload["ramp_default_schedule_complete"])
            self.assertEqual(len(payload["ramp_levels_run"]), 1)
            self.assertTrue(run.turns_output_path(metrics_path).exists())

    def test_hard_error_path_writes_hello_then_error_and_skips_the_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            failure = mock.AsyncMock(
                side_effect=RuntimeError(
                    "ramp level C=1 is invalid (1 failed turn(s) during the level)"
                )
            )
            with mock.patch.object(run, "measure", failure):
                code = asyncio.run(run.main_async(args))
            self.assertEqual(code, 1)
            records = self._records(Path(args.vs_output))
            self.assertEqual([record["kind"] for record in records], ["hello", "error"])
            self.assertIn("invalid", records[1]["message"])
            self.assertFalse(Path(args.vs_output + ".metrics.json").exists())

    def test_explicit_subset_without_ref_reports_normally_as_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, ramp=(1, 4))
            values = {name: 1.0 for name in run.METRICS if name != "p95_ttft_turn2plus_ms_at_ref"}
            values["peak_goodput_tok_s"] = 50.0
            outcome = run.MeasurementOutcome(
                values=values,
                all_results=[],
                levels_run=[_level(concurrency=1), _level(concurrency=4)],
                ref_required=False,
                ref_reached=False,
                used_default_schedule=False,
            )
            with mock.patch.object(run, "measure", mock.AsyncMock(return_value=outcome)):
                code = asyncio.run(run.main_async(args))
            self.assertEqual(code, 0)
            records = self._records(Path(args.vs_output))
            self.assertEqual([record["kind"] for record in records], ["hello", "result"])
            self.assertFalse(
                records[0]["metrics"]["p95_ttft_turn2plus_ms_at_ref"].get("required", True)
            )
            payload = json.loads(Path(args.vs_output + ".metrics.json").read_text())
            self.assertFalse(payload["ramp_default_schedule_complete"])
            self.assertFalse(payload["partial"])

    def test_ramp_stopped_before_ref_level_writes_json_and_fails_the_protocol_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)  # default ramp includes C_REF=16.
            values = {name: 1.0 for name in run.METRICS if name != "p95_ttft_turn2plus_ms_at_ref"}
            levels_run = [_level(concurrency=1), _level(concurrency=2), _level(concurrency=4)]
            outcome = run.MeasurementOutcome(
                values=values,
                all_results=[],
                levels_run=levels_run,
                ref_required=True,
                ref_reached=False,
                used_default_schedule=True,
            )
            with mock.patch.object(run, "measure", mock.AsyncMock(return_value=outcome)):
                code = asyncio.run(run.main_async(args))
            self.assertEqual(code, 0)  # not a hard error: the ramp measured something real.
            records = self._records(Path(args.vs_output))
            self.assertEqual([record["kind"] for record in records], ["hello", "error"])
            self.assertIn("C_REF", records[1]["message"])
            metrics_path = Path(args.vs_output + ".metrics.json")
            self.assertTrue(metrics_path.exists())
            payload = json.loads(metrics_path.read_text())
            self.assertTrue(payload["partial"])

    def test_without_vs_output_only_the_human_readable_json_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.json")
            args = self._args(tmp, vs_output=None, output_json=out)
            outcome = self._make_outcome()
            with mock.patch.object(run, "measure", mock.AsyncMock(return_value=outcome)):
                self.assertEqual(asyncio.run(run.main_async(args)), 0)
            self.assertTrue(Path(out).exists())
            self.assertEqual(list(Path(tmp).glob("*.jsonl")), [run.turns_output_path(Path(out))])


class VsProtocolReportTests(unittest.TestCase):
    """vs_protocol.ProtocolReport: the record stream shape PROTOCOL.md fixes."""

    def _records(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_hello_then_result_matches_the_wire_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            with vs_protocol.ProtocolReport(path) as report:
                report.declare(
                    {
                        "aggregate": vs_protocol.MetricSpec(unit="tok/s", direction="max"),
                        "latency_ms": vs_protocol.MetricSpec(unit="ms", direction="min"),
                    }
                )
                report.emit({"aggregate": 1180.4, "latency_ms": 812})
            self.assertEqual(
                self._records(path),
                [
                    {
                        "kind": "hello",
                        "protocol": 2,
                        "metrics": {
                            "aggregate": {"unit": "tok/s", "direction": "max"},
                            "latency_ms": {"unit": "ms", "direction": "min"},
                        },
                    },
                    {
                        "kind": "result",
                        "label": "",
                        "values": {"aggregate": 1180.4, "latency_ms": 812.0},
                    },
                ],
            )

    def test_bare_spec_writes_an_empty_object_and_omits_required_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            with vs_protocol.ProtocolReport(path) as report:
                report.declare({"primary_value": vs_protocol.MetricSpec()})
                report.emit({"primary_value": 0.0})
            self.assertEqual(self._records(path)[0]["metrics"], {"primary_value": {}})

    def test_hello_is_flushed_before_the_outcome_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            report = vs_protocol.ProtocolReport(path).__enter__()
            try:
                report.declare({"primary_value": vs_protocol.MetricSpec()})
                self.assertEqual([r["kind"] for r in self._records(path)], ["hello"])
            finally:
                report.close()

    def test_error_record_needs_no_hello(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            with vs_protocol.ProtocolReport(path) as report:
                report.fail("server did not become ready")
            self.assertEqual(
                self._records(path),
                [{"kind": "error", "message": "server did not become ready"}],
            )

    def test_optional_metric_is_dropped_when_not_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            with vs_protocol.ProtocolReport(path) as report:
                report.declare(
                    {
                        "kept": vs_protocol.MetricSpec(),
                        "sometimes": vs_protocol.MetricSpec(required=False),
                    }
                )
                report.emit({"kept": 1.0, "sometimes": float("nan")})
            self.assertEqual(self._records(path)[1]["values"], {"kept": 1.0})

    def test_required_metric_that_is_not_finite_is_rejected(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        report.declare({"kept": vs_protocol.MetricSpec()})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "not finite"):
            report.emit({"kept": float("inf")})

    def test_required_metric_that_is_absent_is_rejected(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        report.declare(
            {"kept": vs_protocol.MetricSpec(), "other": vs_protocol.MetricSpec(required=False)}
        )
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "never reported: kept"):
            report.emit({"other": 1.0})

    def test_undeclared_metric_is_rejected(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        report.declare({"kept": vs_protocol.MetricSpec()})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "not declared"):
            report.emit({"kept": 1.0, "surprise": 2.0})

    def test_boolean_is_not_a_number(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        report.declare({"kept": vs_protocol.MetricSpec()})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "not a number"):
            report.emit({"kept": True})

    def test_only_one_outcome_and_one_hello(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        report.declare({"kept": vs_protocol.MetricSpec()})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "already written"):
            report.declare({"kept": vs_protocol.MetricSpec()})
        report.emit({"kept": 1.0})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "already written"):
            report.emit({"kept": 2.0})
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "already written"):
            report.fail("too late")

    def test_whitespace_metric_name_is_rejected(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "invalid metric name"):
            report.declare({"two words": vs_protocol.MetricSpec()})

    def test_result_before_hello_is_rejected(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        with self.assertRaisesRegex(vs_protocol.ProtocolError, "before hello"):
            report.emit({"kept": 1.0})

    def test_not_reporting_writes_nothing_but_still_validates(self) -> None:
        report = vs_protocol.ProtocolReport(None)
        self.assertFalse(report.reporting)
        report.declare({"kept": vs_protocol.MetricSpec()})
        with self.assertRaises(vs_protocol.ProtocolError):
            report.emit({"nope": 1.0})


@unittest.skipIf(run is None, f"could not import run.py: {_IMPORT_ERROR}")
class CheckObjectivesIntegrationTests(unittest.TestCase):
    """Optional integration check against the real framework library, if importable."""

    def _hello(self, *, ref_concurrency_in_levels: bool):
        try:
            from vs_evaluator_protocol.records import Hello
            from vs_evaluator_protocol.records import MetricSpec as ProtoMetricSpec
        except ImportError as exc:  # pragma: no cover - environment dependent.
            self.skipTest(f"vs_evaluator_protocol not importable: {exc}")
        specs = run.build_metric_specs(ref_concurrency_in_levels=ref_concurrency_in_levels)
        return Hello(
            protocol=vs_protocol.PROTOCOL_VERSION,
            metrics={name: ProtoMetricSpec(**spec.as_record()) for name, spec in specs.items()},
        )

    def test_default_schedule_hello_satisfies_check_objectives(self) -> None:
        from vs_evaluator_protocol.measurement import check_objectives

        hello = self._hello(ref_concurrency_in_levels=True)
        check_objectives(hello, {"peak_goodput_tok_s", "p95_ttft_turn2plus_ms_at_ref"})

    def test_subset_without_ref_hello_fails_check_objectives(self) -> None:
        from vs_evaluator_protocol.errors import ProtocolError as VsProtocolError
        from vs_evaluator_protocol.measurement import check_objectives

        hello = self._hello(ref_concurrency_in_levels=False)
        with self.assertRaises(VsProtocolError):
            check_objectives(hello, {"peak_goodput_tok_s", "p95_ttft_turn2plus_ms_at_ref"})


if __name__ == "__main__":
    unittest.main()
