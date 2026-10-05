"""Unit tests for the harness's server argv/env building in ``_server.py``.

Covers ``$VIBESYS_EXTRA_SERVER_ARGS`` handling and, in
``SpeculativeDecodeArgsTest``, the ``[speculative]``-table-driven
``speculative_decode_args()``/``spec_decode_enabled()``, the
``VIBESYS_SPEC_DECODE`` off switch, and the two draft-source branches
(site config's ``paths.draft_model`` set vs. unset). Also covers, in
``SchedulerArgsTest``, the ``[scheduler]``-table-driven
``scheduler_args()``/``overlap_schedule_enabled()`` and the
``VIBESYS_OVERLAP_SCHEDULE`` override. Also covers, in ``TunableOpEnvTest``,
the ``[tunableop]``-table-driven ``PYTORCH_TUNABLEOP_*`` vars in
``build_launch_env()``, the per-device-ordinal file fan-out in
``stage_tunableop_files()``, and the ``VIBESYS_TUNABLEOP`` kill switch.
Also covers, in ``PrefillCudaGraphArgsTest``, the
``[prefill_cuda_graph]``-table-driven ``prefill_cuda_graph_args()``/
``prefill_graph_enabled()`` and the ``VIBESYS_PREFILL_GRAPH`` off switch.
Also covers, in ``TokenizerFastPathEnvDefaultsTest``, the
``[env_defaults]``-table ``SGLANG_TEXT_ONLY_SEND_IDS``/
``SGLANG_INCREMENTAL_TOKENIZE`` platform defaults and the caller-env-wins
``setdefault`` contract they share with ``SGLANG_SKINNY_GEMM``/
``SGLANG_MXFP4_MOE_HIP``.
No SGLang install or cluster access required: ``_server.py`` only shells out to
``python3 -m sglang.launch_server`` as a subprocess, so it is safe to
import and exercise standalone. Run with:

    python3 -m pytest .vibesys/tasks/multiturn/test_server.py

or plain ``unittest``:

    python3 .vibesys/tasks/multiturn/test_server.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _server


class ExtraServerArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._orig is None:
            os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
        else:
            os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = self._orig

    def test_unset_yields_no_extra_args(self) -> None:
        self.assertEqual(_server.extra_server_args(), [])

    def test_empty_string_yields_no_extra_args(self) -> None:
        os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = "   "
        self.assertEqual(_server.extra_server_args(), [])

    def test_shell_word_splitting(self) -> None:
        os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = (
            "--enable-mixed-chunk --chunked-prefill-size 1024"
        )
        self.assertEqual(
            _server.extra_server_args(),
            ["--enable-mixed-chunk", "--chunked-prefill-size", "1024"],
        )

    def test_quoted_value_stays_one_token(self) -> None:
        os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = (
            "--model-loader-extra-config '{\"num_threads\": 4}'"
        )
        self.assertEqual(
            _server.extra_server_args(),
            ["--model-loader-extra-config", '{"num_threads": 4}'],
        )

    def test_appended_after_config_derived_args(self) -> None:
        os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = "--sentinel-ad-hoc-flag"
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertEqual(argv[-1], "--sentinel-ad-hoc-flag")
        # The config-derived KV pool pin is always present, and precedes the
        # extra args regardless of loader path.
        self.assertIn("--max-total-tokens", argv)
        self.assertLess(
            argv.index("--max-total-tokens"), argv.index("--sentinel-ad-hoc-flag")
        )

    def test_no_extra_args_leaves_argv_unchanged_shape(self) -> None:
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertNotIn("--sentinel-ad-hoc-flag", argv)

    def test_platform_extra_args_precede_vibesys_extra_server_args(self) -> None:
        # mi300a-rocm700 (the default platform via VIBESYS_SITE=example) sets
        # extra_args for mixed chunked prefill; argv order must be
        # config-derived flags, then platform extra_args, then
        # $VIBESYS_EXTRA_SERVER_ARGS.
        platform_extra = _server._PLATFORM.extra_args
        self.assertTrue(
            platform_extra, "expected the default platform config to set extra_args"
        )
        os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = "--sentinel-ad-hoc-flag"
        argv = _server.build_launch_argv(model_path="/does/not/matter")

        start = argv.index(platform_extra[0])
        self.assertEqual(argv[start : start + len(platform_extra)], platform_extra)
        self.assertLess(argv.index("--max-total-tokens"), start)
        self.assertLess(
            start + len(platform_extra) - 1, argv.index("--sentinel-ad-hoc-flag")
        )
        self.assertEqual(argv[-1], "--sentinel-ad-hoc-flag")

    def test_platform_extra_args_present_without_vibesys_extra_server_args(self) -> None:
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        for token in _server._PLATFORM.extra_args:
            self.assertIn(token, argv)


class SpeculativeDecodeArgsTest(unittest.TestCase):
    """Covers speculative_decode_args()/spec_decode_enabled() against the
    default site+platform (example/mi300a-rocm700 via $VIBESYS_SITE), which
    ships a [speculative] table -- see config/platforms/mi300a-rocm700.toml.
    """

    def setUp(self) -> None:
        self._orig = os.environ.pop("VIBESYS_SPEC_DECODE", None)
        self.addCleanup(self._restore)
        self.assertIsNotNone(
            _server._PLATFORM.speculative,
            "expected the default platform config to declare [speculative]",
        )

    def _restore(self) -> None:
        if self._orig is None:
            os.environ.pop("VIBESYS_SPEC_DECODE", None)
        else:
            os.environ["VIBESYS_SPEC_DECODE"] = self._orig

    def test_on_by_default(self) -> None:
        self.assertTrue(_server.spec_decode_enabled())
        args = _server.speculative_decode_args()
        self.assertIn("--speculative-algorithm", args)
        self.assertEqual(
            args[args.index("--speculative-algorithm") + 1],
            _server._PLATFORM.speculative.algorithm,
        )

    def test_draft_model_path_comes_from_site_config(self) -> None:
        # example's committed default sets paths.draft_model (see
        # config/sites/example.toml), so the default site+platform picks
        # the sharded-artifact draft source; see the two tests below for
        # both branches in isolation.
        self.assertIsNotNone(
            _server._SITE.paths.draft_model,
            "expected the default site config to set paths.draft_model",
        )
        args = _server.speculative_decode_args()
        idx = args.index("--speculative-draft-model-path")
        self.assertEqual(args[idx + 1], _server._SITE.paths.draft_model)
        fmt_idx = args.index("--speculative-draft-load-format")
        self.assertEqual(args[fmt_idx + 1], "sharded_state")

    def test_draft_model_set_uses_sharded_state_regardless_of_platform_format(
        self,
    ) -> None:
        patched_paths = dataclasses.replace(
            _server._SITE.paths, draft_model="/fake/draft-sharded-tp4"
        )
        with mock.patch.object(
            _server, "_SITE", dataclasses.replace(_server._SITE, paths=patched_paths)
        ):
            args = _server.speculative_decode_args()
        path_idx = args.index("--speculative-draft-model-path")
        self.assertEqual(args[path_idx + 1], "/fake/draft-sharded-tp4")
        fmt_idx = args.index("--speculative-draft-load-format")
        self.assertEqual(args[fmt_idx + 1], "sharded_state")

    def test_draft_model_unset_falls_back_to_model_path_and_platform_format(
        self,
    ) -> None:
        patched_paths = dataclasses.replace(_server._SITE.paths, draft_model=None)
        with mock.patch.object(
            _server, "_SITE", dataclasses.replace(_server._SITE, paths=patched_paths)
        ):
            args = _server.speculative_decode_args()
        path_idx = args.index("--speculative-draft-model-path")
        self.assertEqual(args[path_idx + 1], _server._SITE.paths.model)
        fmt_idx = args.index("--speculative-draft-load-format")
        self.assertEqual(args[fmt_idx + 1], _server._PLATFORM.speculative.draft_load_format)

    def test_env_var_zero_disables(self) -> None:
        os.environ["VIBESYS_SPEC_DECODE"] = "0"
        self.assertFalse(_server.spec_decode_enabled())
        self.assertEqual(_server.speculative_decode_args(), [])

    def test_env_var_false_disables(self) -> None:
        os.environ["VIBESYS_SPEC_DECODE"] = "false"
        self.assertFalse(_server.spec_decode_enabled())

    def test_other_env_value_stays_enabled(self) -> None:
        os.environ["VIBESYS_SPEC_DECODE"] = "1"
        self.assertTrue(_server.spec_decode_enabled())

    def test_disabled_argv_omits_speculative_flags(self) -> None:
        os.environ["VIBESYS_SPEC_DECODE"] = "0"
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertNotIn("--speculative-algorithm", argv)
        self.assertNotIn("--speculative-draft-model-path", argv)

    def test_enabled_argv_places_speculative_flags_after_platform_extra_args_before_vibesys_extra(
        self,
    ) -> None:
        orig_extra = os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
        try:
            os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = "--sentinel-ad-hoc-flag"
            argv = _server.build_launch_argv(model_path="/does/not/matter")
            platform_extra = _server._PLATFORM.extra_args
            spec_idx = argv.index("--speculative-algorithm")
            self.assertGreater(spec_idx, argv.index(platform_extra[-1]))
            self.assertLess(spec_idx, argv.index("--sentinel-ad-hoc-flag"))
        finally:
            if orig_extra is None:
                os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
            else:
                os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = orig_extra


class SchedulerArgsTest(unittest.TestCase):
    """Covers scheduler_args()/overlap_schedule_enabled() against the
    default site+platform (example/mi300a-rocm700 via $VIBESYS_SITE), which
    ships a [scheduler] table -- see config/platforms/mi300a-rocm700.toml.
    """

    def setUp(self) -> None:
        self._orig = os.environ.pop("VIBESYS_OVERLAP_SCHEDULE", None)
        self.addCleanup(self._restore)
        self.assertIsNotNone(
            _server._PLATFORM.scheduler,
            "expected the default platform config to declare [scheduler]",
        )

    def _restore(self) -> None:
        if self._orig is None:
            os.environ.pop("VIBESYS_OVERLAP_SCHEDULE", None)
        else:
            os.environ["VIBESYS_OVERLAP_SCHEDULE"] = self._orig

    def test_disabled_by_default(self) -> None:
        self.assertFalse(_server.overlap_schedule_enabled())
        self.assertEqual(_server.scheduler_args(), ["--disable-overlap-schedule"])

    def test_env_var_one_reenables(self) -> None:
        os.environ["VIBESYS_OVERLAP_SCHEDULE"] = "1"
        self.assertTrue(_server.overlap_schedule_enabled())
        self.assertEqual(_server.scheduler_args(), [])

    def test_env_var_zero_keeps_disabled(self) -> None:
        os.environ["VIBESYS_OVERLAP_SCHEDULE"] = "0"
        self.assertFalse(_server.overlap_schedule_enabled())
        self.assertEqual(_server.scheduler_args(), ["--disable-overlap-schedule"])

    def test_env_var_false_keeps_disabled(self) -> None:
        os.environ["VIBESYS_OVERLAP_SCHEDULE"] = "false"
        self.assertFalse(_server.overlap_schedule_enabled())

    def test_env_var_no_keeps_disabled(self) -> None:
        os.environ["VIBESYS_OVERLAP_SCHEDULE"] = "no"
        self.assertFalse(_server.overlap_schedule_enabled())

    def test_default_argv_includes_disable_flag(self) -> None:
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertIn("--disable-overlap-schedule", argv)

    def test_reenabled_argv_omits_disable_flag(self) -> None:
        os.environ["VIBESYS_OVERLAP_SCHEDULE"] = "1"
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertNotIn("--disable-overlap-schedule", argv)

    def test_disable_flag_placed_after_platform_extra_args_before_speculative(
        self,
    ) -> None:
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        platform_extra = _server._PLATFORM.extra_args
        idx = argv.index("--disable-overlap-schedule")
        self.assertGreater(idx, argv.index(platform_extra[-1]))
        if _server._PLATFORM.speculative is not None and _server.spec_decode_enabled():
            self.assertLess(idx, argv.index("--speculative-algorithm"))


class PrefillCudaGraphArgsTest(unittest.TestCase):
    """Covers prefill_cuda_graph_args()/prefill_graph_enabled() against the
    default site+platform (example/mi300a-rocm700 via $VIBESYS_SITE), which
    ships a [prefill_cuda_graph] table -- see
    config/platforms/mi300a-rocm700.toml.
    """

    def setUp(self) -> None:
        self._orig = os.environ.pop("VIBESYS_PREFILL_GRAPH", None)
        self.addCleanup(self._restore)
        self.assertIsNotNone(
            _server._PLATFORM.prefill_cuda_graph,
            "expected the default platform config to declare [prefill_cuda_graph]",
        )

    def _restore(self) -> None:
        if self._orig is None:
            os.environ.pop("VIBESYS_PREFILL_GRAPH", None)
        else:
            os.environ["VIBESYS_PREFILL_GRAPH"] = self._orig

    def test_on_by_default(self) -> None:
        self.assertTrue(_server.prefill_graph_enabled())

    def test_default_args_lock_breakable_backend_with_13_buckets(self) -> None:
        args = _server.prefill_cuda_graph_args()
        self.assertEqual(args[0], "--cuda-graph-config")
        payload = json.loads(args[1])
        self.assertEqual(
            payload,
            {
                "prefill": {
                    "backend": _server._PLATFORM.prefill_cuda_graph.backend,
                    "bs": _server.GEMM_PAD_WARMUP_TOKEN_COUNTS,
                }
            },
        )
        self.assertEqual(payload["prefill"]["backend"], "breakable")
        self.assertEqual(len(payload["prefill"]["bs"]), 13)
        self.assertEqual(
            payload["prefill"]["bs"],
            [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
        )

    def test_json_round_trips_through_json_loads(self) -> None:
        args = _server.prefill_cuda_graph_args()
        # Round-tripping confirms the payload is valid JSON exactly as
        # sglang's own parse_cuda_graph_config_arg (server_args.py) will
        # parse it -- not merely a string that happens to look like JSON.
        reparsed = json.loads(json.dumps(json.loads(args[1])))
        self.assertEqual(reparsed, json.loads(args[1]))

    def test_default_argv_contains_cuda_graph_config(self) -> None:
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertIn("--cuda-graph-config", argv)
        idx = argv.index("--cuda-graph-config")
        payload = json.loads(argv[idx + 1])
        self.assertEqual(payload["prefill"]["backend"], "breakable")

    def test_env_var_zero_disables(self) -> None:
        os.environ["VIBESYS_PREFILL_GRAPH"] = "0"
        self.assertFalse(_server.prefill_graph_enabled())
        self.assertEqual(_server.prefill_cuda_graph_args(), [])

    def test_env_var_false_disables(self) -> None:
        os.environ["VIBESYS_PREFILL_GRAPH"] = "false"
        self.assertFalse(_server.prefill_graph_enabled())

    def test_env_var_no_disables(self) -> None:
        os.environ["VIBESYS_PREFILL_GRAPH"] = "no"
        self.assertFalse(_server.prefill_graph_enabled())

    def test_other_env_value_stays_enabled(self) -> None:
        os.environ["VIBESYS_PREFILL_GRAPH"] = "1"
        self.assertTrue(_server.prefill_graph_enabled())

    def test_env_var_zero_removes_flag_from_argv(self) -> None:
        os.environ["VIBESYS_PREFILL_GRAPH"] = "0"
        argv = _server.build_launch_argv(model_path="/does/not/matter")
        self.assertNotIn("--cuda-graph-config", argv)

    def test_no_prefill_cuda_graph_table_never_emits_flag(self) -> None:
        with mock.patch.object(
            _server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, prefill_cuda_graph=None)
        ):
            self.assertEqual(_server.prefill_cuda_graph_args(), [])

    def test_flag_placed_after_speculative_before_vibesys_extra(self) -> None:
        orig_extra = os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
        try:
            os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = "--sentinel-ad-hoc-flag"
            argv = _server.build_launch_argv(model_path="/does/not/matter")
            idx = argv.index("--cuda-graph-config")
            self.assertLess(idx, argv.index("--sentinel-ad-hoc-flag"))
            if _server._PLATFORM.speculative is not None and _server.spec_decode_enabled():
                self.assertGreater(idx, argv.index("--speculative-algorithm"))
        finally:
            if orig_extra is None:
                os.environ.pop("VIBESYS_EXTRA_SERVER_ARGS", None)
            else:
                os.environ["VIBESYS_EXTRA_SERVER_ARGS"] = orig_extra


class TunableOpEnvTest(unittest.TestCase):
    """Covers the ``[tunableop]``-table-driven env vars in
    ``build_launch_env()``, the per-device-ordinal file fan-out in
    ``stage_tunableop_files()``, and ``tunableop_enabled()``/
    ``VIBESYS_TUNABLEOP``, against the default site+platform
    (example/mi300a-rocm700 via $VIBESYS_SITE), which ships
    ``[tunableop] enabled = true`` -- see
    config/platforms/mi300a-rocm700.toml.

    Each test that stages files uses its own ``tempfile.TemporaryDirectory``
    as ``run_dir`` so nothing is left behind and runs cannot collide.
    """

    FAKE_TP = 4

    def setUp(self) -> None:
        self._orig = os.environ.pop("VIBESYS_TUNABLEOP", None)
        self.addCleanup(self._restore)
        self.assertIsNotNone(
            _server._PLATFORM.tunableop,
            "expected the default platform config to declare [tunableop]",
        )
        self.assertTrue(
            _server._PLATFORM.tunableop.enabled,
            "expected the default platform config's [tunableop] to be enabled",
        )
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "outdir"

    def _restore(self) -> None:
        if self._orig is None:
            os.environ.pop("VIBESYS_TUNABLEOP", None)
        else:
            os.environ["VIBESYS_TUNABLEOP"] = self._orig

    def test_on_by_default(self) -> None:
        self.assertTrue(_server.tunableop_enabled())

    def test_vars_present_when_enabled(self) -> None:
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("PYTORCH_TUNABLEOP_ENABLED"), "1")
        self.assertEqual(env.get("PYTORCH_TUNABLEOP_TUNING"), "0")
        self.assertIn("PYTORCH_TUNABLEOP_FILENAME", env)
        self.assertEqual(env.get("PYTORCH_TUNABLEOP_VERBOSE"), "1")

    def test_filename_carries_literal_percent_d(self) -> None:
        # PyTorch substitutes the device ordinal for a literal "%d" in this
        # value (torch/cuda/tunable.py, torch 2.9.0a0); the harness must
        # never resolve it itself.
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        filename = env["PYTORCH_TUNABLEOP_FILENAME"]
        self.assertTrue(Path(filename).name == "tunableop%d.csv", filename)

    def test_stages_one_read_only_copy_per_device_ordinal(self) -> None:
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        staged_dir = Path(env["PYTORCH_TUNABLEOP_FILENAME"]).parent
        committed_csv = Path(_server._PLATFORM.tunableop.results_file)
        committed_bytes = committed_csv.read_bytes()
        found = sorted(staged_dir.glob("tunableop*.csv"))
        self.assertEqual(len(found), self.FAKE_TP, found)
        for device in range(self.FAKE_TP):
            per_device = staged_dir / f"tunableop{device}.csv"
            self.assertIn(per_device, found)
            self.assertEqual(per_device.read_bytes(), committed_bytes)
            mode = oct(per_device.stat().st_mode)[-3:]
            self.assertEqual(mode, "444", f"{per_device} mode was {mode}, expected read-only 444")

    def test_staging_dir_lands_under_run_dir_when_given(self) -> None:
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        staged_dir = Path(env["PYTORCH_TUNABLEOP_FILENAME"]).parent
        self.assertEqual(staged_dir.parent, self.run_dir)

    def test_staging_dir_falls_back_to_tempdir_when_run_dir_none(self) -> None:
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=None)
        staged_dir = Path(env["PYTORCH_TUNABLEOP_FILENAME"]).parent
        self.assertTrue(staged_dir.is_dir())
        self.assertTrue(staged_dir.name.startswith("tunableop-"))

    def test_file_count_tracks_tp(self) -> None:
        env = _server.build_launch_env(tp=2, run_dir=self.run_dir)
        staged_dir = Path(env["PYTORCH_TUNABLEOP_FILENAME"]).parent
        self.assertEqual(len(list(staged_dir.glob("tunableop*.csv"))), 2)

    def test_rejects_committed_csv_containing_carriage_return(self) -> None:
        # Regression test for the dense-keys CRLF bug: PyTorch's TunableOp
        # validator is an exact string compare against its five Validator
        # header lines, so a CRLF-terminated line never matches even when
        # both sides print identically, silently discarding the whole
        # results table with only a warning, no boot failure. This must be
        # a loud, immediate error at staging time instead.
        crlf_csv = Path(self._tmp.name) / "crlf_tunableop.csv"
        crlf_csv.write_bytes(
            b"Validator,PT_VERSION,2.9.0\r\n"
            b"GemmTunableOp_BFloat16_TN,tn_512_256_4096_ld_4096_4096_512,Gemm_Rocblas_45,0.1\n"
        )
        patched = dataclasses.replace(_server._PLATFORM.tunableop, results_file=str(crlf_csv))
        with mock.patch.object(_server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=patched)):
            with self.assertRaisesRegex(ValueError, "carriage.return"):
                _server.stage_tunableop_files(tp=self.FAKE_TP, run_dir=self.run_dir)
        # Nothing should have been staged: the check runs before run_dir is
        # even created, so the failed call must not have created it.
        self.assertFalse(self.run_dir.exists())

    def test_accepts_committed_csv_with_plain_lf_line_endings(self) -> None:
        # Same content as the CRLF case above, minus the \r: must stage
        # cleanly, confirming the check is specifically about \r bytes and
        # not, say, line count or content shape.
        lf_csv = Path(self._tmp.name) / "lf_tunableop.csv"
        lf_csv.write_bytes(
            b"Validator,PT_VERSION,2.9.0\n"
            b"GemmTunableOp_BFloat16_TN,tn_512_256_4096_ld_4096_4096_512,Gemm_Rocblas_45,0.1\n"
        )
        patched = dataclasses.replace(_server._PLATFORM.tunableop, results_file=str(lf_csv))
        with mock.patch.object(_server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=patched)):
            filename = _server.stage_tunableop_files(tp=self.FAKE_TP, run_dir=self.run_dir)
        staged_dir = Path(filename).parent
        found = sorted(staged_dir.glob("tunableop*.csv"))
        self.assertEqual(len(found), self.FAKE_TP, found)
        for f in found:
            self.assertEqual(f.read_bytes(), lf_csv.read_bytes())

    def test_env_var_zero_disables(self) -> None:
        os.environ["VIBESYS_TUNABLEOP"] = "0"
        self.assertFalse(_server.tunableop_enabled())
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_ENABLED", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_TUNING", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_FILENAME", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)
        # Disabled means no staging at all, not just the env vars unset.
        self.assertFalse(self.run_dir.exists())

    def test_env_var_false_disables(self) -> None:
        os.environ["VIBESYS_TUNABLEOP"] = "false"
        self.assertFalse(_server.tunableop_enabled())

    def test_env_var_no_disables(self) -> None:
        os.environ["VIBESYS_TUNABLEOP"] = "no"
        self.assertFalse(_server.tunableop_enabled())

    def test_other_env_value_stays_enabled(self) -> None:
        os.environ["VIBESYS_TUNABLEOP"] = "1"
        self.assertTrue(_server.tunableop_enabled())

    def test_tuning_is_always_off_even_when_enabled(self) -> None:
        # The harness must never turn tuning on: the CSV is read-only input.
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("PYTORCH_TUNABLEOP_TUNING"), "0")

    def test_verbose_set_when_enabled(self) -> None:
        # PYTORCH_TUNABLEOP_VERBOSE=1 is what makes PyTorch log the
        # validator check and the "reading tuning results from ..." line
        # per rank; without it, a stale/rejected table is silently
        # unobservable in the server log (see README.md's "Detecting a
        # stale table" section).
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("PYTORCH_TUNABLEOP_VERBOSE"), "1")

    def test_verbose_absent_when_env_var_disables(self) -> None:
        os.environ["VIBESYS_TUNABLEOP"] = "0"
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)

    def test_verbose_absent_when_platform_table_disabled(self) -> None:
        disabled = dataclasses.replace(_server._PLATFORM.tunableop, enabled=False)
        with mock.patch.object(
            _server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=disabled)
        ):
            env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)

    def test_verbose_absent_when_no_tunableop_table(self) -> None:
        with mock.patch.object(
            _server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=None)
        ):
            env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)

    def test_disabled_platform_table_never_sets_vars(self) -> None:
        disabled = dataclasses.replace(_server._PLATFORM.tunableop, enabled=False)
        with mock.patch.object(
            _server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=disabled)
        ):
            env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_ENABLED", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_TUNING", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_FILENAME", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)
        self.assertFalse(self.run_dir.exists())

    def test_no_tunableop_table_never_sets_vars(self) -> None:
        with mock.patch.object(
            _server, "_PLATFORM", dataclasses.replace(_server._PLATFORM, tunableop=None)
        ):
            env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertNotIn("PYTORCH_TUNABLEOP_ENABLED", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_TUNING", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_FILENAME", env)
        self.assertNotIn("PYTORCH_TUNABLEOP_VERBOSE", env)
        self.assertFalse(self.run_dir.exists())


class TokenizerFastPathEnvDefaultsTest(unittest.TestCase):
    """Covers the two ``[env_defaults]``-table tokenizer fast-path switches
    on the default site+platform (example/mi300a-rocm700 via
    ``$VIBESYS_SITE``): ``SGLANG_TEXT_ONLY_SEND_IDS`` (skip the
    decode-then-re-tokenize round trip for text-only chat turns) and
    ``SGLANG_INCREMENTAL_TOKENIZE`` (suffix-only tokenization of a growing
    multi-turn prompt). Both are on by default via
    ``config/platforms/mi300a-rocm700.toml``'s ``[env_defaults]`` table,
    which ``build_launch_env()`` applies with ``os.environ.setdefault()``,
    so an explicit value already present in the caller's environment must
    win over the platform default (the same contract as
    ``SGLANG_SKINNY_GEMM``/``SGLANG_MXFP4_MOE_HIP`` above them in that
    table).
    """

    FAKE_TP = 4
    _VARS = ("SGLANG_TEXT_ONLY_SEND_IDS", "SGLANG_INCREMENTAL_TOKENIZE")

    def setUp(self) -> None:
        self._orig = {name: os.environ.pop(name, None) for name in self._VARS}
        self.addCleanup(self._restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "outdir"

    def _restore(self) -> None:
        for name, value in self._orig.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_both_default_to_one(self) -> None:
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("SGLANG_TEXT_ONLY_SEND_IDS"), "1")
        self.assertEqual(env.get("SGLANG_INCREMENTAL_TOKENIZE"), "1")

    def test_explicit_zero_in_caller_env_wins_for_send_ids(self) -> None:
        os.environ["SGLANG_TEXT_ONLY_SEND_IDS"] = "0"
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("SGLANG_TEXT_ONLY_SEND_IDS"), "0")
        # The other switch is independent and still defaults on.
        self.assertEqual(env.get("SGLANG_INCREMENTAL_TOKENIZE"), "1")

    def test_explicit_zero_in_caller_env_wins_for_incremental_tokenize(self) -> None:
        os.environ["SGLANG_INCREMENTAL_TOKENIZE"] = "0"
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("SGLANG_INCREMENTAL_TOKENIZE"), "0")
        self.assertEqual(env.get("SGLANG_TEXT_ONLY_SEND_IDS"), "1")

    def test_explicit_zero_in_caller_env_wins_for_both(self) -> None:
        os.environ["SGLANG_TEXT_ONLY_SEND_IDS"] = "0"
        os.environ["SGLANG_INCREMENTAL_TOKENIZE"] = "0"
        env = _server.build_launch_env(tp=self.FAKE_TP, run_dir=self.run_dir)
        self.assertEqual(env.get("SGLANG_TEXT_ONLY_SEND_IDS"), "0")
        self.assertEqual(env.get("SGLANG_INCREMENTAL_TOKENIZE"), "0")


if __name__ == "__main__":
    unittest.main()
