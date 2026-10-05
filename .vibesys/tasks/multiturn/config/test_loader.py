"""Unit tests for ``config/loader.py``'s platform config parsing.

Covers ``load_platform``'s handling of the optional ``extra_args`` key:
parsing a valid list of strings, rejecting a non-list or a list with a
non-string element, and the default of an empty list when the key is
absent. Also covers the optional ``[speculative]`` table: absent-by-default,
parsing, key validation, and the mi300a-rocm700 committed default. Also
covers the optional ``[scheduler]`` table with the same coverage. Also
covers the optional ``[tunableop]`` table: absent-by-default, parsing,
``results_file`` resolution relative to ``PLATFORMS_DIR``, the
enabled-but-missing-file error, and the mi300a-rocm700 committed default.
Also covers the optional ``[prefill_cuda_graph]`` table: absent-by-default,
parsing, key validation, and the mi300a-rocm700 committed default. Also
covers ``load_site``'s optional ``paths.draft_model`` key:
absent-by-default, parsing, and the example committed default. No SGLang
install or cluster access required. Run with:

    python3 -m pytest .vibesys/tasks/multiturn/config/test_loader.py

or plain ``unittest``:

    python3 .vibesys/tasks/multiturn/config/test_loader.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import loader

# A minimal platform TOML covering every required top-level key, split
# around the `[hf_loader]` table header so tests can splice a top-level
# `extra_args = [...]` line in between without landing inside that table.
_BASE_PLATFORM_TOML_HEAD = """
attention_backend = "aiter"
page_size = 16
mem_fraction_static = "0.72"
tp = 4
max_total_tokens = 787936
"""
_BASE_PLATFORM_TOML_TAIL = """
[hf_loader]
disable_mmap = true
extra_config = '{"num_threads": 2}'
"""


class LoadPlatformExtraArgsTest(unittest.TestCase):
    def _load_with(self, extra_toml: str) -> loader.PlatformConfig:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            content = _BASE_PLATFORM_TOML_HEAD + extra_toml + _BASE_PLATFORM_TOML_TAIL
            (platforms_dir / "fake.toml").write_text(content)
            with mock.patch.object(loader, "PLATFORMS_DIR", platforms_dir):
                return loader.load_platform("fake")

    def test_absent_extra_args_defaults_to_empty_list(self) -> None:
        platform = self._load_with("")
        self.assertEqual(platform.extra_args, [])

    def test_parses_list_of_strings(self) -> None:
        platform = self._load_with(
            'extra_args = ["--enable-mixed-chunk", "--chunked-prefill-size", "1024"]\n'
        )
        self.assertEqual(
            platform.extra_args,
            ["--enable-mixed-chunk", "--chunked-prefill-size", "1024"],
        )

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load_with('extra_args = "--enable-mixed-chunk"\n')
        self.assertIn("extra_args", str(ctx.exception))

    def test_rejects_list_with_non_string_element(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load_with('extra_args = ["--chunked-prefill-size", 1024]\n')
        self.assertIn("extra_args", str(ctx.exception))

    def test_rejects_unknown_top_level_key_still_works_alongside_extra_args(self) -> None:
        # Sanity check that extra_args being a recognized optional key
        # doesn't loosen unknown-key rejection for other keys.
        with self.assertRaises(ValueError) as ctx:
            self._load_with('extra_args = []\nnot_a_real_key = 1\n')
        self.assertIn("unknown key", str(ctx.exception))

    def test_mi300a_rocm700_sets_mixed_chunk_extra_args(self) -> None:
        # The committed default: mixed chunked prefill on by default (see
        # the comment on extra_args in that file for the measured effect).
        platform = loader.load_platform("mi300a-rocm700")
        self.assertEqual(
            platform.extra_args,
            ["--enable-mixed-chunk", "--chunked-prefill-size", "1024"],
        )


class LoadPlatformSpeculativeTest(unittest.TestCase):
    """Covers the optional ``[speculative]`` table: absent-by-default,
    parsing, key validation, and the mi300a-rocm700 committed default."""

    def _load_with(self, extra_toml: str) -> loader.PlatformConfig:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            content = _BASE_PLATFORM_TOML_HEAD + extra_toml + _BASE_PLATFORM_TOML_TAIL
            (platforms_dir / "fake.toml").write_text(content)
            with mock.patch.object(loader, "PLATFORMS_DIR", platforms_dir):
                return loader.load_platform("fake")

    _SPEC_TABLE = """
[speculative]
algorithm = "NEXTN"
num_steps = 3
eagle_topk = 1
num_draft_tokens = 4
enable_linear_replayssm_spec = true
draft_load_format = "auto"
"""

    def test_absent_speculative_defaults_to_none(self) -> None:
        platform = self._load_with("")
        self.assertIsNone(platform.speculative)

    def test_parses_speculative_table(self) -> None:
        platform = self._load_with(self._SPEC_TABLE)
        self.assertEqual(
            platform.speculative,
            loader.SpeculativeConfig(
                algorithm="NEXTN",
                num_steps=3,
                eagle_topk=1,
                num_draft_tokens=4,
                enable_linear_replayssm_spec=True,
                draft_load_format="auto",
            ),
        )

    def test_rejects_speculative_missing_key(self) -> None:
        incomplete = self._SPEC_TABLE.replace('draft_load_format = "auto"\n', "")
        with self.assertRaises(ValueError) as ctx:
            self._load_with(incomplete)
        self.assertIn("missing required key", str(ctx.exception))

    def test_rejects_speculative_unknown_key(self) -> None:
        extra_key = self._SPEC_TABLE + "not_a_real_key = 1\n"
        with self.assertRaises(ValueError) as ctx:
            self._load_with(extra_key)
        self.assertIn("unknown key", str(ctx.exception))

    def test_mi300a_rocm700_sets_nextn_k3_by_default(self) -> None:
        # The committed default: NEXTN speculative decoding, k=3, on by
        # default (see the comment on [speculative] in that file for the
        # acceptance/probe job numbers behind this choice).
        platform = loader.load_platform("mi300a-rocm700")
        self.assertEqual(
            platform.speculative,
            loader.SpeculativeConfig(
                algorithm="NEXTN",
                num_steps=3,
                eagle_topk=1,
                num_draft_tokens=4,
                enable_linear_replayssm_spec=True,
                draft_load_format="auto",
            ),
        )


class LoadPlatformSchedulerTest(unittest.TestCase):
    """Covers the optional ``[scheduler]`` table: absent-by-default,
    parsing, key validation, and the mi300a-rocm700 committed default."""

    def _load_with(self, extra_toml: str) -> loader.PlatformConfig:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            content = _BASE_PLATFORM_TOML_HEAD + extra_toml + _BASE_PLATFORM_TOML_TAIL
            (platforms_dir / "fake.toml").write_text(content)
            with mock.patch.object(loader, "PLATFORMS_DIR", platforms_dir):
                return loader.load_platform("fake")

    _SCHEDULER_TABLE = """
[scheduler]
disable_overlap_schedule = true
"""

    def test_absent_scheduler_defaults_to_none(self) -> None:
        platform = self._load_with("")
        self.assertIsNone(platform.scheduler)

    def test_parses_scheduler_table(self) -> None:
        platform = self._load_with(self._SCHEDULER_TABLE)
        self.assertEqual(
            platform.scheduler,
            loader.SchedulerConfig(disable_overlap_schedule=True),
        )

    def test_rejects_scheduler_missing_key(self) -> None:
        incomplete = self._SCHEDULER_TABLE.replace(
            "disable_overlap_schedule = true\n", ""
        )
        with self.assertRaises(ValueError) as ctx:
            self._load_with(incomplete)
        self.assertIn("missing required key", str(ctx.exception))

    def test_rejects_scheduler_unknown_key(self) -> None:
        extra_key = self._SCHEDULER_TABLE + "not_a_real_key = 1\n"
        with self.assertRaises(ValueError) as ctx:
            self._load_with(extra_key)
        self.assertIn("unknown key", str(ctx.exception))

    def test_mi300a_rocm700_disables_overlap_schedule_by_default(self) -> None:
        # The committed default: the overlap scheduler off by default (see
        # the comment on [scheduler] in that file for the probe job number
        # behind this choice).
        platform = loader.load_platform("mi300a-rocm700")
        self.assertEqual(
            platform.scheduler,
            loader.SchedulerConfig(disable_overlap_schedule=True),
        )


class LoadPlatformTunableOpTest(unittest.TestCase):
    """Covers the optional ``[tunableop]`` table: absent-by-default,
    parsing, ``results_file`` resolution relative to ``PLATFORMS_DIR``, the
    enabled-but-missing-file error, and the mi300a-rocm700 committed
    default."""

    def _load_with(self, extra_toml: str, *, platforms_dir: Path) -> loader.PlatformConfig:
        content = _BASE_PLATFORM_TOML_HEAD + extra_toml + _BASE_PLATFORM_TOML_TAIL
        (platforms_dir / "fake.toml").write_text(content)
        with mock.patch.object(loader, "PLATFORMS_DIR", platforms_dir):
            return loader.load_platform("fake")

    def test_absent_tunableop_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platform = self._load_with("", platforms_dir=Path(tmp))
        self.assertIsNone(platform.tunableop)

    def test_enabled_with_file_present_resolves_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            (platforms_dir / "tunableop").mkdir()
            results_path = platforms_dir / "tunableop" / "fake.csv"
            results_path.write_text("Validator,PT_VERSION,2.9.0\n")
            platform = self._load_with(
                '[tunableop]\nenabled = true\nresults_file = "tunableop/fake.csv"\n',
                platforms_dir=platforms_dir,
            )
        self.assertIsNotNone(platform.tunableop)
        self.assertTrue(platform.tunableop.enabled)
        self.assertEqual(platform.tunableop.results_file, str(results_path))

    def test_enabled_with_file_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            with self.assertRaises(ValueError) as ctx:
                self._load_with(
                    '[tunableop]\nenabled = true\nresults_file = "tunableop/does-not-exist.csv"\n',
                    platforms_dir=platforms_dir,
                )
        self.assertIn("does-not-exist.csv", str(ctx.exception))

    def test_disabled_does_not_require_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            platform = self._load_with(
                '[tunableop]\nenabled = false\nresults_file = "tunableop/does-not-exist.csv"\n',
                platforms_dir=platforms_dir,
            )
        self.assertIsNotNone(platform.tunableop)
        self.assertFalse(platform.tunableop.enabled)

    def test_rejects_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self._load_with("[tunableop]\nenabled = true\n", platforms_dir=Path(tmp))
        self.assertIn("missing required key", str(ctx.exception))

    def test_rejects_unknown_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            (platforms_dir / "tunableop").mkdir()
            (platforms_dir / "tunableop" / "fake.csv").write_text("x\n")
            with self.assertRaises(ValueError) as ctx:
                self._load_with(
                    '[tunableop]\nenabled = true\nresults_file = "tunableop/fake.csv"\n'
                    "not_a_real_key = 1\n",
                    platforms_dir=platforms_dir,
                )
        self.assertIn("unknown key", str(ctx.exception))

    def test_mi300a_rocm700_enables_tunableop_with_the_committed_csv(self) -> None:
        # The committed default: TunableOp-tuned dense GEMM tiles on by
        # default, pointing at the committed CSV under
        # config/platforms/tunableop/ (see the comment on [tunableop] in
        # that file for the measured speedup and job number).
        platform = loader.load_platform("mi300a-rocm700")
        self.assertIsNotNone(platform.tunableop)
        self.assertTrue(platform.tunableop.enabled)
        expected_path = loader.PLATFORMS_DIR / "tunableop" / "mi300a-rocm700.csv"
        self.assertEqual(platform.tunableop.results_file, str(expected_path))
        self.assertTrue(Path(platform.tunableop.results_file).is_file())


class LoadPlatformPrefillCudaGraphTest(unittest.TestCase):
    """Covers the optional ``[prefill_cuda_graph]`` table: absent-by-default,
    parsing, key validation, and the mi300a-rocm700 committed default."""

    def _load_with(self, extra_toml: str) -> loader.PlatformConfig:
        with tempfile.TemporaryDirectory() as tmp:
            platforms_dir = Path(tmp)
            content = _BASE_PLATFORM_TOML_HEAD + extra_toml + _BASE_PLATFORM_TOML_TAIL
            (platforms_dir / "fake.toml").write_text(content)
            with mock.patch.object(loader, "PLATFORMS_DIR", platforms_dir):
                return loader.load_platform("fake")

    _PREFILL_CUDA_GRAPH_TABLE = """
[prefill_cuda_graph]
backend = "breakable"
"""

    def test_absent_prefill_cuda_graph_defaults_to_none(self) -> None:
        platform = self._load_with("")
        self.assertIsNone(platform.prefill_cuda_graph)

    def test_parses_prefill_cuda_graph_table(self) -> None:
        platform = self._load_with(self._PREFILL_CUDA_GRAPH_TABLE)
        self.assertEqual(
            platform.prefill_cuda_graph,
            loader.PrefillCudaGraphConfig(backend="breakable"),
        )

    def test_rejects_prefill_cuda_graph_missing_key(self) -> None:
        incomplete = self._PREFILL_CUDA_GRAPH_TABLE.replace('backend = "breakable"\n', "")
        with self.assertRaises(ValueError) as ctx:
            self._load_with(incomplete)
        self.assertIn("missing required key", str(ctx.exception))

    def test_rejects_prefill_cuda_graph_unknown_key(self) -> None:
        extra_key = self._PREFILL_CUDA_GRAPH_TABLE + "not_a_real_key = 1\n"
        with self.assertRaises(ValueError) as ctx:
            self._load_with(extra_key)
        self.assertIn("unknown key", str(ctx.exception))

    def test_mi300a_rocm700_locks_prefill_to_breakable_by_default(self) -> None:
        # The committed default: the prefill CUDA graph locked to the
        # "breakable" backend (see the comment on [prefill_cuda_graph] in
        # that file for the acceptance job numbers behind this choice).
        platform = loader.load_platform("mi300a-rocm700")
        self.assertEqual(
            platform.prefill_cuda_graph,
            loader.PrefillCudaGraphConfig(backend="breakable"),
        )


class LoadSiteDraftModelTest(unittest.TestCase):
    """Covers the optional ``paths.draft_model`` key: absent-by-default,
    parsing, and the example committed default (see
    config/sites/example.toml's comment on that field)."""

    _BASE_SITE_TOML_HEAD = """
platform = "mi300a-rocm700"
startup_timeout_s = 900

[paths]
model = "/fake/model"
sharded_artifact = "/fake/sharded"
sharded_artifact_striped = "/fake/sharded-striped"
aiter_jit_dir = "/fake/aiter-jit"
hip_ext_dir = "/fake/hip-ext"
tmpfs_root = "/dev/shm"
log_dir = "/fake/logs"
checkout = "/fake/checkout"
"""
    _BASE_SITE_TOML_TAIL = """
[slurm]
account = "fake-account"
partition = "fake-partition"
edf = "/fake/runtime.toml"

[lustre]
stripe_count = 8
stripe_size = "4M"
"""

    def _load_with(self, extra_paths_toml: str) -> loader.SiteConfig:
        with tempfile.TemporaryDirectory() as tmp:
            sites_dir = Path(tmp)
            content = self._BASE_SITE_TOML_HEAD + extra_paths_toml + self._BASE_SITE_TOML_TAIL
            (sites_dir / "fake.toml").write_text(content)
            with mock.patch.object(loader, "SITES_DIR", sites_dir):
                return loader.load_site("fake")

    def test_absent_draft_model_defaults_to_none(self) -> None:
        site = self._load_with("")
        self.assertIsNone(site.paths.draft_model)

    def test_parses_draft_model(self) -> None:
        site = self._load_with('draft_model = "/fake/draft-model"\n')
        self.assertEqual(site.paths.draft_model, "/fake/draft-model")

    def test_example_sets_draft_model_to_the_sharded_artifact(self) -> None:
        # The committed default: example points draft_model at the
        # draft-only sharded_state artifact produced by
        # manual/draft-shard/save_draft_shard_v5.py (fork commit
        # 1443e6bfdc, job 633762); see the comment on paths.draft_model in
        # that file for the validated boot numbers (job 633763).
        site = loader.load_site("example")
        self.assertEqual(
            site.paths.draft_model,
            "/path/to/models/Qwen3.5-397B-A17B-MXFP4-mtp-sharded-tp4",
        )


if __name__ == "__main__":
    unittest.main()
