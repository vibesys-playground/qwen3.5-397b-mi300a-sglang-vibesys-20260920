"""CPU-only unit tests for ``sglang.srt.layers.hip_extension``'s generic HIP
JIT-extension loader.

No GPU, ROCm, or real HIP toolchain is required: ``load_hip_extension`` only
touches ``torch.utils.cpp_extension.load`` inside the function body, which
these tests stub out with a fake ``load`` rather than building anything.
"""

import os
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@unittest.skipUnless(torch is not None, "requires torch")
class TestLoadHipExtension(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers import hip_extension

        # Each test gets a clean module-level cache so tests don't leak
        # cached extensions into each other via a shared name.
        self._module = hip_extension
        self._orig_cache = dict(hip_extension._extensions)
        hip_extension._extensions.clear()

    def tearDown(self):
        self._module._extensions.clear()
        self._module._extensions.update(self._orig_cache)

    def test_builds_and_caches_by_name(self):
        from sglang.srt.layers import hip_extension

        fake_module = object()
        fake_load = mock.Mock(return_value=fake_module)
        with mock.patch("torch.utils.cpp_extension.load", fake_load):
            first = hip_extension.load_hip_extension(
                "sglang_test_ext", sources=["a.cu"]
            )
            second = hip_extension.load_hip_extension(
                "sglang_test_ext", sources=["a.cu"]
            )

        self.assertIs(first, fake_module)
        self.assertIs(second, fake_module)
        # Cached: the second call must not re-invoke the (expensive) build.
        fake_load.assert_called_once()

    def test_distinct_names_build_independently(self):
        from sglang.srt.layers import hip_extension

        fake_load = mock.Mock(side_effect=[object(), object()])
        with mock.patch("torch.utils.cpp_extension.load", fake_load):
            first = hip_extension.load_hip_extension("ext_one", sources=["a.cu"])
            second = hip_extension.load_hip_extension("ext_two", sources=["b.cu"])

        self.assertIsNot(first, second)
        self.assertEqual(fake_load.call_count, 2)

    def test_build_directory_defaults_to_none_without_env(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers import hip_extension

        fake_load = mock.Mock(return_value=object())
        with mock.patch.object(envs.SGLANG_HIP_EXT_DIR, "get", return_value=None):
            with mock.patch("torch.utils.cpp_extension.load", fake_load):
                hip_extension.load_hip_extension("ext_default_dir", sources=["a.cu"])

        self.assertIsNone(fake_load.call_args.kwargs["build_directory"])
        # No env dir: the source path is forwarded as-is, never touched or
        # copied anywhere (the content-keyed path is opt-in).
        self.assertEqual(fake_load.call_args.kwargs["sources"], ["a.cu"])

    def test_forwards_sources_and_flags(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers import hip_extension

        fake_load = mock.Mock(return_value=object())
        with mock.patch.object(envs.SGLANG_HIP_EXT_DIR, "get", return_value=None):
            with mock.patch("torch.utils.cpp_extension.load", fake_load):
                hip_extension.load_hip_extension(
                    "ext_flags",
                    sources=["a.cu", "b.cu"],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
                )

        _, kwargs = fake_load.call_args
        self.assertEqual(kwargs["name"], "ext_flags")
        self.assertEqual(kwargs["sources"], ["a.cu", "b.cu"])
        self.assertEqual(kwargs["extra_cflags"], ["-O3"])
        self.assertEqual(kwargs["extra_cuda_cflags"], ["-O3", "--offload-arch=gfx942"])


@unittest.skipUnless(torch is not None, "requires torch")
class TestContentKeyedBuildDirectory(unittest.TestCase):
    """Covers the SGLANG_HIP_EXT_DIR content-keyed path: the fix for a
    harness that stages every launch into a fresh tmpfs directory, which
    otherwise defeats cpp_extension.load's path/mtime-based staleness check
    and forces a full rebuild on every boot even with a warm cache.
    """

    def setUp(self):
        from sglang.srt.layers import hip_extension

        self._module = hip_extension
        self._orig_cache = dict(hip_extension._extensions)
        hip_extension._extensions.clear()

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = os.path.join(self._tmp.name, "hip-ext-cache")

        # Two separate "launches" get their own staging directory, mirroring
        # stage_workspace.sh giving each boot a freshly named tmpfs path.
        self.launch1_dir = os.path.join(self._tmp.name, "launch1")
        self.launch2_dir = os.path.join(self._tmp.name, "launch2")
        os.makedirs(self.launch1_dir)
        os.makedirs(self.launch2_dir)
        self.source_content = b"// same kernel source\n"
        self.src1 = os.path.join(self.launch1_dir, "kernel.cu")
        self.src2 = os.path.join(self.launch2_dir, "kernel.cu")
        with open(self.src1, "wb") as f:
            f.write(self.source_content)
        with open(self.src2, "wb") as f:
            f.write(self.source_content)

    def tearDown(self):
        self._module._extensions.clear()
        self._module._extensions.update(self._orig_cache)

    def _load(self, name, source):
        from sglang.srt.environ import envs
        from sglang.srt.layers import hip_extension

        fake_load = mock.Mock(return_value=object())
        with mock.patch.object(
            envs.SGLANG_HIP_EXT_DIR, "get", return_value=self.cache_dir
        ):
            with mock.patch("torch.utils.cpp_extension.load", fake_load):
                # Simulate a fresh process per "launch": load_hip_extension
                # only skips torch.utils.cpp_extension.load for a name it
                # has already cached in *this* process.
                hip_extension._extensions.clear()
                hip_extension.load_hip_extension(name, sources=[source])
        return fake_load

    def test_build_directory_is_hashed_and_under_env_dir(self):
        fake_load = self._load("sglang_test_ext", self.src1)

        build_directory = fake_load.call_args.kwargs["build_directory"]
        self.assertTrue(build_directory.startswith(self.cache_dir + os.sep))
        subdir = os.path.basename(build_directory)
        self.assertTrue(subdir.startswith("sglang_test_ext-"))
        # <name>-<hash12>: a 12 hex-char suffix after the name and dash.
        suffix = subdir[len("sglang_test_ext-") :]
        self.assertEqual(len(suffix), 12)
        int(suffix, 16)  # raises ValueError if not hex

    def test_same_content_different_launch_paths_hash_identically(self):
        """The whole point of the fix: two launches with the same source
        *content* staged at two different *paths* must resolve to the same
        build directory, so a warm cache is recognized across launches."""
        fake_load_1 = self._load("sglang_test_ext", self.src1)
        fake_load_2 = self._load("sglang_test_ext", self.src2)

        dir1 = fake_load_1.call_args.kwargs["build_directory"]
        dir2 = fake_load_2.call_args.kwargs["build_directory"]
        self.assertEqual(dir1, dir2)

    def test_different_content_hashes_differently(self):
        other_src = os.path.join(self.launch1_dir, "other.cu")
        with open(other_src, "wb") as f:
            f.write(b"// different kernel source\n")

        fake_load_1 = self._load("sglang_test_ext", self.src1)
        fake_load_2 = self._load("sglang_test_ext", other_src)

        dir1 = fake_load_1.call_args.kwargs["build_directory"]
        dir2 = fake_load_2.call_args.kwargs["build_directory"]
        self.assertNotEqual(dir1, dir2)

    def test_source_is_copied_into_build_directory_once(self):
        fake_load = self._load("sglang_test_ext", self.src1)

        build_directory = fake_load.call_args.kwargs["build_directory"]
        copied_sources = fake_load.call_args.kwargs["sources"]
        self.assertEqual(len(copied_sources), 1)
        copied = copied_sources[0]
        self.assertEqual(os.path.dirname(copied), build_directory)
        self.assertEqual(os.path.basename(copied), "kernel.cu")
        with open(copied, "rb") as f:
            self.assertEqual(f.read(), self.source_content)

    def test_second_launch_reuses_copy_without_rewriting_it(self):
        """A warm-cache boot must not rewrite the already-copied source: no
        rewrite means no mtime bump, which is what lets ninja treat the
        already-built object file as up to date and skip hipcc entirely."""
        fake_load_1 = self._load("sglang_test_ext", self.src1)
        copied_path_1 = fake_load_1.call_args.kwargs["sources"][0]
        mtime_after_first = os.stat(copied_path_1).st_mtime

        # Second launch, different staged source path, identical content.
        fake_load_2 = self._load("sglang_test_ext", self.src2)
        copied_path_2 = fake_load_2.call_args.kwargs["sources"][0]
        mtime_after_second = os.stat(copied_path_2).st_mtime

        self.assertEqual(copied_path_1, copied_path_2)
        self.assertEqual(mtime_after_first, mtime_after_second)

    def test_changed_content_overwrites_the_copy(self):
        from sglang.srt.layers import hip_extension

        fake_load_1 = self._load("sglang_test_ext", self.src1)
        build_directory = fake_load_1.call_args.kwargs["build_directory"]
        stale_copy = fake_load_1.call_args.kwargs["sources"][0]

        # Corrupt the copy in place (e.g. a partial/stale write from a
        # previous run) and confirm the copy step detects the content
        # mismatch against the true source and repairs it.
        with open(stale_copy, "wb") as f:
            f.write(b"// stale, wrong content\n")
        os.utime(stale_copy, (0, 0))

        hip_extension._copy_source_if_stale(self.src1, build_directory)
        with open(stale_copy, "rb") as f:
            self.assertEqual(f.read(), self.source_content)

    def test_copy_uses_temp_name_and_replace_for_concurrent_safety(self):
        """Guards the concurrent-ranks case (several TP processes loading
        the same extension at once): the copy must never be visible in a
        partially written state, which os.replace from a temp file (rather
        than writing straight to the destination path) guarantees."""
        from sglang.srt.layers import hip_extension

        os.makedirs(self.cache_dir, exist_ok=True)
        real_replace = os.replace
        seen_temp_names = []

        def spying_replace(src, dst):
            seen_temp_names.append(os.path.basename(src))
            return real_replace(src, dst)

        with mock.patch("os.replace", side_effect=spying_replace):
            dest = hip_extension._copy_source_if_stale(self.src1, self.cache_dir)

        self.assertEqual(os.path.basename(dest), "kernel.cu")
        self.assertEqual(len(seen_temp_names), 1)
        self.assertNotEqual(seen_temp_names[0], "kernel.cu")
        self.assertTrue(seen_temp_names[0].startswith(".kernel.cu."))

    def test_flags_change_the_hash(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers import hip_extension

        def load_with(extra_cflags):
            fake_load = mock.Mock(return_value=object())
            with mock.patch.object(
                envs.SGLANG_HIP_EXT_DIR, "get", return_value=self.cache_dir
            ):
                with mock.patch("torch.utils.cpp_extension.load", fake_load):
                    hip_extension._extensions.clear()
                    hip_extension.load_hip_extension(
                        "sglang_test_ext",
                        sources=[self.src1],
                        extra_cflags=extra_cflags,
                    )
            return fake_load.call_args.kwargs["build_directory"]

        dir_no_flags = load_with(None)
        dir_with_flags = load_with(["-O3"])
        self.assertNotEqual(dir_no_flags, dir_with_flags)


if __name__ == "__main__":
    unittest.main()
