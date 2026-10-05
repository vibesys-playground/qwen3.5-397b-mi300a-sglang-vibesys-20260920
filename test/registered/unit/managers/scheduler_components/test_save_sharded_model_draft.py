"""
Regression test for the draft-worker branch of
`SchedulerWeightUpdaterManager.save_sharded_model`.

Bug 1 (fixed by uw-syfi/sglang PR #55, first attempt): `save_sharded_model`
only ever saved the target model's sharded_state artifact, never the draft
worker's, even when a draft worker was configured (unlike `save_remote_model`,
which already branches on `self.draft_worker`).

Bug 2 (found running PR #55's fix against real Qwen3.5 NEXTN/MTP boots, job
633711): the added draft branch reached for `self.draft_worker.model_runner`
directly. `self.draft_worker` at this layer is the *scheduler-level* spec
worker (`EAGLEWorkerV2` / `FrozenKVMTPWorkerV2`), which has no `.model_runner`
of its own -- only the inner draft worker it wraps
(`self.draft_worker._draft_worker`, an `EagleDraftWorker` /
`FrozenKVMTPDraftWorker`) does, via its own `.draft_runner` /
`.draft_model_runner`. The original version of this test's `_fake_worker()`
helper gave the *outer* mock a `.model_runner` attribute directly, which
does not mirror any real worker class and so did not catch the bug: the
mock made the buggy line succeed instead of raising `AttributeError`, so the
draft went unsaved with no error in either the mock test or the real job
(the RPC failure was swallowed between the scheduler subprocess and the
`Engine.save_sharded_model` caller). This test's fakes now mirror the real
nesting (`outer._draft_worker.draft_runner`, matching
`_get_draft_model_runner`'s own lookup), and a separate test drives the real
`EAGLEWorkerV2` / `FrozenKVMTPWorkerV2` classes to confirm the attribute
structure assumption holds outside of any mock.

This test drives `SchedulerWeightUpdaterManager.save_sharded_model` directly
with fake `tp_worker`/`draft_worker` stand-ins (no GPU, no real model) to
check that:
  - with no draft worker, only the target is saved (unchanged behavior).
  - with a draft worker and a `draft_path`, both target and draft are saved,
    each to its own path, through the real nested attribute chain.
  - with a draft worker and no `draft_path`, the call raises rather than
    silently skipping the draft save.
  - with `skip_target=True`, only the draft is saved (the target branch is
    skipped entirely -- this is what lets a rerun write only the draft
    artifact without re-touching the target's).
  - the real `EAGLEWorkerV2` / `FrozenKVMTPWorkerV2` classes expose no
    `.model_runner` of their own (so the old buggy direct-access code would
    fail against them) and do expose the `_draft_worker.draft_runner` chain
    `_get_draft_model_runner` reads.

Bug 3 (found running PR #55's fix -- commit 332634d2c6, unchanged -- against
real Qwen3.5 NEXTN/MTP boots a second time, job 633733): `save_draft_shard_v4.py`
called `llm.save_sharded_model(pattern=None, max_size=..., draft_path=...,
skip_target=True)` (deliberately no `path` key, relying on `skip_target=True`
to short-circuit the target branch) and got back
`AssertionError: 'path'` out of `Engine.collective_rpc`'s own
`assert recv_req.success, recv_req.message`. `TestEngineRpcPathAgainstRealPlumbing`
below replays that exact call through the real `Engine.save_sharded_model` ->
`collective_rpc` -> (fake transport) -> `SchedulerWeightUpdaterManager.
save_sharded_model` (the real 332634d2c6 code) path and it succeeds with no
exception -- proving the fork's own source was never the problem. The
traceback frame job 633733 actually hit
(`File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py"`) is
the container image's own baked-in stock sglang install, not the staged
fork bundle (`/dev/shm/sglang-v3-332634d-310` per that job's own
`stage.txt`) -- an environment/PYTHONPATH bug in the sbatch wrapper, fixed
in `save-draft-shard-v5.sbatch` and self-checked at runtime by
`save_draft_shard_v5.py`'s `_check_sglang_is_the_staged_bundle()`
(`TestSaveDraftShardV5SelfCheck` below exercises that function directly).
`TestSaveDraftShardV5Kwargs` imports `save_draft_shard_v5.py`'s
`build_save_sharded_model_kwargs()` (factored out of the script specifically
so this test can import it without constructing an Engine) and drives it
through the same real RPC plumbing.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

# Importing anything under the `sglang` package (even the CI-registration
# helper) runs sglang/__init__.py, which needs torch plus the rest of the
# repo's runtime dependency stack. Guard the whole block so this test skips
# cleanly, rather than failing collection, in an environment that has
# neither installed (e.g. a plain checkout without `uv sync`).
try:
    import torch

    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
        _get_draft_model_runner,
    )
    from sglang.test.ci.ci_register import register_cpu_ci

    _IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without torch/sglang deps
    _IMPORT_ERROR = e

    def register_cpu_ci(*args, **kwargs):
        return None


# The real worker classes pull in the full scheduler/server-args/HTTP-serving
# import graph (uvicorn, fastapi, setproctitle, ...), heavier than what the
# rest of this file needs. Guard their import separately so an environment
# with torch+sglang's core but not its full serving stack still runs every
# test except the one that specifically needs these classes.
try:
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2

    _WORKER_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without the full serving stack
    _WORKER_IMPORT_ERROR = e


# `Engine` and the RPC io_struct types, for TestEngineRpcPathAgainstRealPlumbing
# and TestSaveDraftShardV5Kwargs below, which drive the real
# Engine.save_sharded_model -> collective_rpc path (with a fake transport,
# no zmq/subprocess) into the real SchedulerWeightUpdaterManager. Guarded
# separately since sglang.srt.entrypoints.engine pulls in the same
# heavier serving-stack import graph as the worker classes above.
try:
    from sglang.srt.entrypoints import engine as _engine_mod
    from sglang.srt.managers.io_struct import RpcReqOutput as _RpcReqOutput

    _ENGINE_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without the full serving stack
    _ENGINE_IMPORT_ERROR = e


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeWeightExporter:
    def __init__(self):
        self.calls: list = []

    def save_sharded_model(self, path, pattern, max_size):
        self.calls.append({"path": path, "pattern": pattern, "max_size": max_size})


def _fake_model(n_tensors: int = 2):
    """A minimal stand-in with a real `state_dict()` of real tensors, since
    `save_sharded_model` now logs tensor/byte counts via
    `ShardedStateLoader._filter_subtensors`, which needs actual `torch.Tensor`
    objects (storage pointers, element size, etc.), not arbitrary mocks."""
    state = {f"w{i}": torch.zeros(4, 4) for i in range(n_tensors)}
    return SimpleNamespace(state_dict=lambda: dict(state))


def _fake_target_worker():
    """Mirrors `TpModelWorker`: `tp_worker.model_runner.weight_exporter`.
    Unaffected by the bug (the target is always reached via
    `self.tp_worker.model_runner` directly), so a flat mock is faithful here.
    """
    exporter = _FakeWeightExporter()
    model_runner = SimpleNamespace(weight_exporter=exporter, model=_fake_model())
    return SimpleNamespace(model_runner=model_runner), exporter


def _fake_draft_worker():
    """Mirrors the REAL nesting `_get_draft_model_runner` reads for the
    EAGLE-family workers this fork actually deploys (EAGLEWorkerV2,
    FrozenKVMTPWorkerV2): the scheduler-level worker has an inner
    `_draft_worker` (EagleDraftWorker / FrozenKVMTPDraftWorker), which alone
    exposes `.draft_runner` (the real `ModelRunner`). The outer worker itself
    has no `.model_runner` -- giving it one here would silently defeat this
    regression test, as it did before this fix.
    """
    exporter = _FakeWeightExporter()
    model_runner = SimpleNamespace(weight_exporter=exporter, model=_fake_model())
    inner = SimpleNamespace(draft_runner=model_runner)
    outer = SimpleNamespace(_draft_worker=inner)
    return outer, exporter


# save_draft_shard_v5.py is an ops script that runs the actual save job on
# the test cluster (manual/draft-shard/ there); it is not part of this repo's own
# tree. TestSaveDraftShardV5SelfCheck and TestSaveDraftShardV5Kwargs below
# load it by file path -- from $SAVE_DRAFT_SHARD_V5_SCRIPT if set, else the
# known location this task ships it to -- purely as an offline check that
# THIS exact script's call site and self-check pass the real Engine-level
# checks, before it is ever run on a cluster node. Absent in a generic
# checkout/CI environment, so both test classes skip cleanly rather than
# fail collection when the file can't be found.
_V5_SCRIPT_CANDIDATES = [
    p
    for p in [
        os.environ.get("SAVE_DRAFT_SHARD_V5_SCRIPT"),
        "/tmp/claude-1000/-mnt-data-shli-vibesys--claude-worktrees-amd-gpu-cluster-access-fa8c50/824cc643-74b5-48ba-a1f5-e2a54cff53d3/scratchpad/loop/draft-shard/save_draft_shard_v5.py",
    ]
    if p
]


def _load_save_draft_shard_v5():
    for path in _V5_SCRIPT_CANDIDATES:
        if not path or not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("save_draft_shard_v5", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"torch/sglang runtime deps not importable in this environment: {_IMPORT_ERROR}",
)
class TestSaveShardedModelDraftBranch(unittest.TestCase):
    def setUp(self):
        self.tp_worker, self.target_exporter = _fake_target_worker()

    def _manager(self, draft_worker):
        return SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=draft_worker,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda *a, **k: True,
            is_fully_idle=lambda *a, **k: True,
        )

    def test_no_draft_worker_saves_target_only(self):
        manager = self._manager(draft_worker=None)

        manager.save_sharded_model(
            {"path": "/tmp/target", "pattern": None, "max_size": 123}
        )

        self.assertEqual(
            self.target_exporter.calls,
            [{"path": "/tmp/target", "pattern": None, "max_size": 123}],
        )

    def test_draft_worker_saves_both_to_distinct_paths(self):
        draft_worker, draft_exporter = _fake_draft_worker()
        manager = self._manager(draft_worker=draft_worker)

        manager.save_sharded_model(
            {
                "path": "/tmp/target",
                "pattern": None,
                "max_size": 123,
                "draft_path": "/tmp/draft",
            }
        )

        self.assertEqual(
            self.target_exporter.calls,
            [{"path": "/tmp/target", "pattern": None, "max_size": 123}],
        )
        self.assertEqual(
            draft_exporter.calls,
            [{"path": "/tmp/draft", "pattern": None, "max_size": 123}],
        )

    def test_draft_worker_without_draft_path_raises(self):
        draft_worker, draft_exporter = _fake_draft_worker()
        manager = self._manager(draft_worker=draft_worker)

        with self.assertRaises(AssertionError):
            manager.save_sharded_model(
                {"path": "/tmp/target", "pattern": None, "max_size": 123}
            )

        # The target save still runs; only the draft side is guarded.
        self.assertEqual(
            self.target_exporter.calls,
            [{"path": "/tmp/target", "pattern": None, "max_size": 123}],
        )
        self.assertEqual(draft_exporter.calls, [])

    def test_draft_worker_missing_draft_runner_raises_loudly(self):
        # An outer worker with neither `.draft_model_runner` nor a
        # `_draft_worker.draft_runner` (e.g. a future spec worker class this
        # helper hasn't been taught about yet) must fail loudly instead of
        # silently writing nothing, which is exactly how the real bug went
        # unnoticed: the draft branch raised `AttributeError` deep inside,
        # got caught and turned into an RPC failure by
        # `Scheduler.handle_rpc_request`, and that failure never surfaced to
        # the caller of `Engine.save_sharded_model`.
        draft_worker = SimpleNamespace()  # no draft_model_runner, no _draft_worker
        manager = self._manager(draft_worker=draft_worker)

        with self.assertRaises(RuntimeError):
            manager.save_sharded_model(
                {
                    "path": "/tmp/target",
                    "pattern": None,
                    "max_size": 123,
                    "draft_path": "/tmp/draft",
                }
            )

    def test_skip_target_only_saves_draft(self):
        draft_worker, draft_exporter = _fake_draft_worker()
        manager = self._manager(draft_worker=draft_worker)

        manager.save_sharded_model(
            {
                "path": "/tmp/target",
                "pattern": None,
                "max_size": 123,
                "draft_path": "/tmp/draft",
                "skip_target": True,
            }
        )

        self.assertEqual(self.target_exporter.calls, [])
        self.assertEqual(
            draft_exporter.calls,
            [{"path": "/tmp/draft", "pattern": None, "max_size": 123}],
        )

    def test_skip_target_defaults_to_false(self):
        # No `skip_target` key at all (the pre-PR-#55 kwarg shape): target
        # must still be saved, matching every existing caller.
        manager = self._manager(draft_worker=None)

        manager.save_sharded_model(
            {"path": "/tmp/target", "pattern": None, "max_size": 123}
        )

        self.assertEqual(
            self.target_exporter.calls,
            [{"path": "/tmp/target", "pattern": None, "max_size": 123}],
        )


@unittest.skipIf(
    _IMPORT_ERROR is not None or _WORKER_IMPORT_ERROR is not None,
    "torch/sglang runtime deps (incl. the full serving stack) not importable "
    f"in this environment: {_IMPORT_ERROR or _WORKER_IMPORT_ERROR}",
)
class TestRealSpecWorkerAttributeStructure(unittest.TestCase):
    """Checks the attribute-shape assumption `_get_draft_model_runner` (and
    the fakes above) rely on against the actual worker classes this fork
    deploys, not a mock -- so a future refactor of these classes' internals
    fails this test instead of silently reintroducing bug 2.
    """

    def _check(self, worker_cls, inner_attr: str):
        # `object.__new__` builds an instance without running `__init__`
        # (which needs a live ServerArgs/ParallelState/GPU), just to probe
        # the class's attribute surface.
        instance = object.__new__(worker_cls)

        # The bug: code that reached for `self.draft_worker.model_runner`
        # directly (`self.draft_worker` here being the *scheduler's*
        # reference to this very instance) must not find one.
        self.assertFalse(
            hasattr(worker_cls, "model_runner"),
            f"{worker_cls.__name__} unexpectedly defines .model_runner; "
            "if this now legitimately exists, save_sharded_model's draft "
            "branch could use it directly again instead of "
            "_get_draft_model_runner.",
        )

        # The fix: _get_draft_model_runner must be able to reach the real
        # ModelRunner via `_draft_worker.<inner_attr>`.
        fake_model_runner = object()
        instance._draft_worker = SimpleNamespace(**{inner_attr: fake_model_runner})
        self.assertIs(_get_draft_model_runner(instance), fake_model_runner)

    def test_eagle_worker_v2_attribute_structure(self):
        self._check(EAGLEWorkerV2, "draft_runner")

    def test_frozen_kv_mtp_worker_v2_attribute_structure(self):
        self._check(FrozenKVMTPWorkerV2, "draft_runner")


class _FakeRpcSocket:
    """Stands in for `Engine.send_to_rpc` (a real zmq socket). Holds the last
    `RpcReqInput` handed to the faked `sock_send` and, on `sock_recv`,
    dispatches it exactly the way `Scheduler.handle_rpc_request` +
    `Scheduler.save_sharded_model` do against a REAL
    `SchedulerWeightUpdaterManager` -- no zmq, no subprocess, no GPU."""

    def __init__(self, manager: "SchedulerWeightUpdaterManager"):
        self.manager = manager
        self.pending = None


def _fake_sock_send(socket: _FakeRpcSocket, obj, flags=0):
    socket.pending = obj


def _fake_sock_recv(socket: _FakeRpcSocket, flags=0):
    recv_req = socket.pending
    # Mirrors Scheduler.handle_rpc_request's try/except/success/message
    # contract exactly (managers/scheduler.py:4420-4437), dispatching into
    # Scheduler.save_sharded_model's own shape
    # (`def save_sharded_model(self, **kwargs): self.weight_updater.
    # save_sharded_model(kwargs)`, managers/scheduler.py:4417-4418).
    success = True
    exc = None
    try:
        method = getattr(recv_req, "method")
        assert method == "save_sharded_model", method
        socket.manager.save_sharded_model(recv_req.parameters or {})
    except Exception as e:  # noqa: BLE001 - mirror the real catch-all
        success = False
        exc = e
    return _RpcReqOutput(success=success, message="" if not exc else str(exc))


@unittest.skipIf(
    _IMPORT_ERROR is not None or _ENGINE_IMPORT_ERROR is not None,
    "torch/sglang runtime deps (incl. sglang.srt.entrypoints.engine) not "
    f"importable in this environment: {_IMPORT_ERROR or _ENGINE_IMPORT_ERROR}",
)
class TestEngineRpcPathAgainstRealPlumbing(unittest.TestCase):
    """Drives the REAL `Engine.save_sharded_model` -> `collective_rpc` code
    (python/sglang/srt/entrypoints/engine.py) against a REAL
    `SchedulerWeightUpdaterManager.save_sharded_model` (commit 332634d2c6),
    with only the zmq transport (`sock_send`/`sock_recv`) faked out. This is
    the whole RPC path job 633733 hit `AssertionError: 'path'` on -- if
    that path were actually broken at the source level, this test would
    catch it exactly the way the real Engine caller does (an
    `AssertionError` carrying `recv_req.message`), no GPU or cluster boot
    required.
    """

    def setUp(self):
        self._orig_sock_send = _engine_mod.sock_send
        self._orig_sock_recv = _engine_mod.sock_recv
        _engine_mod.sock_send = _fake_sock_send
        _engine_mod.sock_recv = _fake_sock_recv
        self.addCleanup(self._restore)

        self.tp_worker, self.target_exporter = _fake_target_worker()
        self.draft_worker, self.draft_exporter = _fake_draft_worker()
        self.manager = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda *a, **k: True,
            is_fully_idle=lambda *a, **k: True,
        )

        # object.__new__ builds a bare Engine instance (no ServerArgs, no
        # subprocess spawn, no GPU) purely to call its real
        # save_sharded_model/collective_rpc methods, which only touch
        # self.send_to_rpc.
        self.engine = object.__new__(_engine_mod.Engine)
        self.engine.send_to_rpc = _FakeRpcSocket(self.manager)

    def _restore(self):
        _engine_mod.sock_send = self._orig_sock_send
        _engine_mod.sock_recv = self._orig_sock_recv

    def test_skip_target_false_saves_both(self):
        self.engine.save_sharded_model(
            path="/tmp/target",
            pattern=None,
            max_size=123,
            draft_path="/tmp/draft",
        )
        self.assertEqual(
            self.target_exporter.calls,
            [{"path": "/tmp/target", "pattern": None, "max_size": 123}],
        )
        self.assertEqual(
            self.draft_exporter.calls,
            [{"path": "/tmp/draft", "pattern": None, "max_size": 123}],
        )

    def test_skip_target_true_saves_draft_only_even_with_no_path_key(self):
        # This is save_draft_shard_v4.py's exact call (job 633733): no
        # `path` key at all, relying on skip_target=True to short-circuit
        # the target branch before params["path"] is ever read. Must NOT
        # raise -- if it does, the fix broke and this reproduces the real
        # bug report's AssertionError directly.
        self.engine.save_sharded_model(
            pattern=None,
            max_size=123,
            draft_path="/tmp/draft",
            skip_target=True,
        )
        self.assertEqual(self.target_exporter.calls, [])
        self.assertEqual(
            self.draft_exporter.calls,
            [{"path": "/tmp/draft", "pattern": None, "max_size": 123}],
        )

    def test_skip_target_true_with_defensive_path_key_still_saves_draft_only(self):
        # v5's actual call shape: path is still supplied (defense in
        # depth), but skip_target=True means it is still never opened.
        self.engine.save_sharded_model(
            path="/tmp/target-should-not-be-touched",
            pattern=None,
            max_size=123,
            draft_path="/tmp/draft",
            skip_target=True,
        )
        self.assertEqual(self.target_exporter.calls, [])
        self.assertEqual(
            self.draft_exporter.calls,
            [{"path": "/tmp/draft", "pattern": None, "max_size": 123}],
        )

    def test_missing_path_without_skip_target_reproduces_the_observed_bug_shape(self):
        # Documents *why* the real failure looked like `AssertionError:
        # 'path'`: any save_sharded_model implementation whose target
        # branch unconditionally reads params["path"] (skip_target=False,
        # or no skip_target support at all -- i.e. the container's stock,
        # pre-PR-#55 sglang, which is what job 633733 actually ran against)
        # raises exactly this shape once the caller stops sending a `path`
        # key. This fork's real code does NOT do this when skip_target=True
        # (see the test above) -- the container's un-forked stock code
        # does, since it has no skip_target branch to short-circuit on.
        with self.assertRaises(AssertionError) as ctx:
            self.engine.save_sharded_model(
                pattern=None,
                max_size=123,
                draft_path="/tmp/draft",
                # skip_target intentionally omitted, AND path omitted --
                # this fork's real code's `if not skip_target:` block is
                # therefore entered and reads params["path"], KeyError'ing.
            )
        self.assertEqual(str(ctx.exception), "'path'")


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"torch/sglang runtime deps not importable in this environment: {_IMPORT_ERROR}",
)
class TestSaveDraftShardV5SelfCheck(unittest.TestCase):
    """Exercises `save_draft_shard_v5.py`'s `_check_sglang_is_the_staged_bundle()`
    directly: the runtime guard added specifically to catch job 633733's
    real bug (the script's `import sglang` silently resolving to the
    container's stock /sgl-workspace/sglang instead of the staged fork
    bundle) offline, in under a second, instead of after a ~1300s Engine
    boot on the next cluster job.
    """

    def setUp(self):
        module = _load_save_draft_shard_v5()
        if module is None:
            self.skipTest(
                "save_draft_shard_v5.py not found via SAVE_DRAFT_SHARD_V5_SCRIPT "
                f"or the known candidate paths: {_V5_SCRIPT_CANDIDATES}"
            )
        self.module = module

    def test_matching_root_passes(self):
        # Point the expected root at wherever `sglang` (this test process's
        # own import) actually resolved from, so the check's "resolved is
        # under expected" comparison passes against the real sglang.__file__.
        import sglang as _sglang

        os.environ["VIBESYS_STAGED_SGLANG_ROOT"] = os.path.dirname(
            os.path.dirname(os.path.abspath(_sglang.__file__))
        )
        try:
            self.module._check_sglang_is_the_staged_bundle()  # must not raise
        finally:
            del os.environ["VIBESYS_STAGED_SGLANG_ROOT"]

    def test_mismatched_root_raises_loudly(self):
        os.environ["VIBESYS_STAGED_SGLANG_ROOT"] = "/sgl-workspace/sglang"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.module._check_sglang_is_the_staged_bundle()
            self.assertIn("NOT under", str(ctx.exception))
        finally:
            del os.environ["VIBESYS_STAGED_SGLANG_ROOT"]

    def test_unset_root_warns_but_does_not_raise(self):
        os.environ.pop("VIBESYS_STAGED_SGLANG_ROOT", None)
        self.module._check_sglang_is_the_staged_bundle()  # must not raise


@unittest.skipIf(
    _IMPORT_ERROR is not None or _ENGINE_IMPORT_ERROR is not None,
    "torch/sglang runtime deps (incl. sglang.srt.entrypoints.engine) not "
    f"importable in this environment: {_IMPORT_ERROR or _ENGINE_IMPORT_ERROR}",
)
class TestSaveDraftShardV5Kwargs(unittest.TestCase):
    """Imports save_draft_shard_v5.py's `build_save_sharded_model_kwargs()`
    -- factored out of the script specifically so this test can import it
    without constructing an Engine -- and drives the exact kwargs it
    returns through the same real RPC plumbing as
    TestEngineRpcPathAgainstRealPlumbing, proving v5's real call site
    passes the real Engine-level checks end to end offline."""

    def setUp(self):
        module = _load_save_draft_shard_v5()
        if module is None:
            self.skipTest(
                "save_draft_shard_v5.py not found via SAVE_DRAFT_SHARD_V5_SCRIPT "
                f"or the known candidate paths: {_V5_SCRIPT_CANDIDATES}"
            )
        self.module = module

        self._orig_sock_send = _engine_mod.sock_send
        self._orig_sock_recv = _engine_mod.sock_recv
        _engine_mod.sock_send = _fake_sock_send
        _engine_mod.sock_recv = _fake_sock_recv
        self.addCleanup(self._restore)

        self.tp_worker, self.target_exporter = _fake_target_worker()
        self.draft_worker, self.draft_exporter = _fake_draft_worker()
        self.manager = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda *a, **k: True,
            is_fully_idle=lambda *a, **k: True,
        )
        self.engine = object.__new__(_engine_mod.Engine)
        self.engine.send_to_rpc = _FakeRpcSocket(self.manager)

    def _restore(self):
        _engine_mod.sock_send = self._orig_sock_send
        _engine_mod.sock_recv = self._orig_sock_recv

    def test_kwargs_shape(self):
        kwargs = self.module.build_save_sharded_model_kwargs()
        self.assertEqual(
            set(kwargs.keys()), {"path", "pattern", "max_size", "draft_path", "skip_target"}
        )
        self.assertTrue(kwargs["skip_target"])
        self.assertIsInstance(kwargs["path"], str)
        self.assertIsInstance(kwargs["draft_path"], str)
        self.assertIsNone(kwargs["pattern"])
        self.assertIsInstance(kwargs["max_size"], int)

    def test_kwargs_pass_the_real_engine_rpc_path(self):
        kwargs = self.module.build_save_sharded_model_kwargs()

        # Must not raise: this is exactly `Engine.save_sharded_model(**kwargs)`,
        # the real call save_draft_shard_v5.py's main() makes.
        self.engine.save_sharded_model(**kwargs)

        # skip_target=True in the real kwargs: target must stay untouched,
        # draft must be saved to the script's DRAFT_OUTPUT_PATH.
        self.assertEqual(self.target_exporter.calls, [])
        self.assertEqual(len(self.draft_exporter.calls), 1)
        self.assertEqual(self.draft_exporter.calls[0]["path"], kwargs["draft_path"])

    def test_full_main_with_stub_engine_dry_run(self):
        """Runs save_draft_shard_v5.py's real `main()` end to end -- module
        import, the staged-bundle self-check, Engine construction, the
        real `llm.save_sharded_model(**kwargs)` call site, and the
        non-weight-file copy -- with `sglang.Engine` monkeypatched to a
        recording stub (no model load, no GPU) and the module's path
        constants redirected into a tmp dir (its real constants are
        /path/to/scratch paths that don't exist on this machine). Proves the
        exact kwargs the script sends are what reaches Engine.
        save_sharded_model, with no piece of main() skipped or
        reimplemented by the test.
        """
        recorded = {}

        class _StubServerArgs:
            mem_fraction_static = 0.72
            max_total_tokens = 787936
            max_running_requests = 8
            disable_cuda_graph = True
            weight_loader_disable_mmap = False
            speculative_algorithm = "EAGLE"
            speculative_num_steps = 3
            speculative_draft_model_path = "/fake/draft"
            speculative_draft_load_format = "auto"

        class _StubEngine:
            def __init__(self, **kwargs):
                recorded["engine_kwargs"] = kwargs
                self.server_args = _StubServerArgs()

            def save_sharded_model(self, **kwargs):
                recorded["save_sharded_model_kwargs"] = kwargs

        with tempfile.TemporaryDirectory() as tmp:
            draft_model_src = os.path.join(tmp, "draft-model-src")
            draft_output = os.path.join(tmp, "draft-output")
            os.makedirs(draft_model_src)
            # A non-weight file so copy_non_weight_files() (called at the
            # end of main(), after save_sharded_model) has something to
            # walk without touching a real checkpoint.
            with open(os.path.join(draft_model_src, "config.json"), "w") as f:
                f.write("{}")

            self.module.DRAFT_MODEL_PATH = draft_model_src
            self.module.DRAFT_OUTPUT_PATH = draft_output
            self.module.TARGET_DISCARD_PATH = os.path.join(tmp, "target-discard")

            real_sglang = sys.modules["sglang"]
            orig_engine = getattr(real_sglang, "Engine", None)
            real_sglang.Engine = _StubEngine
            os.environ.pop("VIBESYS_STAGED_SGLANG_ROOT", None)  # exercise the "unset -> warn" path too
            try:
                self.module.main()
            finally:
                if orig_engine is not None:
                    real_sglang.Engine = orig_engine
                else:  # pragma: no cover - sglang always defines Engine in practice
                    del real_sglang.Engine

        # The exact kwargs build_save_sharded_model_kwargs() returns must be
        # what actually reached Engine.save_sharded_model via main()'s real
        # call site -- not a hand-rolled copy the test constructs itself.
        expected_kwargs = dict(self.module.build_save_sharded_model_kwargs())
        # build_save_sharded_model_kwargs() reads the (now-redirected)
        # module-level path constants at call time, so recompute expected
        # draft/target paths against the redirected values used above.
        self.assertEqual(recorded["save_sharded_model_kwargs"], expected_kwargs)
        self.assertTrue(recorded["save_sharded_model_kwargs"]["skip_target"])
        self.assertEqual(
            recorded["save_sharded_model_kwargs"]["draft_path"], draft_output
        )


if __name__ == "__main__":
    unittest.main()
