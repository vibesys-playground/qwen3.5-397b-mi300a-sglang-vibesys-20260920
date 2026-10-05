"""
Regression test for profile_by_stage stalling on the EXTEND (prefill) stage.

Bug: with profile_by_stage=True, once the DECODE stage's first batch forced
the still-open EXTEND recording to stop (a "force trace flush" when
switching stages), the EXTEND stage could never resume: its start guard only
fired on a raw step-counter equal to zero, which had already been consumed.
EXTEND's step counter kept climbing on every subsequent prefill batch, but
because `torch_profiler` was gone and `profile_in_progress` had been
collapsed to a single flag shared by both stages, EXTEND's "count exceeded
target" branch silently no-op'd instead of exporting a trace. A later
explicit /stop_profile then found `profile_in_progress` False and raised
"Profiling is not in progress."

This test drives SchedulerProfilerManager directly with fake alternating
EXTEND/DECODE batches and a stub torch profiler (no GPU needed) to check
that both stages export their own trace, and that stopping mid-session does
not raise.
"""

import gzip
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Importing anything under the `sglang` package (even the CI-registration
# helper) runs sglang/__init__.py, which needs torch plus the rest of the
# repo's runtime dependency stack. Guard the whole block so this test skips
# cleanly, rather than failing collection, in an environment that has
# neither installed (e.g. a plain checkout without `uv sync`).
try:
    from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
    from sglang.srt.managers.scheduler_components.profiler_manager import (
        SchedulerProfilerManager,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.test.ci.ci_register import register_cpu_ci

    _IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without torch/sglang deps
    _IMPORT_ERROR = e

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTorchProfiler:
    """Stand-in for torch.profiler.profile that needs neither a GPU nor a
    real trace backend. Records lifecycle calls and writes a real (if tiny)
    gzip-compressed Chrome Trace Event Format file on export, tagged with
    this instance's identity, so a test can assert not just that a stage's
    trace file exists but that it actually contains every recording window's
    data rather than only the last one to write that path.

    `recorded_batches` stands in for the kernels a real torch profiler would
    capture: a test appends to it, for whichever `_FakeTorchProfiler`
    instance is `self.torch_profiler` right after a
    `_profile_batch_predicate` call, the mode of the batch that was about to
    run. That mirrors production (the predicate runs before the batch's
    forward pass), so an instance's `recorded_batches` reflects exactly
    which batches' kernels its window was open for -- including, if a stage
    fails to pause another stage's still-open window, batches of the wrong
    mode.
    """

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.exported_to = None
        self.recorded_batches: list = []
        _FakeTorchProfiler.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        self.exported_to = path
        # Real export_chrome_trace writes gzip-compressed JSON when the path
        # ends in .gz; matching that here is what exercises the manager's
        # merge step instead of only its file-existence bookkeeping.
        with gzip.open(path, "wt") as f:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "instance_id": id(self),
                            "recorded_batches": [
                                mode.name for mode in self.recorded_batches
                            ],
                        }
                    ]
                },
                f,
            )


def _fake_batch(forward_mode: "ForwardMode"):
    return SimpleNamespace(forward_mode=forward_mode)


def _load_trace_events(path: str) -> list:
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def _flatten_recorded_batches(events: list) -> list:
    flattened: list = []
    for event in events:
        flattened.extend(event.get("recorded_batches", []))
    return flattened


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"torch/sglang runtime deps not importable in this environment: {_IMPORT_ERROR}",
)
class TestProfileByStage(unittest.TestCase):
    def setUp(self):
        _FakeTorchProfiler.instances = []
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        self.ps = SimpleNamespace(
            tp_rank=0,
            dp_size=1,
            dp_rank=0,
            pp_size=1,
            pp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            gpu_id=0,
        )
        self.manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )

        patcher = patch(
            "sglang.srt.managers.scheduler_components.profiler_manager.torch.profiler.profile",
            _FakeTorchProfiler,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        barrier_patcher = patch("torch.distributed.barrier")
        barrier_patcher.start()
        self.addCleanup(barrier_patcher.stop)

    def _start_profile_by_stage(self, num_steps: int):
        req = ProfileReq(
            req_type=ProfileReqType.START_PROFILE,
            output_dir=self._tmpdir.name,
            num_steps=num_steps,
            activities=["CPU"],
            profile_by_stage=True,
            profile_id="s1",
            with_stack=False,
            record_shapes=False,
        )
        result = self.manager._profile(req)
        self.assertTrue(result.success, result.message)

    def _trace_path(self, stage: str) -> str:
        return os.path.join(self._tmpdir.name, f"s1-TP-0-{stage}.trace.json.gz")

    def _run_sequence(self, modes) -> None:
        """Drive `_profile_batch_predicate` for each mode in order, then
        record that mode against whichever `_FakeTorchProfiler` instance is
        active right after the call -- mirroring that `run_batch` calls the
        predicate before running the batch's forward pass (see
        scheduler.py's `run_batch`), so the profiler open right after the
        predicate call is the one that would capture that batch's kernels.
        A batch that itself pushes a stage's count past its target is
        stopped (and exported) by the predicate before returning, so it is
        correctly left unrecorded here, the same as in production.
        """
        for mode in modes:
            self.manager._profile_batch_predicate(_fake_batch(mode))
            if self.manager.torch_profiler is not None:
                self.manager.torch_profiler.recorded_batches.append(mode)

    def test_both_stages_export_after_interleaving(self):
        """EXTEND must resume and eventually export even though its first
        recording window gets force-flushed by an early DECODE batch."""
        self._start_profile_by_stage(num_steps=2)

        # EXTEND starts, then DECODE interrupts it before it reaches target.
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.DECODE))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.DECODE))
        # 3rd decode batch pushes decode_ct to 3 > target(2): decode finishes.
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.DECODE))

        self.assertTrue(self.manager.profiler_decode_done)
        self.assertTrue(
            os.path.exists(self._trace_path("DECODE")),
            "DECODE stage did not export its trace",
        )

        # EXTEND must be able to resume now that DECODE is done.
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))

        self.assertTrue(
            self.manager.profiler_prefill_done,
            "EXTEND stage never finished: it was abandoned after being "
            "force-flushed by the DECODE stage switch",
        )
        self.assertTrue(
            os.path.exists(self._trace_path("EXTEND")),
            "EXTEND stage did not export its trace",
        )
        self.assertFalse(self.manager.profile_in_progress)

        # EXTEND recorded in two separate windows (batch 1, then batch 5-6
        # after being force-flushed and resumed). Both windows' data must
        # survive into the final file: a fix that simply re-exports to the
        # same canonical path on resume silently overwrites the first
        # window with the second, shrinking the trace instead of completing
        # it.
        extend_events = _load_trace_events(self._trace_path("EXTEND"))
        self.assertEqual(
            len(extend_events),
            2,
            "EXTEND's canonical trace only reflects its last recording "
            "window; the pre-interruption window's data was overwritten "
            "instead of merged in",
        )

    def test_finished_stage_batches_still_pause_an_open_other_stage(self):
        """Regression for job 631900: profile_by_stage=True, num_steps=8.

        DECODE reached its quota within ~2s and was finalized
        (profiler_decode_done=True). EXTEND then took 11.4s to see its 8
        prefill batches, and roughly 99 DECODE batches ran during that
        window. `_profile_batch_predicate` only called
        `_ensure_profiling_stage` (which does the pausing) inside the
        `if not <stage>_done:` guard, so once DECODE was done, every later
        DECODE batch fell straight through with no effect: EXTEND's
        torch_profiler stayed open across them, capturing their kernels
        into what was later exported as EXTEND's trace.

        This test reproduces the same shape with num_steps=2 (a stage's own
        trace ends up with exactly `num_steps` recorded batches: the batch
        that pushes its counter past the target is stopped, and so left
        unrecorded, before its own forward pass runs -- see
        `test_both_stages_export_after_interleaving`'s "3rd decode batch"
        comment for the same off-by-one). DECODE finishes first (its 3rd
        batch). EXTEND then starts and, while it waits for its own 2nd and
        3rd batches, 3 more DECODE batches arrive -- already-done batches
        that must still pause EXTEND rather than being captured by its
        still-open window, and must themselves record nothing since DECODE
        is done.
        """
        self._start_profile_by_stage(num_steps=2)

        self._run_sequence(
            [
                ForwardMode.DECODE,  # 1
                ForwardMode.DECODE,  # 2
                ForwardMode.DECODE,  # 3: decode_ct 3 > target(2), DECODE finishes
                ForwardMode.EXTEND,  # 1: EXTEND starts
                ForwardMode.DECODE,  # DECODE already done: must pause EXTEND, record nothing
                ForwardMode.DECODE,  # same
                ForwardMode.DECODE,  # same
                ForwardMode.EXTEND,  # 2: EXTEND resumes
                ForwardMode.EXTEND,  # 3: prefill_ct 3 > target(2), EXTEND finishes
            ]
        )

        self.assertTrue(self.manager.profiler_decode_done)
        self.assertTrue(self.manager.profiler_prefill_done)
        self.assertFalse(self.manager.profile_in_progress)

        decode_batches = _flatten_recorded_batches(
            _load_trace_events(self._trace_path("DECODE"))
        )
        self.assertEqual(
            decode_batches,
            [ForwardMode.DECODE.name] * 2,
            "DECODE's trace must contain only its own recorded batches",
        )

        extend_batches = _flatten_recorded_batches(
            _load_trace_events(self._trace_path("EXTEND"))
        )
        self.assertEqual(
            extend_batches,
            [ForwardMode.EXTEND.name] * 2,
            "EXTEND's trace leaked batches from DECODE: an already-finished "
            "stage's batch failed to pause EXTEND's still-open window",
        )

    def test_extend_survives_being_interrupted_after_reaching_its_own_target(self):
        """Hand-traced regression for E E D D D E E with num_steps=2.

        EXTEND reaches its own target step count (2) *before* DECODE ever
        interrupts it, but the interruption still arrives before EXTEND's
        own "target reached" check would have fired (that check only runs
        on the next EXTEND batch, which here is DECODE instead). So EXTEND
        gets force-flushed with a fully-sized window, resumes later, and
        immediately exceeds its target on the very first resumed batch
        because its step counter was never reset. A fix that re-exports the
        resumed (nearly empty) window to EXTEND's canonical path clobbers
        the earlier, already-complete window instead of keeping it.
        """
        self._start_profile_by_stage(num_steps=2)

        for mode in (
            ForwardMode.EXTEND,  # 1
            ForwardMode.EXTEND,  # 2: EXTEND's own window is now full
            ForwardMode.DECODE,  # interrupts EXTEND before it can self-stop
            ForwardMode.DECODE,
            ForwardMode.DECODE,  # decode_ct hits 3 > target(2): decode finishes
            ForwardMode.EXTEND,  # resumes; ct 2->3 > target: EXTEND finishes too
        ):
            self.manager._profile_batch_predicate(_fake_batch(mode))

        self.assertTrue(self.manager.profiler_decode_done)
        self.assertTrue(self.manager.profiler_prefill_done)
        self.assertFalse(self.manager.profile_in_progress)

        extend_events = _load_trace_events(self._trace_path("EXTEND"))
        self.assertEqual(
            len(extend_events),
            2,
            "EXTEND's pre-interruption window (a full 2 steps) was "
            "overwritten by the tiny resumed window instead of being kept",
        )

    def test_explicit_stop_does_not_raise_when_stage_open(self):
        """/stop_profile while one stage is still mid-recording must stop
        and export it instead of failing."""
        self._start_profile_by_stage(num_steps=1000)

        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))
        self.assertTrue(self.manager.profile_in_progress)

        result = self.manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))

        self.assertTrue(result.success, result.message)
        self.assertTrue(os.path.exists(self._trace_path("EXTEND")))
        self.assertFalse(self.manager.profile_in_progress)

        # A second stop after the session has already ended must also not
        # raise (mirrors the reported HTTP 500 on a stray /stop_profile).
        result2 = self.manager._profile(
            ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
        )
        self.assertTrue(result2.success, result2.message)

    def test_explicit_stop_after_natural_completion_does_not_raise(self):
        """Once both armed stages finish on their own, a later /stop_profile
        must succeed instead of raising "Profiling is not in progress"."""
        self._start_profile_by_stage(num_steps=1)

        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.EXTEND))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.DECODE))
        self.manager._profile_batch_predicate(_fake_batch(ForwardMode.DECODE))

        self.assertTrue(self.manager.profiler_prefill_done)
        self.assertTrue(self.manager.profiler_decode_done)
        self.assertFalse(self.manager.profile_in_progress)

        result = self.manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))
        self.assertTrue(result.success, result.message)

    def test_non_by_stage_stop_without_start_still_errors(self):
        """profile_by_stage=False behavior is unchanged: stopping without a
        preceding start is still reported as a failure."""
        result = self.manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))
        self.assertFalse(result.success)
        self.assertIn("Call /start_profile first", result.message)


if __name__ == "__main__":
    unittest.main()
