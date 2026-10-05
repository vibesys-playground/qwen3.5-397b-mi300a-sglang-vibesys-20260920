from __future__ import annotations

import gc
import gzip
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
)

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_device
from sglang.srt.utils import is_mps, is_npu
from sglang.srt.utils.profile_merger import ProfileMerger
from sglang.srt.utils.profile_utils import ProfileManager
from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

_is_npu = is_npu()
_is_mps = is_mps()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    apply_torch_npu_patches(torch_npu, patches)
elif _is_mps:
    from sglang.srt.hardware_backend.mlx.profiler import apply_metal_profiler_patches

    apply_metal_profiler_patches()

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SchedulerProfilerManager:
    ps: Any
    dp_tp_cpu_group: Any
    get_forward_ct: Callable[[], int]

    def __post_init__(self) -> None:
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager = ProfileManager(
                ps=self.ps,
                cpu_group=self.dp_tp_cpu_group,
            )
            return

        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[Path] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profile_id: Optional[str] = None

        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None

        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None

        # For profile_by_stage: which stage (if any) currently owns
        # torch_profiler, and whether each armed stage has already reached
        # its target step count and exported its trace.
        self.profiler_current_stage: Optional[ForwardMode] = None
        self.profiler_prefill_done: bool = False
        self.profiler_decode_done: bool = False

        # A by-stage recording can be paused (forced flush on a switch to
        # the other stage) and resumed multiple times before it reaches its
        # own target step count. Each pause exports its still-open window
        # to a private part file here rather than the stage's canonical
        # path, so a later, shorter window can't silently overwrite an
        # earlier one; all of a stage's parts are merged into its single
        # canonical trace once it is finalized.
        self.profiler_prefill_trace_parts: List[Path] = []
        self.profiler_decode_trace_parts: List[Path] = []

        self.profile_by_stage: bool = False
        self.profile_in_progress: bool = False
        self.merge_profiles = False

        # For ROCM
        self.rpd_profiler = None

    def _init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool = False,
        profile_prefix: str = "",
        profile_stages: Optional[List[str]] = None,
    ) -> ProfileReqOutput:
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.configure(
                output_dir=output_dir,
                start_step=start_step,
                num_steps=num_steps,
                activities=activities,
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=profile_by_stage,
                profile_id=profile_id,
                merge_profiles=merge_profiles,
                profile_prefix=profile_prefix,
                profile_stages=profile_stages,
            )

        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage
        self.merge_profiles = merge_profiles

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = Path(output_dir).expanduser()
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id
        self.profile_prefix = profile_prefix

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.get_forward_ct() + 1)

        if num_steps:
            if self.profile_by_stage:
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
                self.profiler_current_stage = None
                self.profiler_prefill_done = False
                self.profiler_decode_done = False
                self.profiler_prefill_trace_parts = []
                self.profiler_decode_trace_parts = []
            elif start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                self.profiler_target_forward_ct = self.get_forward_ct() + num_steps + 1
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def _start_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.manual_start()

        stage_str = f" for {stage.name}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }

        if current_platform.is_out_of_tree():
            if hasattr(
                torch.profiler.ProfilerActivity,
                current_platform.get_torch_profiler_activity_str(),
            ):
                activity_map[current_platform.get_torch_profiler_activity_str()] = (
                    current_platform.get_torch_profiler_activity()
                )
        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            activity_map["XPU"] = torch.profiler.ProfilerActivity.XPU
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:  # for ROCM
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",
            )

            if self.ps.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.dp_tp_cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
            self.profile_in_progress = True
        elif torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                on_trace_ready=(
                    None
                    if not _is_npu
                    else torch_npu.profiler.tensorboard_trace_handler(
                        str(self.torch_profiler_output_dir)
                    )
                ),
                experimental_config=(
                    None
                    if not _is_npu
                    else torch_npu.profiler._ExperimentalConfig(
                        export_type=torch_npu.profiler.ExportType.Text,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None,
                    )
                ),
            )
            try:
                self.torch_profiler.start()
            except RuntimeError as e:
                self.torch_profiler = None
                return ProfileReqOutput(success=False, message=str(e))
            self.profile_in_progress = True

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(
                max_entries=envs.SGLANG_MEM_PROFILE_MAX_ENTRIES.get()
            )
            self.profile_in_progress = True

        if "CUDA_PROFILER" in activities:
            if self.ps.gpu_id == get_device().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def _merge_profile_traces(self) -> str:
        if not self.merge_profiles:
            return ""

        if self.ps.tp_rank != 0:
            return ""
        if self.ps.dp_size > 1 and self.ps.dp_rank != 0:
            return ""
        if self.ps.pp_size > 1 and self.ps.pp_rank != 0:
            return ""
        if self.ps.moe_ep_size > 1 and self.ps.moe_ep_rank != 0:
            return ""

        try:
            logger.info("Starting profile merge...")
            merger = ProfileMerger(self.torch_profiler_output_dir, self.profile_id)
            merged_path = merger.merge_chrome_traces()

            summary = merger.get_merge_summary()
            merge_message = (
                f" Merged trace: {merged_path} "
                f"(Events: {summary.get('total_events', '?')}, "
                f"Files: {summary.get('total_files', '?')})"
            )

            logger.info(f"Profile merge completed: {merged_path}")
        except Exception as e:
            logger.error(f"Failed to merge profiles: {e}", exc_info=True)
            return f" Merge failed: {e!s}"
        else:
            return merge_message

    def _trace_filename(self, stage: Optional[ForwardMode]) -> str:
        if self.profile_prefix:
            stage_prefix = self.profile_prefix + "-"
        else:
            stage_prefix = ""
        stage_suffix = f"-{stage.name}" if stage else ""

        # Build filename with only non-zero ranks to maintain backward compatibility
        filename_parts = [self.profile_id, f"TP-{self.ps.tp_rank}"]

        # Only add other ranks if parallelism is enabled (size > 1)
        if self.ps.dp_size > 1:
            filename_parts.append(f"DP-{self.ps.dp_rank}")
        if self.ps.pp_size > 1:
            filename_parts.append(f"PP-{self.ps.pp_rank}")
        if self.ps.moe_ep_size > 1:
            filename_parts.append(f"EP-{self.ps.moe_ep_rank}")

        return stage_prefix + "-".join(filename_parts) + stage_suffix + ".trace.json.gz"

    def _stage_trace_parts(self, stage: ForwardMode) -> List[Path]:
        if stage.is_prefill():
            return self.profiler_prefill_trace_parts
        if stage.is_decode():
            return self.profiler_decode_trace_parts
        raise RuntimeError(f"unsupported profile stage: {stage}")

    def _merge_chrome_traces(self, parts: List[Path], output_path: Path) -> None:
        """Concatenate the traceEvents of each per-window chrome trace in
        `parts` into one file at `output_path`.

        A profile_by_stage recording can be interrupted and resumed several
        times before it reaches its own target step count; each
        interruption exports its still-open window to its own part file
        (see `_stop_profile`'s `finalize=False` path) instead of the
        stage's canonical path. Merging every part here, rather than only
        keeping the last one, is what makes the final trace actually
        contain every step the stage recorded.
        """
        merged_events: List[Any] = []
        header: dict = {}
        for part in parts:
            with gzip.open(part, "rt") as f:
                doc = json.load(f)
            if not header:
                header = {k: v for k, v in doc.items() if k != "traceEvents"}
            merged_events.extend(doc.get("traceEvents", []))
        header["traceEvents"] = merged_events
        with gzip.open(output_path, "wt") as f:
            json.dump(header, f)

    def _finalize_stage_trace(self, stage: ForwardMode) -> None:
        """Merge and export a stage's already-flushed part files when no
        torch profiler is currently open for it.

        This covers a stage that was force-flushed by a switch to the other
        stage and never got to run again before an explicit /stop_profile
        ended the whole by-stage session: its parts would otherwise be
        dropped once the session is marked finished.
        """
        parts = self._stage_trace_parts(stage)
        if not parts:
            return
        canonical_path = self.torch_profiler_output_dir / self._trace_filename(stage)
        self._merge_chrome_traces(parts, canonical_path)
        for part in parts:
            part.unlink(missing_ok=True)
        parts.clear()
        logger.info("Profiling done. Traces are saved to: %s", canonical_path)

    def _stop_profile(
        self, stage: Optional[ForwardMode] = None, *, finalize: bool = True
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.manual_stop()

        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)

        if self.profile_prefix:
            stage_prefix = self.profile_prefix + "-"
        else:
            stage_prefix = ""

        stage_suffix = f"-{stage.name}" if stage else ""
        logger.info("Stop profiling" + stage_suffix + "...")
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            if not _is_npu:
                filename = self._trace_filename(stage)
                canonical_path = self.torch_profiler_output_dir / filename
                if self.profile_by_stage and stage is not None:
                    # Export this window to its own part file rather than
                    # the canonical path: a mid-target pause
                    # (finalize=False) must not let its later resumption
                    # overwrite it, and `finalize` merges every part
                    # (including this one) into the canonical trace.
                    parts = self._stage_trace_parts(stage)
                    part_path = self.torch_profiler_output_dir / (
                        filename.removesuffix(".trace.json.gz")
                        + f".part{len(parts)}.trace.json.gz"
                    )
                    self.torch_profiler.export_chrome_trace(str(part_path))
                    parts.append(part_path)
                    if finalize:
                        self._merge_chrome_traces(parts, canonical_path)
                        for part in parts:
                            part.unlink(missing_ok=True)
                        parts.clear()
                else:
                    self.torch_profiler.export_chrome_trace(str(canonical_path))
            torch.distributed.barrier(self.dp_tp_cpu_group)

        if self.rpd_profiler is not None:
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()

            torch.distributed.barrier(self.dp_tp_cpu_group)
            if self.ps.tp_rank == 0:
                from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

                rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
            self.rpd_profiler = None
            self.rpd_profile_path = None

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                stage_prefix
                + str(time.time())
                + f"-TP-{self.ps.tp_rank}-memory"
                + stage_suffix
                + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            if self.ps.gpu_id == get_device().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStop()

        if finalize:
            merge_message = self._merge_profile_traces()
            logger.info(
                "Profiling done. Traces are saved to: %s%s",
                self.torch_profiler_output_dir,
                merge_message,
            )
        else:
            # A mid-target pause is not a completion: the stage will resume
            # and its parts stay merged (not yet exported) until finalize.
            merge_message = ""
            logger.info(
                "Paused profiling%s; will resume toward its target step count.",
                stage_suffix,
            )

        if self.torch_profiler is not None:
            self.torch_profiler = None
            gc.collect()

        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message=f"Succeeded.{merge_message}")

    def _ensure_profiling_stage(self, stage: ForwardMode) -> None:
        """Make `stage` the one currently recorded by torch_profiler.

        If another stage currently owns the profiler, pause it first: stop
        it and flush its still-open window to a part file (not its
        canonical trace) so its data isn't silently discarded. It resumes,
        and all of its parts are merged, once it reaches its own target
        step count or the session is explicitly stopped. No-op if `stage`
        is already the active one.
        """
        if self.profiler_current_stage == stage:
            return
        if self.profile_in_progress:
            self._stop_profile(stage=self.profiler_current_stage, finalize=False)
        self._start_profile(stage)
        self.profiler_current_stage = stage

    def _profile_batch_predicate(self, batch: ScheduleBatch):
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager.step(forward_mode=batch.forward_mode)
            return

        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                arriving_stage = ForwardMode.EXTEND
            elif batch.forward_mode.is_decode():
                arriving_stage = ForwardMode.DECODE
            elif batch.forward_mode.is_idle():
                arriving_stage = None
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")

            # A stage's profiler must only be recording while a batch of its
            # own mode runs. Pause any other stage that is currently open as
            # soon as a batch of a different mode arrives, regardless of
            # whether the arriving batch's own stage is done, pending, or
            # itself recording. Gating this pause on the arriving stage's
            # done flag (the PR #9 bug) left an open stage recording every
            # later batch of a mode whose quota had already been reached:
            # DECODE could finish first, and while EXTEND then waited on its
            # own sparse batches, every subsequent DECODE batch fell through
            # with no effect, leaving EXTEND's window open across them and
            # capturing their kernels into its trace.
            if (
                arriving_stage is not None
                and self.profiler_current_stage is not None
                and self.profiler_current_stage != arriving_stage
            ):
                self._stop_profile(stage=self.profiler_current_stage, finalize=False)
                self.profiler_current_stage = None

            if arriving_stage == ForwardMode.EXTEND:
                if not self.profiler_prefill_done:
                    self._ensure_profiling_stage(ForwardMode.EXTEND)
                    self.profiler_prefill_ct += 1
                    if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                        self._stop_profile(stage=ForwardMode.EXTEND)
                        self.profiler_prefill_done = True
                        self.profiler_current_stage = None
            elif arriving_stage == ForwardMode.DECODE:
                if not self.profiler_decode_done:
                    self._ensure_profiling_stage(ForwardMode.DECODE)
                    self.profiler_decode_ct += 1
                    if self.profiler_decode_ct > self.profiler_target_decode_ct:
                        self._stop_profile(stage=ForwardMode.DECODE)
                        self.profiler_decode_done = True
                        self.profiler_current_stage = None
        else:
            # Check profiler
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.get_forward_ct()
            ):
                self._stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.get_forward_ct()
            ):
                self._start_profile()

    def _profile(self, recv_req: ProfileReq):
        if recv_req.req_type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage or recv_req.start_step:
                return self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.profile_stages,
                )
            else:
                self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                )
                return self._start_profile()
        else:
            return self._stop_profile_request()

    def _stop_profile_request(self) -> ProfileReqOutput | None:
        """Handle an explicit /stop_profile call.

        For profile_by_stage, the per-stage bookkeeping in
        `_profile_batch_predicate` can already have torn down
        `profile_in_progress` (e.g. once the currently recorded stage hit
        its own target step count) while another armed stage hasn't run at
        all yet. That is not an error: unlike a plain start/stop session,
        "no stage currently recording" does not mean "nothing was ever
        started". So an explicit stop must stop and export whatever stage is
        still open, and otherwise succeed instead of raising, then end the
        whole by-stage session so the predicate does not re-arm profiling
        for a request that already asked to stop.
        """
        if not self.profile_by_stage:
            return self._stop_profile()

        if self.profile_in_progress:
            result = self._stop_profile(stage=self.profiler_current_stage)
        else:
            # A stage force-flushed by a switch to the other stage keeps its
            # parts unmerged until it resumes and finishes on its own. If it
            # never got to resume before this explicit stop, those parts
            # would otherwise be silently dropped by marking it done below
            # without ever exporting its canonical trace.
            pending_stages = [
                stage
                for stage, done in (
                    (ForwardMode.EXTEND, self.profiler_prefill_done),
                    (ForwardMode.DECODE, self.profiler_decode_done),
                )
                if not done and self._stage_trace_parts(stage)
            ]
            for stage in pending_stages:
                self._finalize_stage_trace(stage)

            if pending_stages:
                result = ProfileReqOutput(
                    success=True,
                    message=(
                        "Profiling stopped; exported pending trace(s) for "
                        f"{', '.join(stage.name for stage in pending_stages)}."
                    ),
                )
            elif self.profiler_prefill_done and self.profiler_decode_done:
                result = ProfileReqOutput(
                    success=True,
                    message="Profiling already finished for all stages.",
                )
            else:
                result = ProfileReqOutput(
                    success=True,
                    message="Profiling stopped; no stage was actively recording.",
                )

        # Mark both stages finished so the predicate does not re-arm
        # profiling on a later batch for a session that was explicitly
        # asked to stop. `self.profile_by_stage` itself is left as-is: the
        # next /start_profile call overwrites it from the new request
        # regardless, and leaving it set makes a repeated /stop_profile
        # call (also idempotent) take this same graceful branch rather than
        # falling through to the plain-profiling "not in progress" error.
        self.profiler_current_stage = None
        self.profiler_prefill_done = True
        self.profiler_decode_done = True
        return result
