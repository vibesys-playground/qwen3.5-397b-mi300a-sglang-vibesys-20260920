#!/usr/bin/env python3
"""Site and platform config loader for the multiturn task harness.

Machine-specific values (paths, scheduler identity, filesystem facts, launch
env/argv) live in committed TOML files under this directory instead of being
hardcoded in the harness scripts:

- ``platforms/<name>.toml``: everything true of any node with a given GPU
  and container image (launch env overlay, attention backend, KV pool pin,
  weight-loader recipe, backend-independent speculative-decoding and
  scheduler flags, TunableOp-tuned dense GEMM tile selection).
- ``sites/<name>.toml``: everything true of one place that runs the harness
  (filesystem paths, Slurm account/partition/EDF, Lustre striping, the
  I/O-dominated startup timeout). Each site names the platform it runs.

Selection is via the ``VIBESYS_SITE`` environment variable (default
``"example"``), with an optional ``VIBESYS_PLATFORM`` override of the
platform a site names; e.g. to test a site against a platform config that
is not (yet) its committed default.

Only the standard library is used. TOML parsing prefers the stdlib
``tomllib`` (Python 3.11+); on older interpreters (this harness's container
runs Python 3.10) it falls back to the ``tomli`` backport, and raises a
clear ``ImportError`` naming the file that needed parsing if neither is
importable.

CLI:

    python3 config/loader.py [--site NAME] [--shell | --json | KEY]

``--shell`` prints ``NAME=value`` lines (shell-quoted) suitable for
``eval "$(python3 config/loader.py --shell)"``. ``--json`` prints the full
merged site+platform config as JSON. ``KEY`` prints a single dotted value,
e.g. ``paths.tmpfs_root`` or ``platform.tp``. With none of the three, the
merged config is printed as JSON (the ``--json`` default).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
PLATFORMS_DIR = CONFIG_DIR / "platforms"
SITES_DIR = CONFIG_DIR / "sites"

DEFAULT_SITE_NAME = "example"


def _load_toml_module():
    try:
        import tomllib

        return tomllib
    except ImportError:
        try:
            import tomli

            return tomli
        except ImportError as exc:
            raise ImportError(
                "no TOML parser available: neither the stdlib `tomllib` "
                "(Python 3.11+) nor the `tomli` backport (needed on this "
                "harness's Python 3.10 container) could be imported"
            ) from exc


def _load_toml(path: Path) -> dict[str, Any]:
    toml_module = None
    try:
        toml_module = _load_toml_module()
    except ImportError as exc:
        raise ImportError(f"cannot parse {path}: {exc}") from exc
    with path.open("rb") as f:
        return toml_module.load(f)


def _check_keys(data: dict[str, Any], *, required: set[str], optional: set[str], context: str) -> None:
    allowed = required | optional
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}"
        )
    missing = required - set(data)
    if missing:
        raise ValueError(f"{context}: missing required key(s) {sorted(missing)}")


@dataclasses.dataclass(frozen=True, slots=True)
class HfLoaderConfig:
    disable_mmap: bool
    extra_config: str


@dataclasses.dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    """Backend-independent NEXTN/MTP speculative-decoding flags.

    Everything here is true of any node with this GPU and image; the draft
    model path is not (it depends on where a given site's checkpoints
    live), so it is not part of this table -- see ``_server.py``'s
    ``speculative_decode_args()``, which sources it from the site config's
    ``paths.model`` instead.
    """

    algorithm: str
    num_steps: int
    eagle_topk: int
    num_draft_tokens: int
    enable_linear_replayssm_spec: bool
    draft_load_format: str


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Backend-independent scheduler flags.

    Split the same way as ``SpeculativeConfig``: everything here is true of
    any node with this GPU and image, so it lives in the platform file
    rather than being site-specific. See ``_server.py``'s
    ``scheduler_args()`` for the argv this produces and
    ``VIBESYS_OVERLAP_SCHEDULE`` for the per-run override.
    """

    disable_overlap_schedule: bool


@dataclasses.dataclass(frozen=True, slots=True)
class TunableOpConfig:
    """PyTorch TunableOp-tuned dense GEMM tile selection.

    Image- and GPU-specific (the tuned CSV's own validator header pins
    PyTorch/ROCm/hipBLASLt/GPU-arch versions), so this lives in the
    platform file, not the site file. ``results_file`` is stored here
    already resolved to an absolute path by ``load_platform`` (the TOML
    value is relative to ``PLATFORMS_DIR``); see ``_server.py``'s
    ``build_launch_env()`` for how the three ``PYTORCH_TUNABLEOP_*`` env
    vars are derived from this field and ``VIBESYS_TUNABLEOP`` for the
    per-run kill switch.
    """

    enabled: bool
    results_file: str


@dataclasses.dataclass(frozen=True, slots=True)
class PrefillCudaGraphConfig:
    """Locks the prefill phase's CUDA-graph backend.

    Image- and workload-specific rather than a general SGLang default (see
    ``config/platforms/<platform>.toml``'s ``[prefill_cuda_graph]`` comment
    for the acceptance numbers and why ``breakable`` is the accepted
    backend on this platform), so this lives in the platform file. The
    bucket list is deliberately not a field here: it is the same 13-value
    capture set the gemm-pad-m warmup already targets
    (``GEMM_PAD_WARMUP_TOKEN_COUNTS`` in ``_server.py``), so
    ``prefill_cuda_graph_args()`` reuses that list rather than a second copy
    drifting out of sync with it.
    """

    backend: str


@dataclasses.dataclass(frozen=True, slots=True)
class PlatformConfig:
    name: str
    env: dict[str, str]
    env_defaults: dict[str, str]
    attention_backend: str
    page_size: int
    mem_fraction_static: str
    tp: int
    max_total_tokens: int
    hf_loader: HfLoaderConfig
    extra_args: list[str]
    speculative: SpeculativeConfig | None
    scheduler: SchedulerConfig | None
    tunableop: TunableOpConfig | None
    prefill_cuda_graph: PrefillCudaGraphConfig | None


@dataclasses.dataclass(frozen=True, slots=True)
class SitePaths:
    model: str
    sharded_artifact: str
    sharded_artifact_striped: str
    aiter_jit_dir: str
    hip_ext_dir: str
    tmpfs_root: str
    log_dir: str
    checkout: str
    # Optional: a draft-only `sharded_state` artifact for the speculative
    # decoding draft model (see `models/<...>-mtp-sharded-tp4` on the test cluster,
    # produced by `manual/draft-shard/save_draft_shard_v5.py`). When set,
    # `_server.py`'s `speculative_decode_args()` loads the draft from this
    # path with `--speculative-draft-load-format sharded_state` instead of
    # from `model` with the platform's `draft_load_format` -- the implicit
    # rule is "draft_model set implies sharded_state", since a
    # `sharded_state` draft path only ever holds one thing: this kind of
    # artifact. Absent (`None`) on any site that has not produced one yet,
    # in which case the draft falls back to the original checkpoint at
    # `model` with the platform's own `draft_load_format` (normally
    # ``"auto"``).
    draft_model: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SiteSlurm:
    account: str
    partition: str
    edf: str


@dataclasses.dataclass(frozen=True, slots=True)
class SiteLustre:
    stripe_count: int
    stripe_size: str


@dataclasses.dataclass(frozen=True, slots=True)
class SiteConfig:
    name: str
    platform: str
    paths: SitePaths
    slurm: SiteSlurm
    lustre: SiteLustre
    startup_timeout_s: int


def load_platform(name: str) -> PlatformConfig:
    """Load and validate ``platforms/<name>.toml``."""
    path = PLATFORMS_DIR / f"{name}.toml"
    if not path.is_file():
        available = sorted(p.stem for p in PLATFORMS_DIR.glob("*.toml"))
        raise ValueError(f"unknown platform {name!r}: no file {path}; available: {available}")
    data = _load_toml(path)

    top_required = {"attention_backend", "page_size", "mem_fraction_static", "tp", "max_total_tokens", "hf_loader"}
    top_optional = {
        "env",
        "env_defaults",
        "extra_args",
        "speculative",
        "scheduler",
        "tunableop",
        "prefill_cuda_graph",
    }
    _check_keys(data, required=top_required, optional=top_optional, context=str(path))

    hf_loader_data = data["hf_loader"]
    _check_keys(
        hf_loader_data,
        required={"disable_mmap", "extra_config"},
        optional=set(),
        context=f"{path} [hf_loader]",
    )

    speculative_data = data.get("speculative")
    speculative: SpeculativeConfig | None = None
    if speculative_data is not None:
        _check_keys(
            speculative_data,
            required={
                "algorithm",
                "num_steps",
                "eagle_topk",
                "num_draft_tokens",
                "enable_linear_replayssm_spec",
                "draft_load_format",
            },
            optional=set(),
            context=f"{path} [speculative]",
        )
        speculative = SpeculativeConfig(**speculative_data)

    scheduler_data = data.get("scheduler")
    scheduler: SchedulerConfig | None = None
    if scheduler_data is not None:
        _check_keys(
            scheduler_data,
            required={"disable_overlap_schedule"},
            optional=set(),
            context=f"{path} [scheduler]",
        )
        scheduler = SchedulerConfig(**scheduler_data)

    tunableop_data = data.get("tunableop")
    tunableop: TunableOpConfig | None = None
    if tunableop_data is not None:
        _check_keys(
            tunableop_data,
            required={"enabled", "results_file"},
            optional=set(),
            context=f"{path} [tunableop]",
        )
        tunableop_enabled = tunableop_data["enabled"]
        results_file = str(PLATFORMS_DIR / tunableop_data["results_file"])
        if tunableop_enabled and not Path(results_file).is_file():
            raise ValueError(
                f"{path} [tunableop]: enabled = true but results_file "
                f"{results_file!r} does not exist"
            )
        tunableop = TunableOpConfig(enabled=tunableop_enabled, results_file=results_file)

    prefill_cuda_graph_data = data.get("prefill_cuda_graph")
    prefill_cuda_graph: PrefillCudaGraphConfig | None = None
    if prefill_cuda_graph_data is not None:
        _check_keys(
            prefill_cuda_graph_data,
            required={"backend"},
            optional=set(),
            context=f"{path} [prefill_cuda_graph]",
        )
        prefill_cuda_graph = PrefillCudaGraphConfig(backend=prefill_cuda_graph_data["backend"])

    env = dict(data.get("env", {}))
    env_defaults = dict(data.get("env_defaults", {}))
    for table_name, table in (("env", env), ("env_defaults", env_defaults)):
        for key, value in table.items():
            if not isinstance(value, str):
                raise ValueError(f"{path} [{table_name}].{key}: expected a string, got {type(value).__name__}")

    extra_args = data.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise ValueError(f"{path} extra_args: expected a list of strings, got {extra_args!r}")

    return PlatformConfig(
        name=name,
        env=env,
        env_defaults=env_defaults,
        attention_backend=data["attention_backend"],
        page_size=data["page_size"],
        mem_fraction_static=data["mem_fraction_static"],
        tp=data["tp"],
        max_total_tokens=data["max_total_tokens"],
        hf_loader=HfLoaderConfig(
            disable_mmap=hf_loader_data["disable_mmap"],
            extra_config=hf_loader_data["extra_config"],
        ),
        extra_args=list(extra_args),
        speculative=speculative,
        scheduler=scheduler,
        tunableop=tunableop,
        prefill_cuda_graph=prefill_cuda_graph,
    )


def load_site(name: str | None = None) -> SiteConfig:
    """Load and validate ``sites/<name>.toml``.

    ``name`` defaults to the ``VIBESYS_SITE`` environment variable, itself
    defaulting to ``"example"``. The site's platform name is overridden by
    ``VIBESYS_PLATFORM`` when that variable is set.
    """
    if name is None:
        name = os.environ.get("VIBESYS_SITE", DEFAULT_SITE_NAME)
    path = SITES_DIR / f"{name}.toml"
    if not path.is_file():
        available = sorted(p.stem for p in SITES_DIR.glob("*.toml"))
        raise ValueError(f"unknown site {name!r}: no file {path}; available: {available}")
    data = _load_toml(path)

    _check_keys(
        data,
        required={"platform", "paths", "slurm", "lustre", "startup_timeout_s"},
        optional=set(),
        context=str(path),
    )

    paths_data = data["paths"]
    _check_keys(
        paths_data,
        required={
            "model",
            "sharded_artifact",
            "sharded_artifact_striped",
            "aiter_jit_dir",
            "hip_ext_dir",
            "tmpfs_root",
            "log_dir",
            "checkout",
        },
        optional={"draft_model"},
        context=f"{path} [paths]",
    )

    slurm_data = data["slurm"]
    _check_keys(
        slurm_data,
        required={"account", "partition", "edf"},
        optional=set(),
        context=f"{path} [slurm]",
    )

    lustre_data = data["lustre"]
    _check_keys(
        lustre_data,
        required={"stripe_count", "stripe_size"},
        optional=set(),
        context=f"{path} [lustre]",
    )

    platform_name = os.environ.get("VIBESYS_PLATFORM") or data["platform"]

    return SiteConfig(
        name=name,
        platform=platform_name,
        paths=SitePaths(**paths_data),
        slurm=SiteSlurm(**slurm_data),
        lustre=SiteLustre(**lustre_data),
        startup_timeout_s=data["startup_timeout_s"],
    )


def _merged_config(site: SiteConfig, platform: PlatformConfig) -> dict[str, Any]:
    return {
        "site_name": site.name,
        "platform_name": site.platform,
        "paths": dataclasses.asdict(site.paths),
        "slurm": dataclasses.asdict(site.slurm),
        "lustre": dataclasses.asdict(site.lustre),
        "startup_timeout_s": site.startup_timeout_s,
        "platform": {
            "tp": platform.tp,
            "attention_backend": platform.attention_backend,
            "page_size": platform.page_size,
            "mem_fraction_static": platform.mem_fraction_static,
            "max_total_tokens": platform.max_total_tokens,
            "env": dict(platform.env),
            "env_defaults": dict(platform.env_defaults),
            "hf_loader": dataclasses.asdict(platform.hf_loader),
            "extra_args": list(platform.extra_args),
            "speculative": dataclasses.asdict(platform.speculative) if platform.speculative else None,
            "scheduler": dataclasses.asdict(platform.scheduler) if platform.scheduler else None,
            "tunableop": dataclasses.asdict(platform.tunableop) if platform.tunableop else None,
            "prefill_cuda_graph": (
                dataclasses.asdict(platform.prefill_cuda_graph) if platform.prefill_cuda_graph else None
            ),
        },
    }


def _lookup_dotted(config: dict[str, Any], key: str) -> Any:
    node: Any = config
    parts = key.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no such key {key!r} (failed at {'.'.join(parts[: i + 1])!r})")
        node = node[part]
    return node


def _shell_quote(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return shlex.quote(str(value))


def _shell_lines(site: SiteConfig, platform: PlatformConfig) -> list[str]:
    pairs: list[tuple[str, Any]] = [
        ("SITE_NAME", site.name),
        ("SITE_MODEL_PATH", site.paths.model),
        ("SITE_DRAFT_MODEL_PATH", site.paths.draft_model or ""),
        ("SITE_SHARDED_ARTIFACT", site.paths.sharded_artifact),
        ("SITE_SHARDED_ARTIFACT_STRIPED", site.paths.sharded_artifact_striped),
        ("SITE_AITER_JIT_DIR", site.paths.aiter_jit_dir),
        ("SITE_HIP_EXT_DIR", site.paths.hip_ext_dir),
        ("SITE_TMPFS_ROOT", site.paths.tmpfs_root),
        ("SITE_LOG_DIR", site.paths.log_dir),
        ("SITE_CHECKOUT", site.paths.checkout),
        ("SITE_SLURM_ACCOUNT", site.slurm.account),
        ("SITE_SLURM_PARTITION", site.slurm.partition),
        ("SITE_SLURM_EDF", site.slurm.edf),
        ("SITE_LUSTRE_STRIPE_COUNT", site.lustre.stripe_count),
        ("SITE_LUSTRE_STRIPE_SIZE", site.lustre.stripe_size),
        ("SITE_STARTUP_TIMEOUT_S", site.startup_timeout_s),
        ("PLATFORM_NAME", platform.name),
        ("PLATFORM_TP", platform.tp),
        ("PLATFORM_ATTENTION_BACKEND", platform.attention_backend),
        ("PLATFORM_PAGE_SIZE", platform.page_size),
        ("PLATFORM_MEM_FRACTION_STATIC", platform.mem_fraction_static),
        ("PLATFORM_MAX_TOTAL_TOKENS", platform.max_total_tokens),
        ("PLATFORM_HF_LOADER_DISABLE_MMAP", platform.hf_loader.disable_mmap),
        ("PLATFORM_HF_LOADER_EXTRA_CONFIG", platform.hf_loader.extra_config),
        ("PLATFORM_EXTRA_ARGS", " ".join(platform.extra_args)),
    ]
    return [f"{name}={_shell_quote(value)}" for name, value in pairs]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default=None, help="Site name (default: $VIBESYS_SITE or 'example').")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shell", action="store_true", help="Print SITE_*/PLATFORM_* shell assignments.")
    mode.add_argument("--json", action="store_true", help="Print the merged config as JSON.")
    parser.add_argument("key", nargs="?", default=None, help="Print one dotted key, e.g. paths.tmpfs_root.")
    args = parser.parse_args(argv)

    if args.key and (args.shell or args.json):
        parser.error("KEY cannot be combined with --shell or --json")

    try:
        site = load_site(args.site)
        platform = load_platform(site.platform)
    except (ValueError, ImportError) as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        return 1

    if args.shell:
        print("\n".join(_shell_lines(site, platform)))
        return 0

    merged = _merged_config(site, platform)

    if args.key:
        try:
            value = _lookup_dotted(merged, args.key)
        except KeyError as exc:
            print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
            return 1
        print(value if isinstance(value, str) else json.dumps(value))
        return 0

    print(json.dumps(merged, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
