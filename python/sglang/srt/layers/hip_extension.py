# SPDX-License-Identifier: Apache-2.0
"""Generic JIT loader for sglang's ad hoc HIP source extensions.

A thin, cached wrapper around ``torch.utils.cpp_extension.load``: each
distinct HIP kernel module a caller adds (the skinny GEMM kernel, say) calls
``load_hip_extension`` with its own module name and source list, and gets
back the compiled extension, built once per process and cached by name.

The build directory is ``SGLANG_HIP_EXT_DIR`` (see ``environ.py``) when set,
otherwise ``torch.utils.cpp_extension``'s own default (a name-keyed
subdirectory of its extensions cache). A harness that wants to avoid paying
the compile on every launch should point ``SGLANG_HIP_EXT_DIR`` at a
persistent, pre-warmed directory, the same role ``AITER_JIT_DIR`` plays for
aiter's own JIT cache (see
``.vibesys/tasks/multiturn/config/sites/example.toml``'s ``paths.hip_ext_dir``).

When ``SGLANG_HIP_EXT_DIR`` is set, the build is keyed on source *content*
rather than source *path*: a harness that stages each launch into a fresh
scratch directory (so the sources' absolute paths and mtimes differ every
time even though their content doesn't) would otherwise defeat
``cpp_extension.load``'s own path/mtime-based staleness check and pay a full
rebuild on every launch. Instead, the sources are hashed and copied (once)
into a hash-named subdirectory of ``SGLANG_HIP_EXT_DIR``, and that stable
copy -- not the caller's original, per-launch path -- is what gets built.
Once a given (name, sources, flags, torch/HIP version) combination has been
built once under a warm ``SGLANG_HIP_EXT_DIR``, later launches see the copy
already present with unchanged content and mtime and ``cpp_extension.load``
returns without invoking the compiler.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from typing import Any, Sequence

from sglang.srt.environ import envs

_lock = threading.Lock()
_extensions: dict[str, Any] = {}

# Fixed mtime (Unix epoch) stamped on every source file copied into a
# hash-named build directory. The copy step only ever writes bytes it has
# already verified (by content comparison) are either new or different, so
# stamping a constant mtime rather than "now" keeps ninja's mtime-based
# staleness check stable across repeated copies of identical content --
# without it, every relaunch would give the copy a fresh "now" mtime and
# ninja would consider the source newer than its already-built object file.
_FIXED_MTIME_EPOCH = 0


def _torch_hip_version_string() -> str:
    """A string identifying the torch build and HIP toolchain in use.

    Imported lazily (module import time should stay torch-free, matching
    ``environ.py``'s own convention) and folded into the content hash so a
    torch/ROCm upgrade invalidates the cache automatically instead of
    silently reusing a binary built against a different ABI/toolchain.
    """
    import torch

    return f"{torch.__version__}|{getattr(torch.version, 'hip', None)}"


def _content_hash(
    sources: Sequence[str],
    extra_cflags: Sequence[str] | None,
    extra_cuda_cflags: Sequence[str] | None,
) -> str:
    """sha256 over the sorted sources' contents, the extra cflags, and the
    torch/HIP version string.

    Sorting the source list (by path) before hashing means the hash depends
    only on *which* file contents are included and not on the order the
    caller happened to list them in.
    """
    digest = hashlib.sha256()
    for path in sorted(sources):
        with open(path, "rb") as f:
            digest.update(f.read())
        digest.update(b"\0")
    for flag in extra_cflags or ():
        digest.update(flag.encode())
        digest.update(b"\0")
    for flag in extra_cuda_cflags or ():
        digest.update(flag.encode())
        digest.update(b"\0")
    digest.update(_torch_hip_version_string().encode())
    return digest.hexdigest()


def _copy_source_if_stale(src: str, dest_dir: str) -> str:
    """Copy ``src`` into ``dest_dir`` (under its basename) unless a file of
    identical content is already there; return the copy's path.

    Concurrency-safe across processes (e.g. several TP ranks all loading the
    same extension at once): the new content is written to a temp file in
    ``dest_dir`` first and moved into place with ``os.replace``, which is
    atomic on a POSIX filesystem, so a concurrent reader of ``dest`` never
    observes a partially written file. Racing writers all write the same
    content (``src``'s), so whichever one's ``os.replace`` lands last still
    leaves ``dest`` correct.
    """
    basename = os.path.basename(src)
    dest = os.path.join(dest_dir, basename)
    with open(src, "rb") as f:
        content = f.read()
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            if f.read() == content:
                return dest
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=f".{basename}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.utime(tmp_path, (_FIXED_MTIME_EPOCH, _FIXED_MTIME_EPOCH))
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return dest


def load_hip_extension(
    name: str,
    sources: Sequence[str],
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    verbose: bool = False,
) -> Any:
    """Build (once per process) and return a HIP source extension.

    Safe to call repeatedly and concurrently: the first call for a given
    ``name`` compiles and caches the module; later calls (any thread) return
    the cached module without re-invoking ``torch.utils.cpp_extension.load``.

    When ``SGLANG_HIP_EXT_DIR`` is set, ``sources`` are first copied (see
    ``_copy_source_if_stale``) into a subdirectory named
    ``<name>-<hash12>``, where the hash covers the sources' content, the
    extra flags, and the torch/HIP version (see ``_content_hash``); the
    copies, not the caller's original paths, are what gets built. This
    keeps the build keyed on content rather than the caller's (possibly
    per-launch, e.g. staged into a fresh tmpfs path every time) source path,
    so a warm cache is recognized as such across launches whose staged
    source paths differ but whose content doesn't.

    Args:
        name: Extension module name, also used as the cache key. Callers
            should pick a name unique across sglang's HIP extensions (the
            underlying loader also uses it to name the build subdirectory
            when ``SGLANG_HIP_EXT_DIR`` is unset).
        sources: Paths to the extension's HIP/C++ source files.
        extra_cflags: Extra host-compiler flags, e.g. ``["-O3"]``.
        extra_cuda_cflags: Extra device-compiler flags, e.g.
            ``["-O3", "--offload-arch=gfx942"]``.
        verbose: Forwarded to ``torch.utils.cpp_extension.load``.
    """
    if name in _extensions:
        return _extensions[name]
    with _lock:
        if name not in _extensions:
            from torch.utils.cpp_extension import load

            build_directory = envs.SGLANG_HIP_EXT_DIR.get()
            load_sources = list(sources)
            if build_directory is not None:
                content_hash = _content_hash(sources, extra_cflags, extra_cuda_cflags)
                build_directory = os.path.join(
                    build_directory, f"{name}-{content_hash[:12]}"
                )
                os.makedirs(build_directory, exist_ok=True)
                load_sources = [
                    _copy_source_if_stale(src, build_directory) for src in load_sources
                ]
            _extensions[name] = load(
                name=name,
                sources=load_sources,
                extra_cflags=list(extra_cflags) if extra_cflags else None,
                extra_cuda_cflags=(
                    list(extra_cuda_cflags) if extra_cuda_cflags else None
                ),
                build_directory=build_directory,
                verbose=verbose,
            )
    return _extensions[name]
