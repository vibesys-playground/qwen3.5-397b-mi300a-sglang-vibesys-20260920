"""Start and stop the SGLang server for the benchmark and accuracy checker.

Same interface as the bespoke-engine bundle's ``launcher.py``, so
``benchmark/run.py`` and ``accuracy_checker/checker.py`` are shared verbatim
between the two tasks. Booting is delegated to ``_server.py``, which builds the
SGLang launch command from ``config/`` (site paths, platform flags, TunableOp
tables, speculative decoding).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _server  # noqa: E402

DEFAULT_HOST = _server.DEFAULT_HOST
DEFAULT_PORT = _server.DEFAULT_PORT
STARTUP_TIMEOUT_SECONDS = _server.STARTUP_TIMEOUT_SECONDS


def resolve_model_path(*, required: bool = True) -> str | None:
    """Return ``$MODEL_PATH`` or the site default; ``None`` when not required and unset."""
    if not required and not os.environ.get("MODEL_PATH"):
        return None
    return _server.resolve_model_path()


@contextlib.asynccontextmanager
async def server_endpoint(
    *,
    base_url: str | None,
    workspace: Path,
    model_path: str | None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_path: Path,
    startup_timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
):
    """Yield a base URL: reuse ``base_url`` if given, else boot SGLang and tear it down."""
    async with _server.server_endpoint(
        base_url=base_url,
        workspace=workspace,
        model_path=model_path if model_path is not None else _server.resolve_model_path(),
        host=host,
        port=port,
        log_path=log_path,
        startup_timeout_seconds=startup_timeout_seconds,
    ) as url:
        yield url
