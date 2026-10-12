"""Host execution through the SDK's Unix local backend (Linux, macOS, WSL)."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.sandbox.entries import LocalDir


if TYPE_CHECKING:
    from agents.sandbox.manifest import Manifest


def check_native_platform() -> None:
    if sys.platform == "win32":
        raise RuntimeError(
            "Native Strix requires Linux, macOS, or WSL2. On Windows, install and "
            "run Strix and OpenCode inside WSL2; Docker is not required."
        )


def session_workspace(session: Any) -> str:
    """The actual workspace, including test/custom backends without SDK state."""
    state = getattr(session, "state", None)
    manifest = getattr(state, "manifest", None)
    return str(getattr(manifest, "root", "/workspace"))


def workspace_path(session: Any, path: str) -> Path:
    """Resolve an orchestrator path against the session's actual workspace."""
    if path == "/workspace":
        return Path(session_workspace(session))
    if path.startswith("/workspace/"):
        return Path(session_workspace(session)) / path.removeprefix("/workspace/")
    return Path(path)


async def native_backend(
    *,
    image: str,
    manifest: Manifest,
    exposed_ports: tuple[int, ...],
    bind_mounts: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    """Start a local session; the caller supplies a persistent per-run root.

    Source trees are copied by manifest materialization, never host mounts.
    No container entrypoint, tool installation, or proxy process is assumed.
    """
    del image, bind_mounts, exposed_ports
    check_native_platform()
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    class NativeClient(UnixLocalSandboxClient):
        async def delete(self, session: Any) -> Any:
            # SDK delete does not stop PTYs for caller-owned roots. Always stop
            # first and retain this run's workspace for review and resume.
            await session.stop()
            return await super().delete(session)

    root = Path(manifest.root)
    sources_file = root.parent / ".state" / "native-sources.json"
    sources = {
        str(key): str(entry.src.resolve())
        for key, entry in manifest.entries.items()
        if isinstance(entry, LocalDir) and entry.src is not None
    }
    previous: dict[str, str] = json.loads(sources_file.read_text()) if sources_file.exists() else {}
    if sources_file.exists() and previous != sources:
        raise ValueError("Native source paths changed; use a new run name for this scope")
    # Reconnecting a run must retain edits to its source copies. Only skip
    # materialization after a successful prior start recorded the same sources.
    if sources_file.exists():
        manifest = manifest.model_copy(
            update={
                "entries": {
                    key: entry
                    for key, entry in manifest.entries.items()
                    if not (root / str(key)).exists()
                },
            }
        )
    client = NativeClient()
    session = await client.create(manifest=manifest)
    try:
        await session.start()
        sources_file.parent.mkdir(parents=True, exist_ok=True)
        sources_file.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    except BaseException:
        with contextlib.suppress(Exception):
            await client.delete(session)
        raise
    return client, session
