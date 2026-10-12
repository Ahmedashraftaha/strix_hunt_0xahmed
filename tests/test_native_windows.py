from __future__ import annotations

import asyncio
import base64
import io
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agents.sandbox.capabilities import Filesystem
from agents.sandbox.capabilities.tools.shell_tool import ExecCommandArgs
from agents.sandbox.entries import File, LocalDir
from agents.sandbox.errors import InvalidManifestPathError
from agents.sandbox.manifest import Manifest
from agents.sandbox.snapshot import resolve_snapshot

from strix.runtime.native_windows import (
    WindowsExecCommandTool,
    WindowsNativeClient,
    WindowsNativeSession,
    WindowsSessionState,
    powershell_command,
)
from strix.runtime.session_manager import build_extra_file_archive, stage_extra_files


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def windows_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[WindowsNativeSession]:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    state = WindowsSessionState(
        manifest=Manifest(root=str(tmp_path / "workspace with spaces")),
        snapshot=resolve_snapshot(None, "windows-test"),
    )
    session = WindowsNativeSession(state)
    await session.start()
    try:
        yield session
    finally:
        await session.stop()


def test_powershell_quoting_unicode_and_profile_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("strix.runtime.native_windows.shutil.which", lambda _name: "powershell.exe")
    script = "Write-Output 'مرحبا'; Write-Output \"a`$b\""
    argv = powershell_command(script)
    assert argv[:5] == [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
    ]
    decoded = base64.b64decode(argv[-1]).decode("utf-16-le")
    assert script in decoded
    assert "UTF8Encoding" in decoded
    assert "LASTEXITCODE" in decoded


async def test_windows_files_source_copy_and_extra_files(
    windows_session: WindowsNativeSession,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "original.txt").write_text("original", encoding="utf-8")
    client = WindowsNativeClient()
    wrapper = await client.create(
        manifest=Manifest(
            root=str(tmp_path / "copied workspace"),
            entries={"repo": LocalDir(src=source), "hello.txt": File(content="مرحبا".encode())},
        )
    )
    try:
        await wrapper.start()
        archive = build_extra_file_archive(
            [{"workspace_path": "/workspace/notes/hello.txt", "content": "context"}]
        )
        assert archive is not None
        await stage_extra_files(wrapper, archive)
        await wrapper.write(Path("repo/original.txt"), io.BytesIO(b"changed"))
        assert (source / "original.txt").read_text() == "original"
        handle = await wrapper.read(Path("notes/hello.txt"))
        with handle:
            assert handle.read() == b"context"
        # The instrumented wrapper must use Windows paths too (image/patch tools).
        root = Path(wrapper.state.manifest.root)
        assert wrapper._workspace_path_policy().relative_path(root / "hello.txt") == Path(
            "hello.txt"
        )
        assert Filesystem(session=wrapper).tools()
    finally:
        await client.delete(wrapper)
    assert (root / "repo/original.txt").read_text() == "changed"
    with pytest.raises(InvalidManifestPathError):
        await windows_session.write(Path("../escape.txt"), io.BytesIO(b"x"))
    with pytest.raises(InvalidManifestPathError):
        windows_session.normalize_path("C:drive-relative.txt")


async def test_source_containing_workspace_does_not_copy_itself(
    windows_session: WindowsNativeSession,
    tmp_path: Path,
) -> None:
    del windows_session
    source = tmp_path / "project"
    source.mkdir()
    (source / "code.txt").write_text("source")
    root = source / "strix_runs" / "test" / "workspace"
    client = WindowsNativeClient()
    session = await client.create(
        manifest=Manifest(root=str(root), entries={"repo": LocalDir(src=source)})
    )
    try:
        await session.start()
        assert (root / "repo/code.txt").read_text() == "source"
        assert not (root / "repo/strix_runs").exists()
    finally:
        await client.delete(session)


async def test_windows_snapshot_rejects_traversal_and_links(
    windows_session: WindowsNativeSession,
) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    archive.seek(0)
    with pytest.raises(InvalidManifestPathError):
        await windows_session.hydrate_workspace(archive)


async def test_windows_pipe_sessions_stream_accept_input_and_close(
    windows_session: WindowsNativeSession,
) -> None:
    update = await windows_session.pty_exec_start(
        sys.executable,
        "-u",
        "-c",
        "import sys; print('ready',flush=True); print(sys.stdin.readline().strip(),flush=True)",
        shell=False,
        yield_time_s=0.25,
    )
    assert update.process_id is not None
    result = await windows_session.pty_write_stdin(
        session_id=update.process_id,
        chars="مرحبا\n",
        yield_time_s=2,
    )
    assert "مرحبا" in result.output.decode()
    assert result.exit_code == 0
    assert result.process_id is None
    assert not windows_session._processes


async def test_windows_timeout_and_cancel_stop_processes(
    windows_session: WindowsNativeSession,
) -> None:
    with pytest.raises(TimeoutError):
        await windows_session.exec(
            sys.executable, "-c", "import time; time.sleep(30)", shell=False, timeout=0.1
        )
    update = await windows_session.pty_exec_start(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        shell=False,
        yield_time_s=0.25,
    )
    assert update.process_id is not None
    process = windows_session._processes[update.process_id].process
    result = await windows_session.pty_write_stdin(
        session_id=update.process_id, chars="\x03", yield_time_s=2
    )
    assert result.process_id is None
    assert process.returncode is not None
    assert not windows_session._processes


async def test_windows_shutdown_cleans_all_sessions_and_rejects_tty(
    windows_session: WindowsNativeSession,
) -> None:
    with pytest.raises(ValueError, match="tty=false"):
        await windows_session.pty_exec_start("ignored", tty=True)
    processes = []
    for _ in range(2):
        update = await windows_session.pty_exec_start(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            shell=False,
            yield_time_s=0.25,
        )
        assert update.process_id is not None
        processes.append(windows_session._processes[update.process_id].process)
    await windows_session.stop()
    assert not windows_session._processes
    assert all(process.returncode is not None for process in processes)


async def test_windows_cancelled_start_does_not_leave_a_session(
    windows_session: WindowsNativeSession,
) -> None:
    task = asyncio.create_task(
        windows_session.pty_exec_start(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            shell=False,
            yield_time_s=10,
        )
    )
    for _ in range(100):
        if windows_session._processes:
            break
        await asyncio.sleep(0.01)
    assert windows_session._processes
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not windows_session._processes


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows PowerShell")
async def test_real_windows_powershell_unicode_exit_codes_and_workdir(
    windows_session: WindowsNativeSession,
) -> None:
    result = await windows_session.exec("Write-Output 'مرحبا'")
    assert result.ok()
    assert "مرحبا" in result.stdout.decode()
    assert (await windows_session.exec("cmd /c exit 7")).exit_code == 7
    assert (await windows_session.exec("throw 'failure'")).exit_code != 0
    directory = Path(windows_session.state.manifest.root) / "a 'quoted' directory"
    directory.mkdir()
    tool = WindowsExecCommandTool(session=windows_session)
    output = await tool.run(ExecCommandArgs(cmd="(Get-Location).Path", workdir=str(directory)))
    assert str(directory) in output
