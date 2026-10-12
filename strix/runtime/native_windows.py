"""Native Windows host sessions, with PowerShell and pollable subprocess pipes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from agents.sandbox.capabilities import Shell
from agents.sandbox.capabilities.shell import ShellToolSet
from agents.sandbox.capabilities.tools import ExecCommandTool, WriteStdinTool
from agents.sandbox.entries import BaseEntry, Dir, File, LocalDir
from agents.sandbox.errors import InvalidManifestPathError, PtySessionNotFoundError
from agents.sandbox.files import EntryKind, FileEntry
from agents.sandbox.session import BaseSandboxSession, SandboxSession, SandboxSessionState
from agents.sandbox.session.pty_output import collect_pty_output
from agents.sandbox.session.pty_types import (
    PTY_PROCESSES_MAX,
    PtyExecUpdate,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
)
from agents.sandbox.session.sandbox_client import BaseSandboxClient
from agents.sandbox.snapshot import resolve_snapshot
from agents.sandbox.types import ExecResult, ExposedPortEndpoint, Permissions, User
from agents.sandbox.workspace_paths import WorkspacePathPolicy


if TYPE_CHECKING:
    from agents.sandbox.capabilities.tools.shell_tool import ExecCommandArgs
    from agents.sandbox.manifest import Manifest
    from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec
    from agents.tool import Tool


_MAX_BUFFER_BYTES = 4 * 1024 * 1024
_READ_CHUNK = 16 * 1024


def resolve_powershell(shell: str | None = None) -> str:
    candidates = [shell] if shell else ["pwsh", "powershell"]
    for candidate in candidates:
        if resolved := shutil.which(candidate):
            return resolved
    raise RuntimeError("PowerShell not found. Install PowerShell 7 or add powershell.exe to PATH.")


def powershell_command(script: str, *, shell: str | None = None) -> list[str]:
    """Use encoded scripts to preserve Windows quoting and Unicode on PS 5.1/7."""
    executable = resolve_powershell(shell)
    name = Path(executable).stem.lower()
    if name not in {"pwsh", "powershell"}:
        raise ValueError("Native Windows commands require pwsh or powershell.exe")
    wrapped = (
        "$ErrorActionPreference = 'Stop'; "
        "$ProgressPreference = 'SilentlyContinue'; "
        "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$global:LASTEXITCODE = 0; try { & {\n"
        + script
        + "\n}; $ok = $?; $code = $global:LASTEXITCODE; "
        "if ($code -ne 0) { exit $code }; if (-not $ok) { exit 1 }; exit 0 "
        "} catch { [Console]::Error.WriteLine($_.ToString()); exit 1 }"
    )
    encoded = base64.b64encode(wrapped.encode("utf-16-le")).decode("ascii")
    return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


class WindowsWorkspacePolicy(WorkspacePathPolicy):
    """Resolve host paths, including drive letters, UNC, spaces, and symlinks."""

    def normalize_path(
        self,
        path: str | PurePath,
        *,
        for_write: bool = False,
        resolve_symlinks: bool = False,
    ) -> Path:
        del for_write, resolve_symlinks
        raw = str(path)
        native = Path(raw)
        windows = PureWindowsPath(raw)
        if windows.drive and not windows.is_absolute():
            raise InvalidManifestPathError(rel=raw, reason="escape_root")
        root = self._root.resolve()
        candidate = (native if native.is_absolute() else root / native).resolve()
        if not candidate.is_relative_to(root):
            raise InvalidManifestPathError(rel=raw, reason="escape_root")
        # Windows alternate data streams and drive-relative paths are not files
        # this workspace API should silently reinterpret.
        if any(":" in part for part in candidate.relative_to(root).parts):
            raise InvalidManifestPathError(rel=raw, reason="escape_root")
        return candidate

    def absolute_workspace_path(self, path: str | PurePath) -> Path:
        return self.normalize_path(path)

    def relative_path(self, path: str | PurePath) -> Path:
        return self.normalize_path(path).relative_to(self._root.resolve())


class WindowsSessionState(SandboxSessionState):
    type: Literal["windows_native"] = "windows_native"  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass
class ProcessEntry:
    process: asyncio.subprocess.Process
    chunks: deque[bytes] = field(default_factory=deque[bytes])
    output_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    notify: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: list[asyncio.Task[None]] = field(default_factory=list[asyncio.Task[None]])
    waiter: asyncio.Task[None] | None = None
    buffered_bytes: int = 0
    dropped_bytes: int = 0


class WindowsNativeSession(BaseSandboxSession):
    state: SandboxSessionState

    def __init__(self, state: WindowsSessionState) -> None:
        self.state = state
        self._running = False
        self._processes: dict[int, ProcessEntry] = {}
        self._process_lock = asyncio.Lock()

    async def start(self) -> None:
        root = Path(self.state.manifest.root)
        if not root.is_absolute() or root == root.parent:
            raise ValueError("Native Windows workspace must be an absolute directory below a root")
        if self.state.manifest.users or self.state.manifest.groups:
            raise ValueError("Native Windows does not provision host users or groups")
        root.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._materialize, self.state.manifest.entries)
        self._running = True
        self.state.workspace_root_ready = True

    def _materialize(
        self, entries: dict[str | Path, BaseEntry], parent: Path | None = None
    ) -> None:
        root = parent or Path(self.state.manifest.root)
        for name, entry in entries.items():
            destination = self.normalize_path(root / str(name))
            if isinstance(entry, LocalDir) and entry.src is not None:

                def ignore_workspace(
                    directory: str,
                    names: list[str],
                    workspace: Path = destination,
                ) -> list[str]:
                    # A local target can contain strix_runs itself. Never recurse
                    # into the workspace currently being created underneath it.
                    return [
                        name
                        for name in names
                        if workspace.is_relative_to((Path(directory) / name).resolve())
                    ]

                shutil.copytree(entry.src, destination, dirs_exist_ok=True, ignore=ignore_workspace)
            elif isinstance(entry, Dir):
                destination.mkdir(parents=True, exist_ok=True)
                self._materialize(entry.children, destination)
            elif isinstance(entry, File):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(entry.content)
            else:
                raise ValueError(f"Unsupported Windows manifest entry: {entry.type}")

    async def running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        await self.pty_terminate_all()
        self._running = False

    async def shutdown(self) -> None:
        await self.stop()

    def supports_pty(self) -> bool:
        # SDK uses this flag to enable yielding/polling; transport is pipes,
        # not ConPTY. tty=true is rejected explicitly by pty_exec_start.
        return True

    def _workspace_path_policy(self) -> WorkspacePathPolicy:
        return WindowsWorkspacePolicy(root=self.state.manifest.root)

    def _prepare_exec_command(
        self,
        *command: str | Path,
        shell: bool | list[str],
        user: str | User | None,
    ) -> list[str]:
        if user is not None:
            raise ValueError("Native Windows runs as the current user")
        parts = [str(part) for part in command]
        if not parts:
            raise ValueError("A command is required")
        if shell is False:
            return parts
        # A caller that passes individual argv values gets PowerShell literal
        # quoting; a single string is deliberately executable script content.
        script = (
            parts[0]
            if len(parts) == 1
            else "& " + " ".join("'" + part.replace("'", "''") + "'" for part in parts)
        )
        return powershell_command(script, shell=shell[0] if isinstance(shell, list) else None)

    async def _spawn(self, argv: list[str]) -> asyncio.subprocess.Process:
        if not self._running:
            raise RuntimeError("Native Windows session is not running")
        environment = os.environ.copy()
        environment.update(await self.state.manifest.environment.resolve())
        environment["PYTHONIOENCODING"] = "utf-8"
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            options["start_new_session"] = True
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.state.manifest.root,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **options,
        )

    async def _exec_internal(
        self, *command: str | Path, timeout: float | None = None
    ) -> ExecResult:
        process = await self._spawn([str(part) for part in command])
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except BaseException:
            await self._terminate(process)
            raise
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=process.returncode or 0)

    async def _pump(self, entry: ProcessEntry, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while chunk := await stream.read(_READ_CHUNK):
            async with entry.output_lock:
                # collect_pty_output consumes the deque, so re-count before each
                # append instead of keeping stale accounting across polls.
                entry.buffered_bytes = sum(len(part) for part in entry.chunks)
                entry.chunks.append(chunk)
                entry.buffered_bytes += len(chunk)
                while entry.buffered_bytes > _MAX_BUFFER_BYTES:
                    removed = entry.chunks.popleft()
                    entry.buffered_bytes -= len(removed)
                    entry.dropped_bytes += len(removed)
            entry.notify.set()

    async def _wait(self, entry: ProcessEntry) -> None:
        try:
            await entry.process.wait()
            await asyncio.gather(*entry.tasks)
        finally:
            entry.notify.set()

    async def _collect(
        self,
        process_id: int,
        entry: ProcessEntry,
        yield_time_s: float | None,
        max_output_tokens: int | None,
    ) -> PtyExecUpdate:
        output, tokens = await collect_pty_output(
            output_chunks=entry.chunks,
            output_lock=entry.output_lock,
            output_notify=entry.notify,
            is_done=lambda: bool(entry.waiter and entry.waiter.done()),
            yield_time_ms=clamp_pty_yield_time_ms(int((yield_time_s or 0.25) * 1000)),
            max_output_tokens=max_output_tokens,
        )
        if entry.dropped_bytes:
            output = (
                f"[Dropped {entry.dropped_bytes} bytes from the output buffer]\n".encode() + output
            )
            entry.dropped_bytes = 0
        done = bool(entry.waiter and entry.waiter.done())
        if done:
            async with self._process_lock:
                self._processes.pop(process_id, None)
            await self._close_entry(entry)
        return PtyExecUpdate(
            process_id=None if done else process_id,
            output=output,
            exit_code=entry.process.returncode if done else None,
            original_token_count=tokens,
        )

    async def pty_exec_start(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
        tty: bool = False,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        if tty:
            raise ValueError("Native Windows uses pipes; set tty=false. ConPTY is not implemented.")
        if timeout is not None:
            raise ValueError("Use yielding/polling for Windows command sessions, not timeout")
        argv = self._prepare_exec_command(*command, shell=shell, user=user)
        async with self._process_lock:
            if len(self._processes) >= PTY_PROCESSES_MAX:
                raise RuntimeError(
                    "Too many active command sessions; finish or interrupt one first"
                )
            process_id = allocate_pty_process_id(set(self._processes))
            entry = ProcessEntry(await self._spawn(argv))
            self._processes[process_id] = entry
            entry.tasks = [
                asyncio.create_task(self._pump(entry, stream))
                for stream in (entry.process.stdout, entry.process.stderr)
            ]
            entry.waiter = asyncio.create_task(self._wait(entry))
        try:
            return await self._collect(process_id, entry, yield_time_s, max_output_tokens)
        except BaseException:
            async with self._process_lock:
                self._processes.pop(process_id, None)
            await self._close_entry(entry, terminate=True)
            raise

    async def pty_write_stdin(
        self,
        *,
        session_id: int,
        chars: str,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        async with self._process_lock:
            entry = self._processes.get(session_id)
        if entry is None:
            raise PtySessionNotFoundError(session_id=session_id)
        if "\x03" in chars:
            await self._terminate(entry.process)
        elif chars:
            if entry.process.stdin is None or entry.process.returncode is not None:
                raise RuntimeError("Command stdin is closed")
            entry.process.stdin.write(chars.encode("utf-8"))
            await entry.process.stdin.drain()
        return await self._collect(session_id, entry, yield_time_s, max_output_tokens)

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def _close_entry(self, entry: ProcessEntry, *, terminate: bool = False) -> None:
        if terminate:
            await self._terminate(entry.process)
        if entry.process.stdin is not None:
            entry.process.stdin.close()
        for task in [*entry.tasks, *([entry.waiter] if entry.waiter else [])]:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *entry.tasks, *([entry.waiter] if entry.waiter else []), return_exceptions=True
        )

    async def pty_terminate_all(self) -> None:
        async with self._process_lock:
            entries = list(self._processes.values())
            self._processes.clear()
        results = await asyncio.gather(
            *(self._close_entry(entry, terminate=True) for entry in entries),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise RuntimeError("Could not close a Windows command session") from result  # noqa: TRY004

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        del user
        return self.normalize_path(path).open("rb")

    async def write(self, path: Path, data: io.IOBase, *, user: str | User | None = None) -> None:
        del user
        destination = self.normalize_path(path, for_write=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = data.read()
        destination.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)

    async def mkdir(
        self, path: Path | str, *, parents: bool = True, user: str | User | None = None
    ) -> None:
        del user
        self.normalize_path(path, for_write=True).mkdir(parents=parents, exist_ok=True)

    async def rm(
        self, path: Path | str, *, recursive: bool = False, user: str | User | None = None
    ) -> None:
        del user
        destination = self.normalize_path(path, for_write=True)
        if destination == Path(self.state.manifest.root).resolve():
            raise ValueError("Cannot remove the native workspace root")
        if destination.is_dir():
            shutil.rmtree(destination) if recursive else destination.rmdir()
        else:
            destination.unlink()

    async def ls(
        self, path: Path | str = Path(), *, user: str | User | None = None
    ) -> list[FileEntry]:
        del user
        directory = self.normalize_path(path)
        return [
            FileEntry(
                path=str(child),
                permissions=Permissions.from_mode(child.stat().st_mode),
                owner="",
                group="",
                size=child.stat().st_size,
                kind=EntryKind.DIRECTORY if child.is_dir() else EntryKind.FILE,
            )
            for child in directory.iterdir()
        ]

    async def persist_workspace(self) -> io.IOBase:
        buffer = io.BytesIO()
        root = Path(self.state.manifest.root)
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path in root.rglob("*"):
                if not path.is_symlink():
                    archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        buffer.seek(0)
        return buffer

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        with tarfile.open(fileobj=data, mode="r:*") as archive:
            for member in archive:
                destination = self.normalize_path(member.name, for_write=True)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    handle = archive.extractfile(member)
                    if handle is not None:
                        with handle:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(handle.read())
                else:
                    raise ValueError("Windows snapshots support only regular files and directories")

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        return ExposedPortEndpoint(host="127.0.0.1", port=port, tls=False)


class WindowsSandboxSession(SandboxSession):
    def _workspace_path_policy(self) -> WorkspacePathPolicy:
        return WindowsWorkspacePolicy(root=self.state.manifest.root)


class WindowsNativeClient(BaseSandboxClient[None]):
    backend_id = "windows_native"
    supports_default_options = True

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: None = None,
    ) -> SandboxSession:
        del options
        if manifest is None:
            raise ValueError("Windows native execution requires an explicit workspace manifest")
        session_id = uuid.uuid4()
        state = WindowsSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
        )
        return WindowsSandboxSession(WindowsNativeSession(state))

    async def delete(self, session: SandboxSession) -> SandboxSession:
        await session.stop()
        return session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, WindowsSessionState):
            raise TypeError("Windows native client requires WindowsSessionState")
        return WindowsSandboxSession(WindowsNativeSession(state))

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return WindowsSessionState.model_validate(payload)


class WindowsExecCommandTool(ExecCommandTool):
    default_shell: ClassVar[str | None] = None
    tool_description: ClassVar[str] = (
        "Runs a native Windows PowerShell command with pollable pipes. "
        "Use tty=false; login is ignored because shell profiles are disabled."
    )

    async def run(self, args: ExecCommandArgs) -> str:
        start = time.monotonic()
        script = args.cmd
        if args.workdir:
            directory = self.session.normalize_path(Path(args.workdir))
            literal = str(directory).replace("'", "''")
            script = f"Set-Location -LiteralPath '{literal}';\n{script}"
        shell = resolve_powershell(args.shell)
        update = await self.session.pty_exec_start(  # noqa: S604 - explicit agent shell tool
            script,
            shell=[shell],
            user=self.user,
            tty=args.tty,
            yield_time_s=args.yield_time_ms / 1000,
            max_output_tokens=args.max_output_tokens,
        )
        fields = [f"Wall time: {time.monotonic() - start:.4f} seconds"]
        if update.process_id is not None:
            fields.append(f"Process running with session ID {update.process_id}")
        if update.exit_code is not None:
            fields.append(f"Process exited with code {update.exit_code}")
        fields.extend(
            [
                "Transport: native Windows pipes (tty=false)",
                "Output:",
                update.output.decode("utf-8", errors="replace"),
            ]
        )
        return "\n".join(fields)


class WindowsShell(Shell):
    def tools(self) -> list[Tool]:
        if self.session is None:
            raise ValueError("Windows shell is not bound to a session")
        toolset = ShellToolSet(
            exec_command=WindowsExecCommandTool(session=self.session, user=self.run_as),
            write_stdin=WriteStdinTool(session=self.session),
        )
        if self.configure_tools is not None:
            self.configure_tools(toolset)
        return [tool for tool in (toolset.exec_command, toolset.write_stdin) if tool is not None]

    async def instructions(self, manifest: Manifest) -> str | None:
        return (
            f"Native Windows workspace: {manifest.root}. Use PowerShell syntax with exec_command "
            "(pwsh or powershell.exe). Set tty=false; commands use pipes, not ConPTY. "
            "Poll or send input using write_stdin. "
            "Sending Ctrl-C cancels the command process tree. "
            "login is ignored: shell profiles are never loaded. Linux commands need Windows "
            "equivalents or separately installed tools."
        )
