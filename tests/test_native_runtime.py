from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from strix.config.loader import apply_config_override
from strix.config.settings import RuntimeSettings
from strix.core.inputs import build_root_task, build_scan_targets
from strix.interface.mcp_server import NativeToolBridge, opencode_config
from strix.interface.scan_setup import build_targets_info
from strix.runtime import session_manager
from strix.runtime.native import check_native_platform, session_workspace


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def _native_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRIX_RUNTIME_BACKEND", "native")
    monkeypatch.setenv("STRIX_TELEMETRY", "false")
    monkeypatch.setenv("STRIX_LLM", "")
    monkeypatch.setenv("STRIX_DEDUPE_MODEL", "")
    monkeypatch.delenv("STRIX_CAIDO_URL", raising=False)
    for proxy_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(proxy_name, raising=False)
    apply_config_override(tmp_path / "missing-config.json")
    yield
    apply_config_override(tmp_path / "missing-config.json")


def test_native_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIX_RUNTIME_BACKEND", raising=False)
    assert RuntimeSettings().backend == "native"


def test_windows_requires_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("strix.runtime.native_windows.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="PowerShell not found"):
        check_native_platform()


def test_native_keeps_localhost(_native_settings: None) -> None:
    args = argparse.Namespace(target=["http://127.0.0.1:8080"], target_list=[])
    build_targets_info(args)
    assert args.targets_info[0]["details"]["target_url"] == "http://127.0.0.1:8080"


def test_scope_and_task_use_actual_workspace() -> None:
    config = {
        "targets": [
            {
                "type": "local_code",
                "details": {
                    "workspace_subdir": "repo",
                    "target_path": "/original/repo",
                },
            }
        ]
    }
    root = "/scan/workspace"
    assert build_scan_targets(config, workspace_root=root) == [f"{root}/repo"]
    task = build_root_task(config, workspace_root=root)
    assert f"{root}/repo" in task
    assert "writable copy" in task
    assert "mounted live" not in task


async def test_native_session_copies_sources_stages_files_and_retains_artifacts(
    _native_settings: None,
    tmp_path: Path,
) -> None:
    source = tmp_path / "original"
    source.mkdir()
    (source / "file.txt").write_text("original")
    scan_id = "native-test"
    try:
        bundle = await session_manager.create_or_reuse(
            scan_id,
            image="unused",
            local_sources=[
                {
                    "source_path": str(source),
                    "workspace_subdir": "repo",
                }
            ],
            extra_files=[{"workspace_path": "/workspace/note.txt", "content": "context"}],
        )
        assert bundle["caido_client"] is None
        session = bundle["session"]
        root = Path(session_workspace(session))
        assert root == tmp_path / "strix_runs" / scan_id / "workspace"
        assert (root / "repo/file.txt").read_text() == "original"
        assert (root / "note.txt").read_text() == "context"
        environment = await session.state.manifest.environment.resolve()
        assert "http_proxy" not in environment
        assert "HOST_GATEWAY" not in environment
        result = await session.exec(
            sys.executable,
            "-c",
            "from pathlib import Path; Path('repo/file.txt').write_text('changed')",
            shell=False,
        )
        assert result.ok()
        assert (source / "file.txt").read_text() == "original"
        assert await session_manager.create_or_reuse(scan_id, image="", local_sources=[]) is bundle
    finally:
        await session_manager.cleanup(scan_id)
    assert (root / "repo/file.txt").read_text() == "changed"
    assert scan_id not in session_manager._SESSION_CACHE
    try:
        await session_manager.create_or_reuse(
            scan_id,
            image="",
            local_sources=[
                {
                    "source_path": str(source),
                    "workspace_subdir": "repo",
                }
            ],
        )
        assert (root / "repo/file.txt").read_text() == "changed"
    finally:
        await session_manager.cleanup(scan_id)


async def test_bridge_tools_execute_without_model_and_restore_notes(
    _native_settings: None,
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        target=[],
        target_list=[],
        run_name="bridge",
        scan_mode="deep",
        instruction="Review only the code I provide.",
    )
    bridge = NativeToolBridge(args)
    try:
        await bridge.start()
        names = {t.name for t in bridge.catalog()}
        assert {
            "exec_command",
            "write_stdin",
            "apply_patch",
            "load_skill",
            "create_note",
            "record_coverage",
            "create_vulnerability_report",
            "finish_scan",
            "get_workflow",
        } <= names
        assert "create_agent" not in names
        assert "call_mcp" not in names
        assert "list_requests" not in names
        result = await bridge.call(
            "exec_command",
            {
                "cmd": "Write-Output native-ok" if sys.platform == "win32" else "printf native-ok",
                "login": False,
            },
        )
        assert "native-ok" in result[0].text
        patch = "*** Begin Patch\n*** Add File: proof.txt\n+bridge works\n*** End Patch"
        await bridge.call("apply_patch", {"patch": patch})
        assert (tmp_path / "strix_runs/bridge/workspace/proof.txt").read_text() == "bridge works"
        result = await bridge.call(
            "create_note",
            {"title": "checkpoint", "content": "restored", "category": "general", "tags": []},
        )
        assert "success" in result[0].text
        with pytest.raises(ValueError, match="Unknown"):
            await bridge.call("missing_tool", {})
    finally:
        await bridge.close()
    restored = NativeToolBridge(args)
    try:
        await restored.start()
        result = await restored.call(
            "list_notes", {"category": None, "tags": None, "search_query": None}
        )
        assert "checkpoint" in result[0].text
    finally:
        await restored.close()
    with pytest.raises(RuntimeError, match="closed"):
        await restored.call("get_workflow", {})


@pytest.mark.parametrize("version", [1, 2])
def test_opencode_config_variants(version: int) -> None:
    config = opencode_config(["strix", "mcp"], version=version)
    servers = config["mcp"] if version == 1 else config["mcp"]["servers"]
    assert servers["strix"]["command"] == ["strix", "mcp"]
    assert ("enabled" in servers["strix"]) == (version == 1)
    assert json.loads(json.dumps(config)) == config


async def test_stdio_mcp_handshake_and_command(_native_settings: None, tmp_path: Path) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "strix.interface.main", "mcp", "--run-name", "stdio-test"],
        cwd=str(tmp_path),
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
        initialized = await client.initialize()
        assert initialized.serverInfo.name == "strix-native"
        catalog = await client.list_tools()
        assert "get_workflow" in {tool.name for tool in catalog.tools}
        result = await client.call_tool(
            "exec_command",
            {
                "cmd": "Write-Output stdio-ok" if sys.platform == "win32" else "printf stdio-ok",
                "login": False,
            },
        )
        assert not result.isError
        assert "stdio-ok" in result.content[0].text
        workflow = await client.call_tool("get_workflow", {})
        assert not workflow.isError
        assert "stdio-test" in workflow.content[0].text
    record = json.loads((tmp_path / "strix_runs/stdio-test/run.json").read_text())
    assert record["model_execution"] == "opencode"
