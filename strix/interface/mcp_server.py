"""Expose native Strix tools over stdio; OpenCode owns model execution."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.sandbox.capabilities import Filesystem, Shell
from agents.tool import CustomTool, FunctionTool, ToolOutputImage
from agents.tool_context import ToolContext
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent
from mcp.types import Tool as McpTool

from strix.agents.factory import base_agent_tools, custom_tool_as_function
from strix.config import load_settings
from strix.config.loader import apply_config_override
from strix.core.inputs import build_root_task, build_scan_targets
from strix.core.paths import runtime_state_dir
from strix.interface.scan_setup import build_targets_info
from strix.interface.utils import clone_repository, collect_local_sources, stage_api_specs
from strix.report.state import ReportState, set_global_report_state
from strix.runtime import session_manager
from strix.runtime.native import check_native_platform, session_workspace
from strix.skills import get_available_skills
from strix.tools.finish.tool import finish_scan
from strix.tools.output_store import bound_text


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

# OpenCode coordinates its own agents and external MCPs. Never start a hidden
# Strix LLM loop or expose lifecycle tools that require an SDK agent runner.
_EXCLUDED_TOOLS = {
    "create_agent",
    "stop_agent",
    "view_agent_graph",
    "send_message_to_agent",
    "wait_for_agents",
    "list_mcps",
    "search_mcp_tools",
    "get_mcp_tool_schema",
    "describe_mcp",
    "call_mcp",
    "web_search",
    "web_get_contents",
    "list_requests",
    "view_request",
    "repeat_request",
    "list_sitemap",
    "view_sitemap_entry",
    "scope_rules",
}


def opencode_config(command: Sequence[str], *, version: int = 1) -> dict[str, Any]:
    server: dict[str, Any] = {"type": "local", "command": list(command)}
    server["timeout"] = 30000 if version == 1 else {"startup": 30000, "catalog": 30000}
    if version == 1:
        server["enabled"] = True
    servers = {"strix": server}
    return {
        "$schema": "https://opencode.ai/config.json",
        "mcp": servers if version == 1 else {"servers": servers},
    }


def content_blocks(result: Any) -> list[TextContent | ImageContent]:
    if isinstance(result, ToolOutputImage):
        url = result.image_url
        if isinstance(url, str) and url.startswith("data:"):
            header, data = url.split(",", 1)
            # Validate the encoded output before handing it to the client.
            base64.b64decode(data, validate=True)
            return [ImageContent(type="image", mimeType=header[5:].split(";")[0], data=data)]
    text = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    )
    limits = load_settings().context
    return [
        TextContent(
            type="text",
            text=bound_text(
                text,
                max_lines=limits.tool_output_max_lines,
                max_bytes=limits.tool_output_max_bytes,
            ),
        )
    ]


class NativeToolBridge:
    """One engagement per stdio process; stateful calls are serialized."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bundle: dict[str, Any] | None = None
        self.tools: dict[str, FunctionTool] = {}
        self.context: dict[str, Any] = {}
        self.report: ReportState | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        if self.bundle is not None:
            return
        if self._closed:
            raise RuntimeError("This MCP bridge is closed")
        settings = load_settings()
        if settings.runtime.backend != "native":
            raise ValueError("strix mcp requires STRIX_RUNTIME_BACKEND=native")
        check_native_platform()
        build_targets_info(self.args)
        for target in self.args.targets_info:
            if target["type"] == "repository":
                details = target["details"]
                details["cloned_repo_path"] = await asyncio.to_thread(
                    clone_repository,
                    details["target_repo"],
                    self.args.run_name,
                    details.get("workspace_subdir"),
                )
        sources = collect_local_sources(self.args.targets_info)
        sources.extend(stage_api_specs(self.args.targets_info, self.args.run_name))
        config = {
            "targets": self.args.targets_info,
            "local_sources": sources,
            "scan_mode": self.args.scan_mode,
            "user_instructions": self.args.instruction or "",
        }
        self.report = ReportState(run_name=self.args.run_name)
        self.report.hydrate_from_run_dir()
        self.report.set_scan_config(config)
        self.report.run_record["model_execution"] = "opencode"
        set_global_report_state(self.report)
        state_dir = runtime_state_dir(self.report.get_run_dir())
        self._hydrate(state_dir)
        self.bundle = await session_manager.create_or_reuse(
            self.args.run_name,
            image="",
            local_sources=sources,
        )
        session = self.bundle["session"]
        root = session_workspace(session)
        self.context = {
            "sandbox_session": session,
            "caido_client": self.bundle["caido_client"],
            "agent_id": "opencode",
            "parent_id": None,
            "scan_targets": build_scan_targets(config, workspace_root=root),
            "native_workspace_root": root,
            "supports_images": True,
        }
        tools = [tool for tool in base_agent_tools() if tool.name not in _EXCLUDED_TOOLS]
        tools.append(finish_scan)
        for capability in (Shell(), Filesystem()):
            capability.bind(session)
            tools.extend(capability.tools())
        for tool in tools:
            adapted = custom_tool_as_function(tool) if isinstance(tool, CustomTool) else tool
            if isinstance(adapted, FunctionTool):
                self.tools[adapted.name] = adapted
        self.report.save_run_data()
        self.context["task"] = build_root_task(config, workspace_root=root)

    @staticmethod
    def _hydrate(state_dir: Path) -> None:
        from strix.tools.coverage.tools import hydrate_coverage_from_disk
        from strix.tools.notes.tools import hydrate_notes_from_disk
        from strix.tools.threat_model.tools import hydrate_threat_models_from_disk
        from strix.tools.todo.tools import hydrate_todos_from_disk

        hydrate_notes_from_disk(state_dir)
        hydrate_todos_from_disk(state_dir)
        hydrate_coverage_from_disk(state_dir)
        hydrate_threat_models_from_disk(state_dir)

    def catalog(self) -> list[McpTool]:
        return [
            McpTool(
                name="get_workflow",
                description="Read scope, native workspace, and Strix workflow.",
                inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            *(
                McpTool(name=t.name, description=t.description, inputSchema=t.params_json_schema)
                for t in self.tools.values()
            ),
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
        async with self._lock:
            if self._closed:
                raise RuntimeError("This MCP bridge is closed")
            if name == "get_workflow":
                return content_blocks(
                    {
                        "task": self.context["task"],
                        "workspace": self.context["native_workspace_root"],
                        "reports": str(self.report.get_run_dir()) if self.report else None,
                        "skills": get_available_skills(),
                        "workflow": (
                            "Use only user-authorized targets. Map surfaces, "
                            "track todos and coverage, "
                            "load relevant skills, test hypotheses, record evidence "
                            "and counterevidence, "
                            "then write validated findings and finish_scan. "
                            "OpenCode owns reasoning "
                            "and agent coordination. Commands execute on the host; tools must be "
                            "installed locally. Source trees are copies. Use the actual workspace "
                            "path, not /workspace. Proxy tools are not exposed by this bridge; "
                            "connect your Burp/Caido MCP to OpenCode separately. LLM deduplication "
                            "is disabled: use list_reports before filing findings."
                        ),
                    }
                )
            tool = self.tools.get(name)
            if tool is None:
                raise ValueError(f"Unknown Strix tool: {name}")
            raw = json.dumps(arguments, ensure_ascii=False)
            ctx = ToolContext(
                context=self.context,
                tool_name=name,
                tool_call_id=uuid.uuid4().hex,
                tool_arguments=raw,
            )
            return content_blocks(await tool.on_invoke_tool(ctx, raw))

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self.bundle is not None:
                await session_manager.cleanup(self.args.run_name)
                self.bundle = None
            if self.report is not None:
                self.report.cleanup()
                set_global_report_state(None)


async def serve(args: argparse.Namespace) -> None:
    bridge = NativeToolBridge(args)
    server = Server("strix-native")

    async def _list_tools() -> list[McpTool]:
        return bridge.catalog()

    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
        return await bridge.call(name, arguments)

    server.list_tools()(_list_tools)  # type: ignore[no-untyped-call]
    server.call_tool()(_call_tool)

    try:
        await bridge.start()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await bridge.close()


def run_mcp(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="strix mcp")
    parser.add_argument("-t", "--target", action="append", dest="target", default=[])
    parser.add_argument("--run-name", default="opencode")
    parser.add_argument("--scan-mode", choices=("quick", "standard", "deep"), default="deep")
    parser.add_argument("--instruction")
    parser.add_argument("--print-opencode-config", action="store_true")
    parser.add_argument("--opencode-version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    args.target_list = []
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run_name):
        parser.error("--run-name must be a simple name, without slashes or traversal")
    if args.print_opencode_config:
        command = [sys.executable, "-m", "strix.interface.main", "mcp", "--run-name", args.run_name]
        for target in args.target:
            command.extend(["--target", target])
        command.extend(["--scan-mode", args.scan_mode])
        if args.instruction:
            command.extend(["--instruction", args.instruction])
        print(json.dumps(opencode_config(command, version=args.opencode_version), indent=2))  # noqa: T201
        return
    os.environ["STRIX_RUNTIME_BACKEND"] = "native"
    os.environ["STRIX_TELEMETRY"] = "false"
    # The bridge must not read a stored model configuration or call its provider
    # for report dedupe. OpenCode is the only model owner on this execution path.

    apply_config_override(Path(os.devnull))
    os.environ["STRIX_LLM"] = ""
    os.environ["STRIX_DEDUPE_MODEL"] = ""
    from agents.tracing.provider import DefaultTraceProvider
    from agents.tracing.setup import set_trace_provider

    provider = DefaultTraceProvider()
    provider.set_disabled(True)
    set_trace_provider(provider)
    asyncio.run(serve(args))
