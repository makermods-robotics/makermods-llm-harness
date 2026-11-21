"""An MCP server for clients whose own model drives the robot (Claude Code, Claude Desktop, Codex, ...).

There is one tool per action, with the names, descriptions and input schemas of ``actions.tools()``. A call
runs ``target.step`` and returns the observation as text followed by one JPEG per camera. A rejected or
faulted action is a normal result with ``isError`` set; its text says what happened and what to do instead.
A call that the client cancels, or that is still running when the client disconnects, ends its motion (the
arm holds). The robot description (``target.describe()``) is the server's instructions.

``target`` is a ``Harness`` or a session ``Client``: anything with ``observe``, ``step(action, cancel)`` and
``describe``.
"""

from __future__ import annotations

import base64
import importlib
import threading
from typing import Any, Optional

from .actions import Observation, tools

ERROR_STATUSES = ("rejected", "fault")


def _mcp():
    try:
        return importlib.import_module("mcp")
    except ImportError as e:
        raise ImportError("the MCP server needs the mcp package: pip install 'makermods-llm-harness[mcp]'") from e


def content(obs: Observation) -> list:
    """The observation as MCP content: its text, then one inline JPEG per camera that has an image."""
    types = _mcp().types
    text = obs.text()
    missing = obs.missing_images()
    if missing:
        text += "\n(no image delivered for " + ", ".join(missing) + ")"
    blocks: list = [types.TextContent(type="text", text=text)]
    for jpeg in obs.jpegs().values():
        blocks.append(types.ImageContent(type="image", data=base64.b64encode(jpeg).decode("ascii"),
                                         mime_type="image/jpeg"))
    return blocks


async def call(target, name: str, arguments: Optional[dict]) -> Any:
    """Run one tool call as an action off the event loop; cancelling the call ends the action (the arm holds)."""
    anyio = importlib.import_module("anyio")
    action = {**(arguments or {}), "action": name}
    cancel = threading.Event()
    try:
        obs = await anyio.to_thread.run_sync(lambda: target.step(action, cancel=cancel), abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        cancel.set()
        raise
    return _mcp().types.CallToolResult(content=content(obs), is_error=obs.result.status in ERROR_STATUSES)


def build_server(target):
    """The MCP server for ``target`` (an ``mcp.server.Server``), not yet serving."""
    mcp = _mcp()
    types = mcp.types
    listed = [types.Tool(name=t["name"], description=t["description"], input_schema=t["input_schema"])
              for t in tools()]

    async def list_tools(ctx, params) -> Any:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx, params) -> Any:
        return await call(target, params.name, params.arguments)

    return mcp.server.Server("makermods-llm-harness", instructions=target.describe(),
                             on_list_tools=list_tools, on_call_tool=call_tool)


def run(target) -> None:
    """Serve ``target`` over stdin/stdout until the client disconnects."""
    server = build_server(target)
    anyio = importlib.import_module("anyio")
    stdio = importlib.import_module("mcp.server.stdio")

    async def serve() -> None:
        async with stdio.stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)
