"""An MCP server for clients whose own model drives the robot (Claude Code, Claude Desktop, Codex, ...).

There is one tool per action, with the names, descriptions and input schemas of ``actions.tools()``. A call
runs ``target.step`` and returns the observation as text followed by one JPEG per camera. A rejected or
faulted action is a normal result with ``isError`` set; its text says what happened and what to do instead.
A call that the client cancels, or that is still running when the client disconnects, ends its motion (the
arm holds). The robot description (``target.describe()``) is the server's instructions.

With a session ``Client`` as target there is also ``run_python`` (``kernel.TOOL``): code in the session's Python
kernel, returning the reply as text followed by the new camera grids, and the kernel guide joins the instructions.
``llm-harness mcp`` without ``--connect`` serves its robot through a session of its own, so the tool is there too.
The kernel runs arbitrary Python as the user who runs the session, in a process of its own.

``target`` is anything with ``observe``, ``step(action, cancel)`` and ``describe``.
"""

from __future__ import annotations

import base64
import importlib
import threading
from typing import Any, Optional

from .actions import Observation, tools
from .kernel import GUIDE, TOOL, format_reply, kernel_for, reply_images

ERROR_STATUSES = ("rejected", "fault")
MAX_IMAGES = 3            # camera grids returned inline by one run_python call (the newest)


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




def reply_content(reply: dict) -> list:
    """A kernel reply as MCP content: its text, then the newest camera grids it names (``kernel.reply_images``)."""
    types = _mcp().types
    blocks: list = [types.TextContent(type="text", text=format_reply(reply))]
    for jpeg in reply_images(reply, MAX_IMAGES):
        blocks.append(types.ImageContent(type="image", data=base64.b64encode(jpeg).decode(), mime_type="image/jpeg"))
    return blocks


async def call(target, name: str, arguments: Optional[dict], kernel=None) -> Any:
    """Run one tool call off the event loop: an action, or with ``kernel`` the run_python tool. Cancelling the
    call ends its motion (the arm holds)."""
    anyio = importlib.import_module("anyio")
    cancel = threading.Event()
    if name == TOOL["name"] and kernel is not None:
        arguments = arguments or {}
        work = lambda: kernel.execute(arguments.get("code", ""), arguments.get("wait_s"), cancel=cancel)  # noqa: E731
    else:
        action = {**(arguments or {}), "action": name}
        work = lambda: target.step(action, cancel=cancel)  # noqa: E731
    try:
        out = await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        cancel.set()
        raise
    if isinstance(out, dict):
        return _mcp().types.CallToolResult(content=reply_content(out), is_error=bool(out["error"]))
    return _mcp().types.CallToolResult(content=content(out), is_error=out.result.status in ERROR_STATUSES)


def build_server(target):
    """The MCP server for ``target`` (an ``mcp.server.Server``), not yet serving."""
    mcp = _mcp()
    types = mcp.types
    kernel = kernel_for(target)
    defs = tools() + ([TOOL] if kernel is not None else [])
    listed = [types.Tool(name=t["name"], description=t["description"], input_schema=t["input_schema"])
              for t in defs]
    instructions = target.describe() + (f"\n\n{GUIDE}" if kernel is not None else "")

    async def list_tools(ctx, params) -> Any:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx, params) -> Any:
        return await call(target, params.name, params.arguments, kernel)

    return mcp.server.Server("makermods-llm-harness", instructions=instructions,
                             on_list_tools=list_tools, on_call_tool=call_tool)


def run(target, on_ready=None) -> None:
    """Serve ``target`` over stdin/stdout until the client disconnects, or until the function given to
    ``on_ready(stop)`` is called (from any thread): ``stop()`` ends serving at once, even while it waits on an idle
    stdin."""
    server = build_server(target)
    anyio = importlib.import_module("anyio")
    for module in ("anyio.from_thread", "anyio.lowlevel"):
        importlib.import_module(module)
    stdio = importlib.import_module("mcp.server.stdio")

    async def serve() -> None:
        with anyio.CancelScope() as scope:
            token = anyio.lowlevel.current_token()

            def stop() -> None:
                try:
                    anyio.from_thread.run_sync(scope.cancel, token=token)
                except RuntimeError:          # serving has already ended
                    pass

            if on_ready is not None:
                on_ready(stop)
            async with stdio.stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)
