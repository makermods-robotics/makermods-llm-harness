import base64
import dataclasses
import os
import sys
import time

import cv2
import numpy as np
import pytest

from makermods_llm_harness import tools
from makermods_llm_harness.kernel import GUIDE, TOOL
from makermods_llm_harness.mcp_server import build_server

mcp = pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")


def session(target, body, mode="auto"):
    """Run ``body(client)`` against the server for ``target``, connected in-process."""
    async def main():
        async with mcp.Client(build_server(target), mode=mode) as client:
            return await body(client)
    return anyio.run(main)


def decode(block):
    return cv2.imdecode(np.frombuffer(base64.b64decode(block.data), np.uint8), cv2.IMREAD_COLOR)


def test_tools_and_instructions_come_from_the_harness(sim_harness):
    h = sim_harness()

    async def body(client):
        return (await client.list_tools()).tools, client.instructions

    listed, instructions = session(h, body)
    expected = tools()
    assert [t.name for t in listed] == [t["name"] for t in expected]
    for got, want in zip(listed, expected):
        assert got.description == want["description"] and got.input_schema == want["input_schema"]
    assert instructions == h.describe()


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_observe_returns_text_and_a_jpeg_per_camera(sim_harness, mode):
    h = sim_harness()
    result = session(h, lambda client: client.call_tool("observe", {}), mode)
    assert not result.is_error
    text, *images = result.content
    assert text.type == "text" and "tool xyz" in text.text and "camera top" in text.text
    assert [b.type for b in images] == ["image", "image"] and {b.mime_type for b in images} == {"image/jpeg"}
    assert all(decode(b).shape == (240, 320, 3) for b in images)


def test_move_tool_moves_the_arm(sim_harness):
    h = sim_harness()

    async def body(client):
        await client.call_tool("home", {})
        return await client.call_tool("move_tool", {"xyz": [0.31, 0.03, 0.54], "pitch_deg": -5})

    result = session(h, body)
    assert not result.is_error, result.content[0].text
    tool = h.observe().arms["arm"].tool
    assert np.allclose(tool.xyz, (0.31, 0.03, 0.54), atol=2e-3)
    assert "[0.310, 0.030, 0.540]" in result.content[0].text


def test_a_rejection_is_a_readable_tool_error(sim_harness):
    h = sim_harness()

    async def body(client):
        await client.call_tool("home", {})
        return await client.call_tool("nudge", {"dxyz": [0, 0, -0.6]})

    result = session(h, body)
    assert result.is_error
    assert result.content[0].text.startswith("rejected: ")
    assert [b.type for b in result.content[1:]] == ["image", "image"]


def test_a_target_with_image_paths_only(sim_harness, tmp_path):
    h = sim_harness(image_dir=tmp_path)

    class PathsOnly:
        """Observations as a session client returns them: image paths, no bytes."""

        def describe(self):
            return h.describe()

        def observe(self):
            return self._strip(h.observe())

        def step(self, action, cancel=None):
            return self._strip(h.step(action, cancel=cancel))

        @staticmethod
        def _strip(obs):
            obs.cameras = {n: dataclasses.replace(c, jpeg=None) for n, c in obs.cameras.items()}
            return obs

    result = session(PathsOnly(), lambda client: client.call_tool("observe", {}))
    images = result.content[1:]
    assert len(images) == 2 and decode(images[0]).shape == (240, 320, 3)


def test_run_serves_on_stdio(harness_home):
    code = ("from makermods_llm_harness import Harness, robot\n"
            "from makermods_llm_harness.mcp_server import run\n"
            "with Harness.open(robot.simulated()) as h:\n"
            "    run(h)\n")
    env = {**os.environ, "LLM_HARNESS_HOME": str(harness_home)}
    server = mcp.StdioServerParameters(command=sys.executable, args=["-c", code], env=env)

    async def main():
        async with mcp.Client(server, read_timeout_seconds=60) as client:
            return await client.call_tool("home", {})

    result = anyio.run(main)
    assert not result.is_error and "tool xyz [0.313, " in result.content[0].text


def test_a_cancelled_call_ends_its_motion(sim_harness):
    from makermods_llm_harness import mcp_server
    from conftest import sim_backend

    h = sim_harness(backend=sim_backend(realtime=True))

    async def main():
        with anyio.move_on_after(0.3):
            await mcp_server.call(h, "home", {})

    anyio.run(main)
    deadline = time.monotonic() + 5
    while h._abort is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    obs = h.observe()
    state = h.backend.read()["arm"]
    assert obs.arms["arm"].enabled and np.array_equal(state.q_target, state.q)    # held where it was
    assert not np.allclose(state.q, h.models["arm"].home_q, atol=0.05)              # short of home


def test_run_python_runs_kernel_code_and_returns_reports_with_their_grid(served):
    server, h = served()

    async def body(client):
        assert [t.name for t in (await client.list_tools()).tools] == [t["name"] for t in tools() + [TOOL]]
        assert client.instructions.endswith(GUIDE)
        first = await client.call_tool("run_python", {"code": "def home(r):\n    r.home()\nkernel.queue(home)"})
        news = [first]
        while not any("[action #1 home] done" in r.content[0].text for r in news):
            news.append(await client.call_tool("run_python", {"code": "", "wait_s": 60}))
        failed = await client.call_tool("run_python", {"code": "r.move([2, 0, 0])"})
        return first, news, failed

    first, news, failed = session(h, body)
    assert not first.is_error and first.content[0].text.startswith("1\n")
    reported = next(r for r in news if "[action #1 home] done" in r.content[0].text)
    text, *images = reported.content
    assert "[state] arm: tool [" in text.text
    assert [b.type for b in images] == ["image"] and decode(images[0]).shape[0] == 240
    assert failed.is_error and "MotionFailed: rejected" in failed.content[0].text
