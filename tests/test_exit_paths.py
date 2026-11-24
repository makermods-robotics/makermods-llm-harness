"""Every way a session the CLI owns can end rests and releases the arms and reports each arm's outcome: quit,
SIGINT, SIGTERM, an exception in the server loop, and the standalone MCP server and model loop. Each runs in its own
process on the kinematic simulator."""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from makermods_llm_harness.log import read_episode
from makermods_llm_harness.session import AddressLock, Client, SessionError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and sockets")

DEADLINE_S = 90.0

SERVE_WITH_FAILING_LOOP = """
import sys
from makermods_llm_harness import cli, session
original = session.Server._accept
accepted = []

def accept(self, listener, channel):
    if channel != "operator":
        return original(self, listener, channel)
    while True:
        conn = listener.accept()
        accepted.append(conn)
        if len(accepted) == 3:
            raise RuntimeError("the server loop failed")
        self._handle_connection(conn, channel)

session.Server._accept = accept
sys.exit(cli.main(sys.argv[1:]))
"""

SERVE_WITH_FAILING_KERNEL_LOOP = """
import sys
from makermods_llm_harness import cli, session
original = session.Server._accept

def accept(self, listener, channel):
    if channel != "kernel":
        return original(self, listener, channel)
    listener.accept()
    raise OSError("the kernel listener failed")

session.Server._accept = accept
sys.exit(cli.main(sys.argv[1:]))
"""

MCP_WITH_FAILING_LOOP = """
import sys
from makermods_llm_harness import cli, mcp_server

def run(target, on_ready=None):
    target.step({"action": "home"})
    raise RuntimeError("the MCP loop failed")

mcp_server.run = run
sys.exit(cli.main(sys.argv[1:]))
"""

RUN_WITH_FAILING_PROVIDER = """
import sys
from makermods_llm_harness import agent, cli
from makermods_llm_harness.agent import Reply, ToolCall

class Provider:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def complete(self, system, messages, tools):
        Provider.calls += 1
        if Provider.calls == 1:
            return Reply(tool_calls=[ToolCall("1", "home", {})])
        raise RuntimeError("the provider failed")

agent.AnthropicProvider = Provider
sys.exit(cli.main(sys.argv[1:]))
"""


def until(condition, timeout=DEADLINE_S):
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = condition()
        except (SessionError, OSError):
            value = None
        if value:
            return value
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def start(args, home, script=None, stdin=None):
    env = {**os.environ, "LLM_HARNESS_HOME": str(home)}
    command = [sys.executable, "-c", textwrap.dedent(script)] if script else \
        [sys.executable, "-m", "makermods_llm_harness.cli"]
    return subprocess.Popen(command + args, env=env, stdin=stdin or subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def finished(proc):
    try:
        out, err = proc.communicate(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"the process did not end:\n{out}\n{err}")
    return proc.returncode, out + err


def assert_rested(run_dir, text):
    events = [e["event"] for e in read_episode(run_dir)]
    assert events[-2:] == ["closing", "end"], (events[-5:], text)
    assert "Every arm is at rest with its motors off." in text, text


@pytest.mark.parametrize("closed", [False, True], ids=["open-output", "closed-output"])
@pytest.mark.parametrize("how", ["quit", "sigint", "sigterm", "server-exception", "kernel-loop-exception"])
def test_every_way_out_of_serve_rests_and_releases(tmp_path, how, closed):
    from multiprocessing.connection import Client as connect
    home = tmp_path / "home"
    folder = tempfile.mkdtemp(prefix="lh")
    address, run_dir = f"{folder}/s.sock", tmp_path / "run"
    args = ["sim", "--address", address, "--run-dir", str(run_dir)]
    script = {"server-exception": SERVE_WITH_FAILING_LOOP,
              "kernel-loop-exception": SERVE_WITH_FAILING_KERNEL_LOOP}.get(how)
    proc = start(args, home, script)
    os.environ["LLM_HARNESS_HOME"] = str(home)
    client = until(lambda: Client(address, timeout_s=60))
    assert client.step({"action": "home"}).result.ok
    if closed:                                           # anything the server prints now fails
        proc.stdout.close()
        proc.stderr.close()
    if how == "quit":
        assert client.quit() == {"arm": "released"}
    elif how == "sigint":
        proc.send_signal(signal.SIGINT)
    elif how == "sigterm":
        proc.terminate()
    elif how == "server-exception":
        with pytest.raises(SessionError):
            Client(address)                              # its third connection ends the server loop
    else:
        connect(address + ".k", family="AF_UNIX").close()   # the kernel listener's first connection fails it
    if closed:
        assert proc.wait(DEADLINE_S) is not None
        events = [e["event"] for e in read_episode(run_dir)]
        assert events[-2:] == ["closing", "end"]
    else:
        code, text = finished(proc)
        assert_rested(run_dir, text)
    AddressLock(address).release()                       # the address is free again


@pytest.mark.parametrize("how", ["quit", "sigterm"])
def test_the_standalone_mcp_server_rests_and_releases(tmp_path, how):
    pytest.importorskip("mcp")
    home = tmp_path / "home"
    proc = start(["mcp", "--sim"], home, stdin=subprocess.PIPE)          # an idle client: stdin stays open
    os.environ["LLM_HARNESS_HOME"] = str(home)
    address = None
    lines = []
    deadline = time.monotonic() + DEADLINE_S
    while address is None and time.monotonic() < deadline:
        line = proc.stderr.readline()
        lines.append(line)
        if line.startswith("session at "):
            address = line.split("session at ", 1)[1].strip()
    assert address, lines
    client = Client(address, timeout_s=60)
    assert client.step({"action": "home"}).result.ok
    if how == "quit":
        assert client.quit() == {"arm": "released"}
    else:
        proc.terminate()
    code, text = finished(proc)
    run_dir = next((home / "runs").iterdir())
    assert_rested(run_dir, "".join(lines) + text)


def test_the_standalone_mcp_server_rests_and_releases_when_its_loop_fails(tmp_path):
    pytest.importorskip("mcp")
    home = tmp_path / "home"
    proc = start(["mcp", "--sim"], home, MCP_WITH_FAILING_LOOP)
    code, text = finished(proc)
    assert code != 0 and "the MCP loop failed" in text
    run_dir = next((home / "runs").iterdir())
    assert_rested(run_dir, text)
    assert any(e.get("action") == {"action": "home"} for e in read_episode(run_dir))


def test_the_standalone_model_loop_rests_and_releases_after_an_exception(tmp_path):
    pytest.importorskip("anthropic")
    home = tmp_path / "home"
    proc = start(["run", "home", "--sim"], home, RUN_WITH_FAILING_PROVIDER)
    code, text = finished(proc)
    assert code != 0 and "the provider failed" in text
    run_dir = next((home / "runs").iterdir())
    assert_rested(run_dir, text)
    assert any(e.get("action") == {"action": "home"} for e in read_episode(run_dir))


def test_a_transport_failure_in_shutdown_still_rests_the_arms(sim_harness, served):
    from makermods_llm_harness.cli import close_session
    server, client = served()
    client.step({"action": "home"})

    def broken():
        raise OSError("socket gone")

    original = server.shutdown
    server.shutdown = broken
    lines = []
    try:
        outcomes, code = close_session(server.harness, server, lines.append)
    finally:
        server.shutdown = original
    assert outcomes == {"arm": "released"} and code == 0
    assert any("did not close cleanly" in line for line in lines)
    assert not Path(server.address + ".key").exists()


def test_the_address_lock_is_held_until_the_outcomes_are_published(sim_harness, served):
    from makermods_llm_harness.cli import close_session
    server, client = served()
    client.step({"action": "home"})
    seen = []
    original = server.harness.close

    def close():
        if not seen:
            with pytest.raises(SessionError):
                AddressLock(server.address)              # still owned while the arm rests
            seen.append(True)
        return original()

    server.harness.close = close
    close_session(server.harness, server, lambda line: None)
    assert seen
    AddressLock(server.address).release()
