import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from multiprocessing.connection import Client as connect, Listener

import numpy as np
import pytest

from makermods_llm_harness.actions import Home, Observation
from makermods_llm_harness.session import FAMILY, Client, Server, SessionError, default_address

from conftest import sim_backend


@pytest.fixture
def address():
    """A short, unique session address (Unix socket paths are limited to about 100 bytes)."""
    if sys.platform == "win32":
        yield rf"\\.\pipe\llm-harness-test-{uuid.uuid4().hex[:12]}"
        return
    folder = tempfile.mkdtemp(prefix="lh")
    yield f"{folder}/s.sock"
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def serving(sim_harness, address):
    """``start(harness=None) -> (server, client)``: a server on a thread; shut down afterwards."""
    started = []

    def start(harness=None):
        server = Server(harness or sim_harness(), address)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return server, Client(address, timeout_s=10)

    yield start
    for server, thread in started:
        server.shutdown()
        thread.join(5)
        assert not thread.is_alive()


def test_round_trips(serving, tmp_path):
    server, client = serving()
    obs = client.observe()
    assert isinstance(obs, Observation) and obs.step == 0 and "arm" in obs.arms
    obs = client.step({"action": "home"})
    assert obs.result.status == "done" and obs.step == 1
    obs = client.step(Home())
    assert obs.result.ok and obs.result.action == {"action": "home"}
    obs = client.step('{"action": "nudge", "dxyz": [0, 0, -0.02]}')
    assert obs.result.ok and obs.step == 3
    assert "Robot robot" in client.describe()
    status = client.status()
    assert status == {"robot": "robot", "arms": ["arm"], "cameras": ["top", "side"], "steps": 3, "run_dir": None,
                      "read_only": False, "busy": False}
    assert client.rest().result.ok
    client.quit()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            Client(server.address)
        except SessionError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the server did not shut down after quit")


def test_observations_carry_image_paths(serving, sim_harness, tmp_path):
    _, client = serving(sim_harness(image_dir=tmp_path))
    obs = client.step({"action": "observe"})
    assert obs.cameras["top"].jpeg is None
    assert open(obs.cameras["top"].path, "rb").read(2) == b"\xff\xd8"


def test_rejected_action_comes_back_as_an_observation(serving):
    _, client = serving()
    obs = client.step({"action": "move_tool", "xyz": [2, 0, 0]})
    assert obs.result.status == "rejected" and obs.result.message
    assert client.step({"action": "fly"}).result.status == "rejected"
    assert client.step({"action": "home"}).result.ok


def test_stop_interrupts_a_running_action(serving, sim_harness):
    server, client = serving(sim_harness(backend=sim_backend(realtime=True)))
    replies = []
    worker = threading.Thread(target=lambda: replies.append(client.step({"action": "home"})))
    worker.start()
    time.sleep(0.3)
    assert client.status()["busy"]
    client.stop()
    worker.join(5)
    assert replies[0].result.status == "stopped"
    assert not client.status()["busy"]


def test_no_server(address):
    with pytest.raises(SessionError, match="no session is running at .*llm-harness serve"):
        Client(address)


def test_second_server_is_refused_and_a_stale_socket_is_reused(serving, sim_harness, address):
    server, _ = serving()
    with pytest.raises(SessionError, match="already running"):
        Server(sim_harness(), address)
    if FAMILY != "AF_UNIX":
        return
    server.shutdown()
    time.sleep(0.2)
    open(address, "w").close()      # a regular file is not taken for a stale socket
    with pytest.raises(SessionError, match="not a socket"):
        Server(sim_harness(), address)


def test_stale_socket_file_is_replaced(sim_harness, address):
    if FAMILY != "AF_UNIX":
        pytest.skip("socket files exist only on POSIX")
    import socket
    dead = socket.socket(socket.AF_UNIX)
    dead.bind(address)
    dead.close()                     # the file stays, nobody listens
    server = Server(sim_harness(), address)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert Client(address).status()["steps"] == 0
    finally:
        server.shutdown()
        thread.join(5)


def test_too_long_socket_path():
    if FAMILY != "AF_UNIX":
        pytest.skip("named pipes have no path limit")
    with pytest.raises(SessionError, match="--address"):
        Server(None, tempfile.mkdtemp() + "/" + "x" * 120)


def test_malformed_requests_and_harness_errors_get_error_replies(serving, monkeypatch):
    server, client = serving()
    for raw in (b"not json", b"[1]", json.dumps({"op": "fly"}).encode(), json.dumps({"op": "step"}).encode()):
        conn = connect(server.address, family=FAMILY)
        conn.send_bytes(raw)
        reply = json.loads(conn.recv_bytes())
        conn.close()
        assert reply["ok"] is False and reply["error"]
    monkeypatch.setattr(server.harness, "observe", lambda: 1 / 0)
    with pytest.raises(SessionError, match="ZeroDivisionError"):
        client.observe()
    monkeypatch.undo()
    assert client.observe().step == 0


def test_server_dying_mid_request_is_a_session_error(address):
    listener = Listener(address, family=FAMILY)

    def answer_then_die():
        listener.accept().close()            # the client's connection check
        conn = listener.accept()
        conn.recv_bytes()
        conn.close()                         # gone before it answers

    thread = threading.Thread(target=answer_then_die, daemon=True)
    thread.start()
    try:
        client = Client(address, timeout_s=5)
        with pytest.raises(SessionError, match="ended before it answered"):
            client.step({"action": "observe"})
    finally:
        thread.join(5)
        listener.close()


def test_default_address_lives_in_the_home(harness_home):
    if FAMILY == "AF_UNIX":
        assert default_address() == str(harness_home / "session" / "llm-harness.sock")
    else:
        assert default_address().startswith(r"\\.\pipe\llm-harness-")


# ── access, limits, cancellation and shutdown ──
def test_the_socket_lives_in_a_directory_only_its_owner_can_enter(sim_harness, monkeypatch, tmp_path):
    if FAMILY != "AF_UNIX":
        pytest.skip("named pipes have no directory")
    home = Path(tempfile.mkdtemp(prefix="lh"))     # short enough for the default socket path
    monkeypatch.setenv("LLM_HARNESS_HOME", str(home))
    server = Server(sim_harness())
    try:
        folder = home / "session"
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert stat.S_IMODE(os.stat(server.address).st_mode) == 0o600
    finally:
        server.shutdown()
        shutil.rmtree(home, ignore_errors=True)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    os.chmod(shared, 0o777)
    with pytest.raises(SessionError, match="chmod 700"):
        Server(sim_harness(), str(shared / "s.sock"))


def test_an_oversized_request_is_refused_and_the_server_goes_on(serving, monkeypatch):
    import makermods_llm_harness.session as session
    monkeypatch.setattr(session, "MAX_REQUEST", 1000)
    server, client = serving()
    conn = connect(server.address, family=FAMILY)
    conn.send_bytes(json.dumps({"op": "step", "action": {"action": "observe", "pad": "x" * 5000}}).encode())
    with pytest.raises((EOFError, OSError)):
        conn.recv_bytes()
    conn.close()
    assert client.status()["steps"] == 0


def test_a_client_that_hangs_up_cancels_its_motion(serving, sim_harness):
    server, client = serving(sim_harness(backend=sim_backend(realtime=True)))
    conn = connect(server.address, family=FAMILY)
    conn.send_bytes(json.dumps({"op": "step", "action": {"action": "home"}}).encode())
    time.sleep(0.3)
    assert client.status()["busy"]
    conn.close()
    deadline = time.monotonic() + 5
    while client.status()["busy"] and time.monotonic() < deadline:
        time.sleep(0.05)
    obs = client.step({"action": "observe"})
    state = server.harness.backend.read()["arm"]
    assert not client.status()["busy"] and obs.step == 2
    assert np.array_equal(state.q_target, state.q)              # held where the hang-up found it
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(SessionError, match="cancelled"):
        client.step({"action": "home"}, cancel=cancel)


def test_quit_answers_after_the_owner_rested_the_arms_and_later_requests_are_refused(sim_harness, address):
    harness = sim_harness()
    server = Server(harness, address, reports_close=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(address, timeout_s=10)
    client.step({"action": "home"})
    replies = []
    quitter = threading.Thread(target=lambda: replies.append(client.quit()))
    quitter.start()
    thread.join(5)
    assert not thread.is_alive() and quitter.is_alive()          # admission closed, quit still waiting
    with pytest.raises(SessionError, match="closing"):
        server._run("step", {"action": {"action": "home"}}, None)
    harness.stop()
    server.drain()
    server.finish(harness.close())
    quitter.join(5)
    assert replies == [{"arm": "released"}]


def test_images_without_a_path_travel_as_bytes(serving):
    _, client = serving()
    obs = client.step({"action": "observe"})
    assert obs.cameras["top"].path is None and obs.cameras["top"].jpeg[:2] == b"\xff\xd8"
