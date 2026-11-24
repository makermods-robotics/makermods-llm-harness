"""The session's Python kernel, on the kinematic simulator: its own process, execution tokens, the queue in the
session, reports, stop, shutdown, authentication, the py command. Orderings are forced with files the kernel's code
waits for or writes (it runs in another process) and with events in this one."""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from makermods_llm_harness import cli
from makermods_llm_harness import kernel as kernel_module
from makermods_llm_harness.kernel import MAX_WAITING, format_reply
from makermods_llm_harness.program import Program
from makermods_llm_harness.session import FAMILY, Client, SessionError

from conftest import sim_backend

DEADLINE_S = 60.0          # generous: CI machines are slow and shared

pytestmark = pytest.mark.skipif(FAMILY == "AF_PIPE", reason="the kernel is not available on Windows")

HELPERS = """
import os, time
def wait_file(path, limit=60):
    end = time.monotonic() + limit
    while not os.path.exists(path):
        if time.monotonic() > end:
            raise TimeoutError(path)
        time.sleep(0.01)
def mark(path):
    open(path, "w").close()
log = []
def step(name, gate=None):
    def fn(r):
        if gate:
            wait_file(gate)
        log.append(name)
    fn.__name__ = name
    return fn
"""


def until(condition, timeout=DEADLINE_S):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def run(client, code, **kwargs):
    reply = client.py(code, **kwargs)
    assert not reply["error"], reply["output"]
    return reply


def drain(client, reports=None, timeout=DEADLINE_S):
    """Every report until nothing runs or waits, after ``reports`` already collected."""
    reports = list(reports or [])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reply = client.py("", wait_s=1.0)
        reports += reply["reports"]
        if not reply["busy"]:
            return reports
    pytest.fail("the queue did not empty")


@pytest.fixture
def kernel(served, tmp_path):
    """``make(**served kwargs) -> (server, client, gate(name) -> path)`` with HELPERS defined in the kernel."""
    def make(**kwargs):
        server, client = served(**kwargs)
        run(client, HELPERS)
        return server, client, lambda name: str(tmp_path / name)
    return make


# ── the process ──
def test_the_kernel_is_a_separate_process_holding_only_a_session_client(kernel):
    server, client, _ = kernel()
    reply = run(client, "import gc\nprint(os.getpid())\nprint(type(r.target).__name__)\n"
                        "print(any(type(o).__name__ in ('Harness', 'SimBackend') for o in gc.get_objects()))")
    pid, target, found = reply["output"].split()
    assert int(pid) != os.getpid() and target == "_Session" and found == "False"
    assert run(client, "x = 41")["output"] == ""
    assert run(client, "x + 1")["output"] == "42\n"
    reply = client.py("1 / 0")
    assert reply["error"] and "ZeroDivisionError" in reply["output"] and "1 / 0" in reply["output"]


def test_a_read_only_session_runs_code_but_moves_nothing(kernel):
    server, client, _ = kernel(read_only=True)
    reply = client.py("print(r.state().text())\nr.home()")
    assert reply["error"] and "read-only" in reply["output"]
    assert not server.harness.state().arms["arm"].enabled


# ── the queue ──
def test_actions_run_in_order_and_each_reports_once(kernel):
    server, client, gate = kernel()
    reply = run(client, f"kernel.queue([step('a', {gate('g')!r}), step('b')])")
    assert reply["output"] == "[1, 2]\n"
    assert run(client, "kernel.queue(step('c'))")["output"] == "3\n"
    until(lambda: client.status()["queue"]["running"] is not None)
    status = client.status()
    assert status["busy"] and status["queue"]["running"]["name"] == "a"
    assert [w["name"] for w in status["queue"]["waiting"]] == ["b", "c"]
    reports = run(client, f"mark({gate('g')!r})")["reports"]
    reports = drain(client, reports)
    assert [(r["id"], r["name"], r["status"]) for r in reports] == [(1, "a", "done"), (2, "b", "done"),
                                                                   (3, "c", "done")]
    assert run(client, "log")["output"] == "['a', 'b', 'c']\n"
    assert all(Path(r["image"]).is_file() and r["arms"]["arm"].startswith("tool [") for r in reports)
    assert client.py("")["reports"] == []
    assert not client.status()["busy"]


def test_replace_from_flush_and_the_waiting_limit(kernel):
    server, client, gate = kernel()
    run(client, f"kernel.queue([step('a', {gate('g')!r}), step('b'), step('c'), step('d')])")
    until(lambda: client.status()["queue"]["running"] is not None)
    reply = client.py("kernel.queue(step('e'))")
    assert reply["error"] and f"at most {MAX_WAITING} actions wait" in reply["output"]
    assert run(client, "kernel.queue([step('x'), step('y')], replace_from=3)")["output"] == "[5, 6]\n"
    assert [w["name"] for w in client.status()["queue"]["waiting"]] == ["b", "x", "y"]
    reply = client.py("kernel.queue(step('z'), replace_from='a')")
    assert reply["error"] and "is not waiting" in reply["output"]
    assert run(client, "kernel.flush()")["output"] == "['#2 b', '#5 x', '#6 y']\n"
    reports = drain(client, run(client, f"mark({gate('g')!r})")["reports"])
    assert [r["name"] for r in reports] == ["a"]
    assert run(client, "log")["output"] == "['a']\n"


def test_an_error_drops_the_queue_and_moves_nothing_more(kernel):
    server, client, _ = kernel()
    run(client, "def fails(r):\n    print('about to fail')\n    raise RuntimeError('precondition')\n"
                "def bad_move(r):\n    r.move([2, 0, 0])\n")
    reports = drain(client, run(client, "kernel.queue([fails, step('never')])")["reports"])
    assert [(r["name"], r["status"]) for r in reports] == [("fails", "error")]
    assert "RuntimeError: precondition" in reports[0]["message"] and reports[0]["output"].startswith("about to fail")
    assert reports[0]["dropped"] == ["#2 never"]
    reports = drain(client, run(client, "kernel.queue([bad_move, step('never')])")["reports"])
    assert reports[0]["status"] == "error" and reports[0]["message"].startswith("rejected: arm: ")
    assert reports[0]["dropped"] == ["#4 never"]
    assert run(client, "log")["output"] == "[]\n" and not server.harness.state().arms["arm"].enabled


def test_stop_halts_the_running_action_and_drops_the_queue(kernel):
    server, client, gate = kernel(backend=sim_backend(realtime=True))
    run(client, f"def slow(r):\n    r.home()\n    mark({gate('moving')!r})\n    r.wait(30)\n")
    run(client, "kernel.queue([slow, step('after')])")
    until(lambda: os.path.exists(gate("moving")))
    client.stop()
    reports = drain(client)
    assert [(r["name"], r["status"]) for r in reports] == [("slow", "stopped")]
    assert reports[0]["dropped"] == ["#2 after"] and reports[0]["seconds"] < 30
    state = server.harness.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)


def test_kernel_stop_also_stops_the_code_that_called_it(kernel):
    server, client, gate = kernel()
    reply = client.py("kernel.stop()\nprint('after')\nkernel.queue(step('late'))")
    assert reply["error"] and reply["stopped"] and "after" not in reply["output"]
    assert drain(client) == [] and run(client, "log")["output"] == "[]\n"


def test_a_motion_from_a_call_stops_the_running_action(kernel):
    server, client, gate = kernel(backend=sim_backend(realtime=True))
    run(client, f"def waits(r):\n    mark({gate('running')!r})\n    r.wait(30)\n")
    run(client, "kernel.queue([waits, step('b')])")
    until(lambda: os.path.exists(gate("running")))
    reports = run(client, "r.home()")["reports"]
    reports = drain(client, reports)
    assert [(r["name"], r["status"]) for r in reports] == [("waits", "stopped")]
    assert reports[0]["dropped"] == ["#2 b"] and "outside the queue" in reports[0]["message"]
    assert server.harness.state().arms["arm"].enabled


# ── authority (stopped or finished code) ──
def test_stopped_code_can_neither_queue_nor_move(kernel):
    server, client, gate = kernel()
    code = (f"try:\n    mark({gate('paused')!r})\n    wait_file({gate('resume')!r})\nexcept BaseException:\n    pass\n"
            "outcome = []\n"
            "for attempt in (lambda: kernel.queue(step('late')), lambda: r.home()):\n"
            "    try:\n        attempt()\n        outcome.append('ran')\n"
            "    except BaseException as e:\n        outcome.append(type(e).__name__)\n"
            "print(outcome)")
    replies = []
    worker = threading.Thread(target=lambda: replies.append(client.py(code)))
    worker.start()
    until(lambda: os.path.exists(gate("paused")))
    client.stop()
    Path(gate("resume")).touch()
    worker.join(DEADLINE_S)
    assert "ran" not in replies[0]["output"]
    assert drain(client) == [] and not server.harness.state().arms["arm"].enabled


def test_a_thread_left_by_a_finished_action_has_no_authority(kernel):
    server, client, gate = kernel()
    run(client, "import threading\n"
                f"def late(r):\n    wait_file({gate('go')!r})\n    try:\n        r.home()\n        result = 'moved'\n"
                f"    except Exception as e:\n        result = type(e).__name__\n"
                f"    open({gate('result')!r}, 'w').write(result)\n"
                "def starts(r):\n    threading.Thread(target=late, args=(r,)).start()\n")
    drain(client, run(client, "kernel.queue(starts)")["reports"])
    Path(gate("go")).touch()
    until(lambda: os.path.exists(gate("result")) and Path(gate("result")).read_text())
    assert Path(gate("result")).read_text() == "Stopped" and not server.harness.state().arms["arm"].enabled


def test_a_call_whose_client_hangs_up_loses_its_authority(kernel):
    from makermods_llm_harness.session import _open, _read_key
    server, client, gate = kernel()
    code = (f"try:\n    mark({gate('paused')!r})\n    wait_file({gate('resume')!r})\nexcept BaseException:\n    pass\n"
            f"try:\n    kernel.queue(step('late'))\n    r.home()\n    out = 'moved'\nexcept BaseException as e:\n"
            f"    out = type(e).__name__\nopen({gate('result')!r}, 'w').write(out)")
    conn = _open(client.address, _read_key(client.address))
    conn.send_bytes(json.dumps({"op": "py", "code": code}).encode())
    until(lambda: os.path.exists(gate("paused")))
    conn.close()
    host = server.kernel()
    until(lambda: not host._cells)                          # the session saw the hang-up and revoked the call
    Path(gate("resume")).touch()
    until(lambda: os.path.exists(gate("result")) and Path(gate("result")).read_text())
    assert Path(gate("result")).read_text() != "moved"
    assert drain(client) == [] and not server.harness.state().arms["arm"].enabled


def test_a_stop_after_admission_ends_a_step_held_up_by_the_queue(kernel):
    server, client, _ = kernel()
    host = server.kernel()
    inside, release = threading.Event(), threading.Event()
    original = host.interrupt

    def slow_interrupt():
        inside.set()
        assert release.wait(DEADLINE_S)
        original()

    host.interrupt = slow_interrupt
    replies = []
    worker = threading.Thread(target=lambda: replies.append(client.step({"action": "home"})))
    worker.start()
    assert inside.wait(DEADLINE_S)
    client.stop()
    release.set()
    worker.join(DEADLINE_S)
    assert replies[0].result.status == "stopped" and "before it started" in replies[0].result.message
    assert not server.harness.state().arms["arm"].enabled


def test_a_failed_action_leaves_no_target_away_from_the_arm(kernel):
    server, client, _ = kernel()
    run(client, "r.home()")
    server.harness.backend.block("arm")
    run(client, "def nudged(r):\n    r.move_joints({'wrist_roll': 4}, relative=True)\n    raise RuntimeError('x')\n")
    reports = drain(client, run(client, "kernel.queue(nudged)")["reports"])
    assert reports[0]["status"] == "error"
    state = server.harness.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)


# ── reports ──
def test_a_report_shows_its_own_action_and_the_next_timeout_runs_from_its_start(kernel, monkeypatch):
    server, client, _ = kernel()
    run(client, "r.home()")
    original_look = server.harness.look

    def slow_look():
        time.sleep(1.5)
        return original_look()

    monkeypatch.setattr(server.harness, "look", slow_look)
    run(client, "def first(r):\n    r.move([0.31, 0, 0.54], pitch=-5)\n"
                "def second(r):\n    time.sleep(3)\n    r.move([0.30, 0, 0.47], pitch=-25)\n")
    reports = drain(client, run(client, "kernel.queue(first)\nkernel.queue(second, timeout=1)")["reports"])
    first, second = reports
    assert "[0.310, 0.000, 0.540]" in first["arms"]["arm"].replace("-0.000", "0.000")
    assert second["status"] == "error" and "timed out after 1 s" in second["message"], (second["message"], second["output"])
    assert abs(server.harness.state().arms["arm"].tool.xyz[2] - 0.54) < 0.005     # second never moved


def test_reports_come_again_until_acknowledged_and_a_client_shows_each_once(kernel):
    server, client, gate = kernel()
    run(client, f"kernel.queue(step('a', {gate('g')!r}))")
    Path(gate("g")).touch()
    until(lambda: not client.status()["busy"])
    other = Client(client.address)
    first = client.py("")["reports"]
    assert [r["name"] for r in first] == ["a"]
    assert [r["name"] for r in other.py("")["reports"]] == ["a"]     # not acknowledged yet: sent again
    assert client.py("")["reports"] == []                              # this call acknowledges it
    assert other.py("")["reports"] == []


def test_reports_are_text_with_the_queue_and_state(kernel):
    server, client, _ = kernel()
    run(client, "def home(r):\n    r.home()\n")
    reports = drain(client, run(client, "kernel.queue(home)")["reports"])
    text = format_reply({"output": "", "arms": {}, "queue": "empty", "reports": reports})
    assert re.search(r"\[action #1 home\] done in \d+\.\d s", text) and "image: " in text


def test_undelivered_reports_are_bounded(kernel, monkeypatch):
    monkeypatch.setattr(kernel_module, "MAX_REPORTS", 2)
    server, client, gate = kernel()
    assert run(client, f"kernel.queue([step('a', {gate('g')!r}), step('b'), step('c'), step('d')])")["reports"] == []
    Path(gate("g")).touch()
    until(lambda: not client.status()["busy"])
    reply = client.py("")
    assert len(reply["reports"]) == 2 and reply["reports_lost"] == 2


# ── limits ──
def test_output_and_code_are_bounded(kernel):
    server, client, _ = kernel()
    reply = run(client, "print('x' * 5_000_000)")
    assert len(reply["output"]) < 25_000 and "characters left out" in reply["output"]
    with pytest.raises(SessionError, match="longer than"):
        client.py("#" * (kernel_module.MAX_CODE + 1))


def test_code_running_past_its_limit_restarts_the_kernel(kernel):
    server, client, _ = kernel()
    reply = client.py("while True:\n    pass", timeout_s=2)
    assert reply["error"] and "restarted" in reply["output"]
    assert run(client, "print('fresh' if 'log' not in globals() else 'old')")["output"] == "fresh\n"


# ── shutdown ──
def test_shutdown_revokes_and_ends_the_kernel_without_waiting_for_its_code(kernel):
    server, client, gate = kernel()
    run(client, f"def hangs(r):\n    mark({gate('hung')!r})\n    time.sleep(1000)\n")
    run(client, "kernel.queue(hangs)")
    until(lambda: os.path.exists(gate("hung")))
    replies = []

    def hung_call():
        try:
            replies.append(client.py(f"mark({gate('cell')!r})\ntime.sleep(1000)"))
        except SessionError as e:
            replies.append(e)

    threading.Thread(target=hung_call, daemon=True).start()
    until(lambda: os.path.exists(gate("cell")))
    host = server.kernel()
    proc = host._proc
    start = time.monotonic()
    server.shutdown()
    server.drain()
    assert time.monotonic() - start < 15 and proc.poll() is not None
    until(lambda: replies)
    assert server.harness.close()["arm"] in ("released", "always_on")


# ── authentication ──
def test_every_connection_must_answer_the_challenge(served):
    from multiprocessing.connection import Client as connect
    from makermods_llm_harness.session import CHALLENGE, key_path
    server, client = served()
    path = key_path(server.address)
    assert path.stat().st_mode & 0o777 == 0o600
    raw = connect(server.address, family=FAMILY)
    assert raw.recv_bytes().startswith(CHALLENGE)
    raw.send_bytes(b"wrong" * 6)
    raw.send_bytes(json.dumps({"op": "py", "code": "print(1)"}).encode())
    with pytest.raises((EOFError, OSError)):
        raw.recv_bytes()
    raw.close()
    assert client.status()["steps"] == 0


def test_py_is_refused_where_the_key_cannot_be_shown_private(served, monkeypatch):
    import makermods_llm_harness.session as session
    server, client = served()
    monkeypatch.setattr(session, "KERNEL_AVAILABLE", False)
    with pytest.raises(SessionError, match="not available on Windows"):
        client.py("print(1)")


# ── the CLI ──
def test_the_py_command(kernel, capsys, monkeypatch):
    server, client, _ = kernel()
    assert cli.main(["py", "--address", client.address, "-c",
                     "def home(r):\n    r.home()\nprint(kernel.queue(home))"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1\n") and "[state] arm: tool [" in out
    seen = out
    while "[action #1 home] done" not in seen:
        assert cli.main(["py", "--address", client.address, "--wait", "5"]) == 0
        seen += capsys.readouterr().out
    monkeypatch.setattr(sys, "stdin", type("Pipe", (), {"isatty": lambda self: False,
                                                        "read": lambda self: "print(1 + 1)\n"})())
    assert cli.main(["py", "--address", client.address]) == 0
    assert capsys.readouterr().out.startswith("2\n")
    assert cli.main(["py", "--address", client.address, "-c", "raise ValueError('no')"]) == 1
    assert "ValueError: no" in capsys.readouterr().out


def test_dry_run_and_until_flags_on_op(served, capsys):
    server, client = served()
    assert cli.main(["op", "--address", client.address, "home", "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert not server.harness.state().arms["arm"].enabled
    assert cli.main(["op", "--address", client.address, "home", "--duration", "3"]) == 0
    capsys.readouterr()
    assert cli.main(["op", "--address", client.address, "nudge", "0", "0", "-0.05", "--until", "contact"]) == 0
    assert "reached the target without contact" in capsys.readouterr().out


def test_the_program_api_over_a_session(served):
    server, client = served()
    r = Program(client)
    assert r.home().ok and r.state().arms["arm"].enabled
    verdict, = r.reachable([{"xyz": np.array([0.31, 0, 0.54]), "pitch": -5}])
    assert verdict["ok"] and r.move([0.31, 0, 0.54], pitch=-5, dry_run=True).ok
    assert r.move(np.array([0.31, 0, 0.54]), pitch=-5).ok
    assert Path(r.look()).is_file()


def test_the_agent_guide_runs_on_the_simulator(kernel):
    server, client, _ = kernel()
    guide = (Path(__file__).parent.parent / "docs" / "AGENT_GUIDE.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", guide, re.S)
    assert len(blocks) >= 5
    reports = []
    for block in blocks:
        reports = drain(client, reports + run(client, block)["reports"])
    assert [(r["name"], r["status"]) for r in reports] == [
        ("to_cube", "done"), ("grasp", "done"), ("lift", "done"), ("put_back", "done")]
    cube = server.harness.backend.objects[0]
    assert cube["held_by"] is None and abs(cube["xyz"][0] - 0.28) < 0.01


def test_status_counts_queued_work_but_not_a_call_waiting_in_code(kernel):
    server, client, gate = kernel()
    replies = []
    worker = threading.Thread(target=lambda: replies.append(client.py(f"mark({gate('in')!r})\nwait_file({gate('out')!r})")))
    worker.start()
    until(lambda: os.path.exists(gate("in")))
    assert not client.status()["busy"]                     # code running, nothing moving or queued
    Path(gate("out")).touch()
    worker.join(DEADLINE_S)
    run(client, f"kernel.queue(step('a', {gate('g')!r}))")
    until(lambda: client.status()["queue"]["running"] is not None)
    assert client.status()["busy"]
    drain(client, run(client, f"mark({gate('g')!r})")["reports"])
    assert not client.status()["busy"]


# ── admission, channels, ownership ──
def test_a_call_waiting_for_the_kernel_to_start_is_cancelled_by_a_stop(served, tmp_path):
    server, client = served()
    host = server.kernel()
    inside, release = threading.Event(), threading.Event()
    start = host._ensure_child

    def slow_start():
        inside.set()
        assert release.wait(DEADLINE_S)
        start()

    host._ensure_child = slow_start
    replies = []
    marker = tmp_path / "ran"
    worker = threading.Thread(target=lambda: replies.append(client.py(f"open({str(marker)!r}, 'w')")))
    worker.start()
    assert inside.wait(DEADLINE_S)
    client.stop()
    release.set()
    worker.join(DEADLINE_S)
    assert replies[0]["stopped"] and "a stop came before" in replies[0]["output"] and not marker.exists()


def test_an_ending_call_or_action_loses_its_authority_as_the_message_arrives(served):
    from makermods_llm_harness.kernel import _Queued
    server, client = served()
    host = server.kernel()
    with host._cond:
        cell = host._new_token("cell")
        host._cell_tokens[99] = cell
        action = _Queued(7, "k", "a", 10.0)
        action.token = host._new_token("action")
        host._running = action
    host._on_message({"event": "cell_done", "id": 99})
    host._on_message({"event": "action_done", "id": 7, "status": "done"})
    assert not host.valid(cell) and not host.valid(action.token)
    with host._cond:
        host._running = None


def test_the_kernel_channel_needs_a_live_token_for_anything_that_moves(served):
    from makermods_llm_harness.session import kernel_address
    server, client = served()
    kernel_client = Client(kernel_address(server.address), key=server.kernel_key)
    assert kernel_client.state().arms["arm"].enabled is False                  # queries are open
    reply = kernel_client.request("step", action={"action": "home"})
    assert reply["revoked"] and reply["observation"]["result"]["status"] == "stopped"
    reply = kernel_client.request("step", action={"action": "home"}, token="cell-forged")
    assert reply["revoked"]
    for op, fields in (("kqueue", {"keys": ["k"], "names": ["n"]}), ("stop", {}), ("py", {"code": "1"}),
                       ("quit", {})):
        with pytest.raises(SessionError):
            kernel_client.request(op, **fields)
    with pytest.raises(SessionError):
        Client(kernel_address(server.address))                                # the operator key is not its key
    assert not server.harness.state().arms["arm"].enabled and client.status()["steps"] == 0


def test_work_queued_by_a_call_whose_client_left_is_stopped(kernel):
    from makermods_llm_harness.session import _open, _read_key
    server, client, gate = kernel()
    code = (f"kernel.queue(step('queued', {gate('g')!r}))\nmark({gate('queued')!r})\nwait_file({gate('end')!r})")
    conn = _open(client.address, _read_key(client.address))
    conn.send_bytes(json.dumps({"op": "py", "code": code}).encode())
    until(lambda: os.path.exists(gate("queued")))
    conn.close()
    until(lambda: not client.status()["busy"])
    Path(gate("g")).touch()
    reports = drain(client)
    assert all(r["status"] != "done" for r in reports) and run(client, "log")["output"] == "[]\n"


def test_status_counts_a_waiting_queue_and_not_an_observation(served):
    from makermods_llm_harness.kernel import _Queued
    server, client = served()
    host = server.kernel()
    with host._cond:
        host._alive = False                          # the worker waits for a kernel process
        host._waiting.append(_Queued(1, "k", "queued", 10.0))
    assert client.status()["busy"]
    with host._cond:
        host._waiting.clear()
    inside, release = threading.Event(), threading.Event()
    frames = server.harness._frames

    def pausing(after):
        inside.set()
        assert release.wait(DEADLINE_S)
        return frames(after)

    server.harness._frames = pausing
    worker = threading.Thread(target=client.observe)
    worker.start()
    assert inside.wait(DEADLINE_S)
    assert not client.status()["busy"]
    release.set()
    worker.join(DEADLINE_S)


def test_kernel_calls_are_in_the_episode_log(served, tmp_path):
    from makermods_llm_harness.log import EpisodeLog, read_episode
    log = EpisodeLog(tmp_path / "run")
    server, client = served(log=log)
    run(client, "x = 1")
    assert any(e["event"] == "py" and e["chars"] == 5 for e in read_episode(tmp_path / "run"))


def test_a_second_server_cannot_take_the_address_or_its_key(served, sim_harness):
    from makermods_llm_harness.session import Server, key_path
    server, client = served()
    key = key_path(server.address).read_bytes()
    with pytest.raises(SessionError, match="already running or starting"):
        Server(sim_harness(), server.address)
    assert key_path(server.address).read_bytes() == key
    client.stop()                                    # the first server's key still works


def test_stalled_connections_do_not_keep_stop_out(served, monkeypatch):
    import makermods_llm_harness.session as session
    from makermods_llm_harness.session import CHALLENGE, MAX_CLIENTS, _digest, _read_key
    from multiprocessing.connection import Client as connect
    monkeypatch.setattr(session, "READ_S", 2.0)
    server, client = served()
    key = _read_key(server.address)
    stalled = []
    for _ in range(MAX_CLIENTS):
        conn = connect(server.address, family=FAMILY)
        challenge = conn.recv_bytes()
        assert challenge.startswith(CHALLENGE)
        conn.send_bytes(_digest(key, challenge))
        conn._send(b"\x00\x00\x10\x00")             # a frame header, and never its body
        stalled.append(conn)
    start = time.monotonic()
    client.stop()                                    # answered by a control slot while all handlers wait
    assert time.monotonic() - start < 2.0
    for conn in stalled:
        conn.close()


def test_a_stop_after_shutdown_began_is_refused_and_the_rest_owns_the_arms(served):
    server, client = served()
    client.step({"action": "home"})
    server.shutdown()
    with pytest.raises(SessionError, match="shutting down"):
        server._run("stop", {}, None)
    with pytest.raises(SessionError, match="shutting down"):
        server._run("step", {"action": {"action": "home"}}, None)
    assert server.harness.close()["arm"] in ("released", "always_on")


def test_closing_ends_a_kernel_process_that_does_not_read_its_pipe(kernel):
    import signal
    server, client, _ = kernel()
    host = server.kernel()
    proc = host._proc
    os.kill(proc.pid, signal.SIGSTOP)
    start = time.monotonic()
    assert host.close() is None
    assert proc.poll() is not None and not host._dispatcher.is_alive() and time.monotonic() - start < 15
    with pytest.raises(SessionError):
        host._ensure_child()


def test_quit_on_a_private_session_rests_the_arms_through_its_owner(sim_harness):
    from makermods_llm_harness.cli import Lifecycle
    from makermods_llm_harness.session import PrivateSession
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("the owner is woken in the main thread")
    harness = sim_harness()
    private = PrivateSession(harness)
    try:
        private.client.step({"action": "home"})
        replies = []
        quitter = threading.Thread(target=lambda: replies.append(Client(private.server.address).quit()))
        with Lifecycle(harness, private.server, say=lambda line: None) as life:
            quitter.start()
            deadline = time.monotonic() + DEADLINE_S
            while time.monotonic() < deadline:       # an owner busy elsewhere (the MCP stdio loop)
                time.sleep(0.05)
        quitter.join(DEADLINE_S)
        assert life.interrupted and replies == [{"arm": "released"}]
    finally:
        private.cleanup()


def test_the_kernel_capability_comes_from_the_session(served, monkeypatch):
    import makermods_llm_harness.session as session
    from makermods_llm_harness.kernel import kernel_for
    server, client = served()
    assert kernel_for(client) is not None
    monkeypatch.setattr(session, "KERNEL_AVAILABLE", False)
    assert client.status()["kernel"] is False and kernel_for(client) is None


def test_a_py_call_waiting_to_reach_the_kernel_is_cancelled_by_a_stop(served, tmp_path):
    server, client = served()
    inside, release = threading.Event(), threading.Event()
    original = server.kernel

    def slow_kernel():
        inside.set()
        assert release.wait(DEADLINE_S)
        return original()

    server.kernel = slow_kernel
    replies = []
    worker = threading.Thread(target=lambda: replies.append(client.py("r.home()")))
    worker.start()
    assert inside.wait(DEADLINE_S)
    client.stop()
    release.set()
    worker.join(DEADLINE_S)
    assert replies[0]["stopped"] and not server.harness.state().arms["arm"].enabled


def test_a_kernel_stop_whose_code_was_revoked_does_not_stop_new_work(kernel):
    server, client, gate = kernel(backend=sim_backend(realtime=True))
    host = server.kernel()
    inside, release = threading.Event(), threading.Event()
    original = server.stop

    def paused_stop(token=None):
        if token is not None:
            inside.set()
            assert release.wait(DEADLINE_S)
        return original(token)

    server.stop = paused_stop
    replies, steps = [], []
    worker = threading.Thread(target=lambda: replies.append(client.py("kernel.stop()")))
    worker.start()
    assert inside.wait(DEADLINE_S)
    original()                                        # an operator stop revokes that code
    server.stop = original
    moving = threading.Thread(target=lambda: steps.append(client.step({"action": "home"})))
    moving.start()
    until(lambda: server.harness._heading)            # the new motion is in flight
    release.set()                                     # the stale stop goes on now
    worker.join(DEADLINE_S)
    moving.join(DEADLINE_S)
    assert steps[0].result.ok and server.harness.state().arms["arm"].enabled     # the new motion was not stopped
    assert host is server.kernel()


def test_kernel_stop_hands_its_caller_a_hold_it_could_not_confirm(kernel, monkeypatch):
    server, client, _ = kernel()
    run(client, "r.home()")
    monkeypatch.setattr(server.harness, "confirm_hold", lambda names=None: "arm: the hold is not confirmed: test")
    reply = client.py("kernel.stop()")
    assert reply["motion"] == "fault" and "the hold is not confirmed" in reply["output"]


def test_actions_queued_by_actions_belong_to_the_call_that_started_them(kernel):
    from makermods_llm_harness.session import _open, _read_key
    server, client, gate = kernel()
    code = (f"def child(r):\n    log.append('child')\n    r.home()\n"
            f"def parent(r):\n    kernel.queue(step('waits', {gate('g')!r}))\n    kernel.queue(child)\n"
            f"    mark({gate('queued')!r})\n"
            f"kernel.queue(parent)\nwait_file({gate('end')!r})")
    conn = _open(client.address, _read_key(client.address))
    conn.send_bytes(json.dumps({"op": "py", "code": code}).encode())
    until(lambda: os.path.exists(gate("queued")))
    conn.close()                                      # the submission never reached its client
    until(lambda: not client.status()["busy"])
    Path(gate("g")).touch()
    drain(client)
    assert "child" not in run(client, "log")["output"] and not server.harness.state().arms["arm"].enabled


def test_replies_name_only_images_the_session_made(kernel, tmp_path):
    server, client, _ = kernel()
    forged = server.harness.image_dir / "grid-9999.jpg"
    forged.write_bytes(b"\xff\xd8not ours")
    reply = run(client, "print(r.look())")
    assert reply["images"] and str(forged) not in reply["images"]
    child_claim = server.kernel()
    assert os.path.realpath(reply["images"][0]) in child_claim._issued


def test_a_failing_request_log_neither_breaks_a_reply_nor_strands_its_work(served, tmp_path):
    server, client = served()

    def broken(line):
        raise OSError("log gone")

    server.say = broken
    marker = tmp_path / "ran"
    reply = client.py(f"def touch(r):\n    open({str(marker)!r}, 'w').close()\nkernel.queue(touch)")
    assert not reply["error"]
    until(lambda: marker.exists())                       # delivered, so the work it queued goes on


def test_a_slow_failed_hold_reaches_the_code_that_asked_for_the_stop_and_its_report(kernel, monkeypatch):
    server, client, gate = kernel()
    run(client, "r.home()")

    def slow_failure(names=None):
        time.sleep(2.5)
        return "arm: the hold is not confirmed: refused by the backend"

    monkeypatch.setattr(server.harness, "confirm_hold", slow_failure)
    reply = client.py("kernel.stop()")
    assert reply["motion"] == "fault" and "refused by the backend" in reply["output"]
    run(client, "def stops(r):\n    kernel.stop()\n")
    reports = drain(client, run(client, "kernel.queue(stops)")["reports"])
    assert reports[0]["status"] == "error" and "refused by the backend" in reports[0]["message"]


def test_a_failed_server_start_keeps_the_callers_address_lock(sim_harness, monkeypatch):
    import makermods_llm_harness.session as session
    from makermods_llm_harness.session import AddressLock, Server, private_address
    address = private_address()
    lock = AddressLock(address)
    created = []
    real = session.Listener

    def second_fails(*args, **kwargs):
        if created:
            raise OSError("no second listener")
        created.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(session, "Listener", second_fails)
    with pytest.raises(OSError):
        Server(sim_harness(), address, lock=lock)
    with pytest.raises(SessionError):
        AddressLock(address)                             # still the caller's until it lets go
    lock.release()
    AddressLock(address).release()


def test_an_observe_step_is_not_arm_work(served):
    server, client = served()
    inside, release = threading.Event(), threading.Event()
    frames = server.harness._frames

    def pausing(after):
        inside.set()
        assert release.wait(DEADLINE_S)
        return frames(after)

    server.harness._frames = pausing
    worker = threading.Thread(target=lambda: client.step({"action": "observe"}))       # what `op observe` sends
    worker.start()
    assert inside.wait(DEADLINE_S)
    assert not client.status()["busy"]
    release.set()
    worker.join(DEADLINE_S)


def test_an_image_changed_after_the_session_wrote_it_is_not_read(kernel):
    from makermods_llm_harness.kernel import reply_images
    server, client, _ = kernel()
    reply = run(client, "print(r.look())")
    assert reply_images(reply, 1)
    Path(reply["images"][0]).write_bytes(b"\xff\xd8 replaced")
    assert reply_images(reply, 1) == []
