import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import yaml

from makermods_llm_harness import Harness, cli
from makermods_llm_harness.cli import main
from makermods_llm_harness.log import read_episode
from makermods_llm_harness.robot import Robot
from makermods_llm_harness.safety import load_calibration
from makermods_llm_harness.session import Client, SessionError

from makermods_llm_harness.backends.sim import SimBackend

from conftest import sim_backend


class Uncalibrated(SimBackend):
    """Stands in for a real robot's backend: it does not know the table height."""

    @property
    def table_z(self):
        return None


@pytest.fixture
def address():
    if sys.platform == "win32":
        yield rf"\\.\pipe\llm-harness-test-{uuid.uuid4().hex[:12]}"
        return
    folder = tempfile.mkdtemp(prefix="lh")
    yield f"{folder}/s.sock"
    shutil.rmtree(folder, ignore_errors=True)


def wait_for_session(address, alive=lambda: True, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return Client(address)
        except SessionError:
            if not alive() or time.monotonic() > deadline:
                pytest.fail(f"no session came up at {address}")
            time.sleep(0.05)


@pytest.fixture
def serve(address, capsys):
    """``serve(*argv)`` runs ``llm-harness <argv> --address A`` (sim or serve) on a thread until it answers;
    returns a dict that gets the exit status. Sessions still running at the end are quit."""
    threads = []

    def start(*argv):
        result = {}
        thread = threading.Thread(target=lambda: result.update(code=main([*argv, "--address", address])),
                                  daemon=True)
        thread.start()
        threads.append(thread)
        wait_for_session(address, thread.is_alive)
        capsys.readouterr()                 # the session's start-up text
        result["thread"] = thread
        return result

    yield start
    for thread in threads:
        if thread.is_alive():
            try:
                Client(address).quit()
            except SessionError:
                pass
        thread.join(10)
        assert not thread.is_alive()


@pytest.fixture
def op(address, capsys):
    """``op(*argv) -> (exit status, stdout)`` for ``llm-harness op <argv> --address A``."""
    capsys.readouterr()

    def run(*argv, command="op"):
        code = main([command, *argv, "--address", address])
        return code, capsys.readouterr().out

    return run


def assert_done(result):
    code, out = result
    assert code == 0, out


# ── session ops ──
def test_every_op_against_a_sim_session(serve, op, tmp_path):
    serve("sim", "--run-dir", str(tmp_path / "run"))
    code, out = op("observe")
    assert code == 0 and "arm (motors off): tool xyz" in out
    image = out.split("camera top: ")[1].splitlines()[0]
    assert image.endswith("step-0001-top.jpg") and Path(image).is_file()
    assert_done(op("home"))
    assert_done(op("move_tool", "0.31", "0", "0.54", "--pitch", "-5"))
    code, out = op("--json", "nudge", "0", "0", "-0.02")
    assert code == 0 and json.loads(out)["arms"]["arm"]["tool"]["xyz"] == pytest.approx([0.31, 0, 0.52], abs=3e-3)
    assert_done(op("nudge", "0.01", "0", "0", "--frame", "tool", "--arm", "arm"))
    assert_done(op("gripper", "open"))
    assert_done(op("gripper", "0.3"))
    assert_done(op("move_joints", "wrist_roll=10", "wrist_flex=-5", "--relative"))
    assert_done(op("wait", "0.1"))
    assert_done(op('{"action": "nudge", "dxyz": [0, 0, 0.01]}'))
    code, out = op("--json", "home")
    assert code == 0 and json.loads(out)["result"]["action"] == {"action": "home"}
    code, out = op("status")
    assert code == 0 and "11 steps" in out and str(tmp_path / "run") in out
    code, out = op("status", "--json")
    assert json.loads(out)["steps"] == 11
    code, out = op("describe")
    assert code == 0 and "Safety envelope" in out
    assert_done(op("stop"))
    assert_done(op("rest"))
    code, out = op("--json", command="rest")
    assert code == 0 and json.loads(out)["result"]["status"] == "done"


def test_exit_codes(serve, op, address):
    code, _ = op("observe")
    assert code == 2                                   # no session yet
    serve("sim")
    code, out = op("move_tool", "2", "0", "0")
    assert code == 1 and out.startswith("rejected: ")
    code, out = op('{"action": "fly"}')
    assert code == 1 and "unknown action" in out
    assert op("nudge", "0", "0")[0] == 2                 # missing DZ
    assert op("gripper", "wide")[0] == 2
    assert op("move_joints", "elbow_flex")[0] == 2
    assert op("fly")[0] == 2
    assert main(["op", "--address", address]) == 2
    assert main(["op", "status", "--address", address]) == 0


def test_ops_run_in_order_and_stop_at_the_first_not_done(serve, op, address):
    serve("sim")
    assert op("home", "fly 1 2", command="ops")[0] == 2   # a typo: nothing is sent
    code, out = op("home", "move_tool 2 0 0", "gripper open", command="ops")
    assert code == 1
    assert "> home" in out and "> move_tool 2 0 0" in out and "rejected" in out and "> gripper open" not in out
    assert Client(address).status()["steps"] == 2
    code, out = op("nudge 0 0 -0.01 --arm arm", "gripper open", "--json", command="ops")
    assert code == 0 and [json.loads(line)["result"]["status"] for line in out.splitlines()] == ["done", "done"]


def test_quit_rests_the_arms_and_ends_serve(serve, op, capsys, tmp_path):
    session = serve("sim", "--run-dir", str(tmp_path / "run"))
    assert_done(op("home"))
    code, out = op("quit")
    assert code == 0 and out.count("Every arm is at rest with its motors off.") == 2     # client and server
    session["thread"].join(10)
    assert session["code"] == 0
    events = read_episode(tmp_path / "run")
    assert events[-2]["action"] == {"action": "home"} and events[-1]["event"] == "end"


class HighTable(SimBackend):
    """A table just under the rest pose, and an arm that once on stands where both ways back to rest cross it."""

    @property
    def table_z(self):
        return 0.12

    def enable(self, arm, cancel=None):
        super().enable(arm, cancel)
        a = self._arms[arm]
        a.q = a.q_target = np.radians([-4.8, -99.3, 26.5, -109.5, 30.8, 123.0])


def test_serve_exits_1_when_an_arm_cannot_be_rested(serve, op, capsys, monkeypatch):
    monkeypatch.setattr(Robot, "open_backend", lambda self: HighTable(self.arm_infos(), cameras=()))
    session = serve("sim")
    op("home")                                     # turns the arm on; the motion itself is refused from there
    assert op("rest")[0] == 1
    code, out = op("quit")
    assert code == 1 and "session ended" in out
    assert out.count("arm is NOT confirmed off (holding: arm: that motion") == 2         # client and server
    session["thread"].join(10)
    assert session["code"] == 1
    assert "Support it" in out and "Every arm is at rest" not in out


def test_serve_resume_continues_the_step_count_and_read_only(serve, op, address):
    assert main(["robot", "init", "simbot", "--sim"]) == 0
    session = serve("serve", "--robot", "simbot")
    assert_done(op("home"))
    assert_done(op("quit"))
    session["thread"].join(10)
    serve("serve", "--robot", "simbot", "--resume", "--read-only")
    status = Client(address).status()
    assert status["steps"] == 1 and status["read_only"]
    code, out = op("home")
    assert code == 1 and "read-only" in out


def test_no_robot_and_no_run_to_resume(capsys):
    assert main(["serve"]) == 2
    assert "robot init" in capsys.readouterr().err
    assert main(["robot", "init", "simbot", "--sim"]) == 0
    assert main(["serve", "--resume", "--address", "unused"]) == 2
    assert "no earlier run of simbot" in capsys.readouterr().err


# ── robot files ──
def write_rig(tmp_path) -> Path:
    rig = tmp_path / "rig.yaml"
    rig.write_text(yaml.safe_dump({
        "arms": {"left": {"backend": "gs_usb", "port": "gs_usb:A"}, "right": {"backend": "gs_usb", "port": "gs_usb:B"}},
        "cameras": {"top": "usb:1", "left_wrist": "usb:2", "right_wrist": "usb:3"}}))
    return rig


def test_robot_init_from_rig(tmp_path, capsys, harness_home):
    rig = write_rig(tmp_path)
    assert main(["robot", "init", "one", "--arm", "metal", "--from-rig", str(rig), "--arms", "left",
                 "--cameras", "top", "left_wrist"]) == 0
    out = capsys.readouterr().out
    assert "bases:" not in out and "llm-harness calibrate --robot one" in out
    saved = yaml.safe_load((harness_home / "robots" / "one.yaml").read_text())
    assert saved["arms"] == {"left": {"backend": "gs_usb", "port": "gs_usb:A"}}
    assert saved["cameras"] == {"top": "usb:1", "left_wrist": "usb:2"}
    assert main(["robot", "init", "two", "--arm", "maker", "--from-rig", str(rig)]) == 0
    assert "edit `bases:` in" in capsys.readouterr().out
    assert main(["robot", "init", "two", "--arm", "maker", "--from-rig", str(rig)]) == 2      # exists
    assert main(["robot", "init", "three", "--from-rig", str(rig)]) == 2                      # no --arm
    assert main(["robot", "init", "three", "--arm", "metal", "--from-rig", str(rig), "--arms", "middle"]) == 2
    assert main(["robot", "init", "three"]) == 2
    assert "no arm named middle" in capsys.readouterr().err


def test_robot_list_show_and_check(tmp_path, capsys, monkeypatch):
    assert main(["robot", "list"]) == 0
    assert "no robots" in capsys.readouterr().out
    assert main(["robot", "init", "simbot", "--sim", "--arm", "maker", "--arms", "left", "right"]) == 0
    assert main(["robot", "list"]) == 0
    assert "simbot: sim maker, arms left right" in capsys.readouterr().out
    assert main(["robot", "show"]) == 0
    assert "arm: maker" in capsys.readouterr().out
    assert main(["robot", "check", "simbot", "--out", str(tmp_path / "check")]) == 0
    out = capsys.readouterr().out
    assert ("arm left: hottest motor 35 C; motors off; 0.0 deg from its rest pose; table z = 0.0000 m from the "
            "simulator") in out and out.rstrip().endswith("OK")
    assert sorted(p.name for p in (tmp_path / "check").iterdir()) == ["look-0001-side.jpg", "look-0001-top.jpg"]

    faulty = sim_backend(arms=("left", "right"))
    faulty.set_fault("right", "motor 3 over temperature")
    monkeypatch.setattr(Robot, "open_backend", lambda self: faulty)
    assert main(["robot", "check", "--out", str(tmp_path / "check2")]) == 1
    assert "PROBLEMS: arm right fault" in capsys.readouterr().out


def test_robot_check_reports_an_uncalibrated_arm_and_a_silent_camera(tmp_path, capsys, monkeypatch):
    class NoTopImage(Uncalibrated):
        def images(self):
            return {k: v for k, v in super().images().items() if k != "top"}

    assert main(["robot", "init", "bench", "--arm", "metal", "--from-rig", str(write_rig(tmp_path)),
                 "--arms", "left", "--cameras", "top"]) == 0
    monkeypatch.setattr(Robot, "open_backend", lambda self: NoTopImage(self.arm_infos(), cameras=["top"]))
    monkeypatch.setattr(cli, "CAMERA_WARMUP_S", 0.0)
    assert main(["robot", "check", "--out", str(tmp_path / "check")]) == 1
    out = capsys.readouterr().out
    assert "not calibrated: run `llm-harness calibrate --robot bench --arm left`" in out
    assert "PROBLEMS: camera top" in out


# ── calibration ──
def test_calibrate_table_z_stores_without_opening_anything(tmp_path, capsys, monkeypatch):
    assert main(["robot", "init", "bench", "--arm", "metal", "--from-rig", str(write_rig(tmp_path))]) == 0
    monkeypatch.setattr(Robot, "open_backend", lambda self: pytest.fail("the robot was opened"))
    assert main(["calibrate", "--table-z", "-0.012"]) == 0
    assert {arm: v["table_z"] for arm, v in load_calibration("bench").items()} == {"left": -0.012, "right": -0.012}
    assert main(["calibrate", "--arm", "right", "--table-z", "0.005"]) == 0
    assert load_calibration("bench")["right"]["table_z"] == 0.005
    assert main(["calibrate", "--arm", "middle", "--table-z", "0"]) == 2
    assert main(["calibrate", "--table-z", "5"]) == 2
    assert "more than 0.5 m from the arm base" in capsys.readouterr().err
    assert load_calibration("bench")["right"]["table_z"] == 0.005
    assert main(["robot", "init", "simbot", "--sim"]) == 0
    assert main(["calibrate", "--robot", "simbot", "--table-z", "0"]) == 2
    assert "simulator robot" in capsys.readouterr().err


class PointingDown(Uncalibrated):
    """Every arm held by hand with the gripper pointing straight down, its tool point 5 cm above z = 0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arm in self._arms.values():
            m = arm.model
            pose = m.pose_from_angles([0.28, 0, 0.05], -math.pi / 2, 0, 0)
            q = np.array(m.chain.ik(m.base_from_world @ pose, m.home_q, limits=(m.ik.lo, m.ik.hi)).q)
            arm.q, arm.q_target = q, q.copy()


def test_calibrate_by_touch(tmp_path, capsys, monkeypatch):
    assert main(["robot", "init", "bench", "--arm", "metal", "--from-rig", str(write_rig(tmp_path))]) == 0
    monkeypatch.setattr(Robot, "open_backend", lambda self: PointingDown(self.arm_infos(), cameras=()))
    said = []
    from makermods_llm_harness import robot as robots
    heights = cli.calibrate(robots.load("bench"), ["left", "right"], ask=lambda prompt: "y", say=said.append)
    with Harness.open(robots.load("bench")) as h:
        q = h.backend.read()["left"].q
        assert h.models["left"].tool_angles(q)[1] == pytest.approx(-math.pi / 2, abs=1e-3)
        tip_z = h.models["left"].fingertip(q)[2]
        assert tip_z == pytest.approx(0.05 - 0.015, abs=1e-3)
        assert h.envelopes["left"].table_z == h.envelopes["right"].table_z == pytest.approx(tip_z)
        assert not h.backend.read()["left"].enabled
    assert heights == {"left": pytest.approx(tip_z), "right": pytest.approx(tip_z)}
    assert "Arm left: point the gripper down at the table, jaws closed." in said
    assert any(s.startswith("left: stored table z = ") for s in said)
    rest = robots.load("bench").rest_q
    assert set(rest) == {"left", "right"} and rest["left"] == pytest.approx(np.degrees(q), abs=0.01)

    monkeypatch.setattr("sys.stdin", io.StringIO("\ny\n"))
    assert main(["calibrate", "--arm", "left"]) == 0
    watched = []
    monkeypatch.setattr(cli, "watch_for_touch", lambda read_z, say: watched.append(read_z()) or (lambda prompt: ""))
    assert main(["calibrate", "--arm", "right", "--watch", "--yes"]) == 0
    assert watched == [pytest.approx(tip_z)]
    assert "right: stored table z" in capsys.readouterr().out


def test_calibrate_refuses_a_touch_with_the_gripper_not_pointing_down(tmp_path, capsys, monkeypatch):
    assert main(["robot", "init", "bench", "--arm", "metal", "--from-rig", str(write_rig(tmp_path))]) == 0
    monkeypatch.setattr(Robot, "open_backend", lambda self: Uncalibrated(self.arm_infos(), cameras=()))
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    assert main(["calibrate", "--arm", "left"]) == 2
    err = capsys.readouterr().err
    assert "left: the gripper points -1 deg from horizontal" in err and "below -30 deg" in err
    assert "Nothing was stored" in err and load_calibration("bench") == {}


# ── logs ──
def test_log_show_and_resume(serve, op, capsys, tmp_path):
    assert main(["log", "show"]) == 2
    session = serve("sim", "--run-dir", str(tmp_path / "run"))
    assert_done(op("home"))
    assert op("move_tool", "2", "0", "0")[0] == 1
    op("quit")
    session["thread"].join(10)
    capsys.readouterr()
    assert main(["log", "show", str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert "   1  home -> done" in out and "   2  move_tool xyz=[2.0, 0.0, 0.0] -> rejected: " in out
    assert main(["log", "resume", str(tmp_path / "run")]) == 0
    assert "2 steps; ended." in capsys.readouterr().out


# ── mcp / run ──
def test_mcp_and_run_dispatch(serve, address, capsys, monkeypatch):
    seen = []
    mcp = types.SimpleNamespace(run=lambda target: seen.append(target))
    agent = types.SimpleNamespace(run=lambda target, task, **options: seen.append((target, task, options))
                                  or types.SimpleNamespace(status=options.get("model", "done"), steps=3, message="ok"))
    monkeypatch.setitem(sys.modules, "makermods_llm_harness.mcp_server", mcp)
    monkeypatch.setitem(sys.modules, "makermods_llm_harness.agent", agent)

    assert main(["mcp", "--sim"]) == 0
    assert isinstance(seen[-1], Harness) and read_episode(seen[-1].log.run_dir)[-1]["event"] == "end"
    assert main(["run", "pick up the cube", "--sim", "--arm", "maker", "--max-steps", "5", "--timeout", "9"]) == 0
    harness, task, options = seen[-1]
    assert task == "pick up the cube" and options == {"max_steps": 5, "timeout_s": 9.0}
    assert {"event": "note", "text": "task: pick up the cube"}.items() <= read_episode(harness.log.run_dir)[1].items()
    assert "done after 3 steps: ok" in capsys.readouterr().out
    assert main(["run", "task", "--sim", "--model", "failed"]) == 1

    serve("sim")
    assert main(["mcp", "--connect", address]) == 0
    assert isinstance(seen[-1], Client) and seen[-1].address == address
    assert main(["mcp", "--sim", "--robot", "x"]) == 2

    monkeypatch.setitem(sys.modules, "makermods_llm_harness.agent", None)
    assert main(["run", "task", "--sim"]) == 2
    assert "makermods-llm-harness[agent]" in capsys.readouterr().err


def test_help_lists_the_vocabulary(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("observe", "move_tool", "nudge", "gripper", "move_joints", "home", "rest", "wait", "calibrate"):
        assert name in out
    assert main(["op", "nudge", "--help"]) == 0
    assert "DX DY DZ" in capsys.readouterr().out


# ── a real process ──
def test_sim_session_in_its_own_process(address, harness_home):
    env = {**os.environ, "LLM_HARNESS_HOME": str(harness_home), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "makermods_llm_harness.cli", "sim", "--address", address],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_session(address, lambda: proc.poll() is None, timeout_s=60)
        assert main(["op", "observe", "--address", address]) == 0
        assert main(["op", "quit", "--address", address]) == 0
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── calibration refusals and the rest pose ──
class HandHeld(PointingDown):
    """PointingDown, with hooks: ``torque`` reported for every arm, the jaws ``opening``."""
    torque, opening = False, 0.0

    def read(self):
        states = super().read()
        for state in states.values():
            state.enabled = self.torque
            state.gripper = self.opening
        return states


@pytest.mark.parametrize("change, words", [
    ({"torque": None}, "not confirmed off"), ({"torque": True}, "torque is on"),
    ({"opening": 0.5}, "close the jaws"), ({"fault": "no fresh feedback from elbow_flex"}, "no fresh feedback"),
])
def test_calibrate_refuses_unconfirmed_torque_open_jaws_and_lost_feedback(tmp_path, monkeypatch, change, words):
    assert main(["robot", "init", "bench", "--arm", "metal", "--from-rig", str(write_rig(tmp_path))]) == 0
    sim = HandHeld(robots_arms("bench"), cameras=())
    for key, value in change.items():
        if key == "fault":
            sim.set_fault("left", value)
        else:
            setattr(sim, key, value)
    monkeypatch.setattr(Robot, "open_backend", lambda self: sim)
    with pytest.raises(cli.CalibrationError, match=words):
        cli.calibrate(load_robot("bench"), ["left"], ask=lambda prompt: "y", say=lambda line: None)
    assert load_calibration("bench") == {}


def robots_arms(name):
    return load_robot(name).arm_infos()


def load_robot(name):
    from makermods_llm_harness import robot as robots
    return robots.load(name)


def test_rest_pose_capture_asks_unless_yes_and_refuses_a_pose_far_past_a_limit(tmp_path, monkeypatch, capsys):
    assert main(["robot", "init", "bench", "--arm", "maker", "--from-rig", str(write_rig(tmp_path)),
                 "--arms", "left"]) == 0
    sim = HandHeld(robots_arms("bench"), cameras=())
    monkeypatch.setattr(Robot, "open_backend", lambda self: sim)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert main(["calibrate", "--rest"]) == 2
    assert "no rest pose recorded" in capsys.readouterr().err and not load_robot("bench").rest_q
    assert main(["calibrate", "--rest", "--yes"]) == 0
    assert "left: rest pose recorded in" in capsys.readouterr().out and load_robot("bench").rest_q["left"]
    arm = sim._arms["left"]
    arm.q = arm.q.copy()
    arm.q[1] = arm.model.hi[1] + math.radians(6)
    assert main(["calibrate", "--rest", "--yes"]) == 2
    assert "shoulder_lift rests 6.0 deg past its limit" in capsys.readouterr().err


def test_resume_refuses_another_robots_episode(tmp_path, capsys):
    assert main(["robot", "init", "simbot", "--sim"]) == 0
    assert main(["robot", "init", "other", "--sim"]) == 0
    from makermods_llm_harness.log import EpisodeLog
    log = EpisodeLog(tmp_path / "run")
    log.start("other", ["arm"], [], {}, {})
    log.close()
    assert main(["serve", "--robot", "simbot", "--resume", str(tmp_path / "run"), "--address", "unused"]) == 2
    assert "an episode of robot 'other', not 'simbot'" in capsys.readouterr().err


# ── the owning lifecycle ──
def test_sigterm_rests_the_arms_and_reports_each_outcome(capsys):
    import signal
    from conftest import sim_backend
    h = Harness.open(backend=sim_backend(cameras=()))
    assert h.step({"action": "home"}).result.ok
    with cli.Lifecycle(h) as life:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
        pytest.fail("SIGTERM did not end the block")
    assert life.interrupted and life.outcomes == {"arm": "released"} and life.code == 0
    assert "Every arm is at rest with its motors off." in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) is not cli._interrupt


def test_run_exits_1_when_an_arm_is_left_on(monkeypatch, capsys):
    monkeypatch.setattr(Robot, "open_backend", lambda self: HighTable(self.arm_infos(), cameras=()))
    agent = types.SimpleNamespace(run=lambda target, task, **options: target.step({"action": "home"}) and
                                  types.SimpleNamespace(status="done", steps=1, message="homed"))
    monkeypatch.setitem(sys.modules, "makermods_llm_harness.agent", agent)
    assert main(["run", "home", "--sim"]) == 1
    out = capsys.readouterr().out
    assert "done after 1 steps: homed" in out and "arm is NOT confirmed off (holding" in out
