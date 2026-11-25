import importlib
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from makermods_llm_harness import Harness, SafetyConfig
from makermods_llm_harness.arm_model import ArmModel
from makermods_llm_harness.backends.base import ArmInfo, ArmTarget
from makermods_llm_harness.backends.real import RealBackend
from makermods_llm_harness.cli import report_outcomes
from makermods_llm_harness.safety import MoveRejected


class FakeCamera:
    """A camera whose newest frame is BGR pure blue, stamped ``age`` seconds ago."""

    def __init__(self, age=0.0, fails=False):
        self.age, self.fails, self.opened, self.closed = age, fails, False, 0

    def open(self):
        if self.fails:
            raise OSError("cannot open camera usb:test")
        self.opened = True

    def newest(self, max_age_s=None):
        if max_age_s is not None and self.age > max_age_s:
            return None
        frame = np.zeros((4, 6, 3), np.uint8)
        frame[..., 0] = 255
        return importlib.import_module("metal_arm.cameras").Frame(frame, time.time() - self.age,
                                                                  time.monotonic() - self.age)

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def harness_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_HARNESS_HOME", str(tmp_path))


@pytest.fixture(params=["metal", "maker"])
def rig(request):
    """(backend, sim, camera) for one simulated arm named "arm" and a fake "top" camera."""
    arm, sim = importlib.import_module(f"{request.param}_arm.testing").simulated_arm()
    camera = FakeCamera()
    backend = RealBackend(request.param, {}, {}, [ArmInfo("arm", request.param)], arms={"arm": arm},
                          cameras={"top": camera}, rest_q={"arm": sim.positions[:6]})
    yield backend, sim, camera
    backend.close()


def wait_for(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return check()


def test_read_before_enable(rig):
    backend, _, camera = rig
    state = backend.read()["arm"]
    assert state.enabled is False and state.fault is None          # torque off confirmed by the motors
    assert state.q.shape == (6,) and np.all(np.isfinite(state.q))
    assert -0.05 <= state.gripper <= 1.05
    assert np.allclose(state.q_target, state.q) and state.gripper_target == state.gripper
    assert np.allclose(state.rest_q, state.q, atol=1e-3)
    assert state.temperature == pytest.approx(30.0)
    assert camera.opened and backend.cameras == ["top"]


def test_targets_move_the_arm_and_read_follows(rig):
    backend, sim, _ = rig
    backend.enable("arm")
    backend.enable("arm")
    start = backend.read()["arm"]
    assert start.enabled
    goal = start.q + np.radians([5, -4, 4, 3, -3, 5])

    def arrived():
        backend.step({"arm": ArmTarget(goal, 0.5)}, 0.02)
        state = backend.read()["arm"]
        return np.abs(state.q - goal).max() < 1e-3 and abs(state.gripper - 0.5) < 0.02

    assert wait_for(arrived)
    state = backend.read()["arm"]
    assert np.allclose(state.q_target, goal, atol=1e-6) and abs(state.gripper_target - 0.5) < 0.03
    assert np.allclose(sim.positions[:6], goal, atol=1e-3)
    assert state.fault is None


def test_refused_target_is_reported(rig):
    backend, _, _ = rig
    q = backend.read()["arm"].q
    backend.step({"arm": ArmTarget(q, 0.0)}, 0.0)
    assert backend.read()["arm"].fault == "targets refused: arm is CONNECTED"
    assert backend.read()["arm"].fault == "targets refused: arm is CONNECTED"     # a read does not consume it
    with pytest.raises(RuntimeError, match="refused the hold"):
        backend.hold("arm")
    backend.enable("arm")
    assert backend.read()["arm"].fault is None
    backend.step({"arm": ArmTarget(q, 0.0)}, 0.0)
    assert backend.read()["arm"].fault is None


def test_motor_fault_and_temperature_show_in_state(rig):
    backend, sim, _ = rig
    sim.set_temperature("elbow_flex", 52.0)
    assert backend.read()["arm"].temperature == pytest.approx(52.0)
    backend.enable("arm")
    sim.fault("wrist_roll")
    assert wait_for(lambda: backend.read()["arm"].fault is not None)
    state = backend.read()["arm"]
    assert "wrist_roll" in state.fault and state.enabled is True       # SDK FAULT: the stiff joints hold
    assert all(sim.motor(j).enabled for j in ("shoulder_pan", "shoulder_lift", "elbow_flex"))


def test_rest_releases_where_the_arm_stands_without_moving_it(rig):
    backend, sim, _ = rig
    backend.enable("arm")
    start = backend.read()["arm"]
    goal = start.q + np.radians([6, 0, 0, 0, 0, 6])
    assert wait_for(lambda: backend.step({"arm": ArmTarget(goal, 0.0)}, 0.02)
                    or np.abs(backend.read()["arm"].q - goal).max() < 1e-3)
    assert backend.rest("arm") == "released"
    state = backend.read()["arm"]
    assert state.enabled is False and np.allclose(state.q, goal, atol=1e-3)
    assert not any(sim.motor(j).enabled for j in range(7))


def test_enable_points_the_sdk_return_at_the_recorded_rest_pose_and_caps_its_speed(rig, monkeypatch):
    backend, sim, _ = rig
    arm = backend.group["arm"]
    seen = {}
    original = arm.enable
    monkeypatch.setattr(arm, "enable", lambda **kw: seen.update(kw) or original(**kw))
    recorded = np.array(sim.positions[:6]) + 0.01
    backend.rest_q["arm"] = recorded
    backend.enable("arm")
    assert 0 < seen["speed"] <= np.radians(5.0) + 1e-9
    assert np.allclose(arm.rest_pose[:6], recorded)


def test_targets_carry_the_speed_cap_to_the_sdk(rig, monkeypatch):
    backend, _, _ = rig
    backend.enable("arm")
    arm = backend.group["arm"]
    speeds = []
    original = arm.move_to
    monkeypatch.setattr(arm, "move_to", lambda q, speed=None, **kw: speeds.append(speed) or original(q, speed, **kw))
    q = backend.read()["arm"].q
    backend.step({"arm": ArmTarget(q, 0.0, 0.1)}, 0.0)
    backend.step({"arm": ArmTarget(q, 0.0, 0.1)}, 0.0)
    assert speeds == [0.1, 0.1]


def test_a_cancelled_enable_leaves_the_motors_off(rig):
    backend, sim, _ = rig
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Exception, match="cancelled"):
        backend.enable("arm", cancel=cancel)
    assert not any(sim.motor(j).enabled for j in range(7)) and backend.read()["arm"].enabled is False


def test_torque_left_on_by_another_program_is_reported_and_blocks_motion():
    testing = importlib.import_module("metal_arm.testing")
    arm, sim = testing.simulated_arm(enabled=True)
    backend = RealBackend("metal", {}, {}, [ArmInfo("arm")], arms={"arm": arm}, rest_q={"arm": sim.positions[:6]})
    h = Harness.open(backend=backend, table_z=-0.5)
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "not from this session" in obs.result.message
    assert obs.arms["arm"].enabled is True
    assert h.close()["arm"].startswith("energized")


def test_old_firmware_torque_state_is_unknown_until_this_session_releases_it():
    testing = importlib.import_module("maker_arm.testing")
    arm, sim = testing.simulated_arm(params_supported=False)
    backend = RealBackend("maker", {}, {}, [ArmInfo("arm", "maker")], arms={"arm": arm},
                          rest_q={"arm": sim.positions[:6]})
    try:
        assert backend.read()["arm"].enabled is None
        backend.enable("arm")
        assert backend.rest("arm") == "released"
        assert backend.read()["arm"].enabled is False
    finally:
        assert backend.close() == {"arm": "released"}


def test_feedback_loss_on_an_arm_that_is_off_is_a_fault_not_a_cached_reading(rig, monkeypatch):
    backend, sim, _ = rig
    monkeypatch.setattr("makermods_llm_harness.backends.real.REFRESH_S", 0.0)
    assert backend.read()["arm"].fault is None
    sim.offline("elbow_flex")
    state = backend.read()["arm"]
    assert "no fresh feedback from elbow_flex" in state.fault and state.enabled is None


def test_images_are_rgb_and_stale_frames_are_left_out():
    arm, _ = importlib.import_module("metal_arm.testing").simulated_arm()
    cameras = {"top": FakeCamera(), "old": FakeCamera(age=5.0), "broken": FakeCamera(fails=True)}
    backend = RealBackend("metal", {}, {}, [ArmInfo("arm")], arms={"arm": arm}, cameras=cameras)
    try:
        frames = backend.images()
        assert set(frames) == {"top"}
        image = frames["top"].image
        assert image.shape == (4, 6, 3) and image.dtype == np.uint8
        assert (image[..., 2] == 255).all() and (image[..., 0] == 0).all()
        assert abs(frames["top"].stamp - time.time()) < 1.0
        assert 0 <= time.monotonic() - frames["top"].captured < 1.0
        assert backend.cameras == ["top", "old", "broken"]
        assert "cannot open camera" in backend.camera_errors["broken"]
        assert backend.read()["arm"].fault is None
    finally:
        backend.close()
    assert cameras["top"].closed == 1 and cameras["broken"].closed == 0


def test_close_leaves_an_enabled_arm_holding_with_torque_on(rig):
    backend, sim, camera = rig
    backend.enable("arm")
    before = np.array(sim.positions[:6])
    backend.close()
    backend.close()
    assert camera.closed == 1
    assert backend.closed_as == {"arm": "detached"}
    assert all(sim.motor(j).enabled for j in range(6))
    time.sleep(0.2)
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - before)).max() < 0.05


def test_close_after_rest_releases(rig):
    backend, sim, _ = rig
    backend.enable("arm")
    assert backend.rest("arm") == "released"
    backend.close()
    assert backend.closed_as == {"arm": "released"}
    assert not any(sim.motor(j).enabled for j in range(6))


def test_a_ramp_of_waypoints_is_tracked(rig):
    backend, sim, _ = rig
    backend.enable("arm")
    start = backend.read()["arm"].q
    goal = start + np.radians([6, -5, 5, 4, -4, 6])
    lag = 0.0
    for u in np.linspace(0, 1, 26)[1:]:
        q = start + u * (goal - start)
        backend.step({"arm": ArmTarget(q, 0.0)}, 0.04)
        lag = max(lag, float(np.abs(backend.read()["arm"].q - q).max()))
    assert np.degrees(lag) < 3.0
    assert wait_for(lambda: np.abs(backend.read()["arm"].q - goal).max() < math.radians(0.2), timeout=5.0)


def test_close_without_enable():
    arm, _ = importlib.import_module("maker_arm.testing").simulated_arm()
    backend = RealBackend("maker", {}, {}, [ArmInfo("arm", "maker")], arms={"arm": arm})
    backend.close()
    backend.close()


def test_step_keeps_a_steady_period(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda s: setattr(clock, "now", clock.now + s)
    monkeypatch.setattr("makermods_llm_harness.backends.real.time", clock)
    arm, _ = importlib.import_module("metal_arm.testing").simulated_arm()
    backend = RealBackend("metal", {}, {}, [ArmInfo("arm")], arms={"arm": arm})
    try:
        start = clock.now
        for _ in range(10):
            backend.step({}, 0.02)
            clock.now += 0.005              # work between steps is absorbed by the period
        assert clock.now - start == pytest.approx(0.205)
        clock.now += 0.1                    # late by several periods: the schedule restarts
        start = clock.now
        backend.step({}, 0.02)
        assert clock.now - start == pytest.approx(0.02)
    finally:
        backend.close()


def test_two_arms_from_robot_description():
    testing = importlib.import_module("metal_arm.testing")
    (left, _), (right, right_sim) = testing.simulated_arm(), testing.simulated_arm()
    robot = SimpleNamespace(arm="metal", arms={"left": {}, "right": {}}, cameras=None,
                            arm_infos=lambda: [ArmInfo("left", base_xyz=(0, 0.25, 0)),
                                               ArmInfo("right", base_xyz=(0, -0.25, 0))])
    robot.rest_q = {}
    backend = RealBackend.from_robot(robot, arms={"left": left, "right": right})
    try:
        assert [a.name for a in backend.arms] == ["left", "right"] and backend.cameras == []
        backend.enable("right")
        state = backend.read()
        assert state["right"].enabled and not state["left"].enabled
        goal = state["right"].q + np.radians([4, -4, 4, 0, 0, 4])
        assert wait_for(lambda: backend.step({"right": ArmTarget(goal, 0.0)}, 0.02)
                        or np.abs(backend.read()["right"].q - goal).max() < 1e-3)
        assert np.allclose(right_sim.positions[:6], goal, atol=1e-3)
    finally:
        backend.close()


def test_a_harness_drives_simulated_sdk_arms_from_rest_to_rest(rig):
    """The SDK's simulated rest pose sits a hair past a joint limit; home, nudge, gripper and rest all pass."""
    backend, sim, _ = rig
    info = backend.arms[0]
    rest = backend.read()["arm"].q
    table = float(ArmModel(info).lowest_z(rest)) - 0.10
    safety = SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40, slow_zone_m=0.03)
    h = Harness.open(backend=backend, table_z=table, safety=safety)
    assert h.envelopes["arm"].table_z == table
    obs = h.step({"action": "home"})
    assert obs.result.ok, obs.result.message
    assert np.allclose(np.radians(list(obs.arms["arm"].joints_deg.values())), h.models["arm"].home_q, atol=0.01)
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.02]})
    assert obs.result.ok, obs.result.message
    obs = h.step({"action": "gripper", "opening": 0.5})
    assert obs.result.ok and obs.arms["arm"].gripper == pytest.approx(0.5, abs=0.05)
    obs = h.rest()
    assert obs.result.ok, obs.result.message
    assert obs.arms["arm"].enabled is False
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - rest)).max() < 2.0
    assert h.close() == {"arm": "released"}
    assert backend.closed_as == {"arm": "released"}


class Sagging(RealBackend):
    """Measures shoulder_lift ``bias`` rad off where the simulated motor stands, once the arm is on: a standing
    error such as gravity leaves on a real arm."""
    bias = math.radians(0.5)

    def read(self):
        states = super().read()
        for state in states.values():
            if state.enabled:
                state.q = state.q.copy()
                state.q[1] += self.bias
        return states


def test_small_nudges_add_up_from_the_commanded_pose_despite_a_standing_error():
    arm, sim = importlib.import_module("metal_arm.testing").simulated_arm()
    backend = Sagging("metal", {}, {}, [ArmInfo("arm")], arms={"arm": arm}, rest_q={"arm": sim.positions[:6]})
    model = ArmModel(backend.arms[0])
    table = float(model.lowest_z(backend.read()["arm"].q)) - 0.10
    h = Harness.open(backend=backend, table_z=table,
                     safety=SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40, slow_zone_m=0.03))
    try:
        assert h.step({"action": "home"}).result.ok
        start = model.tool_pose(sim.positions[:6])[:3, 3]
        measured = model.tool_pose(backend.read()["arm"].q)[:3, 3]
        assert np.linalg.norm(measured - start) > 0.002                    # the standing error, in the tool point
        for _ in range(3):
            obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.007]})
            assert obs.result.ok, obs.result.message
            assert obs.result.error["position_m"] < 0.01
        commanded = model.tool_pose(sim.positions[:6])[:3, 3]
        assert np.allclose(commanded, start + [0, 0, -0.021], atol=0.001)
    finally:
        h.close()


def test_a_faulted_arm_is_neither_moved_nor_released(rig):
    """A gripper fault puts the SDK arm in FAULT with the six joints still holding: rest and close leave it."""
    backend, sim, _ = rig
    rest = backend.read()["arm"].q
    table = float(ArmModel(backend.arms[0]).lowest_z(rest)) - 0.10
    safety = SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40, slow_zone_m=0.03)
    h = Harness.open(backend=backend, table_z=table, safety=safety)
    assert h.step({"action": "home"}).result.ok
    held = np.array(sim.positions[:6])
    sim.fault("gripper")
    assert wait_for(lambda: backend.read()["arm"].fault is not None)
    obs = h.rest()
    assert obs.result.status == "fault" and obs.arms["arm"].enabled is True
    time.sleep(0.1)
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - held)).max() < 0.05
    outcome = h.close()["arm"]
    assert outcome.startswith("detached: ") and "gripper" in outcome
    assert all(sim.motor(j).enabled for j in range(6))


def test_missing_arm_is_refused():
    arm, _ = importlib.import_module("metal_arm.testing").simulated_arm()
    with pytest.raises(ValueError, match="no arm given for right"):
        RealBackend("metal", {}, {}, [ArmInfo("left"), ArmInfo("right")], arms={"left": arm})


def test_rest_from_home_reaches_a_rest_pose_2_mm_above_the_table():
    """The rest pose lies under the floor plus allowance: the arm starts in recovery, rises home, and the way
    back ends at the rest pose all the same (the Metal arm folded at zero, as on the bench)."""
    arm, sim = importlib.import_module("metal_arm.testing").simulated_arm()
    backend = RealBackend("metal", {}, {}, [ArmInfo("arm")], arms={"arm": arm}, rest_q={"arm": sim.positions[:6]})
    rest = backend.read()["arm"].q
    model = ArmModel(backend.arms[0])
    table = float(model.lowest_z(rest)) - 0.002
    safety = SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40, floor_margin_m=0.010, floor_tolerance_deg=2.0,
                          sag_allowance_m=0.005)     # a strict envelope, as on the bench where this was found
    h = Harness.open(backend=backend, table_z=table, safety=safety)
    env = h.envelopes["arm"]
    assert "recovery mode" in env.rest_note(rest)
    goal = np.clip(rest, model.lo, model.hi)
    down = model.plan_joint_path(model.home_q, goal, env.slow_speed, safety.rate_hz)
    with pytest.raises(MoveRejected, match="hard floor"):
        env.check(model.home_q, down, capped=False)
    env.check(model.home_q, down, capped=False, rest_q=goal)
    with pytest.raises(MoveRejected):
        env.check(model.home_q, down[:-5], capped=False, rest_q=goal)    # not ending at the rest pose
    obs = h.step({"action": "home"})
    assert obs.result.ok, obs.result.message
    obs = h.rest()
    assert obs.result.ok, obs.result.message
    assert obs.arms["arm"].enabled is False
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - rest)).max() < 2.0
    assert h.close() == {"arm": "released"}


@pytest.mark.parametrize("release", [False, True])
def test_an_exit_that_cannot_rest_asks_the_operator_and_names_the_rest_command(rig, release):
    backend, sim, _ = rig
    rest = backend.read()["arm"].q
    table = float(ArmModel(backend.arms[0]).lowest_z(rest)) - 0.10
    h = Harness.open(backend=backend, table_z=table, safety=SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40))
    assert h.step({"action": "home"}).result.ok
    sim.fault("gripper")
    assert wait_for(lambda: backend.read()["arm"].fault is not None)
    asked = []
    backend.ask_operator = lambda texts: asked.append(texts) or release
    outcomes = h.close()
    command = f"{backend.cli} rest --arm arm --pose shoulder_pan="
    assert command in asked[0]["arm"] and "the arm drops" in asked[0]["arm"]
    lines = []
    assert report_outcomes(outcomes, lines.append, backend.recovery) == (0 if release else 1)
    if release:
        assert outcomes == {"arm": "released_unrested"} and not any(sim.motor(j).enabled for j in range(6))
        assert "NOT at its rest pose" in lines[0]
    else:
        assert outcomes["arm"].startswith("detached") and all(sim.motor(j).enabled for j in range(6))
        assert command in lines[0] and "release" in lines[0]


@pytest.mark.parametrize("kind", ["metal", "maker"])
def test_an_arm_left_energized_is_attached_and_rested(kind):
    testing = importlib.import_module(f"{kind}_arm.testing")
    first, sim = testing.simulated_arm()
    rest = np.array(sim.positions[:6])
    model = ArmModel(ArmInfo("arm", kind))
    first.connect()
    first.enable()
    assert first.move_to(list(model.home_q) + first.get_joint_positions()[6:], speed=3.0, timeout=10.0)
    first.detach()
    sim.watchdog_timeout("gripper")
    held = np.array(sim.positions[:6])

    def backend_for(attach):
        arm = importlib.import_module(f"{kind}_arm.arm").Arm(sim.config, first.backend)
        return RealBackend(kind, {}, {}, [ArmInfo("arm", kind)], arms={"arm": arm}, rest_q={"arm": rest},
                           attach=attach)

    plain = backend_for(False)
    state = plain.read()["arm"]
    assert state.enabled is True and "--attach" in state.fault and f"{kind}-arm rest --arm arm" in state.fault
    assert plain.close() == {"arm": "energized"}

    backend = backend_for(True)
    state = backend.read()["arm"]
    assert state.enabled is True and state.fault is None
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - held)).max() < 0.05     # taken over without moving
    table = float(model.lowest_z(rest)) - 0.10
    h = Harness.open(backend=backend, table_z=table, safety=SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40))
    obs = h.rest()
    assert obs.result.ok, obs.result.message
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - rest)).max() < 2.0
    assert h.close() == {"arm": "released"}


def test_a_hold_returns_the_sdk_setpoint_and_waits_two_control_periods(rig):
    backend, sim, _ = rig
    backend.enable("arm")
    q, opening = backend.hold("arm")
    commanded = backend.group["arm"].get_commanded_positions()
    assert np.allclose(q, commanded[:6]) and -0.01 <= opening <= 1.01      # the measured opening, unclipped
    rate = backend.group["arm"].config.control_rate_hz
    assert backend.hold_settle_s == pytest.approx(2.0 / rate)


def held_by_a_dead_program(kind="maker"):
    """An arm with motors that cannot report torque state, left holding at home by a session that ended; its
    joints then measure load, as a real arm carrying its weight does (the SDK simulator models none)."""
    testing = importlib.import_module(f"{kind}_arm.testing")
    first, sim = testing.simulated_arm(params_supported=False)
    rest = np.array(sim.positions[:6])
    model = ArmModel(ArmInfo("arm", kind))
    first.connect()
    first.enable()
    assert first.move_to(list(model.home_q) + first.get_joint_positions()[6:], speed=3.0, timeout=10.0)
    first.detach()
    arm = importlib.import_module(f"{kind}_arm.arm").Arm(sim.config, first.backend)
    real = arm.get_joint_torques
    arm.get_joint_torques = lambda: [t + (1.4 if i == 1 else 0.0) for i, t in enumerate(real())]
    return arm, sim, model, rest


def test_an_old_firmware_arm_carrying_load_is_held_by_another_program_and_attach_takes_it_over():
    arm, sim, model, rest = held_by_a_dead_program()
    plain = RealBackend("maker", {}, {}, [ArmInfo("arm", "maker")], arms={"arm": arm}, rest_q={"arm": rest})
    state = plain.read()["arm"]
    assert state.enabled is True and "another program probably holds it" in state.fault
    assert "--attach" in state.fault and "shoulder_lift +1.4" in state.fault
    h = Harness.open(backend=plain, table_z=-0.5)
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "--attach" in obs.result.message        # nothing sent to it
    h.close()
    arm, sim, model, rest = held_by_a_dead_program()
    held = np.array(sim.positions[:6])
    backend = RealBackend("maker", {}, {}, [ArmInfo("arm", "maker")], arms={"arm": arm}, rest_q={"arm": rest},
                          attach=True)
    state = backend.read()["arm"]
    assert state.enabled is True and state.fault is None
    assert np.degrees(np.abs(np.array(sim.positions[:6]) - held)).max() < 0.05      # taken over where it held
    h = Harness.open(backend=backend, table_z=float(model.lowest_z(rest)) - 0.10,
                     safety=SafetyConfig(max_speed_deg_s=120, slow_speed_deg_s=40))
    assert h.rest().result.ok
    assert h.close() == {"arm": "released"}


def test_an_enable_the_sdk_refuses_for_a_held_arm_names_attach(rig, monkeypatch):
    backend, sim, _ = rig
    kind = "maker" if "maker" in type(backend.group["arm"]).__module__ else "metal"
    errors = importlib.import_module(f"{kind}_arm.errors")

    def refuse(*args, **kwargs):
        raise errors.HeldByAnotherProgram("motors 2 (shoulder_lift) are producing torque")
    monkeypatch.setattr(backend.group["arm"], "enable", refuse)
    h = Harness.open(backend=backend, table_z=-0.5)
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "another program holds this arm" in obs.result.message
    assert "llm-harness serve --attach" in obs.result.message and f"{kind}-arm rest --arm arm" in obs.result.message


FAKE_CAMERAS = '''
import time
import numpy as np
from maker_arm.cameras import Frame

class Camera:
    def __init__(self, id):
        self.id = id

    def open(self):
        if self.id == "broken":
            raise OSError("no such camera")

    def newest(self, max_age_s=None):
        if self.id == "old" and max_age_s is not None:
            return None
        if self.id == "stalls":
            time.sleep(3600)
        return Frame(np.full((4, 6, 3), 7, np.uint8), time.time(), time.monotonic())

    def close(self):
        pass
'''


def test_cameras_built_from_ids_run_in_a_helper_process(tmp_path, monkeypatch):
    from makermods_llm_harness.backends.camera_process import CameraProcess
    (tmp_path / "fake_cameras.py").write_text(FAKE_CAMERAS)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    camera = CameraProcess("fake_cameras", "front")
    camera.open()
    try:
        pid = camera._proc.pid
        assert pid != __import__("os").getpid()
        frame = camera.newest(1.0)
        assert frame[0].shape == (4, 6, 3) and int(frame[0][0, 0, 0]) == 7 and frame.captured is not None
    finally:
        camera.close()
    assert camera.newest(1.0) is None                                     # closed: no frame, no error
    old = CameraProcess("fake_cameras", "old")
    old.open()
    assert old.newest(1.0) is None
    old.close()
    with pytest.raises(OSError, match="did not open: no such camera"):
        CameraProcess("fake_cameras", "broken").open()


@pytest.mark.parametrize("then", ["wait", "close"])
def test_a_camera_helper_that_stalls_is_killed_and_never_blocks(tmp_path, monkeypatch, then):
    import os
    import signal
    from makermods_llm_harness.backends import camera_process
    (tmp_path / "fake_cameras.py").write_text(FAKE_CAMERAS)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    frame_s, close_s = getattr(camera_process, "FRAME_S", 1.0), getattr(camera_process, "CLOSE_S", 1.0)
    camera = camera_process.CameraProcess("fake_cameras", "stalls")
    camera.open()
    proc = camera._proc
    got = []
    reader = threading.Thread(target=lambda: got.append(camera.newest(1.0)), daemon=True)
    try:
        start = time.monotonic()
        reader.start()
        if then == "close":                       # quit while a frame request is stuck: close does not wait on it
            time.sleep(0.3)
            closer = threading.Thread(target=camera.close, daemon=True)
            closer.start()
            closer.join(close_s + 1.0)
            assert not closer.is_alive()
        reader.join(frame_s + 1.5)
        assert not reader.is_alive() and got == [None] and time.monotonic() - start < frame_s + 1.5
        deadline = time.monotonic() + 2.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None                                    # killed and reaped
        t = time.monotonic()
        assert camera.newest(1.0) is None and time.monotonic() - t < 0.1  # no frame, no wait, from then on
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
        camera.close()


def test_a_backend_from_camera_ids_uses_helper_processes(monkeypatch):
    built = []

    class Recorder:
        def __init__(self, module, cid):
            built.append((module, cid))

        def open(self):
            pass

        def newest(self, max_age_s=None):
            return None

        def close(self):
            pass
    monkeypatch.setattr("makermods_llm_harness.backends.real.CameraProcess", Recorder)
    arm, sim = importlib.import_module("maker_arm.testing").simulated_arm()
    backend = RealBackend("maker", {}, {"front": "url:http://127.0.0.1:1/front"}, [ArmInfo("arm", "maker")],
                          arms={"arm": arm})
    backend.close()
    assert built == [("maker_arm.cameras", "url:http://127.0.0.1:1/front")]


def test_the_camera_helper_takes_the_slowest_cpus_only_when_they_differ(tmp_path, monkeypatch):
    from makermods_llm_harness.backends.camera_process import slow_cpus
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    for cpu, capacity in enumerate((446, 446, 1024, 1024)):
        (tmp_path / f"cpu{cpu}").mkdir()
        (tmp_path / f"cpu{cpu}" / "cpu_capacity").write_text(f"{capacity}\n")
    assert slow_cpus(str(tmp_path)) == {0, 1}
    for cpu in range(4):
        (tmp_path / f"cpu{cpu}" / "cpu_capacity").write_text("1024\n")
    assert slow_cpus(str(tmp_path)) is None
    assert slow_cpus(str(tmp_path / "missing")) is None
