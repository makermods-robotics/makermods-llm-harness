import math
import threading
import time

import cv2
import numpy as np
import pytest

from makermods_llm_harness import Harness, Rest
from makermods_llm_harness.arm_model import ArmModel
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.log import EpisodeLog, read_episode
from makermods_llm_harness.safety import Envelope, MoveRejected

from conftest import APPROACH, GRASP_PITCH, GRASP_X, GRASP_Z, sim_backend


def move(xyz, pitch=None, arm=None, **extra):
    action = {"action": "move_tool", "xyz": list(xyz), **extra}
    if pitch is not None:
        action["pitch_deg"] = pitch
    if arm is not None:
        action["arm"] = arm
    return action


def assert_done(obs):
    assert obs.result.status == "done", obs.result.message


# ── the whole loop ──
@pytest.mark.parametrize("arm_type", ["metal", "maker"])
def test_scripted_pick(sim_harness, arm_type):
    h = sim_harness(arm=arm_type)
    assert_done(h.step({"action": "home"}))
    for xyz, pitch in APPROACH[arm_type]:
        assert_done(h.step(move(xyz, pitch)))
    assert_done(h.step({"action": "gripper", "opening": "open"}))
    assert_done(h.step(move((GRASP_X[arm_type], 0, GRASP_Z[arm_type]), GRASP_PITCH[arm_type])))
    obs = h.step({"action": "gripper", "opening": "close"})
    assert_done(obs)
    assert obs.arms["arm"].gripper == pytest.approx(0.5)     # the 4 cm cube in the 8 cm jaws
    obs = h.step({"action": "nudge", "dxyz": [0, 0, 0.08]})
    assert_done(obs)
    cube = h.backend.objects[0]
    assert cube["held_by"] == "arm" and cube["xyz"][2] > 0.09
    assert obs.arms["arm"].tool.xyz[2] == pytest.approx(GRASP_Z[arm_type] + 0.08, abs=2e-3)
    assert_done(h.step({"action": "rest"}))
    assert not h.observe().arms["arm"].enabled


def test_every_target_sent_was_checked_or_is_a_hold_or_gripper_command(sim_harness, monkeypatch):
    checked = []
    original = Envelope.check

    def recording_check(self, q_start, path, capped=True, rest_q=None):
        original(self, q_start, path, capped, rest_q)
        checked.extend(np.asarray(path))

    monkeypatch.setattr(Envelope, "check", recording_check)
    sent = []

    class Recording(SimBackend):
        def step(self, targets, dt):
            for name, target in targets.items():
                state = self.read()[name]
                sent.append((np.array(target.q), state.q.copy(), state.q_target.copy()))
            super().step(targets, dt)

    base = sim_backend()
    h = sim_harness(backend=Recording(base.arms, objects=base.objects))
    actions = [{"action": "home"}, *(move(xyz, p) for xyz, p in APPROACH["metal"]),
               {"action": "gripper", "opening": 1}, move((0.28, 0, GRASP_Z["metal"]), -90),
               {"action": "gripper", "opening": 0},
               {"action": "nudge", "dxyz": [0, 0.02, 0.05]}, {"action": "nudge", "dxyz": [0, 0, -0.5]},
               {"action": "move_joints", "joints_deg": {"wrist_roll": 20}, "relative": True},
               move((2, 0, 0)), {"action": "wait", "seconds": 0.2}]
    for action in actions:
        h.step(action)
    h.backend.block("arm")
    assert h.step({"action": "nudge", "dxyz": [0, 0, 0.05]}).result.status == "stopped"
    h.backend.block("arm", False)
    h.step({"action": "rest"})
    assert len(sent) > 100
    checked_rows = {row.tobytes() for row in checked}
    for q, measured, commanded in sent:
        assert q.tobytes() in checked_rows or np.array_equal(q, measured) or np.array_equal(q, commanded), q


# ── arms and frames ──
def test_two_arms_need_a_name_and_rest_rests_both(sim_harness):
    h = sim_harness(arms=("left", "right"))
    obs = h.step({"action": "home"})
    assert obs.result.status == "rejected" and obs.result.message == "name the arm: left or right."
    obs = h.step({"action": "home", "arm": "middle"})
    assert obs.result.status == "rejected" and "left, right" in obs.result.message
    assert_done(h.step({"action": "home", "arm": "left"}))
    assert_done(h.step({"action": "home", "arm": "right"}))
    obs = h.rest()
    assert_done(obs)
    assert not obs.arms["left"].enabled and not obs.arms["right"].enabled
    assert_done(h.step({"action": "wait", "seconds": 0.1}))
    assert "not checked against each other" in h.describe()


def test_world_frame_targets_with_a_yawed_offset_base(sim_harness):
    bases = {"left": {"xyz": [0.1, 0.3, 0.02], "yaw_deg": -60}}
    h = sim_harness(arms=("left", "right"), bases=bases)
    assert_done(h.step({"action": "home", "arm": "left"}))
    model = h.models["left"]
    base = model.world_from_base
    target = base[:3, :3] @ [0.28, 0.04, 0.33] + base[:3, 3]
    obs = h.step(move(target, -30, arm="left"))
    assert_done(obs)
    tool = obs.arms["left"].tool
    assert np.allclose(tool.xyz, target, atol=2e-3)
    away = target[:2] - base[:2, 3]
    assert tool.yaw_deg == pytest.approx(math.degrees(math.atan2(away[1], away[0])), abs=0.5)
    assert tool.pitch_deg == pytest.approx(-30, abs=0.5)


def test_move_tool_defaults_keep_pitch_and_roll(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    before = h.observe().arms["arm"].tool
    obs = h.step(move((0.30, 0.05, 0.55)))
    assert_done(obs)
    after = obs.arms["arm"].tool
    assert after.pitch_deg == pytest.approx(before.pitch_deg, abs=0.5)
    assert after.roll_deg == pytest.approx(before.roll_deg, abs=0.5)
    assert after.yaw_deg == pytest.approx(math.degrees(math.atan2(0.05, 0.30)), abs=0.5)
    obs = h.step(move((0.30, 0.05, 0.53), yaw_deg=0, roll_deg=10))
    assert_done(obs)
    assert obs.arms["arm"].tool.yaw_deg == pytest.approx(0, abs=0.5)
    assert obs.arms["arm"].tool.roll_deg == pytest.approx(10, abs=0.5)


@pytest.mark.parametrize("frame", ["base", "tool"])
def test_nudge_uses_the_frame_axes_and_keeps_orientation(sim_harness, frame):
    h = sim_harness(bases={"arm": {"xyz": [0, 0, 0], "yaw_deg": 90}})
    assert_done(h.step({"action": "home"}))
    model = h.models["arm"]
    q0 = h.backend.read()["arm"].q_target
    pose = model.tool_pose(q0)
    axes = pose[:3, :3] if frame == "tool" else model.world_from_base[:3, :3]
    obs = h.step({"action": "nudge", "dxyz": [0.03, 0, 0], "frame": frame})
    assert_done(obs)
    q1 = h.backend.read()["arm"].q_target
    assert np.allclose(model.tool_pose(q1)[:3, 3] - pose[:3, 3], axes[:, 0] * 0.03, atol=1e-3)
    assert np.allclose(model.tool_pose(q1)[:3, :3], pose[:3, :3], atol=0.01)


# ── joints ──
def test_move_joints_names_ranges_and_relative(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    obs = h.step({"action": "move_joints", "joints_deg": {"elbow": 10}})
    assert obs.result.status == "rejected" and "shoulder_pan, shoulder_lift, elbow_flex" in obs.result.message
    obs = h.step({"action": "move_joints", "joints_deg": {"gripper": 10}})
    assert obs.result.status == "rejected" and "use the gripper action" in obs.result.message
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_yaw": 100}})
    assert obs.result.status == "rejected" and "[-85, 85]" in obs.result.message
    start = obs.arms["arm"].joints_deg["wrist_roll"]
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 15}, "relative": True})
    assert_done(obs)
    assert obs.arms["arm"].joints_deg["wrist_roll"] == pytest.approx(start + 15, abs=0.2)
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 60}})
    assert obs.result.status == "rejected" and "35 deg" in obs.result.message


@pytest.mark.parametrize("xyz, pitch", [((0.30, 0, 0.30), -88), ((0.30, 0, 0.29), -85)])
def test_poses_reported_unreachable_on_the_real_arm_are_reached(sim_harness, xyz, pitch):
    """From the tool at [0.297, 0.001, 0.323] pitched -69 deg, as on the real Metal arm."""
    sim = sim_backend()
    h = sim_harness(backend=sim)
    model = h.models["arm"]
    start = model.pose_from_angles([0.297, 0.001, 0.323], math.radians(-69), math.atan2(0.001, 0.297), 0.0)
    place(sim, model.chain.ik(start, np.radians([0.4, -63.5, 68.9, -75, 0, 0]), limits=(model.ik.lo, model.ik.hi)).q)
    obs = h.step(move(xyz, pitch))
    assert_done(obs)
    assert obs.result.error["position_m"] <= 0.005 and obs.result.error["angle_deg"] <= 2.0
    assert np.allclose(obs.arms["arm"].tool.xyz, xyz, atol=0.005) and obs.arms["arm"].tool.pitch_deg == \
        pytest.approx(pitch, abs=2.0)


def test_home_is_exempt_from_the_excursion_cap(sim_harness):
    h = sim_harness()
    obs = h.step({"action": "home"})
    assert_done(obs)
    assert obs.arms["arm"].joints_deg["elbow_flex"] == pytest.approx(math.degrees(1.5), abs=0.2)


# ── refusals and failures ──
def test_bad_actions_are_rejected_and_counted(sim_harness):
    h = sim_harness()
    obs = h.step({"action": "jump"})
    assert obs.result.status == "rejected" and obs.step == 1 and obs.result.action == {"action": "jump"}
    obs = h.step("not json")
    assert obs.result.status == "rejected" and obs.step == 2
    obs = h.step(move((1.5, 0, 0.2)))
    assert obs.result.status == "rejected" and "out of reach" in obs.result.message
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.5]})
    assert obs.result.status == "rejected"


def test_floor_rejection_moves_nothing(sim_harness):
    h = sim_harness(arm="maker")
    assert_done(h.step({"action": "home"}))
    before = h.observe().arms["arm"].joints_deg
    obs = h.step(move((0.28, 0, 0.005), -45))
    assert obs.result.status == "rejected" and "hard floor" in obs.result.message
    assert obs.arms["arm"].joints_deg == before


def test_read_only_allows_only_observe_and_wait(sim_harness):
    h = sim_harness(read_only=True)
    obs = h.step({"action": "home"})
    assert obs.result.status == "rejected" and "read-only" in obs.result.message
    assert not obs.arms["arm"].enabled
    assert_done(h.step({"action": "observe"}))
    assert_done(h.step({"action": "wait", "seconds": 0.1}))


def test_fault_before_and_during_a_motion(sim_harness):
    h = sim_harness()
    h.backend.set_fault("arm", "motor 4 overheated")
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "motor 4 overheated" in obs.result.message
    assert obs.arms["arm"].fault == "motor 4 overheated"

    class FaultsMidway(SimBackend):
        ticks = 0

        def step(self, targets, dt):
            FaultsMidway.ticks += 1
            if FaultsMidway.ticks == 10:
                self.set_fault("arm", "stale feedback")
            super().step(targets, dt)

    base = sim_backend()
    h = sim_harness(backend=FaultsMidway(base.arms))
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "stale feedback" in obs.result.message


def test_enable_failure_and_backend_errors_are_faults_not_exceptions(sim_harness):
    class Broken(SimBackend):
        def enable(self, arm, cancel=None):
            raise OSError("no CAN adapter")

    base = sim_backend()
    h = sim_harness(backend=Broken(base.arms))
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "no CAN adapter" in obs.result.message

    class Crashing(SimBackend):
        def step(self, targets, dt):
            raise OSError("bus off")

    h = sim_harness(backend=Crashing(base.arms))
    obs = h.step({"action": "home"})
    assert obs.result.status == "fault" and "bus off" in obs.result.message


def test_hot_motor_rejects_motion(sim_harness):
    h = sim_harness()
    h.backend.set_temperature("arm", 75)
    obs = h.step({"action": "home"})
    assert obs.result.status == "rejected" and "75 C" in obs.result.message
    assert not obs.arms["arm"].enabled


def test_blocked_arm_stops_and_holds(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 30}, "relative": True})
    assert obs.result.status == "stopped" and "blocked" in obs.result.message
    state = h.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)


def test_stop_ends_the_motion_and_the_arm_holds(sim_harness):
    class StopsMidway(SimBackend):
        harness = None
        ticks = 0

        def step(self, targets, dt):
            StopsMidway.ticks += 1
            if StopsMidway.ticks == 20:
                threading.Thread(target=StopsMidway.harness.stop).start()
                threading.Event().wait(0.02)
            super().step(targets, dt)

    base = sim_backend()
    h = sim_harness(backend=StopsMidway(base.arms))
    StopsMidway.harness = h
    obs = h.step({"action": "home"})
    assert obs.result.status == "stopped" and "on request" in obs.result.message
    assert StopsMidway.ticks < 30
    assert_done(h.step({"action": "home"}))


def test_rest_ends_at_a_rest_pose_just_above_the_table(sim_harness):
    h = sim_harness(table_z=0.12)
    assert h.observe().arms["arm"].clearance_m == pytest.approx(0.0017, abs=5e-4)
    assert_done(h.step({"action": "home"}))
    obs = h.step({"action": "rest"})
    assert_done(obs)
    assert not obs.arms["arm"].enabled


def test_rest_is_rejected_when_the_way_back_breaks_the_floor(sim_harness):
    sim = sim_backend()
    h = sim_harness(backend=sim, table_z=0.12)
    place(sim, STUCK_Q)
    obs = h.step({"action": "rest"})
    assert obs.result.status == "rejected" and "lift the arm" in obs.result.message
    assert obs.arms["arm"].enabled


def test_rest_reports_an_unconfirmed_return(sim_harness):
    class Unconfirmed(SimBackend):
        def rest(self, arm):
            return "unknown"

    base = sim_backend()
    h = sim_harness(backend=Unconfirmed(base.arms))
    assert_done(h.step({"action": "home"}))
    obs = h.rest()
    assert obs.result.status == "stopped" and "not released (unknown)" in obs.result.message
    assert h.close() == {"arm": "unknown: " + obs.result.message}


# ── observations, log, prompt ──
def test_observation_images_are_jpegs_saved_per_step_and_per_look(sim_harness, tmp_path):
    img = tmp_path / "img"
    h = sim_harness(image_dir=img)
    obs = h.observe()
    assert obs.step == 0 and set(obs.cameras) == {"top", "side"}
    top = obs.cameras["top"]
    assert (top.width, top.height) == (320, 240) and top.path == str(img / "look-0001-top.jpg")
    image = cv2.imdecode(np.frombuffer(top.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (240, 320, 3)
    assert (img / "look-0001-top.jpg").read_bytes() == top.jpeg
    step = h.step({"action": "home"})
    assert step.cameras["side"].path == str(img / "step-0001-side.jpg")
    saved = (img / "step-0001-side.jpg").read_bytes()
    look = h.observe()
    assert look.step == 1 and look.cameras["side"].path == str(img / "look-0002-side.jpg")
    assert (img / "step-0001-side.jpg").read_bytes() == saved == step.cameras["side"].jpeg
    assert sorted(p.name for p in img.iterdir()) == ["look-0001-side.jpg", "look-0001-top.jpg", "look-0002-side.jpg",
                                                      "look-0002-top.jpg", "step-0001-side.jpg", "step-0001-top.jpg"]


def test_missing_camera_frames_are_reported():
    class Blind(SimBackend):
        def images(self):
            return {}

    base = sim_backend()
    with Harness.open(backend=Blind(base.arms)) as h:
        assert h.observe().cameras["top"].error == "no recent image from this camera"


def test_steps_go_to_the_episode_log(sim_harness, tmp_path):
    log = EpisodeLog(tmp_path / "run")
    h = sim_harness(log=log)
    h.step({"action": "home"})
    h.step({"action": "fly"})
    h.close()
    events = read_episode(tmp_path / "run")
    assert [e["event"] for e in events] == ["start", "action", "step", "step", "end"]
    assert events[0]["table_z"] == {"arm": 0.0} and events[0]["safety"]["max_speed_deg_s"] == 20
    assert events[1] == {**events[1], "step": 1, "action": {"action": "home"}}      # written before the motion
    assert events[2]["action"] == {"action": "home"} and events[3]["observation"]["result"]["status"] == "rejected"
    assert "jpeg" not in str(events[2]["observation"]["cameras"])
    assert (tmp_path / "run" / "images" / "step-0001-top.jpg").exists()
    resumed = Harness.open(backend=sim_backend(), log=EpisodeLog.resume(tmp_path / "run"))
    assert resumed.step({"action": "observe"}).step == 3
    resumed.close()


def test_describe_is_short_prompt_material(sim_harness):
    text = sim_harness(arm="maker").describe()
    assert len(text.splitlines()) <= 40
    for words in ("shoulder_pan [-158, 156]", "cameras: top, side", "-90 straight down", "20 deg/s", "5 deg/s",
                  "10 mm", "35 deg", "rejected =", "stopped =", "fault =", "metres and degrees", "table z = 0.000 m"):
        assert words in text, words


def test_close_rests_enabled_arms_and_is_idempotent(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    h.close()
    h.close()
    assert not h.backend.read()["arm"].enabled


def test_harness_open_loads_a_named_robot():
    from makermods_llm_harness import robot as robots

    robots.save(robots.simulated(name="bench"))
    with Harness.open("bench") as h:
        assert h.robot_name == "bench" and list(h.models) == ["arm"]
        assert h.step(Rest()).result.ok


# ── when the motors come on ──
def test_refused_requests_and_an_uncalibrated_gripper_leave_the_motors_off(sim_harness):
    h = sim_harness()
    for action in (move((1.5, 0, 0.2)), move((0.28, 0, -0.05), -90), {"action": "nudge", "dxyz": [0, 0, -0.5]},
                   {"action": "move_joints", "joints_deg": {"wrist_roll": 60}}):
        obs = h.step(action)
        assert obs.result.status == "rejected", action
        assert not obs.arms["arm"].enabled
    obs = sim_harness(table_z={}, backend=Uncalibrated(sim_backend().arms)).step({"action": "gripper", "opening": 1})
    assert obs.result.status == "rejected" and not obs.arms["arm"].enabled
    assert_done(h.step({"action": "home"}))
    assert h.observe().arms["arm"].enabled


class Uncalibrated(SimBackend):
    @property
    def table_z(self):
        return None


# ── stop ──
def test_stop_also_ends_the_actions_waiting_behind(sim_harness):
    h = sim_harness(backend=sim_backend(realtime=True))
    results = {}

    def run(key, action):
        results[key] = h.step(action).result

    first = threading.Thread(target=run, args=("first", {"action": "home"}))
    first.start()
    deadline = time.monotonic() + 5
    while not h.backend.read()["arm"].enabled and time.monotonic() < deadline:
        time.sleep(0.01)
    second = threading.Thread(target=run, args=("second", {"action": "wait", "seconds": 0.1}))
    second.start()
    time.sleep(0.2)
    h.stop()
    first.join(5)
    second.join(5)
    assert results["first"].status == "stopped" and "on request" in results["first"].message
    assert results["second"].status == "stopped"
    assert results["second"].message == "stopped on request before it started."
    assert_done(h.step({"action": "wait", "seconds": 0.04}))


# ── tool angles near vertical ──
def test_yaw_turns_the_gripper_pointing_straight_down(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    for xyz, pitch in APPROACH["metal"][:-1]:
        assert_done(h.step(move(xyz, pitch)))
    for yaw in (0, 30, 60, 30):
        obs = h.step(move((0.28, 0, 0.21), -90, yaw_deg=yaw))
        assert_done(obs)
        tool = obs.arms["arm"].tool
        assert tool.pitch_deg == pytest.approx(-90, abs=0.5) and tool.roll_deg == 0.0
        assert tool.yaw_deg == pytest.approx(yaw, abs=1.0)
        assert np.allclose(tool.xyz, (0.28, 0, 0.21), atol=2e-3)


# ── rest ──
def recording(monkeypatch, base_cls=SimBackend):
    """(backend class that records every target it is sent, the targets, the rows of every checked path)."""
    checked, sent = [], []
    original = Envelope.check

    def recording_check(self, q_start, path, capped=True, rest_q=None):
        original(self, q_start, path, capped, rest_q)
        checked.extend(np.asarray(path))

    monkeypatch.setattr(Envelope, "check", recording_check)

    class Recording(base_cls):
        rests = 0

        def step(self, targets, dt):
            sent.extend(np.array(t.q) for t in targets.values())
            super().step(targets, dt)

        def rest(self, arm):
            Recording.rests += 1
            return super().rest(arm)

    return Recording, sent, checked


def place(sim: SimBackend, q) -> None:
    """Turn the simulated arm on, standing at ``q``."""
    sim.enable("arm")
    arm = sim._arms["arm"]
    arm.q, arm.q_target = np.array(q, dtype=float), np.array(q, dtype=float)


# A pose whose straight joint path to the rest pose dips under the hard floor of a table 0.1 m below the rest
# pose, while the paths up to home and from home to rest clear it.
DETOUR_Q = np.radians([-139.0, -63.0, 6.0, -109.0, -78.0, 20.0])
# A pose 44 mm above a table at z = 0.12 (the rest pose 1.7 mm above it) whose ways to rest, straight and by way of
# home, both cross the table.
STUCK_Q = np.radians([-4.8, -99.3, 26.5, -109.5, 30.8, 123.0])


def test_rest_goes_by_way_of_home_when_the_straight_path_is_refused(sim_harness, monkeypatch):
    Recording, sent, checked = recording(monkeypatch)
    sim = Recording(sim_backend().arms)
    model = ArmModel(sim.arms[0])
    rest_q = sim.read()["arm"].rest_q
    table = float(model.lowest_z(rest_q)) - 0.1
    h = sim_harness(backend=sim, table_z=table)
    env = h.envelopes["arm"]
    with pytest.raises(MoveRejected, match="hard floor"):
        env.check(DETOUR_Q, model.plan_joint_path(DETOUR_Q, rest_q, env.slow_speed, 25.0), capped=False)
    place(sim, DETOUR_Q)
    checked.clear()
    obs = h.rest()
    assert_done(obs)
    assert not obs.arms["arm"].enabled and np.array_equal(sim.read()["arm"].q, rest_q)
    rows = {row.tobytes() for row in checked}
    assert sent and all(q.tobytes() in rows for q in sent)
    assert any(np.allclose(q, model.home_q) for q in sent)


def test_rest_refused_both_ways_moves_nothing(sim_harness, monkeypatch):
    Recording, sent, _ = recording(monkeypatch)
    sim = Recording(sim_backend().arms)
    h = sim_harness(backend=sim, table_z=0.12)
    place(sim, STUCK_Q)
    sent.clear()
    before = sim.read()["arm"].q
    obs = h.step({"action": "rest"})
    assert obs.result.status == "rejected" and "Nothing moved" in obs.result.message
    assert sent == [] and Recording.rests == 0
    assert obs.arms["arm"].enabled and np.array_equal(sim.read()["arm"].q, before)


def test_rest_leaves_a_faulted_arm_holding(sim_harness, monkeypatch):
    Recording, sent, _ = recording(monkeypatch)
    sim = Recording(sim_backend().arms)
    h = sim_harness(backend=sim)
    assert_done(h.step({"action": "home"}))
    sent.clear()
    sim.set_fault("arm", "motor 2 overcurrent")
    obs = h.rest()
    assert obs.result.status == "fault" and "motor 2 overcurrent" in obs.result.message
    assert "support it and release it" in obs.result.message
    assert sent == [] and Recording.rests == 0 and obs.arms["arm"].enabled


def test_an_arm_resting_slightly_past_a_joint_limit_can_home_and_rest(sim_harness):
    class PastLimits(SimBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for arm in self._arms.values():
                m = arm.model
                q = arm.q.copy()
                q[1], q[2] = m.hi[1] + math.radians(0.01), m.lo[2] - math.radians(0.01)
                arm.q, arm.q_target, arm.rest_q = q, q.copy(), q.copy()

    for arm_type in ("metal", "maker"):
        sim = PastLimits(sim_backend(arm_type).arms)
        h = sim_harness(backend=sim, table_z=0.0)
        rest_q = sim.read()["arm"].rest_q
        assert rest_q[1] > h.models["arm"].hi[1] and rest_q[2] < h.models["arm"].lo[2]
        assert_done(h.step({"action": "home"}))
        obs = h.rest()
        assert_done(obs)
        assert not obs.arms["arm"].enabled and np.allclose(sim.read()["arm"].q, rest_q, atol=1e-3)
        assert_done(h.step({"action": "home"}))


# ── close ──
def test_close_rests_through_the_checked_path(sim_harness, monkeypatch):
    Recording, sent, checked = recording(monkeypatch)
    sim = Recording(sim_backend().arms)
    h = sim_harness(backend=sim)
    assert_done(h.step({"action": "home"}))
    sent.clear()
    checked.clear()
    assert h.close() == {"arm": "released"}
    assert not sim.read()["arm"].enabled and Recording.rests == 1
    rows = {row.tobytes() for row in checked}
    assert sent and all(q.tobytes() in rows for q in sent)
    assert h.close() == {}


def test_close_leaves_an_arm_it_cannot_rest_holding(sim_harness):
    sim = sim_backend()
    h = sim_harness(backend=sim, table_z=0.12)
    place(sim, STUCK_Q)
    held = sim.read()["arm"].q
    results = h.close()
    assert list(results) == ["arm"] and results["arm"].startswith("holding: ")
    assert "lift the arm" in results["arm"]
    state = sim.read()["arm"]
    assert state.enabled and np.array_equal(state.q, held)


def test_close_skips_arms_that_are_off(sim_harness):
    h = sim_harness(arms=("left", "right"))
    assert_done(h.step({"action": "home", "arm": "right"}))
    assert h.close() == {"left": "released", "right": "released"}
    assert not any(s.enabled for s in h.backend.read().values())
