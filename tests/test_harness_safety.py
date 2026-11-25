"""The harness's interlocks: the rest pose gate, measured starts, per-tick watching, holds, cancellation,
speed caps, closing, and observations that survive camera and disk failures."""

import math
import threading
import time

import numpy as np
import pytest

from makermods_llm_harness import HarnessClosed
from makermods_llm_harness.harness import Harness
from makermods_llm_harness.backends.base import Frame
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.log import EpisodeLog, read_episode
from makermods_llm_harness import harness as harness_module

from conftest import APPROACH, sim_backend


def assert_done(obs):
    assert obs.result.status == "done", obs.result.message


def move(xyz, pitch=None, **extra):
    return {"action": "move_tool", "xyz": list(xyz), **({"pitch_deg": pitch} if pitch is not None else {}), **extra}


class Offset(SimBackend):
    """Reports wrist_roll ``offset`` rad away from where the simulated joint is, once the arm is on."""
    offset = 0.0

    def read(self):
        states = super().read()
        for state in states.values():
            if state.enabled:
                state.q = state.q.copy()
                state.q[5] += self.offset
        return states


# ── the rest pose gate ──
def test_an_arm_away_from_its_rest_pose_is_not_turned_on(sim_harness):
    sim = sim_backend()
    sim._arms["arm"].q = sim._arms["arm"].q + np.radians([0, 0, 0, 0, 0, 8])     # moved by hand while off
    h = sim_harness(backend=sim)
    for action in ({"action": "home"}, {"action": "gripper", "opening": 1}):
        obs = h.step(action)
        assert obs.result.status == "rejected" and "only at the rest pose" in obs.result.message
        assert "wrist_roll is 8 deg from it" in obs.result.message and not obs.arms["arm"].enabled
    sim._arms["arm"].q[5] -= math.radians(4)                                     # within 5 deg of rest
    assert_done(h.step({"action": "home"}))


def test_without_a_rest_pose_the_motors_stay_off(sim_harness):
    class NoRest(SimBackend):
        def read(self):
            states = super().read()
            for state in states.values():
                state.rest_q = None
            return states

    h = sim_harness(backend=NoRest(sim_backend().arms))
    obs = h.step({"action": "home"})
    assert obs.result.status == "rejected" and "calibrate --rest" in obs.result.message
    assert not obs.arms["arm"].enabled


def test_a_joint_resting_past_a_limit_is_brought_inside_slowly_along_a_checked_path(sim_harness, monkeypatch):
    sent = []

    class PastLimit(SimBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            arm = self._arms["arm"]
            arm.q = arm.q.copy()
            arm.q[1] = arm.model.hi[1] + math.radians(3)        # a folded Maker's shoulder sits past its limit
            arm.q_target, arm.rest_q = arm.q.copy(), arm.q.copy()

        def step(self, targets, dt):
            sent.extend(targets.values())
            arm = self._arms["arm"]
            for target in targets.values():              # reaches each target, clipped as the SDK clips it
                if arm.enabled:
                    arm.q = arm.q_target = np.clip(target.q, arm.model.lo, arm.model.hi)

    sim = PastLimit(sim_backend("maker").arms)
    h = sim_harness(backend=sim)
    hi = h.models["arm"].hi[1]
    assert_done(h.step({"action": "gripper", "opening": 0.0}))
    ramp = [t for t in sent if t.speed is not None]
    assert ramp and all(t.speed == h.envelopes["arm"].slow_speed for t in ramp)
    assert ramp[-1].q[1] == pytest.approx(hi) and sim.read()["arm"].q[1] == pytest.approx(hi)


# ── measured starts and speed caps ──
def test_plans_start_from_the_measured_joints_and_every_motion_target_carries_the_speed_cap(sim_harness, monkeypatch):
    sent, starts = [], []
    original = harness_module.Harness._checked_path

    def recording(self, name, act, q0, *rest, **options):
        starts.append((q0.copy(), self.backend.read()[name].q.copy()))
        return original(self, name, act, q0, *rest, **options)

    monkeypatch.setattr(harness_module.Harness, "_checked_path", recording)

    class Recording(Offset):
        offset = math.radians(1.5)

        def step(self, targets, dt):
            for name, target in targets.items():
                sent.append((target, super().read()[name].q_target.copy()))
            super().step(targets, dt)

    h = sim_harness(backend=Recording(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    assert_done(h.step(move((0.31, 0, 0.54), -5)))
    assert_done(h.step({"action": "gripper", "opening": 1}))
    q0, measured = starts[-1]
    assert np.array_equal(q0, measured)                          # the measured pose, not the command it follows
    limit = h.envelopes["arm"].max_speed
    for target, commanded in sent:
        if target.speed is None:                                 # the gripper: the joints keep their command
            assert np.array_equal(target.q, commanded)
        else:
            assert 0 < target.speed <= limit


# ── watching while moving ──
def test_a_joint_off_its_path_holds_the_arm(sim_harness):
    h = sim_harness(backend=Offset(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 30}, "relative": True})
    assert obs.result.status == "stopped" and "blocked: wrist_roll" in obs.result.message
    h.backend.block("arm", False)
    h.backend.offset = math.radians(8)                       # 8 deg off its path at once
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 5}, "relative": True})
    assert obs.result.status == "stopped" and "blocked: wrist_roll" in obs.result.message
    assert "tolerance 6 deg" in obs.result.message


def test_measured_clearance_under_the_floor_holds_the_arm(sim_harness, monkeypatch):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    env = h.envelopes["arm"]
    monkeypatch.setattr(env, "hold_floor", lambda q0: float(env.clearance(q0)) + 0.001)
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.03]})
    assert obs.result.status == "stopped" and "measured clearance fell" in obs.result.message


def test_a_fault_on_the_other_arm_or_heat_during_a_motion_holds_it(sim_harness):
    class Trouble(SimBackend):
        ticks, what = 0, None

        def step(self, targets, dt):
            Trouble.ticks += 1
            if Trouble.ticks == 8:
                if Trouble.what == "fault":
                    self.set_fault("right", "motor 3 overcurrent")
                else:
                    self.set_temperature("left", 80)
            super().step(targets, dt)

    for what, status, words in (("fault", "fault", "right: motor 3 overcurrent"), ("heat", "stopped", "80 C")):
        Trouble.ticks, Trouble.what = 0, what
        h = sim_harness(backend=Trouble(sim_backend(arms=("left", "right")).arms))
        obs = h.step({"action": "home", "arm": "left"})
        assert obs.result.status == status and words in obs.result.message
        state = h.backend.read()["left"]
        assert np.array_equal(state.q_target, state.q)                        # held where it was
    obs = h.step({"action": "home", "arm": "left"})
    assert obs.result.status == "rejected" and "80 C" in obs.result.message


def test_a_hot_arm_may_still_rest(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    h.backend.set_temperature("arm", 80)
    assert h.step({"action": "nudge", "dxyz": [0, 0, 0.01]}).result.status == "rejected"
    assert_done(h.rest())


def test_a_motion_that_does_not_arrive_holds_and_a_stop_while_settling_is_stopped(sim_harness, monkeypatch):
    h = sim_harness(backend=type("Stuck", (Offset,), {"offset": math.radians(1.5)})(sim_backend().arms))
    assert_done(h.step({"action": "home"}))                     # 1.5 deg off: inside the tracking tolerance
    original = h.backend.step

    def stop_when_settling(targets, dt):
        target = targets.get("arm")
        if target is not None and stop_when_settling.last is not None and np.array_equal(target.q, stop_when_settling.last):
            h.stop()
        stop_when_settling.last = None if target is None else np.array(target.q)
        original(targets, dt)

    stop_when_settling.last = None
    monkeypatch.setattr(h.backend, "step", stop_when_settling)
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_flex": 10}, "relative": True})
    assert obs.result.status == "stopped" and "on request" in obs.result.message


def test_a_tool_move_reports_how_far_off_its_target_it_ended(sim_harness):
    h = sim_harness(backend=type("Close", (Offset,), {"offset": math.radians(1.5)})(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    obs = h.step(move((0.31, 0, 0.54), -5))
    assert_done(obs)
    assert obs.result.error == {"position_m": 0.0, "angle_deg": 1.5, "short_m": 0.0}
    h = sim_harness(backend=type("Far", (Offset,), {"offset": math.radians(4)})(sim_backend().arms))
    assert_done(h.step({"action": "home"}))                    # the 4 deg it stands off its command stays put
    obs = h.step(move((0.31, 0, 0.54), -5))
    assert_done(obs)
    assert obs.result.error == {"position_m": 0.0, "angle_deg": 4.0, "short_m": 0.0}
    assert "ended 0 mm and 4.0 deg from the target" in obs.result.message


class Sag(SimBackend):
    """Reports shoulder_lift ``offset`` rad away from where the simulated joint is, once the arm is on: a standing
    error that leaves the measured tool point above its command."""
    offset = 0.0

    def read(self):
        states = super().read()
        for state in states.values():
            if state.enabled:
                state.q = state.q.copy()
                state.q[1] += self.offset
        return states


def sagging(h, height_m):
    """Set the Sag backend's offset so the measured tool point at home stands about ``height_m`` above its command."""
    model = h.models["arm"]
    q = model.home_q.copy()
    q[1] += 1e-3
    rise = (model.tool_points(q)[2] - model.tool_points(model.home_q)[2]) / 1e-3
    h.backend.offset = height_m / rise


def test_a_descent_that_ends_short_along_its_approach_is_stopped_with_the_shortfall(sim_harness):
    h = sim_harness(backend=Sag(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    sagging(h, 0.018)
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.05]})
    assert obs.result.status == "stopped" and "short of the target along the approach" in obs.result.message
    assert obs.result.error["short_m"] == pytest.approx(0.018, abs=0.004)
    sagging(h, 0.006)                                           # within the contact tolerance: done, as before
    assert_done(h.step({"action": "home"}))
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.05]})
    assert_done(obs)
    assert obs.result.error["short_m"] == pytest.approx(0.006, abs=0.003)


def test_until_contact_that_ends_short_and_still_is_contact(sim_harness):
    def descend(**kwargs):
        h = sim_harness(backend=Sag(sim_backend().arms), **kwargs)
        assert_done(h.step({"action": "home"}))
        sagging(h, 0.018)
        return h, h.step({"action": "nudge", "dxyz": [0, 0, -0.05], "until": "contact"})

    h, obs = descend()                                   # it stops short in the air, its tool point swung aside:
    assert obs.result.status == "stopped" and "to the side of its path" in obs.result.message     # not contact
    rested = h.state().arms["arm"].clearance_m + h.envelopes["arm"].table_z
    h, obs = descend(table_z=rested - 0.002)              # the same ending resting on the stored table: contact
    assert_done(obs)
    assert obs.result.contact is not None and "stopped on contact at the end of the move" in obs.result.message
    assert obs.result.error["short_m"] == pytest.approx(0.018, abs=0.004)     # along the approach


def test_a_nudge_starts_from_the_commanded_pose_until_the_arm_is_held_elsewhere(sim_harness):
    h = sim_harness(backend=type("Close", (Offset,), {"offset": math.radians(1.5)})(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    assert_done(h.step({"action": "nudge", "dxyz": [0, 0, -0.01]}))
    commanded = h._commanded_pose("arm", h.backend.read()["arm"])
    assert commanded is not None
    assert commanded[2, 3] == pytest.approx(h.models["arm"].tool_pose(h.models["arm"].home_q)[2, 3] - 0.01)
    assert_done(h.step({"action": "gripper", "opening": 1}))               # the joints keep their command
    assert h._commanded_pose("arm", h.backend.read()["arm"]) is not None
    h.backend.offset = math.radians(5)                                     # measured beyond the arrival tolerance
    assert h._commanded_pose("arm", h.backend.read()["arm"]) is None
    h.backend.offset = math.radians(1.5)
    h.backend.hold("arm")                                                  # now commanded where it was measured
    assert h._commanded_pose("arm", h.backend.read()["arm"]) is None


def test_an_arm_that_does_not_reach_the_rest_pose_is_not_released(sim_harness, monkeypatch):
    rests = []

    class Counting(Offset):
        offset = math.radians(1.5)

        def rest(self, arm):
            rests.append(arm)
            return super().rest(arm)

    h = sim_harness(backend=Counting(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    monkeypatch.setattr(harness_module, "REST_TOLERANCE_DEG", 1.0)
    obs = h.rest()
    assert obs.result.status == "stopped" and "was not released" in obs.result.message
    assert rests == [] and obs.arms["arm"].enabled


# ── stop and cancel ──
def test_stop_during_a_gripper_action_replaces_its_target(sim_harness):
    h = sim_harness(backend=sim_backend(realtime=True))
    assert_done(h.step({"action": "home"}))
    threading.Timer(0.15, h.stop).start()
    obs = h.step({"action": "gripper", "opening": 1})
    state = h.backend.read()["arm"]
    assert obs.result.status == "stopped" and 0 < state.gripper < 1
    assert state.gripper_target == pytest.approx(state.gripper)


def test_a_stop_while_planning_keeps_the_motors_off(sim_harness, monkeypatch):
    h = sim_harness()
    original = harness_module.Harness._checked_path

    def stop_then_plan(self, *args):
        self.stop()
        return original(self, *args)

    monkeypatch.setattr(harness_module.Harness, "_checked_path", stop_then_plan)
    obs = h.step({"action": "home"})
    assert obs.result.status == "stopped" and "before its motors came on" in obs.result.message
    assert not obs.arms["arm"].enabled


def test_cancel_ends_only_its_own_action(sim_harness):
    h = sim_harness(backend=sim_backend(realtime=True))
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    assert h.step({"action": "home"}, cancel=cancel).result.status == "stopped"
    assert_done(h.step({"action": "home"}))
    done = threading.Event()
    done.set()
    assert h.step({"action": "wait", "seconds": 0.1}, cancel=done).result.status == "stopped"


def test_a_closed_harness_takes_no_more_requests(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    assert h.close() == {"arm": "released"}
    with pytest.raises(HarnessClosed):
        h.step({"action": "home"})
    with pytest.raises(HarnessClosed):
        h.observe()


# ── tool orientation ──
def test_move_tool_without_pitch_or_roll_keeps_the_exact_orientation_near_vertical(sim_harness):
    h = sim_harness()
    assert_done(h.step({"action": "home"}))
    for xyz, pitch in APPROACH["metal"][:-1]:
        assert_done(h.step(move(xyz, pitch)))
    assert_done(h.step(move((0.28, 0, 0.21), -89, yaw_deg=0, roll_deg=20)))
    model = h.models["arm"]
    before = model.tool_pose(h.backend.read()["arm"].q)[:3, :3]
    assert_done(h.step(move((0.28, 0, 0.18), yaw_deg=0)))
    after = model.tool_pose(h.backend.read()["arm"].q)[:3, :3]
    assert np.degrees(np.arccos(np.clip((np.trace(before.T @ after) - 1) / 2, -1, 1))) < 0.6


# ── observations ──
def test_a_camera_failure_after_a_motion_keeps_the_result_and_the_log(sim_harness, tmp_path):
    class BrokenCamera(SimBackend):
        def images(self):
            raise OSError("USB camera gone")

    h = sim_harness(backend=BrokenCamera(sim_backend().arms), log=EpisodeLog(tmp_path / "run"))
    obs = h.step({"action": "home"})
    assert_done(obs)
    assert "USB camera gone" in obs.cameras["top"].error
    events = read_episode(tmp_path / "run")
    assert [e["event"] for e in events] == ["start", "action", "step"]


def test_a_frame_older_than_the_end_of_the_action_is_marked_stale(sim_harness, monkeypatch):
    monkeypatch.setattr(harness_module, "FRESH_FRAME_S", 0.1)

    class OldFrames(SimBackend):
        def images(self):
            return {name: Frame(frame.image, frame.stamp, time.monotonic() - 2.0)
                    for name, frame in super().images().items()}

    h = sim_harness(backend=OldFrames(sim_backend().arms))
    start = time.monotonic()
    obs = h.step({"action": "home"})
    assert time.monotonic() - start < 5
    top = obs.cameras["top"]
    assert top.stale and top.age_s >= 2.0 and "stale" in obs.text()
    assert not h.observe().cameras["top"].stale


def test_a_full_disk_is_reported_in_the_reply(sim_harness, tmp_path, monkeypatch):
    log = EpisodeLog(tmp_path / "run")
    h = sim_harness(log=log)

    def full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(log, "write", full)
    obs = h.step({"action": "home"})
    assert obs.result.status == "rejected" and "nothing moved" in obs.result.message
    assert not obs.arms["arm"].enabled
    monkeypatch.undo()


def test_maker_jaws_do_not_open_at_the_rest_pose(sim_harness):
    h = sim_harness(arm="maker")
    obs = h.step({"action": "gripper", "opening": 1.0})
    assert obs.result.status == "rejected" and "cannot open at the rest pose" in obs.result.message
    assert not h.backend.read()["arm"].enabled                              # refused before the motors came on
    assert h.check({"action": "gripper", "opening": 1.0}).status == "rejected"
    assert_done(h.step({"action": "gripper", "opening": 0.0}))              # closing is fine
    assert_done(h.step({"action": "home"}))
    assert_done(h.step({"action": "gripper", "opening": 1.0}))
    metal = sim_harness(arm="metal")
    assert_done(metal.step({"action": "gripper", "opening": 1.0}))          # its jaws open at rest


def test_a_tighter_contact_tolerance_stops_a_move_short_within_the_arrival_tolerance(sim_harness):
    from makermods_llm_harness import SafetyConfig
    h = sim_harness(backend=Sag(sim_backend().arms), safety=SafetyConfig(contact_tolerance_m=0.003))
    assert_done(h.step({"action": "home"}))
    sagging(h, 0.006)                                     # 6 mm: inside the 10 mm arrival tolerance
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.05]})
    assert obs.result.status == "stopped" and "short of the target along the approach" in obs.result.message
    assert obs.result.error["short_m"] == pytest.approx(0.006, abs=0.003)


def test_a_stop_during_the_final_contact_check_is_not_contact(sim_harness, monkeypatch):
    h = sim_harness(backend=Sag(sim_backend().arms))
    assert_done(h.step({"action": "home"}))
    sagging(h, 0.018)
    real = Harness._arrival

    def arrival(self, *args, **kwargs):                    # an operator stop lands as the move ends
        stopper = threading.Thread(target=self.stop)
        stopper.start()
        stopper.join(10)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Harness, "_arrival", arrival)
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.05], "until": "contact"})
    assert obs.result.status == "stopped" and obs.result.contact is None and "on request" in obs.result.message
