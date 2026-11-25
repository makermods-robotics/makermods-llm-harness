"""Every way a motion ends short of a verified arrival ends in a confirmed hold; stop is acknowledged only after one;
blocked, contact and grasp evidence is judged over elapsed time against the size of the move. Orderings are forced
with events inside the harness's own calls."""

import math
import threading

import numpy as np
import pytest

from makermods_llm_harness import SafetyConfig
from makermods_llm_harness.backends.base import ArmTarget
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.program import Program

from conftest import APPROACH, sim_backend

DEADLINE_S = 60.0


def done(obs):
    assert obs.result.status == "done", obs.result.message
    return obs


def held(h, arm="arm"):
    state = h.backend.read()[arm]
    return np.allclose(state.q_target, state.q, atol=1e-9) and abs(state.gripper_target - state.gripper) < 1e-9


class Sticky(SimBackend):
    """Joint ``joint`` stops ``gap`` rad short of its target, as a joint held back by static friction."""
    joint, gap = 5, math.radians(1.5)

    def step(self, targets, dt):
        before = {name: a.q[self.joint] for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            error = a.q_target[self.joint] - before[name]
            a.q[self.joint] = a.q_target[self.joint] - math.copysign(self.gap, error) if abs(error) > self.gap \
                else before[name]


class TakeUp(SimBackend):
    """Joint ``joint`` moves only once its command is ``slack`` rad away, then trails it by that much while the
    command moves, and catches up at 20 deg/s once it stands still: gear take-up and lag on an unobstructed arm."""
    joint, slack = 1, math.radians(3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_target = None

    def step(self, targets, dt):
        before = {name: a.q[self.joint] for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            target = a.q_target[self.joint]
            moving = self.last_target is not None and abs(target - self.last_target) > 1e-9
            error = target - before[name]
            if moving:
                a.q[self.joint] = target - math.copysign(self.slack, error) if abs(error) > self.slack \
                    else before[name]
            else:
                a.q[self.joint] = before[name] + float(np.clip(error, -math.radians(20) * dt, math.radians(20) * dt))
            self.last_target = target


class Surface(SimBackend):
    """The joints stop following once the tool point would go below ``surface_z``."""
    surface_z = 0.08

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            if a.model.tool_pose(a.q)[2, 3] < self.surface_z:
                a.q = before[name]


def over_the_surface(h):
    r = Program(h)
    r.home()
    for xyz, pitch in APPROACH["metal"]:
        r.move(xyz, pitch=pitch)
    return Program(h, check=False)


# ── stop ──
def test_a_stop_during_an_observation_holds_and_confirms_before_it_returns(sim_harness):
    h = sim_harness(backend=Sticky(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    assert not held(h)                                       # done, its target 1.5 deg past the arm
    inside, release = threading.Event(), threading.Event()
    frames = h._frames

    def pausing(after):
        inside.set()
        assert release.wait(DEADLINE_S)
        return frames(after)

    h._frames = pausing
    worker = threading.Thread(target=h.observe)
    worker.start()
    assert inside.wait(DEADLINE_S)                          # observe() holds the harness
    assert h.stop() is None and held(h)
    release.set()
    worker.join(DEADLINE_S)


def test_a_stop_reports_a_hold_it_cannot_confirm(sim_harness):
    class Ignores(SimBackend):
        def hold(self, arm):
            pass

    h = sim_harness(backend=type("StickyIgnores", (Sticky, Ignores), {"gap": math.radians(3)})(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 20}, "relative": True}))
    assert "not confirmed" in h.stop()


def test_a_stop_after_the_last_rest_waypoint_keeps_the_arm_on(sim_harness):
    h = sim_harness()
    done(h.step({"action": "home"}))
    inside, release = threading.Event(), threading.Event()
    fresh = h._wait_fresh

    def pausing(name, since):
        inside.set()
        assert release.wait(DEADLINE_S)
        return fresh(name, since)

    h._wait_fresh = pausing
    results = []
    worker = threading.Thread(target=lambda: results.append(h.step({"action": "rest"})))
    worker.start()
    assert inside.wait(DEADLINE_S)
    assert h.stop() is None
    release.set()
    worker.join(DEADLINE_S)
    assert results[0].result.status == "stopped" and h.backend.read()["arm"].enabled


# ── confirmed holds on every early end ──
def test_an_arrival_miss_holds_the_arm_where_it_is(sim_harness):
    backend = type("StickyShoulder", (Sticky,), {"joint": 1, "gap": 0.0})(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    backend.gap = math.radians(1.8)                          # it starts holding no standing error
    obs = h.step({"action": "move_tool", "xyz": [0.31, 0, 0.54], "pitch_deg": -5})
    assert obs.result.status == "stopped" and "from the target" in obs.result.message and held(h)


def test_a_gripper_still_moving_at_its_limit_is_held_where_it_is(sim_harness, monkeypatch):
    import makermods_llm_harness.backends.sim as sim
    h = sim_harness()
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1}))
    monkeypatch.setattr(sim, "GRIPPER_SPEED", 0.10)
    obs = h.step({"action": "gripper", "opening": 0})
    assert obs.result.status == "stopped" and obs.result.grip == {
        "opening": obs.result.grip["opening"], "target": 0.0, "holding": False, "settled": False}
    assert held(h)


def test_a_contact_hold_with_bad_feedback_afterwards_is_a_fault(sim_harness):
    class FaultsAfterHold(Surface):
        faulty = False

        def hold(self, arm):
            super().hold(arm)
            self.faulty = True

        def read(self):
            states = super().read()
            if self.faulty:
                states["arm"].fault = "encoder lost"
            return states

    h = sim_harness(backend=FaultsAfterHold(sim_backend().arms))
    r = over_the_surface(h)
    result = r.nudge([0, 0, -0.08], until="contact")
    assert result.status == "fault" and result.contact is None and "encoder lost" in result.message


def test_a_hold_that_leaves_the_gripper_target_is_not_confirmed(sim_harness):
    class KeepsGripper(Sticky):
        def hold(self, arm):
            a = self._arms[arm]
            a.q_target = a.q.copy()                          # joints held, gripper target left as it was

    h = sim_harness(backend=KeepsGripper(sim_backend().arms))
    done(h.step({"action": "home"}))
    h.backend._arms["arm"].gripper_target = 1.0
    assert "gripper is commanded" in h.stop()


# ── evidence over elapsed time, against the size of the move ──
def test_a_fully_blocked_small_slow_move_is_not_done(sim_harness):
    h = sim_harness()
    done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 4}, "relative": True, "speed_deg_s": 5})
    assert obs.result.status == "stopped" and "4.0 deg from its goal" in obs.result.message and held(h)


@pytest.mark.parametrize("slack_deg", [3, 5])
@pytest.mark.parametrize("speed", [5, 20])
def test_take_up_on_an_unobstructed_move_is_neither_contact_nor_blocked(sim_harness, slack_deg, speed):
    backend = type("Slack", (TakeUp,), {"slack": math.radians(slack_deg)})(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    obs = h.step({"action": "move_joints", "joints_deg": {"shoulder_lift": 20}, "relative": True,
                  "speed_deg_s": speed})
    done(obs)
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.1], "until": "contact", "speed_deg_s": speed})
    done(obs)
    assert obs.result.contact is None and "without contact" in obs.result.message


def test_empty_jaws_closing_slowly_are_not_holding(sim_harness, monkeypatch):
    import makermods_llm_harness.backends.sim as sim
    h = sim_harness()
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1}))
    monkeypatch.setattr(sim, "GRIPPER_SPEED", 0.10)
    obs = h.step({"action": "gripper", "opening": 0})
    assert obs.result.grip["holding"] is False


def test_an_obstruction_mid_move_is_contact_with_until_and_blocked_without(sim_harness):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = over_the_surface(h)
    result = r.nudge([0, 0, -0.08], until="contact")
    assert result.ok and result.contact is not None and held(h)
    assert result.contact["xyz"][2] == pytest.approx(0.08, abs=0.006)
    r.nudge([0, 0, 0.03])
    result = r.nudge([0, 0, -0.08])
    assert result.status == "stopped" and "blocked" in result.message and held(h)


def test_lag_alone_at_the_end_of_a_move_is_not_contact(sim_harness):
    class Lagging(SimBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pending = []

        def step(self, targets, dt):
            self.pending.append(dict(targets))
            super().step(self.pending.pop(0) if len(self.pending) > 4 else {}, dt)

        def hold(self, arm):
            self.pending.clear()
            state = SimBackend.read(self)[arm]
            SimBackend.step(self, {arm: ArmTarget(state.q, state.gripper)}, 0.0)

    h = sim_harness(backend=Lagging(sim_backend().arms), safety=SafetyConfig(contact_tolerance_m=0.05))
    done(h.step({"action": "home"}))
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.1], "until": "contact"})
    assert obs.result.contact is None


# ── the way into enable and out to rest ──
def test_a_requested_speed_also_caps_the_enable_and_its_correction(sim_harness):
    sent, enables = [], []

    class PastLimit(SimBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            arm = self._arms["arm"]
            arm.q = arm.q.copy()
            arm.q[1] = arm.model.hi[1] + math.radians(1)
            arm.q_target, arm.rest_q = arm.q.copy(), arm.q.copy()

        def enable(self, arm, cancel=None, speed=None):
            enables.append(speed)
            super().enable(arm, cancel)

        def step(self, targets, dt):
            sent.extend(t for t in targets.values() if t.speed is not None)
            super().step(targets, dt)

    h = sim_harness(backend=PastLimit(sim_backend("maker").arms))
    h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 1}, "relative": True, "speed_deg_s": 2})
    assert enables == [pytest.approx(math.radians(2))]
    assert sent and max(t.speed for t in sent) <= math.radians(2) + 1e-9


def test_rest_waits_for_fresh_feedback_from_every_joint_before_release(sim_harness):
    class Stale(SimBackend):
        fresh = False
        released = 0

        def feedback_ages(self, arm):
            return [0.0] * 7 if self.fresh else [0.0] * 6 + [math.inf]

        def rest(self, arm):
            Stale.released += 1
            return super().rest(arm)

    h = sim_harness(backend=Stale(sim_backend().arms))
    done(h.step({"action": "home"}))
    obs = h.step({"action": "rest"})
    assert obs.result.status == "fault" and "no fresh feedback from" in obs.result.message and Stale.released == 0
    h.backend.fresh = True
    done(h.step({"action": "rest"}))
    assert Stale.released == 1


# ── the end-of-motion decision ──
D = math.radians


@pytest.mark.parametrize("case, start_command, start, goal, end, held, requested, expected", [
    ("arrived", [0, 0], [0, 0], [10, 0], [10, 0], None, None, None),
    ("blocked large", [0, 0], [0, 0], [10, 0], [0, 0], None, None, (0, "short")),
    ("half way", [0, 0], [0, 0], [10, 0], [4, 0], None, None, (0, "short")),
    ("standing error kept", [0, 0], [-3, 0], [10, 0], [7, 0], None, None, None),
    ("standing error grew", [0, 0], [-1, 0], [10, 0], [6.5, 0], None, None, (0, "short")),
    ("small blocked, asked", [0, 0], [0, 0], [0.4, 0], [0, 0], None, [True, False], (0, "undetermined")),
    ("small moved, asked", [0, 0], [0, 0], [0.4, 0], [0.3, 0], None, [True, False], None),
    ("small, its standing error kept", [0, 0], [-1, 0], [0.4, 0], [-0.6, 0], None, [True, False], None),
    ("small incidental, within its allowance", [0, 0], [0, 0], [0.4, 0], [1.5, 0], None, None, None),
    ("small incidental, not asked", [0, 0], [-0.2, 0], [0.1, 0], [-0.2, 0], None, None, None),
    ("held, sagging in place", [0, 0], [0, -2.5], [5, 0], [5, -2.5], [False, True], [True, False], None),
    ("held, drifting", [0, 0], [0, -1], [5, 0], [5, -4], [False, True], [True, False], (1, "drift")),
    # the review's cases
    ("blocked +2.5 with 2.5 standing error", [0, 0], [-2.5, 0], [0, 0], [-2.5, 0], None, [True, False], None),
    ("blocked +2.5 from a command beyond", [0, 0], [-2.5, 0], [2.5, 0], [-2.5, 0], None, [True, False],
     (0, "short")),
    ("wrist 0.5 deg, arrived late", [0, 0], [0, 0], [0.5, 0], [0.5, 0], None, [True, False], None),
    ("wrist 1.5 deg, arrived late", [0, 0], [0, 0], [1.5, 0], [1.45, 0], None, [True, False], None),
    ("wrist_flex gains 1.5 deg during home", [0, 0], [0, 0], [10, 0], [10, 1.5], None, None, None),
    ("wrist_flex gains 2.5 deg during home", [0, 0], [0, 0], [10, 0], [10, 2.5], None, None, (1, "undetermined")),
])
def test_judge_motion_table(case, start_command, start, goal, end, held, requested, expected):
    from makermods_llm_harness.harness import judge_motion
    deg = lambda values: np.radians(np.asarray(values, float))  # noqa: E731
    got = judge_motion(deg(goal), deg(start), deg(start_command), deg(end), D(6),
                       held=None if held is None else np.array(held),
                       requested=None if requested is None else np.array(requested))
    assert got == expected, case


@pytest.mark.parametrize("request_deg", [4.0, 5.0, 8.0])
def test_a_fully_blocked_requested_move_is_not_done(sim_harness, request_deg):
    h = sim_harness()
    done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": request_deg}, "relative": True})
    assert obs.result.status == "stopped" and held(h)


class Sag(SimBackend):
    """Joint ``joint`` stands ``sag`` rad below its command, as a shoulder holding against gravity."""
    joint, sag = 1, math.radians(1.0)

    def step(self, targets, dt):
        super().step(targets, dt)
        for a in self._arms.values():
            if a.enabled:
                a.q[self.joint] = a.q_target[self.joint] - self.sag


@pytest.mark.parametrize("sag_deg", [1.0, 2.5, 3.0])
def test_wrist_moves_with_standing_shoulder_error_neither_stop_nor_let_the_shoulder_sink(sim_harness, sag_deg):
    h = sim_harness(backend=type("S", (Sag,), {"sag": math.radians(sag_deg)})(sim_backend().arms))
    done(h.step({"action": "home"}))
    shoulder = h.backend.read()["arm"].q_target[1]
    for _ in range(3):
        done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    assert h.backend.read()["arm"].q_target[1] == pytest.approx(shoulder)       # its command is kept
    obs = h.step({"action": "move_tool", "xyz": [0.31, 0, 0.54], "pitch_deg": -5})
    done(obs)
    if obs.result.error["position_m"] > 0.010:               # the miss its standing error explains is reported
        assert "standing error" in obs.result.message


@pytest.mark.parametrize("sag_deg", [1.8, 3.0, 5.0])
def test_standing_shoulder_error_in_free_space_is_not_contact(sim_harness, sag_deg):
    h = sim_harness(backend=type("S", (Sag,), {"sag": math.radians(sag_deg)})(sim_backend().arms))
    done(h.step({"action": "home"}))
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.06], "until": "contact"})
    assert obs.result.contact is None and "reached the target" not in obs.result.message
    assert f"ended {obs.result.error['position_m'] * 1000:.0f} mm" in obs.result.message      # 15+ mm measured miss
    assert obs.result.status == ("done" if obs.result.error["position_m"] <= 0.03 else "stopped")


def test_an_invalid_final_reading_is_a_fault_not_an_arrival(sim_harness):
    class BreaksAtTheEnd(SimBackend):
        broken = False

        def read(self):
            states = super().read()
            if self.broken:
                states["arm"].q = np.full(6, np.nan)
                states["arm"].fault = "encoder fault"
            return states

    h = sim_harness(backend=BreaksAtTheEnd(sim_backend().arms))
    done(h.step({"action": "home"}))
    original = h._execute

    def then_break(*args, **kwargs):
        result = original(*args, **kwargs)
        h.backend.broken = True
        return result

    h._execute = then_break
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.02]})
    assert obs.result.status == "fault"


# ── grasps ──
def test_empty_jaws_closing_very_slowly_are_not_a_grasp(sim_harness, monkeypatch):
    import makermods_llm_harness.backends.sim as sim
    h = sim_harness()
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1}))
    monkeypatch.setattr(sim, "GRIPPER_SPEED", 0.025)
    obs = h.step({"action": "gripper", "opening": 0})
    assert obs.result.grip["holding"] is False and obs.result.status == "stopped" and held(h)


def test_a_blocked_opening_is_stopped_and_held(sim_harness):
    class Blocked(SimBackend):
        def _move_gripper(self, a, dt):
            pass

    h = sim_harness(backend=Blocked(sim_backend().arms))
    done(h.step({"action": "home"}))
    obs = h.step({"action": "gripper", "opening": 1})
    assert obs.result.status == "stopped" and "blocked" in obs.result.message and held(h)


# ── stops, holds and rest ──
def test_a_stop_during_the_shutdown_rest_changes_nothing(sim_harness):
    h = sim_harness()
    done(h.step({"action": "home"}))
    inside, release = threading.Event(), threading.Event()
    fresh = h._wait_fresh

    def pausing(name, since):
        inside.set()
        assert release.wait(DEADLINE_S)
        return fresh(name, since)

    h._wait_fresh = pausing
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(h.close()))
    worker.start()
    assert inside.wait(DEADLINE_S)
    assert "closing" in h.stop()
    release.set()
    worker.join(DEADLINE_S)
    assert outcomes == [{"arm": "released"}]


def test_a_hold_confirms_only_the_commands_the_backend_reports(sim_harness):
    class Ignores(SimBackend):
        def hold(self, arm):
            pass

    for setup, words in ((lambda b: None, "commanded"),):
        h = sim_harness(backend=type("StickyIgnores", (Sticky, Ignores), {})(sim_backend().arms))
        done(h.step({"action": "home"}))
        done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
        assert words in h.stop()                           # 1.5 deg is not within command precision

    class StaleAges(SimBackend):
        def feedback_ages(self, arm):
            return [math.inf] * 7

    h = sim_harness(backend=StaleAges(sim_backend().arms))
    done(h.step({"action": "home"}))
    assert "no fresh feedback" in h.stop()

    class Unknown(SimBackend):
        unknown = False

        def read(self):
            states = super().read()
            if self.unknown:
                states["arm"].enabled = None
            return states

    h = sim_harness(backend=Unknown(sim_backend().arms))
    done(h.step({"action": "home"}))
    h.backend.unknown = True
    assert "torque state is not known" in h.stop()


def test_rest_judges_freshness_from_a_clock_read_before_the_ages(sim_harness, monkeypatch):
    import time
    import makermods_llm_harness.harness as harness_module

    class SlowAges(SimBackend):
        """Every joint's last sample was taken at ``sample``; its age is measured as the query starts, and the
        query takes 0.3 s."""
        sample = None

        def feedback_ages(self, arm):
            if self.sample is None:
                return [0.0] * 7
            ages = [time.monotonic() - self.sample] * 7
            time.sleep(0.3)
            return ages

    h = sim_harness(backend=SlowAges(sim_backend().arms))
    done(h.step({"action": "home"}))
    monkeypatch.setattr(harness_module, "FRESH_FEEDBACK_S", 0.5)
    fresh = h._wait_fresh

    def stale_from_before(name, since):
        h.backend.sample = since - 0.05                  # older than the last rest waypoint
        return fresh(name, since)

    h._wait_fresh = stale_from_before
    obs = h.step({"action": "rest"})
    assert obs.result.status == "fault" and "no fresh feedback" in obs.result.message
    assert h.backend.read()["arm"].enabled


def test_a_hold_after_a_failed_action_waits_for_an_observation_to_finish(sim_harness):
    h = sim_harness(backend=Sticky(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    inside, release = threading.Event(), threading.Event()
    frames = h._frames

    def pausing(after):
        inside.set()
        assert release.wait(DEADLINE_S)
        return frames(after)

    h._frames = pausing
    worker = threading.Thread(target=h.observe)
    worker.start()
    assert inside.wait(DEADLINE_S)
    results = []
    holder = threading.Thread(target=lambda: results.append(h.hold()))
    holder.start()
    holder.join(0.5)
    assert holder.is_alive()                              # it waits for the observation
    release.set()
    holder.join(DEADLINE_S)
    worker.join(DEADLINE_S)
    assert results == [None] and held(h)
    assert "could not be held" in Harness_hold_timeout(h)


def Harness_hold_timeout(h):
    h._lock.acquire()
    try:
        return h.hold(timeout_s=0.1)
    finally:
        h._lock.release()


# ── images and refusals ──
def test_one_or_two_cameras_keep_their_own_size_in_the_grid(sim_harness, tmp_path):
    import cv2
    backend = sim_backend(cameras=("top",), size=(640, 480))
    h = sim_harness(backend=backend, image_dir=tmp_path)
    assert cv2.imread(h.look()["path"]).shape == (480, 640, 3)
    backend = sim_backend(cameras=("top", "side", "arm_wrist"), size=(640, 480))
    h = sim_harness(backend=backend, image_dir=tmp_path / "three")
    assert cv2.imread(h.look()["path"]).shape[0] == 240


def test_a_descent_under_the_floor_says_how_far_down_is_allowed(sim_harness):
    import re
    h = sim_harness()
    r = over_the_surface(h)
    result = r.nudge([0, 0, -0.11])
    room = re.search(r"may come down at most (\d+) mm", result.message)
    assert result.status == "rejected" and "hard floor" in result.message and room
    deep = r.nudge([0, 0, -0.13], until="contact", dry_run=True)        # a contact move: room for its depth too
    contact_room = re.search(r"may come down at most (\d+) mm", deep.message)
    assert deep.status == "rejected" and "floor of a move with until contact" in deep.message
    assert int(contact_room.group(1)) == pytest.approx(int(room.group(1)) + 25, abs=1)
    assert r.nudge([0, 0, -(int(contact_room.group(1)) - 2) / 1000], until="contact", dry_run=True).ok
    assert r.nudge([0, 0, -(int(room.group(1)) - 2) / 1000]).ok


# ── holds confirmed against their setpoint, after the control loop had time to send them ──
def test_a_command_still_in_flight_after_the_hold_leaves_it_unconfirmed(sim_harness):
    class InFlight(Sticky):
        """A control tick had copied the previous command before the hold and sends it after."""
        hold_settle_s = 0.01

        def hold(self, arm):
            stale = self._arms[arm].q_target.copy()
            setpoint = super().hold(arm)
            self._arms[arm].q_target = stale
            return setpoint

    h = sim_harness(backend=InFlight(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    problem = h.stop()
    assert "not its hold setpoint" in problem and "wrist_roll" in problem


def test_a_joint_measured_past_its_limit_cannot_be_held_there(sim_harness):
    h = sim_harness()
    done(h.step({"action": "home"}))
    arm = h.backend._arms["arm"]
    arm.q[1] = arm.q_target[1] = arm.model.hi[1] + math.radians(2)
    assert "shoulder_lift is measured" in h.stop() and "cannot be held where it is" in h.stop()


def test_an_ignored_hold_is_not_confirmed_even_when_the_arm_slips_toward_its_target(sim_harness):
    class SlipsAndIgnores(Sticky):
        def hold(self, arm):
            a = self._arms[arm]
            setpoint = (a.q.copy(), float(a.gripper))
            a.q[5] -= math.radians(1.5)                       # slips while the hold is ignored
            return setpoint

    h = sim_harness(backend=SlipsAndIgnores(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    assert "not its hold setpoint" in h.stop()


# ── verdicts with standing error and lag ──
class LaggingSag(Sag):
    """``Sag`` on a joint that follows its command ``delay`` ticks late (40 ms each); a hold applies at once."""
    delay = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = []

    def step(self, targets, dt):
        self.pending.append(dict(targets))
        late = self.pending.pop(0) if len(self.pending) > self.delay else {}
        super().step(late, dt)

    def hold(self, arm):
        self.pending.clear()
        state = SimBackend.read(self)[arm]
        Sag.step(self, {arm: ArmTarget(state.q, state.gripper)}, 0.0)
        return state.q.copy(), float(state.gripper)


@pytest.mark.parametrize("delay", [2, 4])
def test_a_nudge_that_ends_short_by_a_steady_shoulder_error_reports_the_miss(sim_harness, delay):
    backend = type("L", (LaggingSag,), {"sag": math.radians(3), "delay": delay})(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.06]})
    done(obs)
    assert obs.result.error["position_m"] > 0.010 and "standing error" in obs.result.message


def test_a_load_change_in_free_space_is_not_contact(sim_harness):
    class Ramp(LaggingSag):
        """The shoulder's standing error ramps from 1 to 3 deg over 320 ms once the motion is under way."""
        sag, ticks = math.radians(1), 0

        def step(self, targets, dt):
            if targets:
                Ramp.ticks += 1
                self.sag = math.radians(1 + 2 * min(1.0, max(0.0, (Ramp.ticks - 10) / 8)))
            super().step(targets, dt)

    h = sim_harness(backend=Ramp(sim_backend().arms), safety=SafetyConfig(max_speed_deg_s=20))
    done(h.step({"action": "home"}))
    Ramp.ticks = 0
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.1], "until": "contact"})
    assert obs.result.contact is None and "reached the target" not in obs.result.message


@pytest.mark.parametrize("request_deg", [0.5, 0.75, 1.0, 1.5])
def test_small_unobstructed_moves_of_a_lagging_arm_are_done(sim_harness, request_deg):
    backend = type("L", (LaggingSag,), {"sag": 0.0, "delay": 4})(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": request_deg}, "relative": True}))


def test_a_joint_gaining_some_gravity_error_during_home_is_done(sim_harness):
    class Gains(SimBackend):
        """wrist_flex settles 1.5 deg off its command once the arm is on."""
        def step(self, targets, dt):
            super().step(targets, dt)
            for a in self._arms.values():
                if a.enabled:
                    a.q[3] = a.q_target[3] - math.radians(1.5)

    h = sim_harness(backend=Gains(sim_backend().arms))
    done(h.step({"action": "home"}))


def test_a_wrist_move_near_the_table_runs_at_the_speed_its_measured_pose_allows(sim_harness):
    sent = []

    class Recording(Sag):
        def step(self, targets, dt):
            sent.extend(t.speed for t in targets.values() if t.speed is not None)
            super().step(targets, dt)

    h = sim_harness(backend=type("R", (Recording,), {"sag": math.radians(3)})(sim_backend().arms))
    r = Program(h, check=False)
    r.home()
    for xyz, pitch in APPROACH["metal"]:
        r.move(xyz, pitch=pitch)
    r.nudge([0, 0, -0.002])
    env = h.envelopes["arm"]
    state = h.backend.read()["arm"]
    assert env.clearance(state.q) < env.config.slow_zone_m < env.clearance(state.q_target)   # sagged into the zone
    sent.clear()
    h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 5}, "relative": True})
    assert sent and max(sent) <= env.slow_speed + 1e-9


def test_a_final_reading_with_stale_feedback_is_not_an_arrival(sim_harness):
    class Stale(SimBackend):
        stale = False

        def feedback_ages(self, arm):
            return [math.inf] * 7 if self.stale else [0.0] * 7

    h = sim_harness(backend=Stale(sim_backend().arms))
    done(h.step({"action": "home"}))
    original = h._execute

    def then_stale(*args, **kwargs):
        result = original(*args, **kwargs)
        h.backend.stale = True
        return result

    h._execute = then_stale
    assert h.step({"action": "nudge", "dxyz": [0, 0, -0.02]}).result.status == "fault"


# ── gripper endpoints ──
class Jaws(SimBackend):
    """Jaws that close fast to ``stop_at`` and then creep at ``creep`` opening per second, or settle ``short`` of the
    request."""
    stop_at, creep, short = None, 0.0, 0.0

    def _move_gripper(self, a, dt):
        target = a.gripper_target + (self.short if a.gripper_target < a.gripper else -self.short)
        if self.stop_at is not None and a.gripper <= self.stop_at + 1e-9 and target < a.gripper:
            a.gripper = max(target, a.gripper - self.creep * dt)
            return
        step = 2.0 * dt
        a.gripper += float(np.clip(target - a.gripper, -step, step))
        if self.stop_at is not None:
            a.gripper = max(a.gripper, self.stop_at) if target < self.stop_at else a.gripper


def test_empty_jaws_creeping_shut_are_not_a_grasp(sim_harness):
    h = sim_harness(backend=type("J", (Jaws,), {"stop_at": 0.09, "creep": 0.008})(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1}))
    obs = h.step({"action": "gripper", "opening": 0})
    assert obs.result.grip["holding"] is False and obs.result.status == "stopped" and "undetermined" in \
        obs.result.message


@pytest.mark.parametrize("opening, short", [(0.0, 0.03), (1.0, 0.03)])
def test_jaws_settling_near_the_request_are_done(sim_harness, opening, short):
    h = sim_harness(backend=type("J", (Jaws,), {"short": short})(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1 - opening}))
    obs = done(h.step({"action": "gripper", "opening": opening}))
    assert obs.result.grip["holding"] is False


# ── rest ──
def test_rest_without_a_rest_pose_confirms_the_hold_it_reports(sim_harness):
    h = sim_harness(backend=Sticky(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    original = h.backend.read

    def no_rest():
        states = original()
        for s in states.values():
            s.rest_q = None
        return states

    h.backend.read = no_rest
    obs = h.step({"action": "rest"})
    assert obs.result.status == "rejected" and "holds where it is" in obs.result.message and held(h)
