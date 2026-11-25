"""How a motion is watched, held and reported, on the kinematic simulator and fakes of it: holds on stop and on
failure, contact and grasp evidence, the tracking lag in elapsed time, progress, speed caps, durations, setting and
request limits, checks that agree with execution, rest feedback and MuJoCo images across threads."""

import math
import threading
import time

import numpy as np
import pytest

from makermods_llm_harness import Harness, SafetyConfig
from makermods_llm_harness.actions import ActionError, parse_action
from makermods_llm_harness.arm_model import MAX_SECONDS
from makermods_llm_harness.backends.base import ArmTarget
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.program import Program

from conftest import APPROACH, sim_backend

FAST = {"max_speed_deg_s": 60.0, "slow_speed_deg_s": 20.0, "max_excursion_deg": 90.0}


def done(obs):
    assert obs.result.status == "done", obs.result.message
    return obs


def held(h):
    state = h.backend.read()["arm"]
    return np.array_equal(state.q_target, state.q)


# ── holds ──
class Sticky(SimBackend):
    """wrist_roll stops ``gap`` rad short of its target, as a joint held back by static friction."""
    gap = math.radians(1.5)

    def step(self, targets, dt):
        before = {name: a.q[5] for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            error = a.q_target[5] - before[name]
            a.q[5] = a.q_target[5] - math.copysign(self.gap, error) if abs(error) > self.gap else before[name]


def test_a_stop_with_no_motion_running_holds_the_arm_where_it_stands(sim_harness):
    h = sim_harness(backend=Sticky(sim_backend().arms))
    done(h.step({"action": "home"}))
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 10}, "relative": True}))
    state = h.backend.read()["arm"]
    assert not np.allclose(state.q_target, state.q)        # done within tolerance, its target 1.5 deg away
    assert h.stop() is None
    state = h.backend.read()["arm"]
    assert np.allclose(state.q_target, state.q, atol=1e-6)


class Surface(SimBackend):
    """The joints stop following once the tool point would go below ``surface_z``."""
    surface_z = 0.08
    hold_fails = False

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            if a.model.tool_pose(a.q)[2, 3] < self.surface_z:
                a.q = before[name]

    def hold(self, arm):
        if self.hold_fails:
            raise RuntimeError("bus timeout")
        super().hold(arm)


def over_the_surface(h):
    r = Program(h)
    r.home()
    for xyz, pitch in APPROACH["metal"]:
        r.move(xyz, pitch=pitch)
    return Program(h, check=False)


def test_a_contact_whose_hold_fails_is_a_fault(sim_harness):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = over_the_surface(h)
    h.backend.hold_fails = True
    result = r.nudge([0, 0, -0.08], until="contact")
    assert result.status == "fault" and result.contact is None and "hold command failed" in result.message


def test_a_hold_that_does_not_take_is_a_fault(sim_harness):
    class Ignores(SimBackend):
        def hold(self, arm):
            pass

    h = sim_harness(backend=Ignores(sim_backend().arms))
    done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 30}, "relative": True})
    assert obs.result.status == "fault" and "not confirmed" in obs.result.message


# ── contact and grasp evidence ──
class Lagging(SimBackend):
    """Joints that follow their command ``delay`` ticks late; a hold takes effect at once."""
    delay = 6

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
        SimBackend.step(self, {arm: ArmTarget(state.q, state.gripper)}, 0.0)


def test_lag_at_the_end_of_a_move_is_not_contact(sim_harness):
    config = SafetyConfig(settle_s=0, contact_tolerance_m=0.05, tracking_lag_s=0.3)
    h = sim_harness(backend=Lagging(sim_backend().arms), safety=config)
    done(h.step({"action": "home"}))
    obs = h.step({"action": "nudge", "dxyz": [0, 0, -0.1], "until": "contact"})
    assert obs.result.contact is None and obs.result.status in ("stopped", "done")
    assert "contact" not in obs.result.message or "without contact" in obs.result.message


def test_a_gripper_still_moving_at_its_limit_is_not_holding(sim_harness):
    import makermods_llm_harness.backends.sim as sim
    h = sim_harness()
    done(h.step({"action": "home"}))
    done(h.step({"action": "gripper", "opening": 1}))
    original = sim.GRIPPER_SPEED
    sim.GRIPPER_SPEED = 0.2                               # 1.0 -> 0.6 in the 2 s the action waits
    try:
        obs = h.step({"action": "gripper", "opening": 0})
    finally:
        sim.GRIPPER_SPEED = original
    assert obs.result.status == "stopped" and obs.result.grip["holding"] is False
    assert obs.result.grip["settled"] is False and "still moving" in obs.result.message


# ── the tracking lag ──
def test_the_tracking_lag_is_capped():
    with pytest.raises(ValueError, match="tracking_lag_s must be between 0 and 0.5"):
        SafetyConfig(tracking_lag_s=2.0)


def test_a_long_lag_does_not_hide_a_blocked_arm(sim_harness):
    h = sim_harness(safety=SafetyConfig(**FAST, tracking_lag_s=0.5))
    done(h.step({"action": "home"}))
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"shoulder_pan": 80}})
    assert obs.result.status == "stopped" and "blocked" in obs.result.message
    sent = h.backend.read()["arm"]
    assert np.array_equal(sent.q_target, sent.q)


def test_the_lag_is_measured_in_elapsed_time(sim_harness):
    class SlowTicks(Lagging):
        """Each tick takes three periods of wall time, so 4 ticks of delay are 0.48 s."""
        delay = 4

        def step(self, targets, dt):
            super().step(targets, dt)
            if dt:
                time.sleep(3 * dt)

    config = SafetyConfig(**FAST, tracking_lag_s=0.3, tracking_tolerance_deg=1.0)
    h = sim_harness(backend=SlowTicks(sim_backend().arms), safety=config)
    obs = h.step({"action": "home"})
    assert obs.result.status == "stopped" and "blocked" in obs.result.message


# ── speed and duration ──
def recording(cls=SimBackend):
    class Recording(cls):
        targets = []

        def step(self, targets, dt):
            Recording.targets += [t for t in targets.values()]
            super().step(targets, dt)
    return Recording


def test_a_requested_speed_reaches_every_target(sim_harness):
    backend = recording()(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    type(backend).targets = []
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 3}, "relative": True, "speed_deg_s": 2}))
    speeds = [t.speed for t in type(backend).targets if t.speed is not None]
    assert speeds and max(speeds) <= math.radians(2) + 1e-9


def test_duration_stretches_a_short_move(sim_harness):
    backend = recording()(sim_backend().arms)
    h = sim_harness(backend=backend)
    done(h.step({"action": "home"}))
    type(backend).targets = []
    done(h.step({"action": "move_joints", "joints_deg": {"wrist_roll": 0.1}, "relative": True, "duration_s": 8}))
    assert len(type(backend).targets) >= 8 * h.config.rate_hz


# ── limits ──
def test_settings_and_requests_outside_their_range_are_refused_naming_the_limit(sim_harness):
    with pytest.raises(ValueError, match="settle_s must be between 0 and 5"):
        SafetyConfig(settle_s=1e9)
    with pytest.raises(ValueError, match="rate_hz must be between"):
        SafetyConfig(rate_hz=1e6)
    with pytest.raises(ActionError, match="at least 0.5"):
        parse_action({"action": "home", "speed_deg_s": 1e-9})
    with pytest.raises(ActionError, match="at most 60"):
        parse_action({"action": "home", "duration_s": 1e9})
    h = sim_harness(safety=SafetyConfig(**FAST))
    done(h.step({"action": "home"}))
    obs = h.step({"action": "move_joints", "joints_deg": {"shoulder_pan": 80}, "speed_deg_s": 0.5})
    assert obs.result.status == "rejected" and f"more than {MAX_SECONDS:g} s" in obs.result.message


# ── checks that agree with execution ──
def test_reachable_refuses_on_a_read_only_harness(sim_harness):
    h = sim_harness(read_only=True)
    verdict, = h.reachable([{"xyz": [0.31, 0, 0.54], "pitch": -5}])
    assert not verdict["ok"] and "read-only" in verdict["why"]
    assert h.step({"action": "move_tool", "xyz": [0.31, 0, 0.54], "pitch_deg": -5}).result.status == "rejected"


def test_reachable_refuses_an_arm_that_could_not_be_turned_on(sim_harness):
    backend = sim_backend()
    backend._arms["arm"].rest_q = backend._arms["arm"].q + np.radians(20)
    h = sim_harness(backend=backend)
    verdict, = h.reachable([{"xyz": [0.31, 0, 0.54], "pitch": -5}], start="home")
    assert not verdict["ok"] and "rest pose" in verdict["why"]
    assert h.step({"action": "move_tool", "xyz": [0.31, 0, 0.54], "pitch_deg": -5}).result.status == "rejected"


def test_a_rest_dry_run_reports_what_rest_would(sim_harness):
    h = sim_harness()
    done(h.step({"action": "home"}))
    h.backend.set_fault("arm", "motor 3 overcurrent")
    assert h.check({"action": "rest"}).status == "fault" == h.step({"action": "rest"}).result.status
    h.backend.set_fault("arm", None)

    class Unknown(SimBackend):
        def read(self):
            states = super().read()
            for s in states.values():
                s.enabled = None
            return states

    u = sim_harness(backend=Unknown(sim_backend().arms))
    assert u.check({"action": "rest"}).status == "stopped" == u.step({"action": "rest"}).result.status


# ── rest ──
def test_rest_is_not_released_without_fresh_fault_free_feedback(sim_harness):
    for broken in ("nan", "fault"):
        class Broken(SimBackend):
            broken_now = False
            released = 0

            def read(self):
                states = super().read()
                if self.broken_now:
                    if broken == "nan":
                        states["arm"].q = np.full(6, np.nan)
                    else:
                        states["arm"].fault = "encoder lost"
                return states

            def rest(self, arm):
                Broken.released += 1
                return super().rest(arm)

        backend = Broken(sim_backend().arms)
        h = sim_harness(backend=backend)
        done(h.step({"action": "home"}))
        original = h._execute

        def then_break(*args, **kwargs):
            result = original(*args, **kwargs)
            backend.broken_now = kwargs.get("rest", False)
            return result

        h._execute = then_break
        obs = h.step({"action": "rest"})
        assert obs.result.status == "fault" and Broken.released == 0, (broken, obs.result.message)
        backend.broken_now = False


# ── a check while a motion is in flight ──
def test_a_check_during_a_motion_starts_where_that_motion_ends(sim_harness):
    h = sim_harness()
    paused, release, ticks = threading.Event(), threading.Event(), []
    watch = h._watch

    def pausing(name, window=None, *rest, **options):
        state = watch(name, window, *rest, **options)
        ticks.append(1)
        if len(ticks) == 5:                                  # between two ticks of the home motion
            paused.set()
            assert release.wait(60)
        return state

    h._watch = pausing
    worker = threading.Thread(target=lambda: h.step({"action": "home"}))
    worker.start()
    assert paused.wait(60)
    assert h._heading                                       # the motion is in flight
    from_here, = h.reachable([{"xyz": [0.31, 0, 0.54], "pitch": -5}])
    release.set()
    worker.join(60)
    assert from_here["ok"], from_here["why"]


# ── MuJoCo ──
def test_mujoco_images_from_several_threads():
    pytest.importorskip("mujoco")
    from makermods_llm_harness import robot
    from makermods_llm_harness.backends import mujoco as _  # noqa: F401
    with Harness.open(robot.simulated("mujoco")) as h:
        first = h.look()
        if h.backend.render_error:
            pytest.skip(f"offscreen rendering is unavailable here: {h.backend.render_error}")
        looks = []
        worker = threading.Thread(target=lambda: looks.append(h.look()))
        worker.start()
        worker.join(60)
        assert first["errors"] == {} and looks[0]["errors"] == {}


# ── a contact move that ends with a joint blocked short of its goal ──
def blocked_short(h, monkeypatch, gap=0.013):
    """Over the surface, with the surface ``gap`` m above the end of a 3 cm descent and the in-motion contact
    detection kept from firing (as on a Metal arm sliding along the table): the descent ends with wrist_flex about
    3 deg short of its goal on a 7.5 deg move, the tool point 13 mm short along the approach, standing still."""
    from makermods_llm_harness import harness as harness_module
    h.backend.surface_z = -1.0
    r = over_the_surface(h)
    h.backend.surface_z = h.state().arms["arm"].tool.xyz[2] - 0.03 + gap
    monkeypatch.setattr(harness_module, "CONTACT_RATIO", -1.0)
    return r


def test_a_contact_move_ending_with_a_joint_blocked_short_and_standing_still_is_contact(sim_harness, monkeypatch):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = blocked_short(h, monkeypatch)
    result = r.nudge([0, 0, -0.03], until="contact")
    assert result.status == "done" and result.contact is not None, result.message
    assert "stopped on contact at the end of the move" in result.message
    assert result.contact["short_m"] == pytest.approx(0.013, abs=0.003) and held(h)


def test_the_same_ending_without_until_contact_stays_stopped(sim_harness, monkeypatch):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = blocked_short(h, monkeypatch)
    result = r.nudge([0, 0, -0.03])
    assert result.status == "stopped" and result.contact is None
    assert "deg from its goal on a 7.5 deg move" in result.message and held(h)


class GivingWay(Surface):
    """Stopped by the surface, the arm creeps on toward its target once the command has ended (the same target
    twice), ``creep`` of the remaining way per tick (as it was when it met the surface): a support giving way."""
    creep = 0.0
    met = None
    last = None

    def step(self, targets, dt):
        target = {name: np.asarray(t.q, dtype=float).copy() for name, t in targets.items()}
        ended = self.last is not None and all(np.array_equal(target[n], self.last[n]) for n in target)
        self.last = target
        if self.met is not None and ended:
            for name, a in self._arms.items():
                a.q_target = target[name]
                a.q = a.q + self.creep * (a.q_target - self.met)
            return
        super().step(targets, dt)
        for a in self._arms.values():
            if self.met is None and a.model.tool_pose(a.q)[2, 3] < self.surface_z + 0.002:
                self.met = a.q.copy()


def test_the_same_contact_ending_while_still_moving_stays_stopped(sim_harness, monkeypatch):
    h = sim_harness(backend=GivingWay(sim_backend().arms))
    r = blocked_short(h, monkeypatch, gap=0.016)
    h.backend.creep = 0.012                              # 0.17 mm a tick: 1.3 mm over STILL_S, under the watch
    real_window, real_hold, events = Harness._still_window, Harness._hold, []

    def window(measured, seen, last):
        found = real_window(measured, seen, last)
        events.append(("window", found))
        return found

    def hold(self, *args, **kwargs):
        events.append(("hold", None))
        return real_hold(self, *args, **kwargs)
    monkeypatch.setattr(Harness, "_still_window", staticmethod(window))
    monkeypatch.setattr(Harness, "_hold", hold)
    result = r.nudge([0, 0, -0.03], until="contact")
    assert result.status == "stopped" and result.contact is None and "still moving" in result.message, \
        (result.message, result.error)
    kinds = [kind for kind, _ in events]
    assert "window" in kinds and kinds.index("window") < kinds.index("hold")       # judged from samples taken
    q_past, q_last, span = events[kinds.index("window")][1]                         # under the watch, before it
    model = h.models["arm"]                                                          # holds
    assert span >= harness_module_still_s()
    assert np.linalg.norm(model.tool_points(q_last) - model.tool_points(q_past)) >= 0.001


def harness_module_still_s():
    from makermods_llm_harness import harness as harness_module
    return harness_module.STILL_S


def test_the_same_contact_ending_with_a_stop_request_stays_stopped(sim_harness, monkeypatch):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = blocked_short(h, monkeypatch)
    real = Harness._arrival

    def arrival(self, *args, **kwargs):                  # a stop lands as the move ends
        stopper = threading.Thread(target=self.stop)
        stopper.start()
        stopper.join(10)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Harness, "_arrival", arrival)
    result = r.nudge([0, 0, -0.03], until="contact")
    assert result.status == "stopped" and result.contact is None and "on request" in result.message


def test_a_stop_during_the_contact_decision_waits_for_it(sim_harness, monkeypatch):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = blocked_short(h, monkeypatch)
    real_execute, real_hold = Harness._execute, Surface.hold
    seen = {}

    def execute(self, *args, **kwargs):
        out = real_execute(self, *args, **kwargs)
        seen["ended"] = True
        return out

    def hold(self, arm):                                  # the hold of the contact decision: a stop lands now
        if seen.get("ended") and "stopper" not in seen:
            abort = h._abort
            seen["stopper"] = threading.Thread(target=lambda: seen.setdefault("stop", h.stop()))
            seen["stopper"].start()
            seen["stopper"].join(0.5)
            seen["abort_set_meanwhile"] = abort.is_set()
        return real_hold(self, arm)
    monkeypatch.setattr(Harness, "_execute", execute)
    monkeypatch.setattr(Surface, "hold", hold)
    result = r.nudge([0, 0, -0.03], until="contact")
    seen["stopper"].join(10)
    assert seen["abort_set_meanwhile"] is False            # the stop waited for the decision ...
    assert result.status == "done" and result.contact is not None and seen["stop"] is None   # ... then held it
    assert held(h)


def test_a_support_giving_way_after_a_blocked_ending_does_not_reach_the_target(sim_harness, monkeypatch):
    class Gives(Surface):
        give = None
        holding = False

        def read(self):
            if self.give is not None and not self.holding:   # the control loop drives on to the target in 0.3 s
                t0, start = self.give
                f = min(1.0, (time.monotonic() - t0) / 0.3)
                for a in self._arms.values():
                    a.q = start + f * (a.q_target - start)
            return super().read()

        def hold(self, arm):
            self.holding = True
            return super().hold(arm)

    h = sim_harness(backend=Gives(sim_backend().arms))
    r = blocked_short(h, monkeypatch)
    real = Harness._execute

    def execute(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        self.backend.give = (time.monotonic(), self.backend._arms["arm"].q.copy())   # the support gives way now
        return out
    monkeypatch.setattr(Harness, "_execute", execute)
    surface = h.backend.surface_z
    result = r.nudge([0, 0, -0.03], until="contact")
    assert h.state().arms["arm"].tool.xyz[2] >= surface - 0.002, result.message      # held at once, not 13 mm on


def test_a_blocked_ending_that_crept_within_the_tolerance_is_not_contact(sim_harness, monkeypatch):
    class Creeps(Surface):
        reads = 0
        toward = None

        def read(self):
            if self.toward is not None and self.reads < 2:      # the tool creeps 2 x 0.4 mm on toward the target
                self.reads += 1
                for a in self._arms.values():
                    a.q = a.q + 0.04 * (self.toward - self.start)
            return super().read()

    h = sim_harness(backend=Creeps(sim_backend().arms))
    r = blocked_short(h, monkeypatch, gap=0.0105)
    real = Harness._execute

    def execute(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        a = self.backend._arms["arm"]
        self.backend.start, self.backend.toward = a.q.copy(), a.q_target.copy()
        return out
    monkeypatch.setattr(Harness, "_execute", execute)
    result = r.nudge([0, 0, -0.03], until="contact")
    assert result.status == "stopped" and result.contact is None, result.message
    assert result.error["short_m"] <= 0.010                                   # the final reading decides


class ElbowBlocked(SimBackend):
    """elbow_flex stops where the tool point first goes under ``surface_z``; the other joints go on (as a Metal
    arm whose shoulder blocks on the table while the rest slide the fingertips along it)."""
    surface_z = -1.0
    elbow = None

    def step(self, targets, dt):
        super().step(targets, dt)
        for a in self._arms.values():
            if self.elbow is None and a.model.tool_pose(a.q)[2, 3] < self.surface_z:
                self.elbow = a.q[2]
            if self.elbow is not None:
                a.q[2] = self.elbow


def side_slide(sim_harness, monkeypatch, table_z=None):
    """A 3 cm vertical descent whose elbow blocks 16 mm above its end: about 3 deg short on a 6 deg move, the
    tool point 12 mm short along the approach and 16 mm forward of it."""
    from makermods_llm_harness import harness as harness_module
    monkeypatch.setattr(harness_module, "CONTACT_RATIO", -1.0)
    kwargs = {} if table_z is None else {"table_z": table_z}
    h = sim_harness(backend=ElbowBlocked(sim_backend().arms), **kwargs)
    r = over_the_surface(h)
    h.backend.surface_z = h.state().arms["arm"].tool.xyz[2] - 0.03 + 0.016
    return h, r.nudge([0, 0, -0.03], until="contact")


def test_a_blocked_ending_to_the_side_of_its_path_and_clear_of_the_table_is_not_contact(sim_harness, monkeypatch):
    h, result = side_slide(sim_harness, monkeypatch)
    assert result.status == "stopped" and result.contact is None, result.message
    assert "to the side of its path" in result.message and held(h)


def test_a_blocked_ending_that_slid_along_the_stored_table_is_contact(sim_harness, monkeypatch):
    h, result = side_slide(sim_harness, monkeypatch)
    lowest = h.state().arms["arm"].clearance_m + h.envelopes["arm"].table_z     # where the arm came to rest
    h2, result = side_slide(sim_harness, monkeypatch, table_z=lowest - 0.002)  # the table stored just there
    assert result.status == "done" and result.contact is not None, result.message
    assert abs(h2.state().arms["arm"].clearance_m) <= 0.01


def test_a_second_contact_descent_goes_no_deeper_than_the_first_was_held(sim_harness):
    h = sim_harness()                                      # no surface: the arm would sink through the table
    r = over_the_surface(h)
    clearance = h.state().arms["arm"].clearance_m
    first = r.nudge([0, 0, -(clearance + 0.020)], until="contact")
    held_at = h.state().arms["arm"].clearance_m
    assert first.status == "stopped" and held_at == pytest.approx(-0.010, abs=0.002)   # the watch: 10 mm under
    second = r.nudge([0, 0, -0.010], until="contact")
    assert second.status != "done"
    assert h.state().arms["arm"].clearance_m >= held_at - 0.001


# ── a contact move onto a correctly stored table ──
class OnTable(SimBackend):
    """The arm stops where its lowest point would go under the table the harness has stored (``table_z``)."""

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            if a.model.lowest_z(a.q) < self.table_z:
                a.q = before[name]


def test_a_contact_move_may_aim_a_bounded_depth_under_a_correctly_stored_table(sim_harness):
    h = sim_harness(backend=OnTable(sim_backend().arms))
    r = over_the_surface(h)
    clearance = h.state().arms["arm"].clearance_m
    plain = r.nudge([0, 0, -(clearance + 0.012)], dry_run=True)
    assert plain.status == "rejected" and "hard floor" in plain.message           # other moves: not under it
    deep = r.nudge([0, 0, -(clearance + 0.030)], until="contact", dry_run=True)   # past the 25 mm contact depth
    result = r.nudge([0, 0, -(clearance + 0.020)], until="contact")
    assert result.status == "done" and result.contact is not None, result.message
    assert deep.status == "rejected" and "floor of a move with until contact" in deep.message
    assert h.state().arms["arm"].clearance_m == pytest.approx(0.0, abs=0.003)     # it stopped on the table
    assert SafetyConfig().contact_depth_m == 0.025
