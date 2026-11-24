"""The program API on the simulators: reach checks that agree with the moves, dry runs, until contact, gripper hold
detection, speed and duration, joint-space tool moves, camera grids, and the speed settings."""

import collections
import threading

import cv2
import numpy as np
import pytest

from makermods_llm_harness import SafetyConfig
from makermods_llm_harness.backends.base import ArmTarget
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.program import MotionFailed, Program

from conftest import APPROACH, sim_backend


def ready(h, check=True):
    r = Program(h, check=check)
    r.home()
    return r


def ticks_of(backend_cls):
    """A SimBackend subclass counting the targets sent to it."""
    class Counting(backend_cls):
        sent = 0

        def step(self, targets, dt):
            Counting.sent += bool(targets)
            super().step(targets, dt)
    return Counting


# ── reach checks ──
CANDIDATES = [
    {"xyz": [0.31, 0, 0.54], "pitch": -5},            # reachable
    {"xyz": [0.28, 0, 0.12], "pitch": -90},           # too far for one action from home (excursion)
    {"xyz": [2.0, 0, 0.1]},                            # out of reach
    {"xyz": [0.30, 0.0, 0.40], "pitch": -20},          # reachable
    {"xyz": [0.31, 0.05, 0.53], "pitch": -5, "yaw": 20},  # reachable, turned
]


@pytest.mark.parametrize("line", [True, False])
def test_reachable_agrees_with_what_the_move_then_does(sim_harness, line):
    h = sim_harness()
    r = ready(h, check=False)
    verdicts = r.reachable(CANDIDATES, line=line)
    assert [v["ok"] for v in verdicts] == [True, False, False, True, True]
    assert "per-action limit (safety setting max_excursion_deg)" in verdicts[1]["why"]
    assert "out of reach" in verdicts[2]["why"]
    for pose, verdict in zip(CANDIDATES, verdicts):
        r.home()
        before = h.state().arms["arm"].joints_deg
        dry = r.move(pose["xyz"], pitch=pose.get("pitch"), yaw=pose.get("yaw"), line=line, dry_run=True)
        result = r.move(pose["xyz"], pitch=pose.get("pitch"), yaw=pose.get("yaw"), line=line)
        assert result.ok == verdict["ok"] == dry.ok, (pose, verdict, result)
        if verdict["ok"]:
            joints = h.state().arms["arm"].joints_deg
            assert joints == pytest.approx(verdict["joints_deg"], abs=1.5)
            assert verdict["margin_deg"] > 0 and verdict["seconds"] > 0
        else:
            assert result.message == verdict["why"] == dry.message
            assert h.state().arms["arm"].joints_deg == pytest.approx(before, abs=0.2)     # nothing moved


def test_reachable_chains_and_starts_from_home_without_moving(sim_harness):
    h = sim_harness()
    r = Program(h)
    plan = [{"xyz": list(xyz), "pitch": p} for xyz, p in APPROACH["metal"]]
    assert all(v["ok"] for v in r.reachable(plan, chain=True, start="home"))
    unchained = r.reachable(plan, start="home")
    assert unchained[0]["ok"] and not unchained[-1]["ok"]
    broken = r.reachable([{"xyz": [2, 0, 0]}] + plan, chain=True, start="home")
    assert not broken[0]["ok"] and broken[1]["why"] == "not checked: the pose before it is refused"
    assert not h.state().arms["arm"].enabled                     # nothing moved, motors still off
    for xyz, pitch in APPROACH["metal"]:
        if not h.state().arms["arm"].enabled:
            r.home()
        r.move(xyz, pitch=pitch)


def test_a_motion_that_fails_raises_with_its_result(sim_harness):
    h = sim_harness()
    r = ready(h)
    with pytest.raises(MotionFailed) as failed:
        r.move([2, 0, 0])
    assert failed.value.result.status == "rejected"
    assert r.move([0.31, 0, 0.54], pitch=-5).ok


def test_dry_runs_move_nothing_and_leave_the_motors_off(sim_harness):
    h = sim_harness()
    r = Program(h)
    for result in (r.home(dry_run=True), r.gripper("open", dry_run=True), r.rest(dry_run=True)):
        assert result.ok, result.message
    assert "would pass the checks" in r.home(dry_run=True).message
    assert not h.state().arms["arm"].enabled


# ── until contact ──
class Surface(SimBackend):
    """The joints stop following once the tool point would go below ``surface_z``, as on an object."""
    surface_z = 0.08

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            if a.model.tool_pose(a.q)[2, 3] < self.surface_z:
                a.q = before[name]


def test_until_contact_ends_where_the_arm_stops_following_and_holds(sim_harness):
    h = sim_harness(backend=Surface(sim_backend().arms))
    r = ready(h)
    for xyz, pitch in APPROACH["metal"]:
        r.move(xyz, pitch=pitch)                                   # down to z 0.12, over the surface
    result = r.nudge([0, 0, -0.08], until="contact")
    assert result.ok and result.contact is not None and "stopped on contact" in result.message
    assert result.contact["xyz"][2] == pytest.approx(0.08, abs=0.006)
    assert result.contact["short_m"] == pytest.approx(0.04, abs=0.008)
    state = h.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)                 # held where it stopped
    assert r.nudge([0, 0, 0.03]).ok                                 # and it moves on from there
    plain = Program(h, check=False).nudge([0, 0, -0.08])           # without until: the same block stops it
    assert plain.status == "stopped" and plain.contact is None


def test_until_contact_without_contact_reaches_the_target(sim_harness):
    h = sim_harness()
    r = ready(h)
    result = r.nudge([0, 0, -0.05], until="contact")
    assert result.ok and result.contact is None and result.message == "arm: reached the target without contact."


def test_until_contact_on_a_cube_in_mujoco():
    pytest.importorskip("mujoco")
    from makermods_llm_harness import Harness, robot
    with Harness.open(robot.simulated("mujoco", "maker")) as h:
        r = ready(h)
        r.move([0.30, 0.0, 0.125], pitch=-50)
        r.move([0.28, 0, 0.125], pitch=-60)
        r.gripper("close")
        result = Program(h, check=False).move([0.28, 0, 0.045], pitch=-60, until="contact")   # jaws on the cube
        assert not result.ok or result.contact is not None                   # never "reached the target"
        assert h.state().arms["arm"].tool.xyz[2] > 0.05


# ── gripper ──
def test_the_gripper_says_whether_it_holds_something(sim_harness):
    h = sim_harness()
    r = ready(h)
    r.gripper("open")
    empty = r.gripper("close")
    assert empty.grip == {"opening": 0.0, "target": 0.0, "holding": False, "settled": True}
    assert "on nothing" in empty.message
    for xyz, pitch in APPROACH["metal"]:
        r.move(xyz, pitch=pitch)
    r.gripper("open")
    r.move([0.28, 0, 0.035], pitch=-90)
    held = r.gripper("close")
    assert held.grip["holding"] and held.grip["opening"] == pytest.approx(0.5) and "hold something" in held.message


# ── speed, duration, line ──
def test_speed_and_duration_slow_a_motion_down(sim_harness):
    counting = ticks_of(SimBackend)
    h = sim_harness(backend=counting(sim_backend().arms))
    r = Program(h)
    rate = h.config.rate_hz
    for options, at_least in (({}, 0.0), ({"duration": 8.0}, 7.0), ({"speed": 5.0}, None)):
        r.rest()
        counting.sent = 0
        r.home(**options)
        took = counting.sent / rate
        if not options:
            fast = took
        elif at_least is not None:
            assert took >= at_least
        else:
            assert took > 2 * fast
    with pytest.raises(MotionFailed, match="not both"):
        r.home(speed=5, duration=3)


def test_a_joint_space_tool_move_reaches_the_pose(sim_harness):
    h = sim_harness()
    r = ready(h)
    r.move([0.31, 0, 0.54], pitch=-5, line=False)
    assert h.state().arms["arm"].tool.xyz == pytest.approx((0.31, 0, 0.54), abs=2e-3)


# ── looking ──
def test_look_saves_one_labelled_grid_and_motions_capture_no_images(sim_harness, tmp_path):
    h = sim_harness(image_dir=tmp_path, cameras=("top", "side", "arm_wrist", "far"))
    r = Program(h)
    path = r.look()
    grid = cv2.imread(path)
    assert path.endswith("grid-0001.jpg") and grid.shape == (480, 960, 3)
    r.home()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grid-0001.jpg"]
    assert h.state().cameras == {}


def test_state_does_not_wait_for_the_action_in_progress(sim_harness):
    h = sim_harness()
    paused, release, ticks = threading.Event(), threading.Event(), []
    watch = h._watch

    def pausing(name, window=None, *rest, **options):
        state = watch(name, window, *rest, **options)
        ticks.append(1)
        if len(ticks) == 5:
            paused.set()
            assert release.wait(60)
        return state

    h._watch = pausing
    worker = threading.Thread(target=lambda: h.step({"action": "home"}))
    worker.start()
    assert paused.wait(60)
    mid = h.state().arms["arm"].joints_deg                   # answered while the step holds the harness
    release.set()
    worker.join(60)
    assert mid != h.state().arms["arm"].joints_deg


# ── speed settings ──
class Lagging(SimBackend):
    """Joints that follow their command 240 ms late (6 ticks), a little later than the real arms at speed."""
    delay = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = collections.deque()

    def step(self, targets, dt):
        self.pending.append(dict(targets))
        late = self.pending.popleft() if len(self.pending) > self.delay else {}
        super().step(late, dt)

    def hold(self, arm):
        """A hold takes effect at once, as on the real arms."""
        self.pending.clear()
        state = SimBackend.read(self)[arm]
        SimBackend.step(self, {arm: ArmTarget(state.q, state.gripper)}, 0.0)


FAST = {"max_speed_deg_s": 60.0, "slow_speed_deg_s": 20.0}


def test_the_tracking_lag_lets_a_lagging_arm_move_fast_and_still_catches_a_block(sim_harness):
    for lag, status in ((0.0, "stopped"), (0.3, "done")):
        h = sim_harness(backend=Lagging(sim_backend().arms),
                        safety=SafetyConfig(**FAST, max_excursion_deg=90, tracking_lag_s=lag))
        obs = h.step({"action": "home"})
        assert obs.result.status == status, obs.result.message
    assert h.step({"action": "move_joints", "joints_deg": {"shoulder_pan": 60}}).result.ok
    h.backend.block("arm")
    obs = h.step({"action": "move_joints", "joints_deg": {"shoulder_pan": 0}})
    assert obs.result.status == "stopped" and "blocked" in obs.result.message


@pytest.mark.parametrize("backend", ["sim", "mujoco"])
def test_sixty_degrees_per_second_tracks_in_the_simulators(sim_harness, backend):
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        from makermods_llm_harness import Harness, robot
        h = Harness.open(robot.simulated("mujoco", "metal"), safety=SafetyConfig(**FAST))
    else:
        h = sim_harness(safety=SafetyConfig(**FAST))
    try:
        r = Program(h)
        r.home()
        for xyz, pitch in APPROACH["metal"]:
            r.move(xyz, pitch=pitch)
        r.move_joints({"shoulder_pan": 30})
        r.move_joints({"shoulder_pan": -5})
        r.home()
    finally:
        h.close()


def test_speed_settings_come_from_the_robot_file_safety_block():
    config = SafetyConfig.from_dict({"max_speed_deg_s": 60, "slow_speed_deg_s": 15, "slow_zone_m": 0.05,
                                     "max_excursion_deg": 60, "settle_s": 0.3, "tracking_tolerance_deg": 8,
                                     "tracking_lag_s": 0.15, "contact_tolerance_m": 0.006})
    assert (config.settle_s, config.tracking_lag_s, config.contact_tolerance_m) == (0.3, 0.15, 0.006)
    defaults = SafetyConfig()
    assert (defaults.max_speed_deg_s, defaults.slow_speed_deg_s, defaults.max_excursion_deg, defaults.settle_s,
            defaults.tracking_lag_s, defaults.contact_tolerance_m) == (20.0, 5.0, 35.0, 1.0, 0.2, 0.010)
    assert SafetyConfig(settle_s=0, tracking_lag_s=0).settle_s == 0
    with pytest.raises(ValueError):
        SafetyConfig(tracking_lag_s=-1)
    with pytest.raises(ValueError):
        SafetyConfig(contact_tolerance_m=0)

