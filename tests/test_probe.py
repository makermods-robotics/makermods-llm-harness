"""The powered table calibration on the kinematic simulator: an arm whose fingertips cannot go below a table that
rises with the reach, an arm that trails its command, a table lower than the probe may go, and the refusals before
anything moves."""

import math

import numpy as np
import pytest

from makermods_llm_harness import cli, probe
from makermods_llm_harness import robot as robots
from makermods_llm_harness.cli import main
from makermods_llm_harness.robot import Robot
from makermods_llm_harness.safety import CalibrationError, load_calibration

from test_cli import Uncalibrated, write_rig

REACH, HEIGHT = (0.26, 0.40), (0.0487, 0.0602)     # a table the model sees higher further out
POINTS = [(0.26, 0.06), (0.33, -0.08), (0.40, 0.07)]


def table_at(reach):
    return float(np.interp(reach, REACH, HEIGHT))


class TableTop(Uncalibrated):
    """The fingertips stop where they would go below the table under them."""

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            tip = a.model.fingertip(a.q)
            if tip[2] < table_at(math.hypot(tip[0], tip[1])):
                a.q = before[name]


class Trailing(TableTop):
    """Also trails its command: each tick covers only a third of the way to it."""

    def step(self, targets, dt):
        before = {name: a.q.copy() for name, a in self._arms.items()}
        super().step(targets, dt)
        for name, a in self._arms.items():
            a.q = before[name] + (a.q - before[name]) / 3


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "STILL_S", 0.0)
    assert main(["robot", "init", "bench", "--arm", "maker", "--from-rig", str(write_rig(tmp_path)),
                 "--arms", "left", "--cameras", "top"]) == 0
    made = []

    def backend(cls):
        def open_backend(self):
            b = cls(self.arm_infos(), cameras=())
            made.append(b)
            return b
        monkeypatch.setattr(Robot, "open_backend", open_backend)
    backend(TableTop)
    rest = TableTop(robots.load("bench").arm_infos(), cameras=())._arms["left"].rest_q
    robot = robots.load("bench")
    robot.rest_q = {"left": [round(math.degrees(v), 3) for v in rest]}
    robots.save(robot)
    return backend, made


@pytest.mark.parametrize("backend_cls", [TableTop, Trailing])
def test_the_powered_calibration_finds_a_table_that_rises_with_the_reach(bench, backend_cls):
    backend, made = bench
    backend(backend_cls)
    said = []
    tables = cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS, expect=0.05, yes=True,
                                   say=said.append)
    stored = load_calibration("bench")["left"]
    measured = dict(stored["table_reach"])
    for (x, y) in POINTS:
        reach = math.hypot(x, y)
        z = min(measured.items(), key=lambda kv: abs(kv[0] - reach))[1]
        assert z == pytest.approx(table_at(reach), abs=0.003), (reach, z)
    assert tables["left"].reach and stored["table_z"] == pytest.approx(np.mean(list(measured.values())))
    arm = made[-1]._arms["left"]
    assert not arm.enabled and np.max(np.abs(arm.q - arm.rest_q)) < math.radians(5)     # back at rest, off
    assert any("table at z = " in s for s in said)


def test_no_table_above_the_limit_ends_at_rest_and_stores_nothing(bench, monkeypatch):
    backend, made = bench
    monkeypatch.setattr(TableTop, "step", Uncalibrated.step)          # nothing to touch
    with pytest.raises(CalibrationError, match="no table found"):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, yes=True,
                              say=lambda s: None)
    assert load_calibration("bench") == {}
    arm = made[-1]._arms["left"]
    assert not arm.enabled and np.max(np.abs(arm.q - arm.rest_q)) < math.radians(5)
    lowest = min(arm.model.fingertip(q)[2] for q in [arm.q])
    assert lowest > 0.05 - probe.BELOW_M - 0.003


def test_a_table_higher_than_expected_is_refused_not_pressed(bench):
    backend, made = bench
    with pytest.raises(CalibrationError, match="higher than expected|above the probe point"):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=-0.03, yes=True,
                              say=lambda s: None)
    arm = made[-1]._arms["left"]
    assert not arm.enabled and np.max(np.abs(arm.q - arm.rest_q)) < math.radians(5) and load_calibration("bench") == {}


@pytest.mark.parametrize("change, words", [("away", "starts only at the recorded rest pose"),
                                           ("hot", "let it cool")])
def test_the_powered_calibration_starts_only_at_rest_and_cool(bench, monkeypatch, change, words):
    backend, made = bench
    real = TableTop.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        arm = self._arms["left"]
        if change == "away":
            arm.q = arm.q + np.radians([0, 10, 0, 0, 0, 0])
        else:
            arm.temperature = probe.MAX_START_C + 1
    monkeypatch.setattr(TableTop, "__init__", init)
    with pytest.raises(CalibrationError, match=words):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, yes=True,
                              say=lambda s: None)
    assert not made[-1]._arms["left"].enabled and load_calibration("bench") == {}


def test_powered_needs_its_own_options_and_asks_first(bench, monkeypatch, capsys):
    assert main(["calibrate", "--points", "0.3,0"]) == 2
    assert "--points and --expect go with --powered" in capsys.readouterr().err
    with pytest.raises(CalibrationError, match="not calibrated"):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, ask=lambda p: "n",
                              say=lambda s: None)


def _probe_steps(monkeypatch, change):
    """Wrap Harness.step: ``change(harness, obs)`` on every probe step; returns the commanded fingertip targets."""
    from makermods_llm_harness.harness import Harness
    real, targets = Harness.step, []

    def step(self, action, *args, **kwargs):
        obs = real(self, action, *args, **kwargs)
        if isinstance(action, dict) and action.get("speed_deg_s") == probe.PROBE_SPEED_DEG_S:
            model = self.models["left"]
            p, w = math.radians(action["pitch_deg"]), math.radians(action["yaw_deg"])
            pointing = np.array([math.cos(p) * math.cos(w), math.cos(p) * math.sin(w), math.sin(p)])
            targets.append(float(action["xyz"][2] + model.gripper.finger_length_m * pointing[2]))
            change(self, obs)
        return obs
    monkeypatch.setattr(Harness, "step", step)
    return targets


def test_a_probe_step_stopped_short_is_a_reading_not_a_failure(bench, monkeypatch):
    stopped = []

    def change(harness, obs):                          # as a real arm whose free offset is large: steps resting on
        short = (obs.result.error or {}).get("short_m")    # the table end stopped short and held where they stand
        if obs.result.ok and short is not None and short > 0.003:
            harness.confirm_hold(["left"])
            obs.result.status, obs.result.message = "stopped", "left: ended short of the target along the approach"
            obs.result.error = dict(obs.result.error, ended_short=True)
            stopped.append(short)
    _probe_steps(monkeypatch, change)
    cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:2], expect=0.05, yes=True, say=lambda s: None)
    assert stopped
    measured = dict(load_calibration("bench")["left"]["table_reach"])
    for (x, y) in POINTS[:2]:
        reach = math.hypot(x, y)
        z = min(measured.items(), key=lambda kv: abs(kv[0] - reach))[1]
        assert z == pytest.approx(table_at(reach), abs=0.003)


def test_any_other_stop_of_a_probe_step_ends_the_probe(bench, monkeypatch):
    def change(harness, obs):                          # a tracking stop also says how far short the move ended
        if len(seen) == probe.FREE_STEPS + 1:
            harness.confirm_hold(["left"])
            obs.result.status, obs.result.message = "stopped", "left: shoulder_lift fell behind its command"
            obs.result.error = dict(obs.result.error or {}, short_m=0.004)
        seen.append(obs.result.status)
    seen = []
    _probe_steps(monkeypatch, change)
    with pytest.raises(CalibrationError, match="fell behind its command"):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, yes=True,
                              say=lambda s: None)
    assert len(seen) == probe.FREE_STEPS + 2 and load_calibration("bench") == {}


def test_a_blocked_fingertip_is_never_commanded_far_under_where_it_stands(bench, monkeypatch):
    from makermods_llm_harness import harness as harness_module
    monkeypatch.setattr(probe, "CONTACT_M", 1.0)       # the probe's contact and pressure checks and the harness's
    monkeypatch.setattr(probe, "PRESS_M", 1.0)         # blocked-joint check see nothing: the bound alone remains
    monkeypatch.setattr(harness_module, "judge_motion", lambda *a, **k: None)
    measured = []

    def change(harness, obs):
        measured.append(float(harness.models["left"].fingertip(harness.backend.read()["left"].q)[2]))
    targets = _probe_steps(monkeypatch, change)
    with pytest.raises(CalibrationError, match="stopped before pressing harder"):
        cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, yes=True,
                              say=lambda s: None)
    lowest = min(measured)
    assert lowest == pytest.approx(table_at(math.hypot(*POINTS[0])), abs=0.003)
    assert min(targets) >= lowest - probe.BELOW_CONTACT_M - 1e-6
    assert min(targets) > 0.05 - probe.BELOW_M + 0.01                 # far above the limit it used to reach


@pytest.mark.parametrize("stored", ["none", "other setup"])
def test_without_expect_the_powered_calibration_needs_a_table_stored_for_the_present_setup(bench, stored):
    from makermods_llm_harness.safety import Table, save_calibration
    backend, made = bench
    robot = robots.load("bench")
    if stored == "other setup":
        setup = robot.setup("left")
        setup["base_xyz"] = [0.0, 0.0, 0.03]                       # the arm base was moved since
        save_calibration("bench", "left", Table(0.05), setup)
    before = load_calibration("bench")
    with pytest.raises(CalibrationError, match="give --expect Z"):
        cli.calibrate_powered(robot, ["left"], points=POINTS[:1], yes=True, say=lambda s: None)
    assert not made and load_calibration("bench") == before            # nothing opened, nothing stored


def test_a_stored_table_per_reach_sets_the_expectation_and_limit_at_each_points_own_reach(bench, monkeypatch):
    from makermods_llm_harness.safety import Table, save_calibration
    robot = robots.load("bench")
    stored = Table(0.058, ((0.25, 0.048), (0.40, 0.068)))              # mean 0.058; 2 cm higher out at 0.40 m
    save_calibration("bench", "left", stored, robot.setup("left"))
    real, seen = probe._probe_point, []

    def probe_point(harness, arm, x, y, expect_z, limit_z, say):
        floor = harness.envelopes[arm].table_under([x, y, 0.0])
        seen.append((math.hypot(x, y), expect_z, limit_z, floor))
        return real(harness, arm, x, y, expect_z, limit_z, say)
    monkeypatch.setattr(probe, "_probe_point", probe_point)
    cli.calibrate_powered(robot, ["left"], points=[POINTS[0], POINTS[2]], yes=True, say=lambda s: None)
    assert len(seen) == 2
    for reach, expect_z, limit_z, floor in seen:
        want = float(stored.at(np.array([[reach, 0.0, 0.0]]), (0.0, 0.0))[0])
        assert expect_z == pytest.approx(want) and limit_z == pytest.approx(want - probe.BELOW_M)
        assert floor == pytest.approx(limit_z)
    assert seen[1][2] > stored.z - probe.BELOW_M + 0.01                 # far out: well above the mean's limit


def test_the_probe_judges_its_steps_with_its_own_contact_tolerance(bench, monkeypatch):
    robot = robots.load("bench")
    robot.safety = dict(robot.safety or {}, contact_tolerance_m=0.003)      # tighter, set for other work
    robots.save(robot)
    real, seen = probe._probe_point, []

    def probe_point(harness, arm, *args):
        seen.append(harness.config.contact_tolerance_m)
        return real(harness, arm, *args)
    monkeypatch.setattr(probe, "_probe_point", probe_point)
    cli.calibrate_powered(robots.load("bench"), ["left"], points=POINTS[:1], expect=0.05, yes=True,
                          say=lambda s: None)
    assert seen == [0.010] and probe.SHORT_M == 0.010
