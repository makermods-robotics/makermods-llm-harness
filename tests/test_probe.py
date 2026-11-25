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
