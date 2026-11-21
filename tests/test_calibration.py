import json

import pytest

from makermods_llm_harness import Harness
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.safety import (CalibrationError, calibration_path, load_calibration, measure_table,
                                          save_calibration, watch_for_touch)

from conftest import sim_backend


def readings(*values):
    it = iter(values)
    return lambda: next(it)


def test_measure_table_takes_two_agreeing_readings():
    said = []
    z = measure_table(readings(0.101, 0.103), ask=lambda prompt: "", say=said.append, gap_s=0)
    assert z == 0.103
    assert said[-1] == "Table at z = 0.1030 m."


def test_measure_table_retries_when_the_arm_moves():
    said = []
    z = measure_table(readings(0.10, 0.12, 0.05, 0.051), ask=lambda prompt: "", say=said.append, gap_s=0)
    assert z == 0.051
    assert any("moved 20.0 mm" in s and "Attempt 1/3" in s for s in said)


def test_measure_table_gives_up_after_three_attempts():
    with pytest.raises(CalibrationError, match="did not hold still"):
        measure_table(readings(0.0, 0.01, 0.0, 0.01, float("nan"), 0.0), ask=lambda p: "", say=lambda s: None, gap_s=0)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def test_watch_for_touch_returns_after_a_move_and_a_still_hold():
    clock = FakeClock()
    z = lambda: 0.30 if clock.now < 2 else 0.05 + (0.001 if int(clock.now * 10) % 2 else 0.0)  # noqa: E731
    ask = watch_for_touch(z, say=lambda s: None, clock=clock, sleep=clock.sleep)
    assert ask("Press Enter") == ""
    assert 6.5 <= clock.now <= 8.0


def test_watch_for_touch_times_out():
    clock = FakeClock()
    ask = watch_for_touch(lambda: 0.3, say=lambda s: None, timeout_s=10, clock=clock, sleep=clock.sleep)
    with pytest.raises(CalibrationError, match="timed out"):
        ask("Press Enter")


def test_calibration_store_keeps_each_arm():
    assert load_calibration("bench") == {}
    save_calibration("bench", "left", 0.01, {"entry": {"port": "a"}}, measured_at=5.0)
    save_calibration("bench", "right", -0.02, {})
    data = load_calibration("bench")
    assert data["left"] == {"table_z": 0.01, "measured_at": 5.0, "setup": {"entry": {"port": "a"}}}
    assert data["right"]["table_z"] == -0.02
    calibration_path("bench").write_text("{not json")
    with pytest.raises(CalibrationError, match="llm-harness calibrate"):
        load_calibration("bench")
    with pytest.raises(CalibrationError):
        save_calibration("x", "left", float("inf"), {})
    calibration_path("bench").write_text('{"left": {"table_z": NaN, "measured_at": 1, "setup": {}}}')
    with pytest.raises(CalibrationError, match="not a finite number"):
        load_calibration("bench")


class Uncalibrated(SimBackend):
    """A simulator that, like a real robot, does not know its table."""

    @property
    def table_z(self):
        return None


def uncalibrated_backend():
    sim = sim_backend(arms=("left", "right"))
    return Uncalibrated(sim.arms, cameras=())


def test_harness_refuses_motion_until_calibrated(sim_harness):
    from makermods_llm_harness.robot import simulated

    robot = simulated(arms=("left", "right"), name="bench")
    h = sim_harness(backend=uncalibrated_backend())
    obs = h.step({"action": "nudge", "arm": "left", "dxyz": [0, 0, 0.01]})
    assert obs.result.status == "rejected" and "llm-harness calibrate" in obs.result.message
    assert obs.arms["left"].clearance_m is None and not obs.arms["left"].enabled
    assert "llm-harness calibrate" in h.describe()
    obs = h.step({"action": "gripper", "arm": "left", "opening": 1})
    assert obs.result.status == "rejected" and "llm-harness calibrate" in obs.result.message
    assert not obs.arms["left"].enabled and obs.arms["left"].gripper == 0.0

    save_calibration("bench", "left", 0.0, robot.setup("left"))
    calibrated = Harness.open(robot, backend=uncalibrated_backend())
    try:
        assert calibrated.envelopes["left"].table_z == 0.0 and calibrated.envelopes["right"].table_z is None
        assert calibrated.step({"action": "nudge", "arm": "left", "dxyz": [0, 0, 0.01]}).result.ok
        assert calibrated.step({"action": "nudge", "arm": "right", "dxyz": [0, 0, 0.01]}).result.status == "rejected"
    finally:
        calibrated.close()
    assert json.loads(calibration_path("bench").read_text())["left"]["table_z"] == 0.0


def test_table_z_argument_overrides_per_arm(sim_harness):
    h = sim_harness(arms=("left", "right"), table_z={"right": -0.01})
    assert h.envelopes["left"].table_z == 0.0 and h.envelopes["right"].table_z == -0.01
    assert sim_harness(table_z=0.02).envelopes["arm"].table_z == 0.02


def test_a_stored_height_is_used_only_on_the_setup_it_was_measured_with():
    from makermods_llm_harness.robot import simulated

    robot = simulated(arms=("left", "right"), name="bench")
    save_calibration("bench", "left", 0.0, robot.setup("left"))
    moved = simulated(arms=("left", "right"), name="bench")
    moved.bases["left"]["xyz"] = [0.0, 0.25, 0.03]
    longer = simulated(arms=("left", "right"), name="bench")
    longer.safety = {"finger_length_m": 0.03}
    for other in (moved, longer):
        with Harness.open(other, backend=uncalibrated_backend()) as h:
            assert h.envelopes["left"].table_z is None
    with Harness.open(robot, backend=uncalibrated_backend()) as h:
        assert h.envelopes["left"].table_z == 0.0
