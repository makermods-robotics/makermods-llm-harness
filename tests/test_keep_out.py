"""Keep-out zones: validation, the planned-path check along the whole arm, recovery from inside a zone, the
execution-time hold, observations, robot files and the CLI."""

import math

import numpy as np
import pytest
import yaml

from makermods_llm_harness import Harness
from makermods_llm_harness import robot as robots
from makermods_llm_harness.arm_model import ArmModel
from makermods_llm_harness.backends.base import ArmInfo
from makermods_llm_harness.cli import main
from makermods_llm_harness.robot import RobotError
from makermods_llm_harness.safety import Envelope, KeepOut, MoveRejected, SafetyConfig, keep_out_zones

from conftest import sim_backend

MODEL = ArmModel(ArmInfo("arm", "metal"))
RATE = 25.0
SPEED = math.radians(20)
# Pointing straight down at [0.28, 0, 0.15]: the jaws open along world y.
DOWN = np.array(MODEL.chain.ik(MODEL.pose_from_angles([0.28, 0, 0.15], -math.pi / 2, 0, 0), MODEL.home_q,
                               limits=(MODEL.ik.lo, MODEL.ik.hi)).q)


def box(name, centre, half):
    centre, half = np.asarray(centre, float), np.broadcast_to(np.asarray(half, float), (3,))
    return KeepOut(name, tuple(centre - half), tuple(centre + half))


def envelope(*zones, model=MODEL, margin=0.03):
    return Envelope(model, SafetyConfig(keep_out_margin_m=margin), -0.5, zones)


def swung_in(joint="shoulder_pan", deg=10.0):
    """(start, path): ``joint`` turned ``deg`` away, then back to DOWN."""
    start = DOWN.copy()
    start[MODEL.joint_names.index(joint)] += math.radians(deg)
    return start, MODEL.plan_joint_path(start, DOWN, SPEED, RATE)


def nudged(q0, dxyz, model=MODEL):
    goal = model.tool_pose(q0)
    goal[:3, 3] += dxyz
    return model.plan_tool_path(q0, goal, SPEED, RATE)


def assert_done(obs):
    assert obs.result.status == "done", obs.result.message


def test_zones_are_validated():
    zones = keep_out_zones([{"name": "mount", "min": [-0.1, -0.42, 0], "max": [0.2, -0.21, 1.2]}])
    assert zones == [KeepOut("mount", (-0.1, -0.42, 0.0), (0.2, -0.21, 1.2))]
    for bad, words in (([{"name": "a", "min": [0, 0, 0], "max": [1, 1, float("nan")]}], "finite"),
                       ([{"name": "a", "min": [0, 0, 0], "max": [1, 0, 1]}], "below max"),
                       ([{"name": "a", "min": [0, 0], "max": [1, 1, 1]}], "three finite"),
                       ([{"name": "a", "min": [0, 0, 0]}], "each zone is"),
                       ([{"name": "", "min": [0, 0, 0], "max": [1, 1, 1]}], "needs a name"),
                       ([{"name": "a", "min": [0, 0, 0], "max": [1, 1, 1]}] * 2, "two zones")):
        with pytest.raises(ValueError, match=words):
            keep_out_zones(bad)


def test_a_point_entering_a_zone_is_rejected_and_one_passing_by_is_not():
    tip = MODEL.fingertip(DOWN)
    env = envelope(box("mount", tip + [0.07, 0, 0], 0.01))
    env.check(DOWN, nudged(DOWN, [-0.03, 0, 0]))
    env.check(DOWN, nudged(DOWN, [0.0, 0, 0.02]))
    with pytest.raises(MoveRejected, match=r"into keep-out zone mount \(its box grown by the 30 mm margin\)\. Go "
                                           "around it"):
        env.check(DOWN, nudged(DOWN, [0.05, 0, 0]))
    name, distance = env.nearest_keep_out(DOWN)
    assert name == "mount" and distance == pytest.approx(0.06, abs=0.005)


def test_a_thin_box_across_a_link_is_caught_between_the_joints():
    nodes = MODEL._body_nodes(DOWN[None])[0]
    a, b = nodes[2], nodes[3]                       # elbow_flex joint to wrist_flex joint
    axis = int(np.argmax(np.abs(b - a)))
    half = np.full(3, 0.01)
    half[axis] = 0.025                               # 5 cm thick along the link
    zone = box("bar", (a + b) / 2, half)
    lo, hi = np.array(zone.min), np.array(zone.max)
    assert not np.any(np.all((nodes >= lo) & (nodes <= hi), axis=1))      # no joint or gripper point inside
    start, path = swung_in()
    envelope(zone, margin=0.0).check(start, path[:1])
    with pytest.raises(MoveRejected, match="link from the elbow_flex joint to the wrist_flex joint"):
        envelope(zone, margin=0.0).check(start, path)


def test_the_open_jaws_are_checked():
    nodes = MODEL._body_nodes(DOWN[None])[0]
    zone = box("cup", nodes[7] + 0.6 * (nodes[8] - nodes[7]), 0.005)    # between the fingertips and a jaw tip
    lo, hi = np.array(zone.min), np.array(zone.max)
    assert not np.any(np.all((MODEL.points(DOWN) >= lo) & (MODEL.points(DOWN) <= hi), axis=1))
    start, path = swung_in("shoulder_lift", 8.0)
    envelope(zone, margin=0.0).check(start, path[:1])
    with pytest.raises(MoveRejected, match="the open jaw"):
        envelope(zone, margin=0.0).check(start, path)


def test_an_arm_inside_a_zone_may_only_move_out_and_rest_may_end_in_one():
    tip = MODEL.fingertip(DOWN)
    env = envelope(box("tray", tip - [0, 0, 0.03], [0.15, 0.15, 0.04]))
    assert env.keep_out_inside(DOWN) and "keep-out zone tray" in env.keep_out_inside(DOWN)[0]
    env.check(DOWN, nudged(DOWN, [0, 0, 0.04]))                            # straight up and out
    with pytest.raises(MoveRejected, match="only motions that keep moving out of the zone"):
        env.check(DOWN, nudged(DOWN, [0, 0, -0.01]))
    above = nudged(DOWN, [0, 0, 0.10])[-1]
    down = MODEL.plan_joint_path(above, DOWN, SPEED, RATE)
    with pytest.raises(MoveRejected, match="into keep-out zone tray"):
        env.check(above, down, capped=False)
    env.check(above, down, capped=False, rest_q=DOWN)                     # the way down to a rest pose inside


def test_two_arms_with_different_bases_see_the_same_zone_in_the_world_frame():
    left = ArmModel(ArmInfo("left", "metal", base_xyz=(0.0, 0.25, 0.0)))
    right = ArmModel(ArmInfo("right", "metal", base_xyz=(0.0, -0.25, 0.0)))
    zone = box("mount", right.fingertip(DOWN) + [0.07, 0, 0], 0.01)
    env_left, env_right = envelope(zone, model=left), envelope(zone, model=right)
    assert env_right.nearest_keep_out(DOWN)[1] < 0.07 < env_left.nearest_keep_out(DOWN)[1]
    env_left.check(DOWN, nudged(DOWN, [0.05, 0, 0], left))
    with pytest.raises(MoveRejected, match="right: that motion would bring"):
        env_right.check(DOWN, nudged(DOWN, [0.05, 0, 0], right))


def test_the_harness_rejects_lists_and_observes_zones(sim_harness):
    h = sim_harness(keep_out=[KeepOut("mount", (0.25, -0.30, 0.0), (0.40, -0.15, 1.2))])
    assert "mount from [0.25, -0.30, 0.00] to [0.40, -0.15, 1.20] m" in h.describe()
    obs = h.step({"action": "home"})
    assert_done(obs)
    assert obs.arms["arm"].nearest_keep_out["name"] == "mount"
    assert "keep-out zone mount" in obs.text() and obs.to_dict()["arms"]["arm"]["nearest_keep_out"]["distance_m"] > 0
    before = obs.arms["arm"].joints_deg
    obs = h.step({"action": "move_joints", "joints_deg": {"shoulder_pan": -30}, "relative": True})
    assert obs.result.status == "rejected" and "keep-out zone mount" in obs.result.message
    assert obs.arms["arm"].joints_deg == before


def test_entering_a_zone_while_moving_holds_the_arm(sim_harness, monkeypatch):
    h = sim_harness(keep_out=[KeepOut("mount", (0.25, -0.40, 0.0), (0.40, -0.25, 1.2))])
    assert_done(h.step({"action": "home"}))
    env = h.envelopes["arm"]
    # Held as soon as the arm comes 1 mm closer than it started (the plan itself stays clear).
    monkeypatch.setattr(env, "keep_out_hold", lambda q0, end=None: env.zone_distances(q0)[0] - 0.001)
    obs = h.step({"action": "nudge", "dxyz": [0, -0.03, 0]})
    assert obs.result.status == "stopped" and "entered keep-out zone mount" in obs.result.message
    state = h.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)


def test_robot_file_round_trip_and_validation():
    robot = robots.simulated(name="zoned")
    robot.keep_out = [{"name": "camera_mount", "min": [-0.10, -0.42, 0.0], "max": [0.20, -0.21, 1.20]}]
    path = robots.save(robot)
    assert yaml.safe_load(path.read_text())["keep_out"] == robot.keep_out
    loaded = robots.load("zoned")
    assert loaded.to_dict() == robot.to_dict() and loaded.zones()[0].name == "camera_mount"
    with Harness.open(loaded) as h:
        assert [z.name for z in h.envelopes["arm"].zones] == ["camera_mount"]
    for bad in ({"name": "a", "min": [0, 0, 0], "max": [0, 1, 1]}, "box"):
        with pytest.raises(RobotError, match="keep_out"):
            robots.Robot.from_dict("x", {**robot.to_dict(), "keep_out": [bad] if isinstance(bad, dict) else bad})


def test_keep_out_cli_and_robot_check(tmp_path, capsys):
    assert main(["robot", "init", "zoned", "--sim"]) == 0
    capsys.readouterr()
    assert main(["robot", "keep-out", "add", "mount", "--min", "0.25", "-0.30", "0", "--max", "0.40", "-0.15",
                 "1.2"]) == 0
    assert "added keep-out zone mount from [0.25, -0.30, 0.00] to [0.40, -0.15, 1.20] m" in capsys.readouterr().out
    assert main(["robot", "keep-out", "add", "mount", "--robot", "zoned", "--min", "0", "0", "0", "--max", "1", "1",
                 "1"]) == 2
    assert main(["robot", "keep-out", "add", "flat", "--min", "0", "0", "0", "--max", "1", "1", "0"]) == 2
    assert "below max" in capsys.readouterr().err
    rest = sim_backend().read()["arm"].rest_q
    tip = MODEL.fingertip(rest)
    assert main(["robot", "keep-out", "add", "stand", "--min", *(f"{v:.4f}" for v in tip - 0.02), "--max",
                 *(f"{v:.4f}" for v in tip + 0.02)]) == 0
    assert main(["robot", "keep-out", "list"]) == 0
    out = capsys.readouterr().out
    assert "mount from" in out and "stand from" in out
    assert main(["robot", "check", "--out", str(tmp_path / "check")]) == 0
    out = capsys.readouterr().out
    assert "arm arm: the " in out and "inside keep-out zone stand" in out and "it may only move out of it" in out
    assert "arm arm: at its recorded rest pose the " in out
    assert main(["robot", "keep-out", "remove", "stand"]) == 0
    assert main(["robot", "keep-out", "remove", "stand"]) == 2
    assert [z["name"] for z in robots.load("zoned").keep_out] == ["mount"]
