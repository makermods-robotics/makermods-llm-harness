"""The Harness on MuJoCo physics: reaching, grasping, lifting, images, two arms and the table floor."""

import cv2
import numpy as np
import pytest

from makermods_llm_harness import Harness, robot
from makermods_llm_harness.backends import mujoco as _  # noqa: F401  sets MUJOCO_GL before mujoco is first imported

pytest.importorskip("mujoco")


@pytest.fixture
def mujoco_harness():
    opened = []

    def make(arm="metal", arms=("arm",)) -> Harness:
        h = Harness.open(robot.simulated("mujoco", arm, arms))
        opened.append(h)
        return h

    yield make
    for h in opened:
        h.close()


def done(h, action):
    obs = h.step(action)
    assert obs.result.status == "done", (action, obs.result.message)
    return obs


def test_reach_grasp_and_lift(mujoco_harness):
    h = mujoco_harness("maker")
    done(h, {"action": "home"})
    target = (0.30, 0.0, 0.125)
    done(h, {"action": "move_tool", "xyz": list(target), "pitch_deg": -50})
    assert np.linalg.norm(h.backend.tool_pose("arm")[:3, 3] - target) < 0.004
    done(h, {"action": "move_tool", "xyz": [0.28, 0, 0.125], "pitch_deg": -60})
    done(h, {"action": "gripper", "opening": "open"})
    done(h, {"action": "move_tool", "xyz": [0.28, 0, 0.035], "pitch_deg": -60})
    obs = done(h, {"action": "gripper", "opening": "close"})
    assert 0.35 < obs.arms["arm"].gripper < 0.55          # the jaws stop on the 4 cm cube
    done(h, {"action": "wait", "seconds": 1.0})
    obs = done(h, {"action": "nudge", "dxyz": [0, 0, 0.06]})
    assert obs.arms["arm"].gripper > 0.35
    assert h.backend.object_xyz("red_cube")[2] > 0.07


def test_images_are_320x240_jpegs(mujoco_harness):
    h = mujoco_harness()
    obs = h.observe()
    if h.backend.render_error:
        pytest.skip(f"offscreen rendering is unavailable here: {h.backend.render_error}")
    assert set(obs.cameras) == {"top", "arm_wrist"}
    for cam in obs.cameras.values():
        assert (cam.width, cam.height) == (320, 240)
        assert cv2.imdecode(np.frombuffer(cam.jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)


def test_two_arms_step_one_at_a_time(mujoco_harness):
    h = mujoco_harness(arms=("left", "right"))
    start = h.observe().arms
    obs = done(h, {"action": "home", "arm": "left"})
    assert obs.arms["left"].tool.xyz != start["left"].tool.xyz
    assert obs.arms["right"].tool.xyz == pytest.approx(start["right"].tool.xyz, abs=1e-3)
    obs = done(h, {"action": "home", "arm": "right"})
    assert obs.arms["right"].tool.xyz[2] == pytest.approx(obs.arms["left"].tool.xyz[2], abs=2e-3)


def test_a_move_into_the_table_is_rejected(mujoco_harness):
    h = mujoco_harness("maker")
    done(h, {"action": "home"})
    before = h.observe().arms["arm"].joints_deg
    obs = h.step({"action": "move_tool", "xyz": [0.28, 0, 0.005], "pitch_deg": -45})
    assert obs.result.status == "rejected" and "hard floor" in obs.result.message
    assert obs.arms["arm"].joints_deg == pytest.approx(before, abs=0.2)
