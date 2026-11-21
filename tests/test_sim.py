import math
import time

import numpy as np
import pytest

from makermods_llm_harness.arm_model import ArmModel
from makermods_llm_harness.backends.base import ArmInfo, ArmTarget
from makermods_llm_harness.backends.sim import GRIPPER_GAP_M, JOINT_SPEED, MOUNT_CLEARANCE_M, SimBackend

DT = 0.04


def backend(arm_type="metal", **kwargs) -> SimBackend:
    kwargs.setdefault("objects", [{"name": "cube", "xyz": [0.28, 0.0, 0.02], "size": 0.04, "color": "red"}])
    return SimBackend([ArmInfo("arm", arm_type)], **kwargs)


def run(sim, q, gripper, ticks=100):
    for _ in range(ticks):
        sim.step({"arm": ArmTarget(np.asarray(q, dtype=float), gripper)}, DT)


@pytest.mark.parametrize("arm_type", ["metal", "maker"])
def test_starts_at_zero_clipped_into_limits_motors_off(arm_type):
    sim = backend(arm_type)
    model = ArmModel(ArmInfo("arm", arm_type))
    state = sim.read()["arm"]
    assert np.array_equal(state.q, np.clip(np.zeros(6), model.lo, model.hi))
    assert not state.enabled and state.gripper == 0.0 and state.fault is None
    assert np.array_equal(state.rest_q, state.q) and np.array_equal(state.q_target, state.q)
    assert sim.table_z == 0.0 and sim.cameras == ["top", "side"]


def test_a_disabled_arm_ignores_targets():
    sim = backend()
    q0 = sim.read()["arm"].q
    run(sim, q0 + 0.2, 1.0, ticks=5)
    assert np.array_equal(sim.read()["arm"].q, q0)


def test_joints_follow_targets_at_a_bounded_speed():
    sim = backend()
    sim.enable("arm")
    q0 = sim.read()["arm"].q
    goal = q0.copy()
    goal[0] += 0.5
    run(sim, goal, 0.0, ticks=1)
    assert math.isclose(sim.read()["arm"].q[0] - q0[0], JOINT_SPEED * DT)
    run(sim, goal, 0.0, ticks=50)
    assert np.allclose(sim.read()["arm"].q, goal)


def test_grasp_carry_and_release():
    sim = backend()
    sim.enable("arm")
    model = ArmModel(ArmInfo("arm", "metal"))
    q = model.chain.ik(model.pose_from_angles([0.28, 0, 0.028], -math.pi / 2, 0, 0), model.home_q).q
    run(sim, sim.read()["arm"].q, 1.0)
    run(sim, q, 1.0)
    run(sim, q, 0.0)
    state = sim.read()["arm"]
    assert math.isclose(state.gripper, 0.04 / GRIPPER_GAP_M) and state.gripper_target == 0.0
    assert sim.objects[0]["held_by"] == "arm"
    lifted = model.chain.ik(model.pose_from_angles([0.25, 0.05, 0.15], -math.pi / 2, 0, 0), q).q
    run(sim, lifted, 0.0)
    assert np.allclose(sim.objects[0]["xyz"], [0.25, 0.05, 0.142], atol=2e-3)
    run(sim, lifted, 1.0)
    obj = sim.objects[0]
    assert obj["held_by"] is None and np.allclose(obj["xyz"], [0.25, 0.05, 0.02], atol=2e-3)


def test_closing_away_from_a_box_closes_fully():
    sim = backend()
    sim.enable("arm")
    run(sim, sim.read()["arm"].q, 1.0)
    run(sim, sim.read()["arm"].q, 0.0)
    assert sim.read()["arm"].gripper == 0.0 and sim.objects[0]["held_by"] is None


def test_block_fault_and_temperature_hooks():
    sim = backend()
    sim.enable("arm")
    q0 = sim.read()["arm"].q
    sim.block("arm")
    run(sim, q0 + 0.1, 0.0, ticks=5)
    assert np.array_equal(sim.read()["arm"].q, q0)
    assert sim.rest("arm") == "holding"
    sim.block("arm", False)
    sim.set_fault("arm", "motor 3 overheated")
    run(sim, q0 + 0.1, 0.0, ticks=5)
    state = sim.read()["arm"]
    assert state.fault == "motor 3 overheated" and np.array_equal(state.q, q0)
    sim.set_temperature("arm", 80)
    assert sim.read()["arm"].temperature == 80
    with pytest.raises(RuntimeError, match="overheated"):
        sim.enable("arm")


def test_rest_releases_without_moving_and_close_moves_nothing():
    sim = backend()
    sim.enable("arm")
    rest = sim.read()["arm"].rest_q
    run(sim, rest + 0.1, 0.0, ticks=10)
    there = sim.read()["arm"].q
    assert sim.rest("arm") == "released"
    state = sim.read()["arm"]
    assert not state.enabled and np.array_equal(state.q, there)
    assert sim.rest("arm") == "released"
    sim.enable("arm")
    run(sim, rest + 0.1, 0.0, ticks=10)
    held = sim.read()["arm"].q
    sim.close()
    state = sim.read()["arm"]
    assert state.enabled and np.array_equal(state.q, held)


def test_a_high_table_raises_the_mount():
    sim = backend(table_z=0.3, objects=())
    info = sim.arms[0]
    model = ArmModel(info)
    assert info.base_xyz[2] > 0.0
    assert math.isclose(model.lowest_z(sim.read()["arm"].q), 0.3 + MOUNT_CLEARANCE_M)
    assert backend(objects=()).arms[0].base_xyz == (0.0, 0.0, 0.0)


def test_images_for_every_camera_are_cheap():
    sim = SimBackend([ArmInfo("left", base_xyz=(0, 0.25, 0)), ArmInfo("right", "maker", base_xyz=(0, -0.25, 0))],
                     cameras=("top", "side", "left_wrist", "front"),
                     objects=[{"name": "cube", "xyz": [0.28, 0.25, 0.02], "size": 0.04, "color": "blue"}])
    start = time.perf_counter()
    frames = sim.images()
    assert time.perf_counter() - start < 0.5
    assert list(frames) == ["top", "side", "left_wrist", "front"]
    for frame in frames.values():
        assert frame.image.shape == (240, 320, 3) and frame.image.dtype == np.uint8 and frame.stamp > 0
    assert np.array_equal(frames["top"].image, frames["front"].image)
    blue = (frames["top"].image == (40, 90, 220)).all(axis=2)
    assert blue.sum() > 50


def test_the_top_view_shows_the_gripper_opening():
    sim = backend(cameras=("top",), objects=())
    sim.enable("arm")
    closed = sim.images()["top"].image
    run(sim, sim.read()["arm"].q, 1.0)
    assert not np.array_equal(closed, sim.images()["top"].image)
