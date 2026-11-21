import math

import numpy as np
import pytest

from makermods_llm_harness.arm_model import (GRIPPERS, POSE_TOL_DEG, POSE_TOL_M, ArmModel, Gripper, Unreachable,
                                             angles_from_rotation, pose_error, resample, rotation_from_angles)
from makermods_llm_harness.backends.base import ArmInfo

RATE = 25.0
SPEED = math.radians(20)
TYPES = ("metal", "maker")


def model(arm_type="metal", xyz=(0.0, 0.0, 0.0), yaw=0.0) -> ArmModel:
    return ArmModel(ArmInfo("arm", arm_type, base_xyz=xyz, base_yaw=yaw))


def random_q(m: ArmModel, rng) -> np.ndarray:
    return m.lo + (m.hi - m.lo) * rng.uniform(0.1, 0.9, 6)


def test_angle_convention_axes():
    rot = rotation_from_angles(0.0, 0.0, 0.0)
    assert np.allclose(rot[:, 2], [1, 0, 0]) and np.allclose(rot[:, 0], [0, 1, 0]) and np.allclose(rot[:, 1], [0, 0, 1])
    rot = rotation_from_angles(math.radians(30), math.radians(90), 0.0)
    assert np.allclose(rot[:, 2], [0, math.cos(math.radians(30)), 0.5])
    assert np.allclose(rot[:, 0], [-1, 0, 0])
    rolled = rotation_from_angles(0.0, 0.0, math.radians(90))
    assert np.allclose(rolled[:, 2], [1, 0, 0]) and np.allclose(rolled[:, 0], [0, 0, 1])


def test_angles_round_trip_exactly():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p, y, r = rng.uniform(-1.5, 1.5), rng.uniform(-3.1, 3.1), rng.uniform(-3.1, 3.1)
        assert np.allclose(angles_from_rotation(rotation_from_angles(p, y, r)), (p, y, r), atol=1e-9)


@pytest.mark.parametrize("pitch", [-90.0, 90.0])
def test_vertical_reports_roll_zero_and_keeps_the_pose(pitch):
    rot = rotation_from_angles(math.radians(pitch), math.radians(30), math.radians(20))
    p, y, r = angles_from_rotation(rot)
    assert math.isclose(math.degrees(p), pitch) and r == 0.0
    assert np.allclose(rotation_from_angles(p, y, r), rot, atol=1e-12)
    assert math.isclose(abs(math.degrees(y)), 10.0 if pitch < 0 else 50.0, abs_tol=1e-9)


@pytest.mark.parametrize("arm_type", TYPES)
def test_model_angles_round_trip_through_poses(arm_type):
    m, rng = model(arm_type, (0.1, -0.2, 0.05), math.radians(40)), np.random.default_rng(2)
    for _ in range(30):
        q = random_q(m, rng)
        xyz, p, y, r = m.tool_angles(q)
        assert np.allclose(m.pose_from_angles(xyz, p, y, r), m.tool_pose(q), atol=1e-9)


@pytest.mark.parametrize("arm_type, home", [("metal", (0.313, 0.0, 0.581)), ("maker", (0.316, 0.0, 0.118))])
def test_profile_facts(arm_type, home):
    m = model(arm_type)
    assert m.joint_names == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
    assert m.gripper_name == "gripper"
    assert np.allclose(m.tool_pose(m.home_q)[:3, 3], home, atol=1e-3)


def test_metal_zero_pose_points_forward():
    xyz, pitch, yaw, _ = model().tool_angles(np.zeros(6))
    assert np.allclose(xyz, [0.174, 0, 0.189], atol=1e-3)
    assert abs(math.degrees(pitch)) < 1.5 and abs(math.degrees(yaw)) < 0.5


def test_world_pose_uses_the_base():
    q = model().home_q
    base, moved = model(), model("metal", (0.2, 0.3, 0.1), math.radians(90))
    local = base.tool_pose(q)[:3, 3]
    assert np.allclose(moved.tool_pose(q)[:3, 3], [0.2 - local[1], 0.3 + local[0], 0.1 + local[2]])
    assert np.allclose(moved.points(q)[4], moved.tool_pose(q)[:3, 3])
    assert np.allclose(moved.skeleton(q)[0], [0.2, 0.3, 0.1])


@pytest.mark.parametrize("arm_type", TYPES)
def test_points_and_lowest_z_for_one_q_and_a_path(arm_type):
    m = model(arm_type)
    q = m.home_q
    pts = m.points(q)
    assert pts.shape == (8, 3) and len(m.point_names) == 8
    assert m.point_names[4] == "tool point" and m.point_names[5:] == ["gripper fingertips"] * 3
    path = np.vstack([q, np.clip(np.zeros(6), m.lo, m.hi)])
    assert m.points(path).shape == (2, 8, 3)
    assert np.allclose(m.points(path)[1], m.points(path[1]))
    assert np.allclose(m.lowest_z(path), [m.lowest_z(path[0]), m.lowest_z(path[1])])
    assert math.isclose(m.lowest_z(q), pts[:, 2].min())


@pytest.mark.parametrize("arm_type", TYPES)
def test_fingertip_and_open_jaw_points(arm_type):
    m = ArmModel(ArmInfo("arm", arm_type, base_xyz=(0.1, 0.2, 0.0), base_yaw=0.5), Gripper(0.02, 0.03, 0.005))
    q = m.home_q
    pose = m.tool_pose(q)
    centre = pose[:3, 3] + 0.025 * pose[:3, 2]
    pts = m.points(q)
    assert np.allclose(m.fingertip(q), pose[:3, 3] + 0.02 * pose[:3, 2])     # what touches the table: no margin
    assert np.allclose(pts[5], centre)
    assert np.allclose(pts[6], centre + 0.035 * pose[:3, 0]) and np.allclose(pts[7], centre - 0.035 * pose[:3, 0])
    bare = ArmModel(ArmInfo("arm", arm_type), Gripper(0.0, 0.0, 0.0))
    assert np.allclose(bare.points(q)[5:], bare.tool_pose(q)[:3, 3])


def test_gripper_geometry_per_arm_type():
    assert ArmModel(ArmInfo("arm", "maker")).gripper == GRIPPERS["maker"]
    assert GRIPPERS["maker"].finger_length_m >= 0.034 and GRIPPERS["maker"].jaw_half_width_m >= 0.0725
    assert ArmModel(ArmInfo("arm", "metal")).gripper == GRIPPERS["metal"]


@pytest.mark.parametrize("arm_type", TYPES)
def test_lowest_z_within_bounds_every_pose_within_the_tolerance(arm_type):
    m = model(arm_type)
    q = m.home_q
    tol = math.radians(2.0)
    guarded = m.lowest_z_within(q, tol)
    assert guarded < m.lowest_z(q)
    corners = np.array(np.meshgrid(*[[-tol, tol]] * 6)).reshape(6, -1).T
    assert m.lowest_z(q + corners).min() >= guarded - 1e-3
    assert m.lowest_z_within(q, 0.0) == pytest.approx(m.lowest_z(q))


@pytest.mark.parametrize("pitch", [-89.6, 89.7])
def test_near_vertical_reports_roll_zero(pitch):
    rot = rotation_from_angles(math.radians(pitch), math.radians(30), math.radians(20))
    p, y, r = angles_from_rotation(rot)
    assert r == 0.0 and math.isclose(math.degrees(p), pitch, abs_tol=1e-9)
    assert np.allclose(rotation_from_angles(p, y, r), rot, atol=0.01)


def test_angles_round_trip_outside_half_a_degree_of_vertical():
    for pitch in (-88.5, -89.4, 87.0):
        rot = rotation_from_angles(math.radians(pitch), 0.0, math.pi)
        assert np.allclose(rotation_from_angles(*angles_from_rotation(rot)), rot, atol=1e-9)


def test_joint_path_ends_on_goal_within_speed():
    m = model()
    q0, q1 = np.zeros(6), m.home_q
    path = m.plan_joint_path(q0, q1, SPEED, RATE)
    assert np.array_equal(path[-1], q1)
    assert np.max(np.abs(np.diff(np.vstack([q0, path]), axis=0))) <= SPEED / RATE
    assert len(m.plan_joint_path(q0, q0, SPEED, RATE)) >= 1


def test_resample_keeps_the_last_row_and_the_speed():
    rows = np.cumsum(np.full((7, 6), 0.1), axis=0)
    out = resample(np.zeros(6), rows, 10.0, 25.0)
    assert np.array_equal(out[-1], rows[-1])
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(6), out]), axis=0))) <= 0.1 * 10 / 25 + 1e-12


@pytest.mark.parametrize("arm_type", TYPES)
@pytest.mark.parametrize("offset", [(0.0, 0.0, -0.05), (0.0, 0.04, 0.0)])
def test_tool_path_is_straight_and_ends_on_the_goal(arm_type, offset):
    m = model(arm_type, (0.3, -0.1, 0.02), math.radians(-30))
    q0 = m.home_q
    start = m.tool_pose(q0)
    goal = start.copy()
    goal[:3, 3] += offset
    path = m.plan_tool_path(q0, goal, SPEED, RATE)
    assert np.max(np.abs(np.diff(np.vstack([q0, path]), axis=0))) <= SPEED / RATE + 1e-9
    end = m.tool_pose(path[-1])
    assert np.linalg.norm(end[:3, 3] - goal[:3, 3]) < 1e-3
    assert np.allclose(end[:3, :3], goal[:3, :3], atol=0.01)
    a, b = start[:3, 3], goal[:3, 3]
    u = (b - a) / np.linalg.norm(b - a)
    pts = np.array([m.tool_pose(q)[:3, 3] for q in path]) - a
    assert np.max(np.linalg.norm(pts - np.outer(pts @ u, u), axis=1)) < 2e-3


@pytest.mark.parametrize("arm_type", TYPES)
def test_world_frame_ik_round_trip_with_yawed_offset_base(arm_type):
    m = model(arm_type, (-0.1, 0.4, 0.03), math.radians(120))
    xyz, p, y, r = m.tool_angles(m.home_q)
    goal = m.pose_from_angles(xyz + m.world_from_base[:3, :3] @ [0.0, 0.03, -0.03], p, y, r)
    path = m.plan_tool_path(m.home_q, goal, SPEED, RATE)
    assert np.allclose(m.tool_angles(path[-1])[0], goal[:3, 3], atol=1e-3)


def pitched_start(m: ArmModel, pitch_deg: float = -69.0) -> np.ndarray:
    """The real Metal arm's joints (shoulder_pan 0.4, shoulder_lift -63.5, elbow_flex 68.9 deg) with wrist_flex
    turned until the tool pitches ``pitch_deg``."""
    q = np.radians([0.4, -63.5, 68.9, -75.0, 0.0, 0.0])
    for _ in range(3):
        q[3] += math.radians(pitch_deg) - m.tool_angles(q)[1]
    return q


@pytest.mark.parametrize("xyz, pitch", [((0.30, 0, 0.30), -88), ((0.30, 0, 0.29), -85)])
def test_poses_a_millimetre_from_an_exact_solution_are_planned(xyz, pitch):
    """Reported on the real arm as out of reach (wrist_flex at its limit, 1 mm off) and as a straight path ending
    1 mm off (the tick solve's pull towards the profile's rest pose)."""
    m = model()
    q0 = pitched_start(m)
    assert math.degrees(m.tool_angles(q0)[1]) == pytest.approx(-69.0, abs=0.01)
    goal = m.pose_from_angles(xyz, math.radians(pitch), 0.0, 0.0)
    path = m.plan_tool_path(q0, goal, SPEED, RATE)
    position, angle = pose_error(m.tool_pose(path[-1]), goal)
    assert position <= POSE_TOL_M and math.degrees(angle) <= POSE_TOL_DEG
    assert np.max(np.abs(np.diff(np.vstack([q0, path]), axis=0))) <= SPEED / RATE + 1e-9


def test_unreachable_says_how_far_off():
    m = model()
    with pytest.raises(Unreachable, match="out of reach .* mm"):
        m.plan_tool_path(m.home_q, m.pose_from_angles([1.5, 0, 0.2], 0, 0, 0), SPEED, RATE)
