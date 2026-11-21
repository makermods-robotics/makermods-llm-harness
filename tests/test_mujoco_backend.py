import importlib
import math

import numpy as np
import pytest

from makermods_llm_harness.backends.base import ArmInfo, ArmTarget
from makermods_llm_harness.backends.mujoco import MujocoBackend

mujoco = pytest.importorskip("mujoco")

PITCH = {"metal": -90.0, "maker": -60.0}   # grasp tilt each arm reaches from the table to 12 cm up
ABOVE_CENTRE = {"metal": 0.005, "maker": 0.02}   # tool point above the cube centre: the longer Maker fingers
                                                  # reach down past it


@pytest.fixture(autouse=True)
def harness_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_HARNESS_HOME", str(tmp_path))


def make(arm_type="metal", **kwargs):
    return MujocoBackend([ArmInfo("arm", arm_type)], **kwargs)


def tool_pose(xyz, pitch_deg):
    p = math.radians(pitch_deg)
    a, x = np.array([math.cos(p), 0.0, math.sin(p)]), np.array([0.0, 1.0, 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(a, x), a])
    pose[:3, 3] = xyz
    return pose


def drive(backend, q0, q1, opening, speed_deg_s=20.0, dt=0.04, settle=10):
    ticks = max(1, math.ceil(np.max(np.abs(q1 - q0)) / math.radians(speed_deg_s) / dt))
    for k in range(1, ticks + settle + 1):
        backend.step({"arm": ArmTarget(q0 + (q1 - q0) * min(1.0, k / ticks), opening)}, dt)
    return q1


@pytest.mark.parametrize("arm_type", ["metal", "maker"])
def test_tool_site_matches_sdk_chain(arm_type):
    yaw, xyz = 0.4, (0.1, -0.2, 0.05)
    b = MujocoBackend([ArmInfo("a", arm_type, base_xyz=xyz, base_yaw=yaw)], table_z=0.05)
    sdk = importlib.import_module(f"{arm_type}_arm.rig").ArmSpec(backend="socketcan").config()
    chain = importlib.import_module(f"{arm_type}_arm.kinematics").chain_from_config(sdk)
    lo, hi = np.array([j.lo for j in sdk.joints[:6]]), np.array([j.hi for j in sdk.joints[:6]])
    world = np.eye(4)
    world[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    world[:3, 3] = xyz
    names = [f"a/{n}" for n in chain.joint_names]
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        for name, value, j in zip(names, q, sdk.joints):
            b.data.qpos[b.model.joint(name).qposadr[0]] = value * j.urdf_sign + j.urdf_offset
        mujoco.mj_kinematics(b.model, b.data)
        expected = world @ chain.fk(q)
        assert np.linalg.norm(b.tool_pose("a")[:3, 3] - expected[:3, 3]) < 1e-3
        assert np.allclose(b.tool_pose("a")[:3, :3], expected[:3, :3], atol=1e-3)
        mujoco.mj_forward(b.model, b.data)
        assert np.allclose(b.read()["a"].q, q, atol=1e-9)


@pytest.mark.parametrize("arm_type", ["metal", "maker"])
def test_starts_at_zero_clipped_without_arm_contacts(arm_type):
    b = make(arm_type)
    state = b.read()["arm"]
    assert np.allclose(state.q, state.rest_q, atol=1e-9)
    assert state.gripper == pytest.approx(0.0, abs=1e-9) and state.enabled and state.fault is None
    b.step({}, 0.2)
    arm_geoms = {g for g in range(b.model.ngeom) if b.model.body(b.model.geom_bodyid[g]).name.startswith("arm/")}
    touching = {c.geom1 for c in b.data.contact[:b.data.ncon]} | {c.geom2 for c in b.data.contact[:b.data.ncon]}
    assert not touching & arm_geoms
    assert b.table_z == 0.0


def test_joints_reach_targets():
    b = make()
    q0 = b.read()["arm"].q
    goal = q0 + np.radians([20, -25, 30, -15, 20, 30])
    drive(b, q0, goal, 0.0, speed_deg_s=40)
    state = b.read()["arm"]
    assert np.allclose(state.q_target, goal)
    assert np.degrees(np.abs(state.q - goal)).max() < 1.0


def test_gripper_opening_tracks_command():
    b = make()
    q = b.read()["arm"].q
    for opening in (1.0, 0.5, 0.0):
        for _ in range(25):
            b.step({"arm": ArmTarget(q, opening)}, 0.04)
        state = b.read()["arm"]
        assert state.gripper_target == opening
        assert state.gripper == pytest.approx(opening, abs=0.02)


def test_cube_rests_on_table():
    b = make(table_z=0.1, objects=[{"name": "box", "xyz": (0.3, 0.05, 0.12), "size": 0.04, "color": "green"}])
    for _ in range(50):
        b.step({}, 0.02)
    assert np.allclose(b.object_xyz("box"), (0.3, 0.05, 0.12), atol=0.002)


@pytest.mark.parametrize("arm_type", ["metal", "maker"])
def test_grasp_and_lift_cube(arm_type):
    b = make(arm_type)
    arm = b.arms[0]
    sdk = importlib.import_module(f"{arm_type}_arm.rig").ArmSpec(backend="socketcan").config()
    chain = importlib.import_module(f"{arm_type}_arm.kinematics").chain_from_config(sdk)
    limits = (np.array([j.lo for j in sdk.joints[:6]]), np.array([j.hi for j in sdk.joints[:6]]))
    cube = b.object_xyz("cube")
    grasp = cube + [0, 0, ABOVE_CENTRE[arm_type]]
    q = b.read()["arm"].q

    def solve(xyz):
        result = chain.ik(tool_pose(xyz, PITCH[arm_type]), q, limits=limits, max_iters=300)
        assert result.ok, (arm.type, xyz, result.pos_err)
        return np.array(result.q)

    q = drive(b, q, solve(grasp + [0, 0, 0.10]), 1.0)
    q = drive(b, q, solve(grasp), 1.0)
    drive(b, q, q, 0.0, settle=25)
    held = b.object_xyz("cube") - b.tool_pose("arm")[:3, 3]
    assert 0.35 < b.read()["arm"].gripper < 0.55       # the jaws stop on the 4 cm cube
    q = drive(b, q, solve(grasp + [0, 0, 0.10]), 0.0)
    drive(b, q, q, 0.0, settle=25)
    lifted = b.object_xyz("cube")
    assert lifted[2] - cube[2] > 0.09
    assert np.linalg.norm(lifted - b.tool_pose("arm")[:3, 3] - held) < 0.01


def test_rest_reports_always_on_and_moves_nothing():
    b = make()
    q0 = b.read()["arm"].q
    drive(b, q0, q0 + np.radians([15, -20, 20, 0, 0, 10]), 0.0, speed_deg_s=60)
    before = b.read()["arm"].q
    assert b.rest("arm") == "always_on"
    assert np.array_equal(b.read()["arm"].q, before) and b.read()["arm"].enabled


def test_two_arms_build_and_step():
    b = MujocoBackend([ArmInfo("left", "metal", base_xyz=(0, 0.25, 0)),
                       ArmInfo("right", "maker", base_xyz=(0, -0.25, 0))])
    assert [a.name for a in b.arms] == ["left", "right"]
    assert b.cameras == ["top", "left_wrist", "right_wrist"]
    assert [o["name"] for o in b.objects] == ["cube_left", "cube_right"]
    start = b.read()
    goal = start["left"].q + np.radians([10, -10, 10, 0, 0, 0])
    for _ in range(30):
        b.step({"left": ArmTarget(goal, 0.5)}, 0.04)
    state = b.read()
    assert np.degrees(np.abs(state["left"].q - goal)).max() < 1.0
    assert np.allclose(state["right"].q, start["right"].q, atol=1e-3)


def test_unknown_camera_is_refused():
    with pytest.raises(ValueError, match="cameras: top, arm_wrist"):
        make(cameras=["side"])


def test_images_are_rgb_frames():
    b = make()
    frames = b.images()
    if b.render_error:
        pytest.skip(f"offscreen rendering is unavailable here: {b.render_error}")
    try:
        assert set(frames) == {"top", "arm_wrist"}
        for frame in frames.values():
            assert frame.image.shape == (240, 320, 3) and frame.image.dtype == np.uint8
            assert frame.stamp == b.time
        assert frames["top"].image.std() > 5
    finally:
        b.close()
        b.close()


def test_without_a_renderer_images_are_empty_with_a_reason(monkeypatch):
    b = make()

    def no_gl(*args, **kwargs):
        raise RuntimeError("no OpenGL context")

    monkeypatch.setattr(mujoco, "Renderer", no_gl)
    with pytest.warns(RuntimeWarning, match="no OpenGL context"):
        assert b.images() == {}
    assert "MUJOCO_GL" in b.render_error
    assert b.images() == {}
    b.step({}, 0.04)
    assert b.time == pytest.approx(0.04)


def test_keep_out_zones_are_drawn_without_collisions():
    from makermods_llm_harness.safety import KeepOut
    backend = make(keep_out=[KeepOut("mount", (0.25, -0.30, 0.0), (0.40, -0.15, 1.2))], cameras=[])
    m = backend.model
    boxes = [g for g in range(m.ngeom) if m.geom_rgba[g][3] < 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert len(boxes) == 1
    g = boxes[0]
    assert np.allclose(m.geom_pos[g], [0.325, -0.225, 0.6]) and np.allclose(m.geom_size[g], [0.075, 0.075, 0.6])
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    backend.close()
