"""One arm's geometry: its SDK profile and kinematic chain placed in the world frame.

Tool angles: the pointing direction (tool z) is ``(cos p cos y, cos p sin y, sin p)`` for pitch p and yaw y.
With roll 0 the tool x axis is horizontal, ``(-sin y, cos y, 0)``, and tool y = z cross x; roll turns the tool
about its z axis. Within 0.5 deg of straight up or down, the angles report roll 0 and put the whole turn in yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .backends.base import ArmInfo
from .robot import sdk

MIN_SECONDS = 0.25          # shortest planned move
MAX_SECONDS = 120.0         # longest planned move: no plan holds more waypoints than this at its rate
PLAN_STEP = math.radians(2.0)   # largest joint step per IK planning tick
POSE_TOL_M = 0.005          # a planned tool path must end this close to the requested pose ...
POSE_TOL_DEG = 2.0          # ... and within this angle of it
SAMPLE_M = 0.01             # spacing of the points sampled along the links for the keep-out zones
LINKS_AFTER = (2, 3, 4, 5, 6)    # joints whose child-link origins are tracked (drawing: all; envelope: from 3)
MONITORED_AFTER = (3, 4, 5, 6)   # joints whose child-link origins the envelope watches, besides the tool point
VERTICAL = math.sin(math.radians(0.5))   # |cos pitch| below this counts as pointing straight up or down


@dataclass(frozen=True)
class Gripper:
    """Fingertip geometry in the tool frame. ``finger_length_m``: how far the fingertips reach past the tool point
    along the pointing direction; ``jaw_half_width_m``: how far the open jaws reach from the tool axis along tool x
    (the closing direction); ``margin_m`` is added to both for the envelope's monitored points."""
    finger_length_m: float
    jaw_half_width_m: float
    margin_m: float


GRIPPERS = {
    # The SDK URDF's gripper_left/right collision meshes in the tool frame: closed or open, the jaws reach 34.2 mm
    # past the tool point; open (prismatic limit 52.4 mm per jaw), they reach 72.5 mm from the tool axis ahead of
    # the tool point.
    "maker": Gripper(0.0342, 0.0725, 0.003),
    # The SDK URDF lumps the gripper into its last link without a mesh, so these are the MuJoCo model's fingers
    # (tips 15 mm past the tool point, open outer faces 50 mm from the axis) with a 10 mm margin.
    "metal": Gripper(0.015, 0.050, 0.010),
}


class Unreachable(Exception):
    """No joint path reaches the requested tool pose. The message says how far off it got."""


def rotation_from_angles(pitch: float, yaw: float, roll: float) -> np.ndarray:
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    z = np.array([cp * cy, cp * sy, sp])
    x = np.array([-sy, cy, 0.0])
    rot = np.column_stack([x, np.cross(z, x), z])
    cr, sr = math.cos(roll), math.sin(roll)
    return rot @ np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])


def angles_from_rotation(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in rad of a rotation, inverse of ``rotation_from_angles``."""
    z = rot[:, 2]
    pitch = math.asin(max(-1.0, min(1.0, z[2])))
    if math.hypot(z[0], z[1]) < VERTICAL:
        x = rot[:, 0]
        return pitch, math.atan2(-x[0], x[1]), 0.0
    yaw = math.atan2(z[1], z[0])
    turn = rotation_from_angles(pitch, yaw, 0.0).T @ rot
    return pitch, yaw, math.atan2(turn[1, 0], turn[0, 0])


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance (m) and rotation angle (rad) between two 4x4 poses."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.acos(max(-1.0, min(1.0, cos)))


def within_pose_tol(position_m: float, angle_rad: float) -> bool:
    return position_m <= POSE_TOL_M and math.degrees(angle_rad) <= POSE_TOL_DEG


def minimum_jerk(ticks: int) -> np.ndarray:
    """(ticks,) progress 0..1 from just after the start to the end, still at both ends."""
    u = np.arange(1, ticks + 1) / ticks
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def resample(q0, rows: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Rows at ``from_hz`` after ``q0``, linearly interpolated to ``to_hz``; the last row is kept. A joint's
    speed between rows is never exceeded."""
    full = np.vstack([q0, rows])
    t_from = np.arange(len(full)) / from_hz
    t_to = np.arange(1, int(math.ceil(len(rows) * to_hz / from_hz - 1e-9)) + 1) / to_hz
    t_to[-1] = t_from[-1]
    return np.column_stack([np.interp(t_to, t_from, full[:, j]) for j in range(full.shape[1])])


@lru_cache(maxsize=None)
def _profile(arm_type: str, profile: Optional[str]):
    """Config, chain, link sub-chains and trajectory IK of a profile, built once per process."""
    arm_sdk = sdk(arm_type)
    config = arm_sdk.rig.ArmSpec(backend="socketcan", profile=profile).config()
    kin = arm_sdk.kinematics
    chain = kin.chain_from_config(config)
    urdf = kin.resolve_urdf(config.urdf)
    joints = config.joints[:chain.n]
    subs = [kin.Chain.from_urdf(urdf, joints=chain.joint_names[:k], sign=[j.urdf_sign for j in joints[:k]],
                                offset=[j.urdf_offset for j in joints[:k]], base_rpy=tuple(config.urdf_base_rpy))
            for k in LINKS_AFTER]
    ik = arm_sdk.trajectory_ik.from_config(config, chain=chain, solver="numpy")
    return config, chain, subs, ik


class ArmModel:
    """Kinematics of one arm in the world frame. Joints ``q`` are the 6 arm joints in rad."""

    def __init__(self, info: ArmInfo, gripper: Optional[Gripper] = None):
        """``gripper``: the fingertip geometry (default: ``GRIPPERS[info.type]``)."""
        self.info = info
        self.gripper = gripper or GRIPPERS[info.type]
        self.name = info.name
        self.config, self.chain, self._subs, self.ik = _profile(info.type, info.profile)
        self._trajectory_ik = sdk(info.type).trajectory_ik
        joints = self.config.joints
        self.joint_names = [j.name for j in joints[:6]]
        self.gripper_name = joints[6].name
        self.point_names = [f"{self.joint_names[k - 1]} joint" for k in MONITORED_AFTER] + ["tool point"] + ["gripper fingertips"] * 3
        self.lo = np.array([j.lo for j in joints[:6]])
        self.hi = np.array([j.hi for j in joints[:6]])
        self.home_q = np.array(self.config.ik["rest_pose"], dtype=float)
        self.raised_q = self.home_q.copy()         # the SDK's raised pose: the IK seed and the way round to rest
        self.rate_hz = float(self.config.control_rate_hz)
        c, s = math.cos(info.base_yaw), math.sin(info.base_yaw)
        self.world_from_base = np.eye(4)
        self.world_from_base[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        self.world_from_base[:3, 3] = info.base_xyz
        self.base_from_world = np.linalg.inv(self.world_from_base)
        # Keep-out samples: points every SAMPLE_M along the links from the base to the tool point, the fingers and
        # the open jaws (node indices into _body_nodes).
        nodes = ["base"] + [f"{self.joint_names[k - 1]} joint" for k in LINKS_AFTER] + ["tool point"]
        segments = [(i, i + 1, f"link from the {nodes[i]} to the {nodes[i + 1]}") for i in range(len(nodes) - 1)]
        segments += [(6, 7, "fingers"), (10, 8, "open jaws"), (11, 9, "open jaws"), (8, 9, "open jaw tips")]
        at_home = self._body_nodes(self.raised_q[None])[0]
        a, b, frac, self.body_names = [], [], [], []
        for i, j, label in segments:
            n = max(1, int(math.ceil(np.linalg.norm(at_home[j] - at_home[i]) / SAMPLE_M)))
            a += [i] * (n + 1)
            b += [j] * (n + 1)
            frac += list(np.linspace(0.0, 1.0, n + 1))
            self.body_names += [label] * (n + 1)
        self._body = (np.array(a), np.array(b), np.array(frac)[:, None])

    def set_home(self, xyz, pitch_deg: float) -> None:
        """Make ``home`` the pose with the tool point at ``xyz`` (the arm's base frame, m) pointing straight ahead
        at ``pitch_deg``; Unreachable when the arm cannot take it."""
        target = np.eye(4)
        target[:3, :3] = rotation_from_angles(math.radians(pitch_deg), 0.0, 0.0)
        target[:3, 3] = np.asarray(xyz, dtype=float)
        self.home_q = np.asarray(self._solve(self.raised_q, target).q, dtype=float)

    # ── poses ──
    def tool_pose(self, q) -> np.ndarray:
        """4x4 tool pose in the world frame."""
        return self.world_from_base @ self.chain.fk(np.asarray(q, dtype=float))

    def tool_angles(self, q) -> tuple[np.ndarray, float, float, float]:
        """Tool point xyz (world) and pitch, yaw, roll (rad)."""
        pose = self.tool_pose(q)
        return (pose[:3, 3].copy(), *angles_from_rotation(pose[:3, :3]))

    @staticmethod
    def pose_from_angles(xyz, pitch: float, yaw: float, roll: float) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = rotation_from_angles(pitch, yaw, roll)
        pose[:3, 3] = xyz
        return pose

    def points(self, q) -> np.ndarray:
        """World positions of the monitored points: (k, 3) for one q (6,), (T, k, 3) for a path (T, 6). They are
        the elbow and wrist link origins, the tool point, the fingertip centre and the two tips of the open jaws,
        the last three with the gripper's margin."""
        links = self._link_points(q, self._subs[len(LINKS_AFTER) - len(MONITORED_AFTER):])
        g = self.gripper
        tips = self._gripper_points(q, g.finger_length_m + g.margin_m, g.jaw_half_width_m + g.margin_m)
        return np.concatenate([links, tips], axis=-2)

    def tool_points(self, q) -> np.ndarray:
        """World xyz of the tool point: (3,) for one q, (T, 3) for a path."""
        pts = self.world_from_base @ self.chain.fk_jacobian_many(np.atleast_2d(np.asarray(q, dtype=float)))[0]
        return pts[0, :3, 3] if np.ndim(q) == 1 else pts[:, :3, 3]

    def fingertip(self, q) -> np.ndarray:
        """World xyz of the point midway between the fingertips (no margin): what touches the table."""
        return self._gripper_points(q, self.gripper.finger_length_m, 0.0)[0]

    def _gripper_points(self, q, length: float, half_width: float) -> np.ndarray:
        """Fingertip centre and the two open-jaw tips: (3, 3) for one q, (T, 3, 3) for a path."""
        poses = self.world_from_base @ self.chain.fk_jacobian_many(np.atleast_2d(np.asarray(q, dtype=float)))[0]
        centre = poses[:, :3, 3] + length * poses[:, :3, 2]
        side = half_width * poses[:, :3, 0]
        pts = np.stack([centre, centre + side, centre - side], axis=1)
        return pts[0] if np.ndim(q) == 1 else pts

    def skeleton(self, q) -> np.ndarray:
        """(7, 3) world points from the base origin through the shoulder, elbow and wrist to the tool point."""
        return np.vstack([self.world_from_base[:3, 3], self._link_points(q, self._subs)])

    def _link_points(self, q, subs) -> np.ndarray:
        qs = np.atleast_2d(np.asarray(q, dtype=float))
        pts = [sub.fk_jacobian_many(qs[:, :sub.n])[0][:, :3, 3] for sub in subs]
        pts.append(self.chain.fk_jacobian_many(qs)[0][:, :3, 3])
        world = np.stack(pts, axis=1) @ self.world_from_base[:3, :3].T + self.world_from_base[:3, 3]
        return world[0] if np.ndim(q) == 1 else world

    def _body_nodes(self, qs: np.ndarray) -> np.ndarray:
        """(T, 12, 3): base, link origins, tool point, fingertip centre, open-jaw tips and the jaw roots beside the
        tool point (the gripper points with the gripper's margin)."""
        g = self.gripper
        base = np.broadcast_to(self.world_from_base[:3, 3], (len(qs), 1, 3))
        skeleton = np.concatenate([base, self._link_points(qs, self._subs)], axis=1)
        tips = self._gripper_points(qs, g.finger_length_m + g.margin_m, g.jaw_half_width_m + g.margin_m)
        side = (tips[:, 1] - tips[:, 0])[:, None]
        tool = skeleton[:, 6:7]
        return np.concatenate([skeleton, tips, tool + side, tool - side], axis=1)

    def body_points(self, q) -> np.ndarray:
        """World points sampled along the whole arm (``body_names`` says which part each is on): (K, 3) for one
        q, (T, K, 3) for a path."""
        a, b, frac = self._body
        nodes = self._body_nodes(np.atleast_2d(np.asarray(q, dtype=float)))
        pts = nodes[:, a] + frac * (nodes[:, b] - nodes[:, a])
        return pts[0] if np.ndim(q) == 1 else pts

    def lowest_z(self, q):
        """Lowest monitored height (world z): a float for one q, (T,) for a path."""
        return self.points(q)[..., 2].min(axis=-1)

    def lowest_z_within(self, q, tolerance: float):
        """Lowest monitored height when every joint may be up to ``tolerance`` rad off ``q``: each point lowered
        by the sum over joints of its drop when that joint alone moves by +-tolerance (a first-order worst case).
        A float for one q, (T,) for a path."""
        qs = np.atleast_2d(np.asarray(q, dtype=float))
        steps = tolerance * np.vstack([np.eye(6), -np.eye(6)])
        z = self.points(np.vstack([qs] + [qs + d for d in steps]))[..., 2].reshape(13, len(qs), -1)
        drop = np.maximum(0.0, z[0] - np.minimum(z[1:7], z[7:])).sum(axis=0)
        lowest = (z[0] - drop).min(axis=-1)
        return float(lowest[0]) if np.ndim(q) == 1 else lowest

    # ── plans ──
    def plan_joint_path(self, q0, q1, speed_rad_s: float, rate_hz: float) -> np.ndarray:
        """(T, 6) minimum-jerk joint interpolation from just after ``q0`` to ``q1``; no tick moves a joint
        more than ``speed_rad_s / rate_hz``."""
        q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        span = float(np.max(np.abs(q1 - q0)))
        # Minimum jerk peaks at 1.875x the mean speed.
        ticks = max(MIN_SECONDS * rate_hz, 1.875 * span * rate_hz / speed_rad_s * 1.01)
        limit_ticks(self.name, ticks, rate_hz)
        ticks = int(math.ceil(ticks))
        path = q0 + minimum_jerk(ticks)[:, None] * (q1 - q0)
        path[-1] = q1
        return path

    def _solve(self, q0, target: np.ndarray):
        """The SDK IK solution for ``target`` (4x4, base frame) seeded from ``q0`` and home; Unreachable when it
        misses by more than POSE_TOL_M or POSE_TOL_DEG."""
        lo, hi = self.ik.bounds(q0)
        end = min((self.chain.ik(target, np.clip(seed, lo, hi), limits=(lo, hi)) for seed in (q0, self.raised_q)),
                  key=lambda r: (not within_pose_tol(r.pos_err, r.rot_err), r.pos_err + 0.1 * r.rot_err))
        if not within_pose_tol(end.pos_err, end.rot_err):
            raise Unreachable(f"{self.name}: that pose is out of reach (the closest reachable pose is "
                              f"{end.pos_err * 1000:.0f} mm and {math.degrees(end.rot_err):.1f} deg off). "
                              "Choose a closer point or another pitch.")
        return end

    def solve_pose(self, q0, goal) -> np.ndarray:
        """Joints that put the tool at ``goal`` (4x4, world), solved from ``q0``; Unreachable when out of reach."""
        q0 = np.asarray(q0, dtype=float)
        return np.asarray(self._solve(q0, self.base_from_world @ np.asarray(goal, dtype=float)).q, dtype=float)

    def plan_tool_path(self, q0, goal, speed_rad_s: float, rate_hz: float) -> np.ndarray:
        """(T, 6) joints moving the tool point in a straight line to ``goal`` (4x4, world), minimum jerk, solved
        tick by tick by the SDK trajectory IK with no joint faster than ``speed_rad_s``, then finished with a short
        joint move onto the exact IK solution. Unreachable when the path cannot end within POSE_TOL_M and
        POSE_TOL_DEG of the goal within the joint limits."""
        q0 = np.asarray(q0, dtype=float)
        target = self.base_from_world @ np.asarray(goal, dtype=float)
        ik = self.ik
        lo, hi = ik.bounds(q0)
        end = self._solve(q0, target)
        start = self.chain.fk(q0)
        speeds = np.minimum(ik.max_velocity, speed_rad_s)
        seconds = max(MIN_SECONDS, 1.25 * float(np.max(1.875 * np.abs(np.asarray(end.q) - q0) / speeds)))
        # The IK is solved at a coarse rate (PLAN_STEP per tick at most) and interpolated to rate_hz.
        plan_hz = min(rate_hz, speed_rad_s / PLAN_STEP)
        limit_ticks(self.name, seconds * rate_hz, rate_hz)
        ticks = int(math.ceil(seconds * plan_hz))
        for _ in range(6):
            # A path the speed cap shortens is retimed slower.
            poses = self._trajectory_ik.straight_line(start, target, ticks)
            plan = ik.solve(np.concatenate([poses, target[None], target[None]]), q0, rate_hz=plan_hz,
                            speed_cap=speed_rad_s, method="tick")
            if not plan.capped.any():
                break
            ticks = int(math.ceil(ticks * 1.5))
            limit_ticks(self.name, ticks * rate_hz / plan_hz, rate_hz)
        path = resample(q0, plan.q, plan_hz, rate_hz)
        # The tick solve leans towards the profile's rest pose and stops a little short of the goal.
        finish = self.chain.ik(target, np.clip(path[-1], lo, hi), limits=(lo, hi))
        if finish.pos_err + 0.1 * finish.rot_err < plan.pos_err[-1] + 0.1 * plan.rot_err[-1]:
            path = np.vstack([path, self.plan_joint_path(path[-1], finish.q, speed_rad_s, rate_hz)])
        position, angle = pose_error(self.chain.fk(path[-1]), target)
        if not within_pose_tol(position, angle):
            raise Unreachable(f"{self.name}: the straight path to that pose ends {position * 1000:.0f} mm and "
                              f"{math.degrees(angle):.1f} deg off it. Try an intermediate point.")
        return path


def limit_ticks(name: str, ticks: float, rate_hz: float) -> None:
    """Unreachable when a plan would need more than MAX_SECONDS of waypoints."""
    if not ticks <= MAX_SECONDS * rate_hz:
        raise Unreachable(f"{name}: that motion would take more than {MAX_SECONDS:g} s (the longest planned move) at "
                          "this speed. Move faster or split it.")
