"""MuJoCo physics: one or two arms on a table with graspable cubes, rendered offscreen.

Each arm is built from the URDF of its SDK profile: the joint frames, axes and inertials of the
chain, capsules between the joint origins as link shapes, and two box fingers on slide joints that
close along the tool frame's x axis around the tool point. Joints are position servos on
gravity-compensated links. ``q`` is in SDK joint coordinates, as a real arm reports it.

Images need offscreen OpenGL. On Linux, importing this module sets ``MUJOCO_GL=egl`` unless it is
set already (it must be set before mujoco is first imported). When no renderer can be created,
``images()`` returns ``{}`` and ``render_error`` says why.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from typing import Mapping, Optional, Sequence

import numpy as np

from ..arm_model import GRIPPERS
from .base import ArmInfo, ArmState, ArmTarget, Backend, Frame

TIMESTEP = 0.002                       # s
MAX_GAP = 0.08                         # m between the fingers at opening 1
FINGER = (0.005, 0.012, 0.0225)        # finger box half sizes along tool x, y, z; the tips reach the arm type's
                                       # finger length (arm_model.GRIPPERS) past the tool point
LINK_RADIUS = 0.022
ARM_KP = (300, 300, 200, 80, 40, 40)   # N*m/rad
ARM_KV = (30, 30, 20, 6, 3, 3)         # N*m*s/rad
FINGER_KP, FINGER_FORCE = 500.0, 10.0   # N/m, N
FINGER_DAMPING = 20.0                  # N*s/m: the fingers close at about 0.5 m/s, as a geared gripper does
# Collision groups (contype, conaffinity): the table touches objects and fingers; links touch objects only.
TABLE, OBJECT, LINK, FINGERS = (1, 10), (2, 15), (4, 2), (8, 3)
COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.2, 0.7, 0.25), "blue": (0.2, 0.35, 0.85),
          "yellow": (0.95, 0.8, 0.15), "orange": (1.0, 0.55, 0.1), "purple": (0.55, 0.3, 0.75),
          "white": (0.95, 0.95, 0.95), "black": (0.1, 0.1, 0.1)}
CUBE_DENSITY = 0.05 / 0.06 ** 3        # kg/m^3: a 6 cm cube would weigh 50 g


if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"     # read by mujoco when it is first imported


def _mujoco():
    try:
        return importlib.import_module("mujoco")
    except ImportError as e:
        raise ImportError("the MuJoCo backend needs MuJoCo: pip install 'makermods-llm-harness[mujoco]'") from e


def _v(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.ravel(values))


def default_objects(arms: Sequence[ArmInfo], table_z: float = 0.0) -> list[dict]:
    """A 4 cm cube 0.28 m in front of each of the first two arms, resting on the table."""
    names = ["cube"] if len(arms) == 1 else [f"cube_{a.name}" for a in arms[:2]]
    return [{"name": name, "color": color, "size": 0.04,
             "xyz": (a.base_xyz[0] + 0.28 * math.cos(a.base_yaw), a.base_xyz[1] + 0.28 * math.sin(a.base_yaw),
                     table_z + 0.02)}
            for name, color, a in zip(names, ("red", "blue"), arms)]


class _Arm:
    """One arm: its SDK profile and chain, its MJCF, and where its joints sit in the model."""

    def __init__(self, info: ArmInfo):
        rig = importlib.import_module(f"{info.type}_arm.rig")
        self.kinematics = importlib.import_module(f"{info.type}_arm.kinematics")
        self.info = info
        self.config = rig.ArmSpec(backend="socketcan", profile=info.profile).config()
        self.chain = self.kinematics.chain_from_config(self.config)
        joints = self.config.joints[:6]
        self.sign = np.array([j.urdf_sign for j in joints], float)
        self.offset = np.array([j.urdf_offset for j in joints], float)
        self.lo, self.hi = np.array([j.lo for j in joints]), np.array([j.hi for j in joints])
        self.rest_q = np.clip(0.0, self.lo, self.hi)
        self.finger_z = GRIPPERS[info.type].finger_length_m - FINGER[2]   # finger centre on tool z
        self.palm_z = self.finger_z - FINGER[2] - 0.003                    # palm plate centre (3 mm half thickness)
        self.q_target, self.gripper_target = self.rest_q.copy(), 0.0

    def theta(self, q) -> np.ndarray:
        """URDF joint angles for SDK joints ``q`` (clipped into the limits)."""
        return np.clip(np.asarray(q, float), self.lo, self.hi) * self.sign + self.offset

    def xml(self) -> str:
        """The arm as an MJCF body tree: the URDF path from its root to the last chain joint, then the tool."""
        cfg, p, names = self.config, self.info.name + "/", self.chain.joint_names
        root = ET.parse(self.kinematics.resolve_urdf(cfg.urdf)).getroot()
        links = {node.get("name"): node for node in root.findall("link")}
        parent_joint = {node.find("child").get("link"): node for node in root.findall("joint")}
        path = [next(j for j in root.findall("joint") if j.get("name") == names[-1])]
        while path[0].find("parent").get("link") in parent_joint:
            path.insert(0, parent_joint[path[0].find("parent").get("link")])
        tcp_rot = self.kinematics.rpy_to_matrix(cfg.tcp_rpy)
        palm = np.asarray(cfg.tcp_xyz, float) + tcp_rot[:, 2] * (self.palm_z - 0.003 - LINK_RADIUS)
        lo, hi = np.sort([self.theta(self.lo), self.theta(self.hi)], axis=0)

        def link(name: str, to) -> str:
            out, inertial = "", links[name].find("inertial")
            if inertial is not None and float(inertial.find("mass").get("value")) > 0:
                inertia = inertial.find("inertia")
                diag = [max(abs(float(inertia.get(k, "0"))), 1e-6) for k in ("ixx", "iyy", "izz")]
                out += (f'<inertial pos="{_v(_origin(inertial)[0])}" mass="{inertial.find("mass").get("value")}" '
                        f'diaginertia="{_v(diag)}"/>')
            if np.linalg.norm(to) > 1e-3:
                out += (f'<geom type="capsule" fromto="0 0 0 {_v(to)}" size="{LINK_RADIUS}" rgba="0.75 0.77 0.8 1" '
                        f'contype="{LINK[0]}" conaffinity="{LINK[1]}"/>')
            return out

        root_link = path[0].find("parent").get("link")
        out = (f'<body name="{p}base" pos="{_v(self.info.base_xyz)}" euler="0 0 {self.info.base_yaw}">'
               f'<body name="{p}{root_link}" euler="{_v(cfg.urdf_base_rpy)}">' + link(root_link, _origin(path[0])[0]))
        for k, joint in enumerate(path):
            pos, rpy = _origin(joint)
            child = joint.find("child").get("link")
            out += f'<body name="{p}{child}" pos="{_v(pos)}" euler="{_v(rpy)}" gravcomp="1">'
            if joint.get("name") in names:
                i = names.index(joint.get("name"))
                axis = joint.find("axis").get("xyz")
                out += (f'<joint name="{p}{names[i]}" axis="{axis}" range="{lo[i]:.6f} {hi[i]:.6f}" '
                        f'armature="0.01" damping="0.2"/>')
            out += link(child, _origin(path[k + 1])[0] if k + 1 < len(path) else palm)
        out += (f'<body name="{p}tool" pos="{_v(cfg.tcp_xyz)}" euler="{_v(cfg.tcp_rpy)}" gravcomp="1">'
                f'<site name="{p}tool" size="0.004" rgba="1 0.9 0 1"/>'
                f'<geom type="box" pos="0 0 {self.palm_z}" size="{MAX_GAP / 2 + 0.012} 0.016 0.003" '
                f'mass="0.02" rgba="0.3 0.3 0.33 1" contype="{LINK[0]}" conaffinity="{LINK[1]}"/>'
                f'<camera name="{self.info.name}_wrist" pos="0 0.06 -0.06" xyaxes="-1 0 0 0 0.878 0.479" fovy="75"/>')
        for side, sign in (("l", 1), ("r", -1)):
            out += (f'<body name="{p}finger_{side}" gravcomp="1">'
                    f'<joint name="{p}finger_{side}" type="slide" axis="{sign} 0 0" range="0 {MAX_GAP / 2}" '
                    f'damping="{FINGER_DAMPING}"/>'
                    f'<geom type="box" pos="{sign * FINGER[0]} 0 {self.finger_z}" size="{_v(FINGER)}" mass="0.02" '
                    f'rgba="0.2 0.2 0.22 1" condim="4" friction="1.2 0.01 0.001" '
                    f'contype="{FINGERS[0]}" conaffinity="{FINGERS[1]}"/></body>')
        return out + "</body>" * (len(path) + 3)

    def actuators_xml(self) -> str:
        p = self.info.name + "/"
        out = "".join(f'<position name="{p}{n}" joint="{p}{n}" kp="{ARM_KP[i]}" kv="{ARM_KV[i]}" forcerange="-80 80"/>'
                      for i, n in enumerate(self.chain.joint_names))
        return out + "".join(f'<position name="{p}finger_{s}" joint="{p}finger_{s}" kp="{FINGER_KP}" kv="5" '
                             f'forcerange="-{FINGER_FORCE} {FINGER_FORCE}"/>' for s in ("l", "r"))


def _origin(node) -> tuple[np.ndarray, np.ndarray]:
    origin = node.find("origin")
    get = (lambda key: origin.get(key, "0 0 0")) if origin is not None else (lambda key: "0 0 0")
    return np.array(get("xyz").split(), float), np.array(get("rpy").split(), float)


class MujocoBackend(Backend):
    """Physics simulation of ``arms`` on a table at ``table_z`` with ``objects`` (dicts with ``name``,
    ``xyz`` (centre, world frame), ``size`` (cube edge, m), ``color`` (a name or RGB 0..1), optional
    ``mass`` (kg)). Cameras: ``top`` (overhead) and ``<arm>_wrist`` (on the tool, looking along it). ``keep_out``
    zones (``safety.KeepOut``) are drawn as translucent boxes that nothing collides with."""

    def __init__(self, arms: Sequence[ArmInfo], table_z: float = 0.0, objects: Optional[Sequence[dict]] = None,
                 cameras: Optional[Sequence[str]] = None, size: tuple[int, int] = (320, 240), keep_out: Sequence = ()):
        self._mj = _mujoco()
        self._arms = {info.name: _Arm(info) for info in arms}
        self._table_z = float(table_z)
        self.objects = list(default_objects(arms, table_z) if objects is None else objects)
        self.size = (int(size[0]), int(size[1]))
        self.keep_out = list(keep_out)
        self.model = self._mj.MjModel.from_xml_string(self._xml())
        self.data = self._mj.MjData(self.model)
        self._cameras = list(cameras) if cameras is not None else ["top"] + [f"{n}_wrist" for n in self._arms]
        for name in self._cameras:
            if self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"no camera {name!r}; cameras: top, " + ", ".join(f"{n}_wrist" for n in self._arms))
        self._renderer = None
        self.render_error: Optional[str] = None
        for arm in self._arms.values():
            p, names = arm.info.name + "/", arm.chain.joint_names
            arm.qpos = [self.model.joint(p + n).qposadr[0] for n in names]
            arm.fingers = [self.model.joint(f"{p}finger_{s}").qposadr[0] for s in ("l", "r")]
            arm.ctrl = [self.model.actuator(p + n).id for n in names]
            arm.finger_ctrl = [self.model.actuator(f"{p}finger_{s}").id for s in ("l", "r")]
            arm.site = self.model.site(p + "tool").id
            self.data.qpos[arm.qpos] = arm.theta(arm.rest_q)
            self._command(arm)
        self._mj.mj_forward(self.model, self.data)

    def _xml(self) -> str:
        bases = np.array([a.info.base_xyz for a in self._arms.values()], float)
        cx, cy = bases[:, 0].mean() + 0.25, bases[:, 1].mean()
        half_y = 0.45 + np.ptp(bases[:, 1]) / 2
        z, (w, h) = self._table_z, self.size
        body = (f'<light pos="{cx} {cy} {z + 1.5}" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7"/>'
                f'<geom name="table" type="box" pos="{cx} {cy} {z - 0.02}" size="0.5 {half_y} 0.02" '
                f'rgba="0.55 0.45 0.35 1" contype="{TABLE[0]}" conaffinity="{TABLE[1]}"/>'
                f'<camera name="top" pos="{cx} {cy} {z + 1.0}" xyaxes="0 -1 0 1 0 0" fovy="55"/>')
        for obj in self.objects:
            size = float(obj.get("size", 0.04))
            color = obj.get("color", "red")
            rgb = COLORS[color] if isinstance(color, str) else tuple(color)[:3]
            mass = float(obj.get("mass", CUBE_DENSITY * size ** 3))
            body += (f'<body name="{obj["name"]}" pos="{_v(obj["xyz"])}"><freejoint name="{obj["name"]}"/>'
                     f'<geom type="box" size="{_v([size / 2] * 3)}" mass="{mass}" rgba="{_v(rgb)} 1" condim="4" '
                     f'friction="1.2 0.01 0.001" contype="{OBJECT[0]}" conaffinity="{OBJECT[1]}"/></body>')
        for zone in self.keep_out:
            lo, hi = np.array(zone.min), np.array(zone.max)
            body += (f'<geom type="box" pos="{_v((lo + hi) / 2)}" size="{_v((hi - lo) / 2)}" '
                     'rgba="1 0.3 0.1 0.2" contype="0" conaffinity="0"/>')
        body += "".join(arm.xml() for arm in self._arms.values())
        actuators = "".join(arm.actuators_xml() for arm in self._arms.values())
        return (f'<mujoco model="llm-harness"><compiler angle="radian" eulerseq="XYZ" autolimits="true"/>'
                f'<option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>'
                f'<visual><global offwidth="{max(w, 640)}" offheight="{max(h, 480)}"/></visual>'
                f'<worldbody>{body}</worldbody><actuator>{actuators}</actuator></mujoco>')

    def _command(self, arm: _Arm) -> None:
        self.data.ctrl[arm.ctrl] = arm.theta(arm.q_target)
        self.data.ctrl[arm.finger_ctrl] = min(1.0, max(0.0, arm.gripper_target)) * MAX_GAP / 2

    @property
    def arms(self) -> list[ArmInfo]:
        return [arm.info for arm in self._arms.values()]

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    @property
    def table_z(self) -> float:
        return self._table_z

    @property
    def time(self) -> float:
        """Simulated seconds since the start."""
        return float(self.data.time)

    def object_xyz(self, name: str) -> np.ndarray:
        """World-frame centre of an object."""
        return self.data.body(name).xpos.copy()

    def tool_pose(self, arm: str) -> np.ndarray:
        """World-frame tool pose (4x4) of an arm, as simulated."""
        a, pose = self._arms[arm], np.eye(4)
        pose[:3, :3] = self.data.site_xmat[a.site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[a.site]
        return pose

    def read(self) -> dict[str, ArmState]:
        out = {}
        for name, arm in self._arms.items():
            q = (self.data.qpos[arm.qpos] - arm.offset) * arm.sign
            gap = float(self.data.qpos[arm.fingers].sum())
            out[name] = ArmState(q=q, gripper=gap / MAX_GAP, q_target=arm.q_target.copy(),
                                 gripper_target=arm.gripper_target, enabled=True, rest_q=arm.rest_q.copy())
        return out

    def step(self, targets: Mapping[str, ArmTarget], dt: float) -> None:
        for name, target in targets.items():
            arm = self._arms[name]
            arm.q_target = np.clip(np.asarray(target.q, float), arm.lo, arm.hi)
            arm.gripper_target = float(target.gripper)
            self._command(arm)
        self._mj.mj_step(self.model, self.data, nstep=max(1, round(dt / TIMESTEP)))

    def rest(self, arm: str) -> str:
        """The servos stay on: a simulated arm has no torque off."""
        return "always_on"

    def images(self) -> dict[str, Frame]:
        if self._renderer is None and self.render_error is None:
            try:
                self._renderer = self._mj.Renderer(self.model, self.size[1], self.size[0])
            except Exception as e:
                self.render_error = (f"MuJoCo cannot render offscreen (MUJOCO_GL={os.environ.get('MUJOCO_GL', '')!r}):"
                                     f" {e}. Import this module before mujoco, or set MUJOCO_GL to egl or osmesa"
                                     f" (with that library installed) before mujoco is first imported.")
                warnings.warn(self.render_error, RuntimeWarning, stacklevel=2)
        if self._renderer is None:
            return {}
        frames = {}
        for name in self._cameras:
            self._renderer.update_scene(self.data, camera=name)
            frames[name] = Frame(self._renderer.render().copy(), float(self.data.time))
        return frames

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
