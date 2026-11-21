"""A kinematic simulator: joints follow their targets at a bounded speed, boxes on a table can be picked up,
and the cameras are cheap synthetic drawings. No physics and no sleeping (unless ``realtime``), so a whole
episode runs in a test in well under a second.

Cameras: ``top`` (looking down, +x up the image), ``side`` (looking along +y, +x to the right) and
``<arm>_wrist`` (from just behind the gripper along its pointing direction); any other name shows the top view.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from ..arm_model import ArmModel
from .base import ArmInfo, ArmState, ArmTarget, Backend, Frame

JOINT_SPEED = math.radians(180.0)   # rad/s, how fast simulated joints follow their targets
GRIPPER_SPEED = 2.0                 # opening per second
GRIPPER_GAP_M = 0.08                # jaw gap at opening 1
MOUNT_CLEARANCE_M = 0.02            # the start pose keeps at least this above the table
TEMPERATURE_C = 35.0
COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 90, 220), "yellow": (230, 200, 40),
          "orange": (240, 130, 30), "purple": (150, 60, 190), "white": (240, 240, 240), "black": (20, 20, 20),
          "gray": (128, 128, 128)}
ARM_COLORS = ((45, 55, 75), (30, 95, 70))
TABLE = (196, 170, 130)
GRID = (176, 150, 112)


@dataclass
class _Arm:
    model: ArmModel
    q: np.ndarray
    rest_q: np.ndarray
    gripper: float = 0.0
    gripper_target: float = 0.0
    enabled: bool = False
    fault: Optional[str] = None
    temperature: float = TEMPERATURE_C
    blocked: bool = False
    holding: Optional[dict] = None
    grip: Optional[np.ndarray] = None      # held object's centre in the tool frame (homogeneous)

    def __post_init__(self):
        self.q_target = self.q.copy()


class SimBackend(Backend):
    """Kinematic simulator of one or more arms over a table at ``table_z`` with box ``objects``
    (``{name, xyz, size, color}``, xyz the box centre). Arms start at all-zero joints clipped into their limits,
    motors off, gripper closed; an arm base is raised when that pose would reach below the table."""

    def __init__(self, arms: Sequence[ArmInfo], cameras: Sequence[str] = ("top", "side"), table_z: float = 0.0,
                 objects: Sequence[dict] = (), realtime: bool = False, size: tuple[int, int] = (320, 240)):
        self._table_z = float(table_z)
        self._cameras = list(cameras)
        self.realtime = realtime
        self.size = size
        self._arms: dict[str, _Arm] = {}
        for info in arms:
            model = ArmModel(info)
            q = np.clip(np.zeros(6), model.lo, model.hi)
            sink = self._table_z + MOUNT_CLEARANCE_M - float(model.lowest_z(q))
            if sink > 0:
                x, y, z = info.base_xyz
                model = ArmModel(ArmInfo(info.name, info.type, info.profile, (x, y, z + sink), info.base_yaw))
            self._arms[info.name] = _Arm(model, q, q.copy())
        self._objects = [{"name": str(o["name"]), "xyz": np.array(o["xyz"], dtype=float),
                          "size": float(o.get("size", 0.06)), "color": o.get("color", "red")} for o in objects]
        bases = np.array([a.model.world_from_base[:3, 3] for a in self._arms.values()])
        x_lo, x_hi = bases[:, 0].min() - 0.3, bases[:, 0].max() + 0.6
        self._top = _View((x_lo, x_hi, bases[:, 1].min() - 0.45, bases[:, 1].max() + 0.45), size, flip_u=True)
        self._side = _View((x_lo, x_hi, self._table_z - 0.05, self._table_z + 0.75), size, flip_u=False)

    # ── Backend ──
    @property
    def arms(self) -> list[ArmInfo]:
        return [arm.model.info for arm in self._arms.values()]

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    @property
    def table_z(self) -> Optional[float]:
        return self._table_z

    def enable(self, arm: str, cancel=None) -> None:
        a = self._arms[arm]
        if a.fault:
            raise RuntimeError(f"{arm} cannot be enabled: {a.fault}")
        if not a.enabled:
            a.enabled, a.q_target, a.gripper_target = True, a.q.copy(), a.gripper

    def read(self) -> dict[str, ArmState]:
        return {name: ArmState(a.q.copy(), a.gripper, a.q_target.copy(), a.gripper_target, a.enabled, a.fault,
                               a.temperature, a.rest_q.copy()) for name, a in self._arms.items()}

    def step(self, targets: Mapping[str, ArmTarget], dt: float) -> None:
        for name, target in targets.items():
            a = self._arms[name]
            if a.enabled and not a.fault:
                a.q_target = np.clip(np.asarray(target.q, dtype=float), a.model.lo, a.model.hi)
                a.gripper_target = min(1.0, max(0.0, float(target.gripper)))
        for a in self._arms.values():
            if a.enabled and not a.fault:
                if not a.blocked:
                    a.q = a.q + np.clip(a.q_target - a.q, -JOINT_SPEED * dt, JOINT_SPEED * dt)
                self._move_gripper(a, dt)
            if a.holding is not None:
                a.holding["xyz"] = (a.model.tool_pose(a.q) @ a.grip)[:3]
        if self.realtime:
            time.sleep(dt)

    def images(self) -> dict[str, Frame]:
        stamp = time.time()
        frames = {}
        for name in self._cameras:
            arm = name[:-len("_wrist")] if name.endswith("_wrist") else None
            if arm in self._arms:
                image = self._draw_wrist(self._arms[arm])
            else:
                image = self._draw_side() if name == "side" else self._draw_top()
            frames[name] = Frame(image, stamp)
        return frames

    def rest(self, arm: str) -> str:
        """Motors off where the arm stands; a held box drops to the table. A faulted or blocked arm keeps holding."""
        a = self._arms[arm]
        if not a.enabled:
            return "released"
        if a.blocked or a.fault:
            return "holding"
        self._release(a)
        a.enabled = False
        return "released"

    # ── test hooks ──
    def set_fault(self, arm: str, text: Optional[str]) -> None:
        """Make the arm report a fault (None clears it). A faulted arm ignores targets and holds."""
        self._arms[arm].fault = text

    def set_temperature(self, arm: str, celsius: float) -> None:
        self._arms[arm].temperature = float(celsius)

    def block(self, arm: str, blocked: bool = True) -> None:
        """Stop the joints following their targets, as if the arm hit something."""
        self._arms[arm].blocked = blocked

    @property
    def objects(self) -> list[dict]:
        """``{name, xyz, size, color, held_by}`` of every box."""
        held = {id(a.holding): name for name, a in self._arms.items() if a.holding is not None}
        return [{"name": o["name"], "xyz": o["xyz"].tolist(), "size": o["size"], "color": o["color"],
                 "held_by": held.get(id(o))} for o in self._objects]

    # ── gripper ──
    def _move_gripper(self, a: _Arm, dt: float) -> None:
        reach = GRIPPER_SPEED * dt
        new = a.gripper + min(reach, max(-reach, a.gripper_target - a.gripper))
        if a.holding is not None:
            touch = a.holding["size"] / GRIPPER_GAP_M
            if new > touch + 0.02:
                self._release(a)
            else:
                new = max(new, touch)
        elif new < a.gripper:
            tool = a.model.tool_pose(a.q)
            tips = a.model.fingertip(a.q)
            for obj in self._objects:
                touch = obj["size"] / GRIPPER_GAP_M
                offset = tips - obj["xyz"]
                # Grasped when the fingertips close on it from inside its outline, below its top.
                if (new < touch <= a.gripper and np.hypot(offset[0], offset[1]) <= obj["size"] / 2
                        and abs(offset[2]) <= obj["size"] / 2 and not self._held(obj)):
                    new, a.holding = touch, obj
                    a.grip = np.linalg.inv(tool) @ np.append(obj["xyz"], 1.0)
                    break
        a.gripper = new

    def _held(self, obj: dict) -> bool:
        return any(a.holding is obj for a in self._arms.values())

    def _release(self, a: _Arm) -> None:
        if a.holding is not None:
            a.holding["xyz"][2] = self._table_z + a.holding["size"] / 2
            a.holding, a.grip = None, None

    # ── drawing ──
    def _jaws(self, a: _Arm) -> list[np.ndarray]:
        """Two finger segments (2, 3) each, world frame."""
        tool = a.model.tool_pose(a.q)
        p, x, z = tool[:3, 3], tool[:3, 0], tool[:3, 2]
        half = max(a.gripper, 0.05) * GRIPPER_GAP_M / 2
        reach = a.model.gripper.finger_length_m
        return [np.array([p + s * half * x - 0.03 * z, p + s * half * x + reach * z]) for s in (-1, 1)]

    def _draw_scene(self, view: "_View", axes: tuple[int, int], top: bool) -> np.ndarray:
        image = np.full((self.size[1], self.size[0], 3), TABLE if top else (215, 225, 235), np.uint8)
        if top:
            view.grid(image, GRID)
        else:
            cv2.rectangle(image, (0, view.px((0.0, self._table_z))[1]), (self.size[0], self.size[1]), TABLE, -1)
        held = [a.holding for a in self._arms.values() if a.holding is not None]
        for obj in sorted(self._objects, key=lambda o: o["xyz"][2]):
            if not any(obj is h for h in held):
                view.box(image, obj, axes)
        for i, (name, a) in enumerate(self._arms.items()):
            color = ARM_COLORS[i % len(ARM_COLORS)]
            pts = a.model.skeleton(a.q)[:, axes]
            cv2.polylines(image, [np.array([view.px(p) for p in pts], np.int32)], False, color, 3 if top else 5,
                          cv2.LINE_AA)
            for p in pts[:-1]:
                cv2.circle(image, view.px(p), 3, (200, 200, 200), -1)
            if a.holding is not None:
                view.box(image, a.holding, axes)
            jaw_color = (230, 120, 20) if a.gripper < 0.15 or a.holding is not None else (40, 200, 90)
            for jaw in self._jaws(a):
                cv2.line(image, view.px(jaw[0, axes]), view.px(jaw[1, axes]), jaw_color, 3, cv2.LINE_AA)
                cv2.circle(image, view.px(jaw[1, axes]), 3, jaw_color, -1)
            state = "holding " + a.holding["name"] if a.holding is not None else f"gripper {a.gripper:.2f}"
            cv2.putText(image, f"{name}: {state}", (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                        cv2.LINE_AA)
        for obj in self._objects:
            if not any(obj is h for h in held):
                view.box(image, obj, axes, fill=False)     # outlines stay visible under the arm
        return image

    def _draw_top(self) -> np.ndarray:
        return self._draw_scene(self._top, (0, 1), True)

    def _draw_side(self) -> np.ndarray:
        return self._draw_scene(self._side, (0, 2), False)

    def _draw_wrist(self, a: _Arm) -> np.ndarray:
        """The table (checkered, 5 cm squares) and boxes seen from just behind the jaws, which show at the bottom."""
        w, h = self.size
        tool = a.model.tool_pose(a.q)
        right, down, forward = -tool[:3, 0], -tool[:3, 1], tool[:3, 2]
        eye = tool[:3, 3] - 0.08 * forward
        f = w / 2 / math.tan(math.radians(40))
        gw, gh = w // 4, h // 4
        u, v = np.meshgrid((np.arange(gw) + 0.5) * 4 - w / 2, (np.arange(gh) + 0.5) * 4 - h / 2)
        rays = forward * f + right * u[..., None] + down * v[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self._table_z - eye[2]) / rays[..., 2]
        hit = eye + t[..., None] * rays
        checker = (np.floor(hit[..., 0] / 0.05) + np.floor(hit[..., 1] / 0.05)) % 2
        small = np.where((t > 0)[..., None], np.where(checker[..., None] > 0, TABLE, GRID), (215, 225, 235))
        image = cv2.resize(small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        for obj in sorted(self._objects, key=lambda o: -np.linalg.norm(o["xyz"] - eye)):
            corners = obj["xyz"] + obj["size"] / 2 * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                                                                 for sz in (-1, 1)])
            rel = corners - eye
            depth = rel @ forward
            if np.all(depth > 0.01):
                pix = np.stack([w / 2 + f * (rel @ right) / depth, h / 2 + f * (rel @ down) / depth], axis=1)
                hull = cv2.convexHull(np.clip(pix, -4 * w, 4 * w).astype(np.int32))
                cv2.fillConvexPoly(image, hull, COLORS.get(obj["color"], COLORS["gray"]), cv2.LINE_AA)
        gap = int(max(a.gripper, 0.05) * w * 0.35)
        for x in (w // 2 - gap - 14, w // 2 + gap):
            cv2.rectangle(image, (x, h - 60), (x + 14, h), (70, 70, 70), -1)
        return image


class _View:
    """Orthographic projection of two world axes (a, b) onto the image, fitted to bounds with equal scale.
    Top view (``flip_u``): a (x) runs up the image and b (y) to the left. Side view: a to the right, b up."""

    def __init__(self, bounds, size: tuple[int, int], flip_u: bool):
        self.bounds = bounds
        lo_a, hi_a, lo_b, hi_b = bounds
        self.w, self.h = size
        self.flip_u = flip_u
        span_u, span_v = (hi_b - lo_b, hi_a - lo_a) if flip_u else (hi_a - lo_a, hi_b - lo_b)
        self.scale = min(self.w / span_u, self.h / span_v)
        self.ca, self.cb = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2

    def px(self, p) -> tuple[int, int]:
        da, db = (float(p[0]) - self.ca) * self.scale, (float(p[1]) - self.cb) * self.scale
        if self.flip_u:
            return int(round(self.w / 2 - db)), int(round(self.h / 2 - da))
        return int(round(self.w / 2 + da)), int(round(self.h / 2 - db))

    def grid(self, image: np.ndarray, color) -> None:
        """Lines every 10 cm."""
        lo_a, hi_a, lo_b, hi_b = self.bounds
        for k in range(math.floor(min(lo_a, lo_b) * 10) - 2, math.ceil(max(hi_a, hi_b) * 10) + 3):
            x, _ = self.px((self.ca, k / 10))
            _, y = self.px((k / 10, self.cb))
            cv2.line(image, (x, 0), (x, self.h), color, 1)
            cv2.line(image, (0, y), (self.w, y), color, 1)

    def box(self, image: np.ndarray, obj: dict, axes: tuple[int, int], fill: bool = True) -> None:
        c, half = obj["xyz"][list(axes)], obj["size"] / 2
        p0, p1 = self.px(c - half), self.px(c + half)
        color = COLORS.get(obj["color"], COLORS["gray"])
        if fill:
            cv2.rectangle(image, p0, p1, color, -1)
            cv2.rectangle(image, p0, p1, (30, 30, 30), 1)
        else:
            cv2.rectangle(image, p0, p1, color, 2)
