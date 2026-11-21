"""Robot description files: which arms (and SDK), where their bases sit, cameras, safety settings, and
for simulators the scene.

One YAML file per robot in ``<home>/robots/<name>.yaml``; home is ``$LLM_HARNESS_HOME`` or
``~/.makermods-llm-harness``::

    backend: real            # real | sim | mujoco
    arm: metal               # metal | maker
    arms:                    # one or two, any names (left/right by convention)
      left:  {backend: gs_usb, port: "gs_usb:<serial>"}     # real: SDK rig arm entries; simulators: {}
      right: {backend: gs_usb, port: "gs_usb:<serial>"}
    bases:                   # optional: arm base poses in the world frame (default origin, yaw 0)
      left:  {xyz: [0.0, 0.25, 0.0], yaw_deg: 0}
      right: {xyz: [0.0, -0.25, 0.0], yaw_deg: 0}
    cameras: {top: "usb:<path>", left_wrist: "usb:<path>"}   # real: SDK camera ids; simulators: names or omit
    rest_q:                  # real: each arm's folded rest pose, joint degrees in profile order (llm-harness calibrate)
      left: [0.4, -1.2, 2.9, 10.3, 0.1, -0.8]
    safety: {max_speed_deg_s: 20}                            # SafetyConfig overrides
    keep_out:                # boxes in the world frame (metres) no part of an arm may enter
      - {name: camera_mount, min: [-0.10, -0.42, 0.0], max: [0.20, -0.21, 1.20]}
    scene: {table_z: 0.0, objects: [{name: cube, xyz: [0.28, 0.0, 0.02], size: 0.04, color: red}]}
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, Union

import yaml

from .backends.base import ARM_TYPES, ArmInfo, Backend

BACKENDS = ("real", "sim", "mujoco")
SIM_BACKENDS = ("sim", "mujoco")
KEYS = ("backend", "arm", "arms", "bases", "cameras", "rest_q", "safety", "keep_out", "scene")
OBJECT_KEYS = ("name", "xyz", "size", "color")


class RobotError(ValueError):
    """A robot description is invalid or missing. The message says how to fix it."""


def home() -> Path:
    """$LLM_HARNESS_HOME, else ~/.makermods-llm-harness."""
    env = os.environ.get("LLM_HARNESS_HOME")
    return Path(env).expanduser() if env else Path.home() / ".makermods-llm-harness"


def robots_dir() -> Path:
    return home() / "robots"


def robot_names() -> list[str]:
    return sorted(p.stem for p in robots_dir().glob("*.yaml"))


def sdk(arm_type: str) -> ModuleType:
    """The SDK package for an arm type (``metal_arm`` or ``maker_arm``) with its submodules imported."""
    if arm_type not in ARM_TYPES:
        raise RobotError(f"arm must be one of {', '.join(ARM_TYPES)}, got {arm_type!r}")
    package = f"{arm_type}_arm"
    for module in ("rig", "arm", "kinematics", "trajectory_ik", "cameras", "testing"):
        importlib.import_module(f"{package}.{module}")
    return importlib.import_module(package)


@dataclass
class Robot:
    name: str
    backend: str = "real"
    arm: str = "metal"
    arms: dict[str, dict] = field(default_factory=dict)
    bases: dict[str, dict] = field(default_factory=dict)
    cameras: dict[str, str] = field(default_factory=dict)
    rest_q: dict[str, list[float]] = field(default_factory=dict)     # degrees
    safety: dict[str, float] = field(default_factory=dict)
    scene: dict = field(default_factory=dict)
    path: Optional[Path] = None
    keep_out: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, raw: dict, path: Optional[Path] = None) -> "Robot":
        where = path or name
        if not isinstance(raw, dict):
            raise RobotError(f"{where}: expected a mapping with {', '.join(KEYS)}")
        unknown = sorted(set(raw) - set(KEYS))
        if unknown:
            raise RobotError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
        cameras = raw.get("cameras") or {}
        if isinstance(cameras, (list, tuple)):
            cameras = {str(c): "" for c in cameras}
        robot = cls(name, raw.get("backend", "real"), raw.get("arm", "metal"),
                    {str(k): dict(v or {}) for k, v in (raw.get("arms") or {}).items()},
                    {str(k): dict(v or {}) for k, v in (raw.get("bases") or {}).items()},
                    {str(k): str(v or "") for k, v in cameras.items()},
                    {str(k): v for k, v in (raw.get("rest_q") or {}).items()},
                    dict(raw.get("safety") or {}), dict(raw.get("scene") or {}), path,
                    raw.get("keep_out") or [])
        robot.validate()
        return robot

    def to_dict(self) -> dict:
        out: dict = {"backend": self.backend, "arm": self.arm, "arms": {k: dict(v) for k, v in self.arms.items()}}
        if self.bases:
            out["bases"] = {k: dict(v) for k, v in self.bases.items()}
        if self.cameras:
            out["cameras"] = dict(self.cameras) if any(self.cameras.values()) else list(self.cameras)
        if self.rest_q:
            out["rest_q"] = {k: [round(float(v), 2) for v in q] for k, q in self.rest_q.items()}
        if self.safety:
            out["safety"] = dict(self.safety)
        if self.keep_out:
            out["keep_out"] = [z.to_dict() for z in self.zones()]
        if self.scene:
            out["scene"] = dict(self.scene)
        return out

    def validate(self) -> None:
        where = self.path or self.name
        if self.backend not in BACKENDS:
            raise RobotError(f"{where}: backend must be one of {', '.join(BACKENDS)}, got {self.backend!r}")
        if self.arm not in ARM_TYPES:
            raise RobotError(f"{where}: arm must be one of {', '.join(ARM_TYPES)}, got {self.arm!r}")
        if not 1 <= len(self.arms) <= 2 or any(not n or n == "all" for n in self.arms):
            raise RobotError(f"{where}: arms must name one or two arms (not 'all'), got {sorted(self.arms)}")
        for name, entry in self.arms.items():
            if self.backend == "real":
                rig = sdk(self.arm).rig
                try:
                    rig.ArmSpec(**entry)
                except (TypeError, rig.RigError) as e:
                    raise RobotError(f"{where}: arms.{name}: {e}") from e
            elif set(entry) - {"profile"}:
                raise RobotError(f"{where}: arms.{name}: a simulated arm takes only 'profile', got {sorted(entry)}")
        if self.backend == "real":
            self._validate_buses(where)
        for name, q in self.rest_q.items():
            if name not in self.arms:
                raise RobotError(f"{where}: rest_q.{name}: no arm named {name!r}")
            if not _numbers(q, 6):
                raise RobotError(f"{where}: rest_q.{name} must be six joint angles in degrees, got {q}")
        for name, base in self.bases.items():
            if name not in self.arms:
                raise RobotError(f"{where}: bases.{name}: no arm named {name!r}")
            if set(base) - {"xyz", "yaw_deg"} or not _numbers(base.get("xyz", [0, 0, 0]), 3) \
                    or not _numbers([base.get("yaw_deg", 0)], 1):
                raise RobotError(f"{where}: bases.{name} must be {{xyz: [x, y, z], yaw_deg: d}}, got {base}")
        if self.backend == "real" and not all(self.cameras.values()):
            raise RobotError(f"{where}: a real robot needs a camera id per camera, got {self.cameras}")
        if self.scene and self.backend == "real":
            raise RobotError(f"{where}: scene is for simulators; a real robot measures its table "
                             "(llm-harness calibrate)")
        self._validate_scene(where)
        from .safety import SafetyConfig
        try:
            SafetyConfig.from_dict(self.safety)
        except ValueError as e:
            raise RobotError(f"{where}: safety: {e}") from e
        try:
            self.zones()
        except ValueError as e:
            raise RobotError(f"{where}: keep_out: {e}") from e

    def zones(self) -> list:
        """The keep-out zones as ``safety.KeepOut``; ValueError when the list is invalid."""
        from .safety import keep_out_zones
        if not isinstance(self.keep_out, list):
            raise ValueError(f"expected a list of zones, got {self.keep_out!r}")
        return keep_out_zones(self.keep_out)

    def _validate_buses(self, where) -> None:
        """Each arm on its own bus, named explicitly when there are two: two names must never reach the same
        motors."""
        rig = sdk(self.arm).rig
        try:
            rig.Rig.from_dict({"arms": self.arms})
        except rig.RigError as e:
            raise RobotError(f"{where}: arms: {e}") from e
        buses = [rig.ArmSpec(**entry) for entry in self.arms.values()]
        if len(buses) > 1 and len({(b.backend, b.address) for b in buses}) < len(buses):
            raise RobotError(f"{where}: arms: two arms would use the same bus "
                             f"({', '.join(b.address for b in buses)}); give each its own channel or port")

    def setup(self, arm: str, safety=None) -> dict:
        """What a table height measured with this arm depends on: arm type, bus and profile, base pose and the
        finger length (``safety``: the SafetyConfig in use, default this robot's)."""
        from .safety import SafetyConfig
        config = safety or SafetyConfig.from_dict(self.safety)
        info = next(i for i in self.arm_infos() if i.name == arm)
        return {"arm": self.arm, "entry": dict(sorted(self.arms[arm].items())), "base_xyz": list(info.base_xyz),
                "base_yaw": info.base_yaw, "finger_length_m": config.gripper(self.arm).finger_length_m}

    def _validate_scene(self, where) -> None:
        unknown = sorted(set(self.scene) - {"table_z", "objects"})
        if unknown:
            raise RobotError(f"{where}: scene: unknown keys {unknown} (known: table_z, objects)")
        if not _numbers([self.scene.get("table_z", 0.0)], 1):
            raise RobotError(f"{where}: scene.table_z must be a number")
        for obj in self.scene.get("objects") or []:
            if not isinstance(obj, dict) or set(obj) - set(OBJECT_KEYS) or not obj.get("name") \
                    or not _numbers(obj.get("xyz"), 3) or not _numbers([obj.get("size", 0.06)], 1):
                raise RobotError(f"{where}: scene.objects: each is {{name, xyz: [x, y, z], size, color}}, got {obj}")

    def arm_infos(self) -> list[ArmInfo]:
        infos = []
        for name, entry in self.arms.items():
            base = self.bases.get(name, {})
            xyz = tuple(float(v) for v in base.get("xyz", (0, 0, 0)))
            infos.append(ArmInfo(name, self.arm, entry.get("profile"), xyz, math.radians(float(base.get("yaw_deg", 0)))))
        return infos

    def open_backend(self, **options) -> Backend:
        """The backend this robot runs on. A real robot's devices open here; ``options`` go to RealBackend."""
        if self.backend == "real":
            from .backends.real import RealBackend
            return RealBackend.from_robot(self, **options)
        table_z = float(self.scene.get("table_z", 0.0))
        objects = list(self.scene.get("objects") or [])
        cameras = {"cameras": list(self.cameras)} if self.cameras else {}
        if self.backend == "sim":
            from .backends.sim import SimBackend
            return SimBackend(self.arm_infos(), table_z=table_z, objects=objects, **cameras)
        from .backends.mujoco import MujocoBackend
        return MujocoBackend(arms=self.arm_infos(), table_z=table_z, objects=objects, keep_out=self.zones(), **cameras)


def _numbers(value, count: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == count
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))


def load(name: Union[str, Path, None] = None) -> Robot:
    """The named robot, a robot file given by path, or the only robot there is."""
    if name is None:
        names = robot_names()
        if len(names) != 1:
            raise RobotError("no robot defined: run 'llm-harness robot init NAME'" if not names
                             else f"several robots ({', '.join(names)}): pass the robot's name")
        name = names[0]
    path = Path(name).expanduser()
    if not (path.suffix in (".yaml", ".yml") and path.is_file()):
        path = robots_dir() / f"{name}.yaml"
        if not path.is_file():
            known = ", ".join(robot_names()) or "none"
            raise RobotError(f"no robot named {name!r} (robots: {known})")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise RobotError(f"{path}: not valid YAML ({e})") from e
    return Robot.from_dict(path.stem, raw, path)


def save(robot: Robot) -> Path:
    """Write ``<home>/robots/<name>.yaml``."""
    robot.validate()
    if not robot.name or any(c in robot.name for c in "/\\:") or robot.name.startswith("."):
        raise RobotError(f"invalid robot name {robot.name!r}")
    path = robots_dir() / f"{robot.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(robot.to_dict(), sort_keys=False))
    tmp.replace(path)
    robot.path = path
    return path


def from_rig(name: str, arm_type: str, rig_path: Union[str, Path, None] = None,
             arms: Optional[Sequence[str]] = None, cameras: Optional[Sequence[str]] = None) -> Robot:
    """A real robot from an SDK rig file (default: the SDK's rig path): its arms (all, or ``arms``) and
    cameras (all, or ``cameras``) under the rig's names. Reads the file only; opens no device."""
    rig = sdk(arm_type).rig.Rig.load(rig_path, missing_ok=False)
    wanted = list(arms or rig.arm_names)
    missing = [a for a in wanted if a not in rig.arms]
    if missing or not wanted:
        raise RobotError(f"rig {rig.path} has no arm named {', '.join(missing) or 'at all'}")
    missing = [c for c in cameras or () if c not in rig.cameras]
    if missing:
        raise RobotError(f"rig {rig.path} has no camera named {', '.join(missing)}")
    entries = {}
    for arm in wanted:
        entry = rig.arms[arm].to_yaml()
        if entry.get("profile") and rig.base:
            entry["profile"] = str(rig.base / Path(entry["profile"]).expanduser())
        entries[arm] = entry
    robot = Robot(name, "real", arm_type, entries,
                  cameras={c: rig.cameras[c] for c in (cameras if cameras is not None else rig.cameras)})
    robot.validate()
    return robot


def simulated(backend: str = "sim", arm: str = "metal", arms: Sequence[str] = ("arm",), name: str = "sim") -> Robot:
    """An in-memory simulator robot: a table at z = 0 and a 4 cm cube in front of each arm, within reach.
    Two arms stand side by side, 0.5 m apart, facing +x."""
    arms = list(arms)
    bases, objects = {}, []
    colors = ("red", "blue")
    for i, arm_name in enumerate(arms):
        y = 0.0 if len(arms) == 1 else (0.25 if i == 0 else -0.25)
        if len(arms) > 1:
            bases[arm_name] = {"xyz": [0.0, y, 0.0], "yaw_deg": 0.0}
        objects.append({"name": f"{colors[i % 2]}_cube", "xyz": [0.28, y, 0.02], "size": 0.04, "color": colors[i % 2]})
    robot = Robot(name, backend, arm, {a: {} for a in arms}, bases,
                  scene={"table_z": 0.0, "objects": objects})
    robot.validate()
    return robot
