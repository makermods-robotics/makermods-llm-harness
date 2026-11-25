"""What an LLM can ask for (actions) and what it gets back (observations), as typed objects and JSON.

Every action is a flat JSON object: ``{"action": "move_tool", "arm": "left", "xyz": [0.3, 0, 0.1]}``.
``tools()`` gives the same actions as tool definitions (name, description, JSON Schema) for a model
API or an MCP server, and ``SCHEMA`` is one JSON Schema covering all of them.

Units on this side are metres, degrees and seconds. ``arm`` may be left out when the robot has one arm.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union


class ActionError(ValueError):
    """The action is malformed. The message is written for whoever sent it."""


# ── actions ──
@dataclass(frozen=True)
class Observe:
    """Look: fresh camera images and the state of every arm. Moves nothing."""
    name: ClassVar[str] = "observe"


@dataclass(frozen=True)
class MoveTool:
    """Move the tool point (between the gripper fingers) in a straight line to xyz in the world frame.
    pitch_deg is the tilt of the gripper's pointing direction (-90 straight down, 0 horizontal; default:
    keep the current tilt). yaw_deg is its compass heading about the vertical (default: pointing away
    from the arm's base). roll_deg turns the gripper about its pointing direction (default: keep).
    A path into a keep-out zone (listed in the robot description) is rejected: go around it. With line false
    the joints move straight to the pose instead (the tool point then sweeps a curve). until "contact" ends the
    move early, holding, where the arm stops following its command. speed_deg_s caps the joint speed below the
    envelope's; duration_s stretches the move to about that long."""
    name: ClassVar[str] = "move_tool"
    xyz: tuple[float, float, float]
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    arm: Optional[str] = None
    line: Optional[bool] = None
    until: Optional[str] = None
    speed_deg_s: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class Nudge:
    """Move the tool point by dxyz metres, keeping its orientation. frame "base" uses the arm's base
    axes (x forward, y left, z up); frame "tool" uses the gripper's own axes (z is its pointing direction).
    The offset is added to the last commanded tool pose while the arm holds still within 10 mm and 3 deg of
    it, so repeated small nudges add up; after a stop or fault, or when the arm is further off, it is added
    to the measured pose. until, speed_deg_s and duration_s work as for move_tool."""
    name: ClassVar[str] = "nudge"
    dxyz: tuple[float, float, float]
    frame: str = "base"
    arm: Optional[str] = None
    until: Optional[str] = None
    speed_deg_s: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class Gripper:
    """Set the gripper opening: 0 closed, 1 fully open ("open" and "close" also work). The observation
    reports the opening actually reached, and the result says whether the jaws stopped on something."""
    name: ClassVar[str] = "gripper"
    opening: float
    arm: Optional[str] = None


@dataclass(frozen=True)
class MoveJoints:
    """Move joints to angles in degrees, by joint name. Joints left out stay where they are. With
    relative, the angles are added to the current ones."""
    name: ClassVar[str] = "move_joints"
    joints_deg: dict[str, float]
    relative: bool = False
    arm: Optional[str] = None
    speed_deg_s: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class Home:
    """Move to the ready pose: gripper forward, above the table."""
    name: ClassVar[str] = "home"
    arm: Optional[str] = None
    speed_deg_s: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class Rest:
    """Return to the rest pose and turn the motors off. Without arm, every arm rests."""
    name: ClassVar[str] = "rest"
    arm: Optional[str] = None


@dataclass(frozen=True)
class Wait:
    """Do nothing for some seconds, then observe."""
    name: ClassVar[str] = "wait"
    seconds: float = 1.0


Action = Union[Observe, MoveTool, Nudge, Gripper, MoveJoints, Home, Rest, Wait]
ACTIONS: dict[str, type] = {cls.name: cls for cls in (Observe, MoveTool, Nudge, Gripper, MoveJoints, Home, Rest, Wait)}
FRAMES = ("base", "tool")
MAX_WAIT_S = 30.0
MAX_DURATION_S = 60.0
MIN_SPEED_DEG_S = 0.5
UNTIL = ("contact",)
MOTIONS = ("move_tool", "nudge", "move_joints", "home")    # the actions that take speed_deg_s and duration_s

_ARM = {"type": "string", "description": "Arm name. Optional when the robot has one arm."}
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_SPEED = {"speed_deg_s": {"type": "number", "minimum": MIN_SPEED_DEG_S,
                          "description": "Joint speed cap in deg/s, below the envelope's limit."},
          "duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_DURATION_S,
                         "description": "Take about this long (longer when the speed limits need it)."}}
_UNTIL = {"until": {"type": "string", "enum": list(UNTIL),
                    "description": "contact: end early, holding, where the arm stops following its command."}}
_PROPERTIES: dict[str, dict[str, Any]] = {
    "observe": {},
    "move_tool": {
        "xyz": {**_XYZ, "description": "Target of the tool point: x, y, z in metres, world frame (z up)."},
        "pitch_deg": {"type": "number", "minimum": -90, "maximum": 90,
                      "description": "Tilt of the pointing direction: -90 straight down, 0 horizontal."},
        "yaw_deg": {"type": "number", "description": "Heading of the pointing direction about the vertical."},
        "roll_deg": {"type": "number", "description": "Turn of the gripper about its pointing direction."},
        "arm": _ARM,
        "line": {"type": "boolean", "default": True,
                 "description": "true: the tool point moves in a straight line; false: the joints move straight."},
        **_UNTIL, **_SPEED,
    },
    "nudge": {
        "dxyz": {**_XYZ, "description": "Offset in metres along the frame's x, y, z."},
        "frame": {"type": "string", "enum": list(FRAMES), "default": "base"},
        "arm": _ARM,
        **_UNTIL, **_SPEED,
    },
    "gripper": {
        "opening": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "string", "enum": ["open", "close"]}],
                    "description": "0 closed .. 1 open, or \"open\" / \"close\"."},
        "arm": _ARM,
    },
    "move_joints": {
        "joints_deg": {"type": "object", "additionalProperties": {"type": "number"},
                       "description": "Joint name -> angle in degrees."},
        "relative": {"type": "boolean", "default": False},
        "arm": _ARM,
        **_SPEED,
    },
    "home": {"arm": _ARM, **_SPEED},
    "rest": {"arm": _ARM},
    "wait": {"seconds": {"type": "number", "minimum": 0, "maximum": MAX_WAIT_S, "default": 1.0}},
}
_REQUIRED = {"move_tool": ["xyz"], "nudge": ["dxyz"], "gripper": ["opening"], "move_joints": ["joints_deg"]}


def _description(cls: type) -> str:
    return " ".join((cls.__doc__ or "").split())


def tools() -> list[dict[str, Any]]:
    """One tool definition per action: ``{"name", "description", "input_schema"}``."""
    return [{"name": name, "description": _description(cls),
             "input_schema": {"type": "object", "properties": _PROPERTIES[name],
                              "required": _REQUIRED.get(name, []), "additionalProperties": False}}
            for name, cls in ACTIONS.items()]


def _schema() -> dict[str, Any]:
    variants = []
    for tool in tools():
        inner = tool["input_schema"]
        variants.append({"type": "object", "description": tool["description"],
                         "properties": {"action": {"const": tool["name"]}, **inner["properties"]},
                         "required": ["action", *inner["required"]], "additionalProperties": False})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Action", "oneOf": variants}


SCHEMA = _schema()


def _show(value: Any) -> str:
    """A short repr for an error message."""
    try:
        text = repr(value)
    except ValueError:      # an int too long to print
        return "a number too large to print"
    return text if len(text) <= 80 else text[:77] + "..."


def _number(value: Any, what: str) -> float:
    try:
        number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
    except OverflowError:
        number = math.nan
    if not math.isfinite(number):
        raise ActionError(f"{what} must be a finite number, got {_show(value)}")
    return number


def _triple(value: Any, what: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ActionError(f"{what} must be three numbers [x, y, z], got {_show(value)}")
    return tuple(_number(v, what) for v in value)  # type: ignore[return-value]


def _optional(raw: dict, key: str) -> Optional[float]:
    return None if raw.get(key) is None else _number(raw[key], key)


def _timing(name: str, raw: dict) -> dict:
    """``speed_deg_s`` and ``duration_s`` of a motion, checked; at most one of them."""
    speed, duration = _optional(raw, "speed_deg_s"), _optional(raw, "duration_s")
    if speed is not None and duration is not None:
        raise ActionError(f"{name}: give speed_deg_s or duration_s, not both")
    if speed is not None and speed < MIN_SPEED_DEG_S:
        raise ActionError(f"speed_deg_s must be at least {MIN_SPEED_DEG_S:g}, got {speed:g}")
    if duration is not None and not 0 < duration <= MAX_DURATION_S:
        raise ActionError(f"duration_s must be above 0 and at most {MAX_DURATION_S:g}, got {duration}")
    return {"speed_deg_s": speed, "duration_s": duration}


def _until(raw: dict) -> Optional[str]:
    until = raw.get("until")
    if until is not None and until not in UNTIL:
        raise ActionError(f"until must be one of {', '.join(UNTIL)}, got {_show(until)}")
    return until


def _flag(raw: dict, key: str) -> Optional[bool]:
    value = raw.get(key)
    if value is not None and not isinstance(value, bool):
        raise ActionError(f"{key} must be true or false, got {_show(value)}")
    return value


def parse_action(raw: Union[Action, dict, str]) -> Action:
    """An action from its JSON object, JSON text or typed instance, all checked the same way. A field that is
    omitted or null takes its default. ActionError says what is wrong."""
    if isinstance(raw, tuple(ACTIONS.values())):
        raw = action_to_dict(raw)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as e:
            raise ActionError(f"not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ActionError(f"an action is a JSON object, got {_show(raw)}")
    raw = {k: v for k, v in raw.items() if v is not None}
    name = raw.pop("action", None)
    if not isinstance(name, str) or name not in ACTIONS:
        raise ActionError(f"unknown action {_show(name)}; expected one of {', '.join(ACTIONS)}")
    unknown = sorted(str(k) for k in set(raw) - set(_PROPERTIES[name]))
    if unknown:
        raise ActionError(f"{name}: unknown fields {unknown}; it takes {sorted(_PROPERTIES[name]) or 'no fields'}")
    missing = [k for k in _REQUIRED.get(name, []) if k not in raw]
    if missing:
        raise ActionError(f"{name}: missing {', '.join(missing)}")
    arm = raw.get("arm")
    if arm is not None and (not isinstance(arm, str) or not arm):
        raise ActionError(f"arm must be a name, got {_show(arm)}")
    if name == "observe":
        return Observe()
    if name == "move_tool":
        pitch = _optional(raw, "pitch_deg")
        if pitch is not None and not -90 <= pitch <= 90:
            raise ActionError(f"pitch_deg must be between -90 (straight down) and 90, got {pitch}")
        return MoveTool(_triple(raw["xyz"], "xyz"), pitch, _optional(raw, "yaw_deg"), _optional(raw, "roll_deg"), arm,
                        _flag(raw, "line"), _until(raw), **_timing(name, raw))
    if name == "nudge":
        frame = raw.get("frame", "base")
        if not isinstance(frame, str) or frame not in FRAMES:
            raise ActionError(f"frame must be one of {', '.join(FRAMES)}, got {_show(frame)}")
        return Nudge(_triple(raw["dxyz"], "dxyz"), frame, arm, _until(raw), **_timing(name, raw))
    if name == "gripper":
        opening = {"open": 1.0, "close": 0.0, "closed": 0.0}.get(raw["opening"]) if isinstance(raw["opening"], str) \
            else _number(raw["opening"], "opening")
        if opening is None or not 0 <= opening <= 1:
            raise ActionError(f"opening must be between 0 (closed) and 1 (open), or \"open\" / \"close\", "
                              f"got {_show(raw['opening'])}")
        return Gripper(opening, arm)
    if name == "move_joints":
        joints = raw["joints_deg"]
        if not isinstance(joints, dict) or not joints or not all(isinstance(k, str) for k in joints):
            raise ActionError("joints_deg must map joint names to degrees, for example {\"elbow_flex\": 30}")
        relative = raw.get("relative", False)
        if not isinstance(relative, bool):
            raise ActionError(f"relative must be true or false, got {_show(relative)}")
        return MoveJoints({k: _number(v, f"joints_deg.{k}") for k, v in joints.items()}, relative, arm,
                          **_timing(name, raw))
    if name == "home":
        return Home(arm, **_timing(name, raw))
    if name == "rest":
        return Rest(arm)
    seconds = _number(raw.get("seconds", 1.0), "seconds")
    if not 0 <= seconds <= MAX_WAIT_S:
        raise ActionError(f"seconds must be between 0 and {MAX_WAIT_S:g}, got {seconds}")
    return Wait(seconds)


def action_to_dict(action: Action) -> dict[str, Any]:
    """The JSON object of an action; fields left at None are dropped."""
    out: dict[str, Any] = {"action": action.name}
    for f in dataclasses.fields(action):
        value = getattr(action, f.name)
        if value is not None:
            out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


# ── observations ──
@dataclass
class Result:
    """How the last action ended. ``status`` is "done", "rejected" (refused before anything moved: fix the
    request and try again), "stopped" (ended early, the arm holds where it is) or "fault" (the arm reports
    a fault). ``message`` explains anything but "done" in plain words. ``error``: after a tool move, how far the
    measured tool pose ended from the requested one, ``{"position_m", "angle_deg", "short_m"}`` (``short_m``: how far
    short of the target along the approach, negative past it). ``contact``: a move with
    until "contact" that ended on contact, ``{"xyz", "short_m"}`` (where the tool point stopped, how far short of
    the target). ``grip``: after a gripper action, ``{"opening", "target", "holding"}`` (the measured and the
    requested opening, and whether the jaws stopped on something)."""
    status: str = "done"
    message: str = ""
    action: Optional[dict[str, Any]] = None
    error: Optional[dict[str, float]] = None
    contact: Optional[dict[str, Any]] = None
    grip: Optional[dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == "done"


@dataclass
class ToolPose:
    """The tool point in the world frame, with the same angles ``move_tool`` takes."""
    xyz: tuple[float, float, float]
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


@dataclass
class ArmObservation:
    joints_deg: dict[str, float]
    gripper: float                       # opening, 0 closed .. 1 open
    tool: ToolPose
    clearance_m: Optional[float]         # lowest monitored point above the table; None without a table height
    enabled: Optional[bool] = True       # motors on; None when the torque state is not known
    fault: Optional[str] = None
    temperature_c: Optional[float] = None
    nearest_keep_out: Optional[dict[str, Any]] = None   # {"name", "distance_m"}: the closest zone; None without zones


@dataclass
class CameraImage:
    """One camera's picture, JPEG-encoded once. ``path`` is set when the harness saves images. ``stale``: no frame
    captured after the action ended arrived in time, so this one (``age_s`` old) may predate its end."""
    stamp: float = 0.0
    width: int = 0
    height: int = 0
    path: Optional[str] = None
    jpeg: Optional[bytes] = field(default=None, repr=False)
    error: Optional[str] = None
    age_s: Optional[float] = None
    stale: bool = False


@dataclass
class Observation:
    step: int
    time: float
    arms: dict[str, ArmObservation]
    cameras: dict[str, CameraImage]
    result: Result = field(default_factory=Result)

    def to_dict(self, images: bool = False) -> dict[str, Any]:
        """JSON-ready. Image bytes are left out unless ``images`` (then base64 under ``jpeg_base64``)."""
        cameras = {}
        for name, cam in self.cameras.items():
            entry = {k: v for k, v in dataclasses.asdict(cam).items() if k != "jpeg" and v is not None}
            if images and cam.jpeg is not None:
                entry["jpeg_base64"] = base64.b64encode(cam.jpeg).decode("ascii")
            cameras[name] = entry
        arms = {}
        for name, arm in self.arms.items():
            entry = dataclasses.asdict(arm)
            entry["tool"]["xyz"] = list(arm.tool.xyz)
            arms[name] = entry
        return {"step": self.step, "time": self.time, "result": dataclasses.asdict(self.result),
                "arms": arms, "cameras": cameras}

    def to_json(self, images: bool = False) -> str:
        return json.dumps(self.to_dict(images))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Observation":
        arms = {}
        for name, a in raw.get("arms", {}).items():
            tool = dict(a["tool"])
            tool["xyz"] = tuple(tool["xyz"])
            arms[name] = ArmObservation(**{**a, "tool": ToolPose(**tool)})
        cameras = {}
        for name, c in raw.get("cameras", {}).items():
            c = dict(c)
            encoded = c.pop("jpeg_base64", None)
            cameras[name] = CameraImage(**c, jpeg=base64.b64decode(encoded) if encoded else None)
        return cls(raw["step"], raw["time"], arms, cameras, Result(**raw.get("result", {})))

    def jpegs(self) -> dict[str, bytes]:
        """Camera name -> JPEG bytes, read from the image file when only its path is known."""
        out = {}
        for name, cam in self.cameras.items():
            if cam.jpeg is not None:
                out[name] = cam.jpeg
            elif cam.path and Path(cam.path).is_file():
                out[name] = Path(cam.path).read_bytes()
        return out

    def missing_images(self) -> list[str]:
        """Cameras that report an image whose bytes are not here and whose file cannot be read."""
        return [name for name, cam in self.cameras.items()
                if cam.error is None and cam.jpeg is None and not (cam.path and Path(cam.path).is_file())]

    def text(self) -> str:
        """A compact description for a model or a terminal: result, then one block per arm, then images."""
        lines = [] if self.result.ok and not self.result.message else [f"{self.result.status}: {self.result.message}"]
        for name, arm in self.arms.items():
            x, y, z = arm.tool.xyz
            clearance = "unknown" if arm.clearance_m is None else f"{arm.clearance_m:.3f} m"
            state = {True: "", False: " (motors off)"}.get(arm.enabled, " (torque state unknown)")
            zone = arm.nearest_keep_out
            lines.append(f"{name}{state}: tool xyz [{x:.3f}, {y:.3f}, {z:.3f}] m, pitch {arm.tool.pitch_deg:.0f}, "
                         f"yaw {arm.tool.yaw_deg:.0f}, roll {arm.tool.roll_deg:.0f} deg; gripper {arm.gripper:.2f}; "
                         f"clearance above table {clearance}"
                         + (f"; keep-out zone {zone['name']} {zone['distance_m']:.3f} m away" if zone else ""))
            lines.append("  joints (deg): " + ", ".join(f"{k} {v:.1f}" for k, v in arm.joints_deg.items()))
            if arm.fault:
                lines.append(f"  FAULT: {arm.fault}")
        for name, cam in self.cameras.items():
            where = cam.path or f"{cam.width}x{cam.height} image"
            if cam.stale:
                where += f" (stale: {cam.age_s:.1f} s old, may predate the end of the action)"
            lines.append(f"camera {name}: {cam.error or where}")
        return "\n".join(lines)
