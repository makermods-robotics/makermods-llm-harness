"""The interface between the harness and a robot: joint targets in, joint state and images out.

A backend is real arms, the built-in kinematic simulator, MuJoCo, or your own simulator. The harness
does everything else (actions, safety envelope, inverse kinematics, logging), so a new simulator is
one subclass of ``Backend``. See docs/backends.md for a worked example.

Units: radians, metres, seconds. Joints are the arm joints in profile order, without the gripper;
the gripper is one number, its opening from 0 (closed) to 1 (open).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

ARM_TYPES = ("metal", "maker")
OFF = ("released", "always_on")   # rest() outcomes after which nothing holds torque the harness owns


@dataclass(frozen=True)
class ArmInfo:
    """One arm of the robot. ``type`` selects the SDK profile (joint names, limits, kinematics);
    ``profile`` names another profile file. The base pose places the arm in the world frame (z up):
    its base origin at ``base_xyz``, turned ``base_yaw`` about z."""
    name: str
    type: str = "metal"
    profile: Optional[str] = None
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_yaw: float = 0.0

    def __post_init__(self):
        if self.type not in ARM_TYPES:
            raise ValueError(f"arm type must be one of {', '.join(ARM_TYPES)}, got {self.type!r}")


@dataclass
class ArmState:
    """What an arm reports. ``q_target`` and ``gripper_target`` are the command the arm is following
    (equal to the measurement before the first command). ``enabled``: True while its motors hold torque,
    False when torque off is confirmed, None when it is not known. ``fault`` is None while the arm can
    move."""
    q: np.ndarray
    gripper: float
    q_target: np.ndarray
    gripper_target: float
    enabled: Optional[bool] = True
    fault: Optional[str] = None
    temperature: Optional[float] = None     # hottest motor, °C
    rest_q: Optional[np.ndarray] = None     # the folded rest pose: the only pose where the arm is turned on or off


@dataclass
class ArmTarget:
    """A joint command for one arm. ``speed`` (rad/s) caps every joint on the way to it; None leaves the
    backend's own limit (sent only with ``q`` equal to the arm's current command, for the gripper)."""
    q: np.ndarray
    gripper: float
    speed: Optional[float] = None


@dataclass
class Frame:
    """One camera image: RGB, uint8, (height, width, 3), with its wall-clock time. ``captured`` is its
    ``time.monotonic()`` capture time; None means it was made when ``images()`` was called."""
    image: np.ndarray
    stamp: float
    captured: Optional[float] = None


class Backend(ABC):
    """A robot the harness can drive. Methods are called from one thread at a time."""

    @property
    @abstractmethod
    def arms(self) -> list[ArmInfo]:
        """The arms, in a fixed order."""

    @property
    def cameras(self) -> list[str]:
        """Camera names."""
        return []

    @property
    def table_z(self) -> Optional[float]:
        """World-frame height of the table when the backend knows it (a simulator does). None means
        the harness needs a table calibration before it moves an arm."""
        return None

    def enable(self, arm: str, cancel: Optional[threading.Event] = None) -> None:
        """Make the arm ready to follow targets, holding its current pose (real arms: torque on); a set
        ``cancel`` aborts it. The harness calls it at the rest pose, before the first motion of an arm
        whose state does not report ``enabled`` True."""

    @abstractmethod
    def read(self) -> dict[str, ArmState]:
        """The current state of every arm."""

    @abstractmethod
    def step(self, targets: Mapping[str, ArmTarget], dt: float) -> None:
        """Send ``targets`` (only the arms named; the others keep theirs) and let ``dt`` seconds pass:
        a simulator advances its clock, real hardware sleeps. The harness sends consecutive waypoints of a
        checked path, one per call, so an arm should reach each target within about ``dt``."""

    def hold(self, arm: str) -> None:
        """Freeze the arm at its measured joints and gripper opening, replacing every outstanding target."""
        state = self.read()[arm]
        self.step({arm: ArmTarget(state.q, state.gripper)}, 0.0)

    def images(self) -> dict[str, Frame]:
        """The newest image of each camera. A camera without a recent image is left out."""
        return {}

    def rest(self, arm: str) -> str:
        """Release the arm, which the harness has brought to ``rest_q``, without moving it. Returns
        "released" (torque off confirmed), "always_on" (a simulator without torque off), "holding" (not
        released: it keeps its pose) or an error text."""
        return "always_on"

    def close(self) -> Optional[dict[str, str]]:
        """Free the devices without moving anything. The harness rests the arms first; an arm still on is left
        holding its pose where the hardware can do that. May return each arm's torque outcome as ``rest``
        words it, plus "unknown" or "energized" (on, but not by this session)."""
