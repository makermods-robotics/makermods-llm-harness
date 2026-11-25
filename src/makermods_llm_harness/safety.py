"""The safety envelope every planned arm motion must pass, and the table calibration it depends on.

``Envelope.check`` vets a joint path before any of it is sent: joint limits, a per-move excursion cap,
a hard floor above the table (every monitored point lowered by what the tracking tolerance and sag allow;
with recovery when the arm already sits under it, and the final approach to the recorded rest pose exempt when
the rest pose is the lowest point of the path) and the per-tick speed, which drops to the slow speed
whenever the path comes near the table, and keep-out boxes no point along the arm may enter (with the same
kind of recovery for an arm that starts inside one). The table height is measured by touching it with the fingertips
(``measure_table``) and stored per robot and arm, with the setup it was measured on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, ClassVar, Optional, Sequence

import numpy as np

from .arm_model import GRIPPERS, ArmModel, Gripper
from .robot import home

TOUCH_TOLERANCE_M = 0.003    # two touch readings must agree this well
TOUCH_ATTEMPTS = 3
RECOVERY_EPS_M = 5e-4        # path-following noise of a planned lift: a passing dip this small is not a descent
REST_TOLERANCE_DEG = 5.0     # per joint: how close to the recorded rest pose an arm must be to turn on or off


@dataclass(frozen=True)
class Table:
    """The table height under one arm (world z, m): ``z`` everywhere, or with ``reach`` a height that depends on the
    horizontal distance from the arm base, linear between ``(reach_m, z_m)`` points and held flat beyond them (an
    arm whose kinematic model drifts with extension sees a flat table at different heights). ``z`` is the height
    reported in words; with ``reach`` it is their mean."""
    z: float
    reach: tuple = ()

    def __post_init__(self):
        rows = tuple(sorted((float(r), float(h)) for r, h in self.reach))
        if not all(math.isfinite(v) for row in rows for v in row) or not math.isfinite(float(self.z)):
            raise ValueError(f"the table height must be finite numbers, got {self.z}, {self.reach}")
        if len({r for r, _ in rows}) != len(rows):
            raise ValueError(f"the table's reach points must differ, got {[r for r, _ in rows]}")
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "reach", rows)

    @classmethod
    def measured(cls, points) -> "Table":
        """From ``(reach_m, z_m)`` points: a flat table for one, a height per reach for more."""
        rows = [(float(r), float(h)) for r, h in points]
        if not rows:
            raise ValueError("no table points")
        return cls(rows[0][1]) if len(rows) == 1 else cls(float(np.mean([h for _, h in rows])), tuple(rows))

    def at(self, points: np.ndarray, base_xy) -> np.ndarray:
        """The table height under each of ``points`` (..., 3), world frame, for an arm based at ``base_xy``."""
        points = np.asarray(points, dtype=float)
        if not self.reach:
            return np.full(points.shape[:-1], self.z)
        reach = np.hypot(points[..., 0] - base_xy[0], points[..., 1] - base_xy[1])
        r, h = zip(*self.reach)
        return np.interp(reach, r, h)

    def text(self) -> str:
        if not self.reach:
            return f"z = {self.z:.3f} m"
        return ("z = " + ", ".join(f"{h:.4f} m at {r:.2f} m" for r, h in self.reach)
                + " from the base (linear between, flat beyond)")


@dataclass(frozen=True)
class KeepOut:
    """A named axis-aligned box in the world frame (metres) that no part of an arm may enter."""
    name: str
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    def to_dict(self) -> dict:
        return {"name": self.name, "min": list(self.min), "max": list(self.max)}

    def text(self) -> str:
        return (f"{self.name} from [{', '.join(f'{v:.2f}' for v in self.min)}] to "
                f"[{', '.join(f'{v:.2f}' for v in self.max)}] m")


def keep_out_zones(raw) -> list[KeepOut]:
    """Zones from a robot file's ``keep_out`` list of ``{name, min: [x, y, z], max: [x, y, z]}``. ValueError says
    what is wrong."""
    zones = []
    for entry in raw or []:
        if not isinstance(entry, dict) or set(entry) != {"name", "min", "max"}:
            raise ValueError(f"each zone is {{name, min: [x, y, z], max: [x, y, z]}}, got {entry!r}")
        name, corners = entry["name"], []
        if not isinstance(name, str) or not name:
            raise ValueError(f"a zone needs a name, got {name!r}")
        for key in ("min", "max"):
            value = entry[key]
            if not (isinstance(value, (list, tuple)) and len(value) == 3 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
                raise ValueError(f"zone {name}: {key} must be three finite numbers [x, y, z], got {value!r}")
            corners.append(tuple(float(v) for v in value))
        if not all(a < b for a, b in zip(*corners)):
            raise ValueError(f"zone {name}: min must be below max on every axis, got {list(corners[0])} and "
                             f"{list(corners[1])}")
        if name in (z.name for z in zones):
            raise ValueError(f"two zones are named {name!r}")
        zones.append(KeepOut(name, *corners))
    return zones


def signed_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(..., Z) distance of each point (..., 3) from each box (Z, 3): positive outside, minus the depth inside."""
    d = np.maximum(lo - points[..., None, :], points[..., None, :] - hi)
    inside = d.max(axis=-1)
    return np.where(inside > 0, np.linalg.norm(np.maximum(d, 0.0), axis=-1), inside)


# Accepted range of every setting (inclusive). Settings outside ZERO_OK must also be above 0.
LIMITS: dict[str, tuple[float, float]] = {
    "max_speed_deg_s": (0.5, 180.0), "slow_speed_deg_s": (0.5, 180.0), "slow_zone_m": (0.0, 0.5),
    "floor_margin_m": (0.0, 0.2), "max_excursion_deg": (1.0, 360.0), "max_temp_c": (20.0, 100.0),
    "tracking_tolerance_deg": (0.5, 30.0), "tracking_lag_s": (0.0, 0.5), "settle_s": (0.0, 5.0),
    "contact_tolerance_m": (0.001, 0.05), "floor_tolerance_deg": (0.0, 10.0), "sag_allowance_m": (0.0, 0.05),
    "contact_allowance_m": (0.0, 0.05), "rate_hz": (5.0, 200.0), "keep_out_margin_m": (0.0, 0.2),
    "finger_length_m": (0.0, 0.3), "jaw_half_width_m": (0.0, 0.3), "finger_margin_m": (0.0, 0.05),
}


class MoveRejected(Exception):
    """The motion breaks a rule of the envelope. The message says what to do instead."""


class CalibrationError(RuntimeError):
    """The table height could not be measured."""


@dataclass(frozen=True)
class SafetyConfig:
    max_speed_deg_s: float = 20.0      # joint speed limit
    slow_speed_deg_s: float = 5.0      # joint speed limit within slow_zone_m of the table
    slow_zone_m: float = 0.0762        # height above the table under which the slow speed applies
    floor_margin_m: float = 0.0        # hard floor: least planned clearance above the table (0: may touch it)
    max_excursion_deg: float = 35.0    # farthest any joint may travel from its start in one action
    max_temp_c: float = 70.0           # no motion while a motor is hotter
    # Farthest a measured joint may be from the planned path before the motion is held as blocked (the real
    # Metal arm tracks within about 3 deg at 15 deg/s).
    tracking_tolerance_deg: float = 6.0
    # How long a measured joint may trail its command: it is compared with every waypoint sent within this much
    # elapsed time, so the allowance along the path grows with the commanded speed (speed x lag) and the tolerance
    # above applies across it. At most 0.5 s.
    tracking_lag_s: float = 0.2
    settle_s: float = 1.0              # longest wait after the last waypoint for the joints to reach the goal
    # A move with until "contact" ends once the measured tool point is this far from every tool point commanded
    # within the tracking lag.
    contact_tolerance_m: float = 0.010
    # Per-joint error the planned-path floor check allows for (every joint this far off, summed as a worst case).
    floor_tolerance_deg: float = 0.0
    # Height the arm may sit below its kinematic model under its own weight (link and gear compliance).
    sag_allowance_m: float = 0.0
    # A moving arm is held once a measured monitored point is this far below the floor (a compliant fingertip
    # pressing on the table does not stop it).
    contact_allowance_m: float = 0.010
    rate_hz: float = 25.0              # joint targets per second while moving
    keep_out_margin_m: float = 0.03    # planned paths keep every point along the arm this far from keep-out zones
    # Gripper geometry; None takes arm_model.GRIPPERS for the arm type. Measure them on the installed tool.
    finger_length_m: Optional[float] = None
    jaw_half_width_m: Optional[float] = None
    finger_margin_m: Optional[float] = None
    OPTIONAL: ClassVar[tuple[str, ...]] = ("finger_length_m", "jaw_half_width_m", "finger_margin_m")
    ZERO_OK: ClassVar[tuple[str, ...]] = ("floor_margin_m", "floor_tolerance_deg", "sag_allowance_m",
                                          "contact_allowance_m", "keep_out_margin_m", "tracking_lag_s", "settle_s")

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.name in self.OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be a finite number, got {value!r}")
            if value < 0 or (value == 0 and f.name not in self.ZERO_OK + self.OPTIONAL):
                raise ValueError(f"{f.name} must be > 0, got {value}")
            lo, hi = LIMITS[f.name]
            if not lo <= value <= hi:
                raise ValueError(f"{f.name} must be between {lo:g} and {hi:g}, got {value:g}")
        if self.slow_speed_deg_s > self.max_speed_deg_s:
            raise ValueError("slow_speed_deg_s must not exceed max_speed_deg_s")

    def gripper(self, arm_type: str) -> Gripper:
        """The gripper geometry for an arm type: the table's, with this config's overrides."""
        base = GRIPPERS[arm_type]
        pick = lambda value, default: default if value is None else float(value)  # noqa: E731
        return Gripper(pick(self.finger_length_m, base.finger_length_m),
                       pick(self.jaw_half_width_m, base.jaw_half_width_m), pick(self.finger_margin_m, base.margin_m))

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "SafetyConfig":
        raw = dict(raw or {})
        known = [f.name for f in fields(cls)]
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"unknown settings {unknown} (known: {', '.join(known)})")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


class Envelope:
    """The rules for one arm. ``table_z`` is the world-frame table height, a number or a ``Table``; without it no
    motion passes."""

    def __init__(self, model: ArmModel, config: SafetyConfig, table_z,
                 keep_out: Sequence[KeepOut] = ()):
        if isinstance(table_z, Table):
            self.table = table_z
        elif table_z is None:
            self.table = None
        elif not math.isfinite(table_z):
            raise ValueError(f"{model.name}: the table height must be a finite number, got {table_z}")
        else:
            self.table = Table(float(table_z))
        self.model = model
        self.config = config
        self.table_z = None if self.table is None else self.table.z
        self._base_xy = model.world_from_base[:2, 3].copy()
        self.max_speed = math.radians(config.max_speed_deg_s)
        self.slow_speed = math.radians(config.slow_speed_deg_s)
        self.tolerance = math.radians(config.tracking_tolerance_deg)
        self.floor_tolerance = math.radians(config.floor_tolerance_deg)
        self.zones = list(keep_out)
        self._lo = np.array([z.min for z in self.zones], dtype=float).reshape(-1, 3)
        self._hi = np.array([z.max for z in self.zones], dtype=float).reshape(-1, 3)

    def clearance(self, q):
        """Height of the lowest monitored point above the table under it (a float for q (6,), (T,) for a path), or
        None without a table height."""
        if self.table is None:
            return None
        if not self.table.reach:
            return self.model.lowest_z(q) - self.table_z
        return self._above(self.model.points(q))

    def guarded_clearance(self, q):
        """``clearance`` with every joint up to the floor tolerance off and the sag allowance taken off."""
        self.require_table()
        if not self.table.reach:
            return self.model.lowest_z_within(q, self.floor_tolerance) - self.config.sag_allowance_m - self.table_z
        return self._above(self.model.points_within(q, self.floor_tolerance)) - self.config.sag_allowance_m

    def table_under(self, xyz) -> float:
        """The table height (world z) under a world point."""
        self.require_table()
        return float(self.table.at(np.asarray(xyz, dtype=float), self._base_xy))

    def _above(self, points: np.ndarray):
        """Lowest height of ``points`` (..., k, 3) above the table under each: a float for (k, 3)."""
        lowest = (points[..., 2] - self.table.at(points, self._base_xy)).min(axis=-1)
        return float(lowest) if np.ndim(lowest) == 0 else lowest

    def hold_floor(self, q_start) -> float:
        """The measured clearance under which a motion from ``q_start`` is held: the hard floor, or for an arm
        starting below it, its start less the sag allowance; either less the contact allowance."""
        cfg = self.config
        return min(cfg.floor_margin_m, float(self.clearance(q_start)) - cfg.sag_allowance_m) - cfg.contact_allowance_m

    def zone_distances(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Per zone, the signed distance of the arm's closest sampled point (negative inside) and that point's
        index into ``model.body_names``: (Z,) each for one q, (T, Z) for a path."""
        sd = signed_distance(self.model.body_points(q), self._lo, self._hi)
        return sd.min(axis=-2), sd.argmin(axis=-2)

    def nearest_keep_out(self, q) -> Optional[tuple[str, float]]:
        """The zone closest to the arm at ``q`` and the signed distance to it, or None without zones."""
        if not self.zones:
            return None
        d, _ = self.zone_distances(q)
        z = int(np.argmin(d))
        return self.zones[z].name, float(d[z])

    def keep_out_inside(self, q) -> list[str]:
        """What it means for each zone the arm at ``q`` is inside (with the margin)."""
        if not self.zones:
            return []
        d, part = self.zone_distances(q)
        margin = self.config.keep_out_margin_m
        return [f"the {self.model.body_names[part[z]]} is {(margin - d[z]) * 1000:.0f} mm inside keep-out zone "
                f"{zone.name} (its box grown by the {margin * 1000:.0f} mm margin)"
                for z, zone in enumerate(self.zones) if d[z] < margin]

    def keep_out_hold(self, q_start, end=None) -> np.ndarray:
        """Per zone, the measured distance under which a motion from ``q_start`` is held: 0 (entering the box), or
        for an arm starting inside it, its start less the contact allowance; with ``end`` (the way to a rest pose)
        no stricter than that pose less the allowance."""
        if not self.zones:
            return np.zeros(0)
        allowance = self.config.contact_allowance_m
        limit = np.minimum(0.0, self.zone_distances(q_start)[0] - allowance)
        return limit if end is None else np.minimum(limit, self.zone_distances(end)[0] - allowance)

    def require_table(self) -> None:
        if self.table is None:
            raise MoveRejected(f"{self.model.name}: the table height is not known for this setup (never measured, "
                               "or measured with another bus, profile, base pose or finger length), so the arm "
                               "may not move. The operator must run `llm-harness calibrate` first.")

    def speed_for(self, q_start, path) -> float:
        """Joint speed limit (rad/s) for a path: the slow speed when the start or any waypoint is within the
        slow zone, else the maximum."""
        self.require_table()
        clearance = self.clearance(np.vstack([q_start, path]))
        return self.slow_speed if float(np.min(clearance)) < self.config.slow_zone_m else self.max_speed

    def check(self, q_start, path, capped: bool = True, rest_q=None) -> None:
        """Raise MoveRejected unless every waypoint of ``path`` (T, 6), sent in order after ``q_start``, keeps
        every rule. ``capped=False`` lifts the excursion cap (moves to the known home and rest poses). A path
        that ends at ``rest_q``, the recorded rest pose where the arm sits supported, may end under the floor
        (see ``_rest_approach``)."""
        self.require_table()
        name, joints, cfg = self.model.name, self.model.joint_names, self.config
        q_start = np.asarray(q_start, dtype=float)
        path = np.asarray(path, dtype=float).reshape(-1, 6)
        if not len(path) or not np.all(np.isfinite(path)) or not np.all(np.isfinite(q_start)):
            raise MoveRejected(f"{name}: the motion contains a value that is not a finite number.")
        # A joint that starts past a limit (a rest pose lying on its end stop) may stay there, never go further.
        lo, hi = np.minimum(self.model.lo, q_start) - 1e-9, np.maximum(self.model.hi, q_start) + 1e-9
        outside = np.argwhere((path < lo) | (path > hi))
        if len(outside):
            t, j = outside[0]
            lo_deg, hi_deg = math.degrees(self.model.lo[j]), math.degrees(self.model.hi[j])
            raise MoveRejected(f"{name}: {joints[j]} would go to {math.degrees(path[t, j]):.1f} deg, outside its "
                               f"range [{lo_deg:.0f}, {hi_deg:.0f}] deg.")
        if capped:
            span = np.max(np.abs(path - q_start), axis=0)
            j = int(np.argmax(span))
            if span[j] > math.radians(cfg.max_excursion_deg) + 1e-9:
                raise MoveRejected(f"{name}: {joints[j]} would move {math.degrees(span[j]):.0f} deg in one action, "
                                   f"over the {cfg.max_excursion_deg:g} deg per-action limit (safety setting "
                                   "max_excursion_deg). Split the motion into smaller steps.")
        try:
            self._check_floor(q_start, path)
        except MoveRejected:
            if rest_q is None or not self._rest_approach(q_start, path, rest_q):
                raise
        self._check_keep_out(q_start, path, rest_q)
        speed = self.speed_for(q_start, path)
        steps = np.max(np.abs(np.diff(np.vstack([q_start, path]), axis=0)), axis=0)
        j = int(np.argmax(steps))
        if steps[j] > speed / cfg.rate_hz * (1 + 1e-6) + 1e-12:
            zone = " near the table" if speed == self.slow_speed else ""
            raise MoveRejected(f"{name}: the motion moves {joints[j]} at {math.degrees(steps[j]) * cfg.rate_hz:.1f} "
                               f"deg/s, over the {math.degrees(speed):.0f} deg/s limit{zone}.")



    def rest_floor(self, rest_q) -> float:
        """The measured clearance under which a motion ending at the rest pose is held: the hard floor, or the
        rest pose's own guarded clearance when that is lower; either less the contact allowance."""
        cfg = self.config
        return min(cfg.floor_margin_m, float(self.guarded_clearance(rest_q))) - cfg.contact_allowance_m

    def rest_note(self, rest_q) -> Optional[str]:
        """What it means when the recorded rest pose lies under the floor for this table height, else None."""
        if self.table_z is None or self.guarded_clearance(rest_q) >= self.config.floor_margin_m:
            return None
        return (f"{self.model.name}: the recorded rest pose is {self.clearance(rest_q) * 1000:.0f} mm above the table "
                f"({self.guarded_clearance(rest_q) * 1000:.0f} mm allowing for tracking error and sag), under the "
                f"{self.config.floor_margin_m * 1000:.0f} mm hard floor. The arm starts in recovery mode there: it "
                "may only rise until it clears the floor. The rest action can still bring it back down to this pose.")

    def _rest_approach(self, q_start, path, rest_q) -> bool:
        """Does the path pass as an approach to the rest pose ``rest_q``? It must end there at the slow speed,
        never go lower than the rest pose itself, and dip under the floor only at its start (an arm already
        under it) and on its final approach."""
        if not np.allclose(path[-1], np.asarray(rest_q, dtype=float), atol=1e-9):
            return False
        ext = np.vstack([q_start, path])
        nominal = self.clearance(ext)
        if not np.all(np.isfinite(nominal)) or np.min(nominal) < nominal[-1] - RECOVERY_EPS_M:
            return False
        if self.speed_for(q_start, path) != self.slow_speed:
            return False
        return _only_at_ends(self.guarded_clearance(ext) < self.config.floor_margin_m)

    def _check_keep_out(self, q_start, path, rest_q=None) -> None:
        """No sampled point of the arm comes within the margin of a zone. From inside that (recovery), the
        distance may never drop below the largest it has reached and once outside it stays out. A path ending at
        the rest pose may end inside, if it never goes deeper than the rest pose and is inside only at its ends."""
        if not self.zones:
            return
        name, margin = self.model.name, self.config.keep_out_margin_m
        dist, part = self.zone_distances(np.vstack([q_start, path]))
        for z, zone in enumerate(self.zones):
            d = dist[:, z]
            inside = d < margin
            if d[0] >= margin:
                bad = bool(np.any(inside))
            else:
                best = np.maximum.accumulate(d)
                out = np.flatnonzero(~inside)
                bad = bool(np.any(d[1:] < best[:-1] - RECOVERY_EPS_M) or d[-1] < d[0]
                           or (len(out) and np.any(inside[out[0]:])))
            if not bad or (rest_q is not None and np.allclose(path[-1], rest_q, atol=1e-9)
                           and np.min(d) >= d[-1] - RECOVERY_EPS_M and _only_at_ends(inside)):
                continue
            t = int(np.argmin(d[1:])) + 1
            what, depth = self.model.body_names[part[t, z]], (margin - d[t]) * 1000
            if d[0] >= margin:
                raise MoveRejected(f"{name}: that motion would bring the {what} {depth:.0f} mm into keep-out zone "
                                   f"{zone.name} (its box grown by the {margin * 1000:.0f} mm margin). Go around "
                                   "it: move away from the zone first, then approach from another side.")
            raise MoveRejected(f"{name}: the arm is {(margin - d[0]) * 1000:.0f} mm inside keep-out zone {zone.name} "
                               f"(with its {margin * 1000:.0f} mm margin), and that motion would take the {what} "
                               f"{depth:.0f} mm in. From here only motions that keep moving out of the zone are "
                               "allowed: move away from it first.")

    def _check_floor(self, q_start, path) -> None:
        """Every waypoint's guarded clearance stays above the floor. From under it (recovery), the clearance may
        never drop below the highest it has reached (by more than RECOVERY_EPS_M on the way, at all at the end), and
        once a waypoint is above the floor the rest stay there."""
        name, floor = self.model.name, self.config.floor_margin_m
        guarded = self.guarded_clearance(np.vstack([q_start, path]))
        nominal = self.clearance(np.vstack([q_start, path]))
        if not np.all(np.isfinite(guarded)):
            raise MoveRejected(f"{name}: the clearance of that motion is not a finite number.")

        def refuse(t: int, text: str):
            what = self.model.point_names[int(np.argmin(self.model.points(path[t - 1])[:, 2]))]
            raise MoveRejected(text.format(what=what, mm=nominal[t] * 1000, guarded=guarded[t] * 1000,
                                           floor=floor * 1000, start=nominal[0] * 1000))

        if guarded[0] >= floor:
            t = int(np.argmin(guarded[1:])) + 1
            if guarded[t] < floor:
                room = (guarded[0] - floor) * 1000
                refuse(t, f"{name}: that motion would bring the {{what}} to {{mm:.0f}} mm above the table "
                          "({guarded:.0f} mm allowing for tracking error and sag), under the {floor:.0f} mm hard "
                          f"floor. From here the lowest point may come down at most {room:.0f} mm (its clearance); "
                          "aim at that or higher.")
            return
        best = np.maximum.accumulate(nominal)
        dips = np.flatnonzero(nominal[1:] < best[:-1] - RECOVERY_EPS_M)
        if nominal[-1] < nominal[0]:
            dips = np.append(dips, len(nominal) - 2)
        above = np.flatnonzero(guarded >= floor)
        later = np.flatnonzero(guarded[above[0]:] < floor) + above[0] if len(above) else np.array([], int)
        if len(dips) or len(later):
            t = int(np.argmin(nominal[1:])) + 1 if len(dips) else int(later[0])
            refuse(t, f"{name}: the arm is {{start:.0f}} mm above the table, under the {{floor:.0f}} mm hard floor "
                      "once tracking error and sag are allowed for, and that motion would take the {what} down to "
                      "{mm:.0f} mm. From here only motions that keep gaining height are allowed: lift first (for "
                      "example nudge dxyz [0, 0, 0.02]).")


def _only_at_ends(flags: np.ndarray) -> bool:
    """True when every set flag belongs to the run at the start or the run at the end."""
    first, last = 0, len(flags)
    while first < last and flags[first]:
        first += 1
    while last > first and flags[last - 1]:
        last -= 1
    return not np.any(flags[first:last])


# ── table calibration ──
def measure_table(read_z: Callable[[], float], ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                  gap_s: float = 0.3) -> float:
    """The table height from the fingertips resting on it: ``read_z()`` gives their world z. Two
    readings ``gap_s`` apart must agree within 3 mm; three attempts, then CalibrationError."""
    say("Move the arm by hand until the fingertips rest on the table top, and keep it there.")
    for attempt in range(1, TOUCH_ATTEMPTS + 1):
        ask("Press Enter when the fingertips touch the table... ")
        first = float(read_z())
        time.sleep(gap_s)
        second = float(read_z())
        if math.isfinite(first) and math.isfinite(second) and abs(first - second) <= TOUCH_TOLERANCE_M:
            table_z = max(first, second)
            say(f"Table at z = {table_z:.4f} m.")
            return table_z
        say(f"The reading moved {abs(first - second) * 1000:.1f} mm between samples (limit "
            f"{TOUCH_TOLERANCE_M * 1000:.0f} mm): hold the arm still. Attempt {attempt}/{TOUCH_ATTEMPTS}.")
    raise CalibrationError("the table height could not be measured: the arm did not hold still")


def watch_for_touch(read_z: Callable[[], float], say: Callable[[str], None] = print, move_m: float = 0.03,
                    still_m: float = 0.003, still_s: float = 5.0, timeout_s: float = 480.0, poll_s: float = 0.2,
                    clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """An ``ask`` for ``measure_table`` that needs no keyboard: it returns once the reading has moved by
    ``move_m`` and then stayed within ``still_m`` for ``still_s``. CalibrationError after ``timeout_s``."""
    def ask(prompt: str) -> str:
        say(f"{prompt.strip()} (watching: move the gripper onto the table, then hold it still for {still_s:.0f} s)")
        start_z, started = read_z(), clock()
        moved = False
        window: list[tuple[float, float]] = []
        while clock() - started < timeout_s:
            z, now = read_z(), clock()
            if not moved:
                moved = abs(z - start_z) > move_m
            else:
                window = [(t, v) for t, v in window if now - t <= still_s] + [(now, z)]
                values = [v for _, v in window]
                if now - window[0][0] >= still_s - poll_s and max(values) - min(values) <= still_m:
                    return ""
            sleep(poll_s)
        raise CalibrationError("timed out waiting for the gripper to be placed on the table and held still")
    return ask


def calibration_path(robot_name: str) -> Path:
    return home() / "calibration" / f"{robot_name}.json"


def load_calibration(robot_name: str) -> dict[str, dict]:
    """``{arm: {"table_z": m, "measured_at": unix time, "setup": {...}}}`` for the robot, plus ``"table_reach":
    [[reach_m, z_m], ...]`` for a table measured at several reaches; empty when never calibrated. ``setup`` is what
    the height was measured with (``Robot.setup``)."""
    path = calibration_path(robot_name)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
        data = {str(arm): {"table_z": float(v["table_z"]), "measured_at": float(v["measured_at"]),
                           "setup": dict(v["setup"])} for arm, v in raw.items()}
        for arm, v in raw.items():
            if v.get("table_reach"):
                data[str(arm)]["table_reach"] = [[float(r), float(z)] for r, z in v["table_reach"]]
                table(data[str(arm)])
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise CalibrationError(f"{path} is damaged ({e}); run `llm-harness calibrate` again") from e
    if not all(math.isfinite(v["table_z"]) for v in data.values()):
        raise CalibrationError(f"{path} holds a table height that is not a finite number; run `llm-harness "
                               "calibrate` again")
    return data


def save_calibration(robot_name: str, arm: str, table_z, setup: dict,
                     measured_at: Optional[float] = None) -> Path:
    """Store one arm's table height (a number or a ``Table``) and the setup it was measured with, keeping the other
    arms'."""
    table_obj = table_z if isinstance(table_z, Table) else None
    table_z = table_obj.z if table_obj else table_z
    if not math.isfinite(table_z):
        raise CalibrationError(f"table_z must be finite, got {table_z}")
    data = load_calibration(robot_name)
    data[arm] = {"table_z": float(table_z), "measured_at": time.time() if measured_at is None else measured_at,
                 "setup": json.loads(json.dumps(setup))}
    if table_obj is not None and table_obj.reach:
        data[arm]["table_reach"] = [[r, z] for r, z in table_obj.reach]
    path = calibration_path(robot_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return path


def table(entry: dict) -> Table:
    """The ``Table`` of one arm's stored calibration entry."""
    if entry.get("table_reach"):
        return Table(float(entry["table_z"]), tuple(tuple(row) for row in entry["table_reach"]))
    return Table(float(entry["table_z"]))
