"""The powered table calibration: an arm finds the table itself by a slow, guarded descent at a few reaches.

For each probe point the fingertips go above it with the gripper pointing down (as steeply as the arm can there,
jaws closed), then the fingertip target goes down STEP_M at a time at PROBE_SPEED_DEG_S. After each step, once its
command has ended, the measured fingertip height is read twice, STILL_S apart. Contact: over the last STILL_STEPS
steps the command went down while the measured fingertips moved less than STILL_M, they stand still, and they stand
more than CONTACT_M above their command beyond the offset they held while descending freely over the first
FREE_STEPS steps. Lag closes once a command has ended, so it never passes; a support does not. The table there is
the measured fingertip height. The descent never commands the fingertips lower than ``limit_z``, and the harness it
runs on has its table at ``limit_z``, so its own floor refuses anything lower. The arm starts at rest with its
motors cool and is rested at the end, also after a failure.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Sequence

import numpy as np

from .arm_model import Unreachable
from .harness import Harness
from .safety import REST_TOLERANCE_DEG, CalibrationError

STEP_M = 0.002               # fingertip target step
ABOVE_M = 0.06               # the descent starts this far above the expected table
BELOW_M = 0.04               # and never commands the fingertips lower than this below it
FREE_STEPS = 3               # steps that must descend freely to set the free offset
STILL_STEPS = 3              # contact: over this many steps ...
STILL_M = 0.001              # ... the measured fingertips moved less than this
CONTACT_M = 0.004            # ... and stand this far above their command beyond the free offset
FREE_FRACTION = 0.5          # free descent: the measured fingertips follow at least this fraction of the command
STILL_S = 0.3                # the second reading after a step, to see the fingertips stand still
PROBE_SPEED_DEG_S = 3.0
MAX_START_C = 50.0           # no probe starts with a motor this hot
PITCHES_DEG = (-90.0, -75.0, -60.0, -45.0)
DEFAULT_REACHES_M = (0.25, 0.32, 0.39)


def default_points(harness: Harness, arm: str, reaches: Sequence[float] = DEFAULT_REACHES_M) -> list[tuple]:
    """World (x, y) straight ahead of the arm base at ``reaches``."""
    model = harness.models[arm]
    base, heading = model.world_from_base[:2, 3], model.world_from_base[:2, 0]
    return [tuple(float(v) for v in base + r * heading) for r in reaches]


def probe_table(harness: Harness, arm: str, points: Sequence[tuple], expect_z: float, limit_z: float,
                say: Callable[[str], None] = print) -> list[tuple[float, float, list]]:
    """Find the table under ``arm`` at each world (x, y) of ``points``: ``[(reach_m, z_m, fingertip_xyz), ...]``.
    ``expect_z``: roughly where the table is (world z); ``limit_z``: the lowest the fingertips are ever sent. The
    harness's table height for the arm must be ``limit_z``. CalibrationError when the arm is not ready, a point
    has no pose, or no table is found before ``limit_z``; the arm rests in every case."""
    model = harness.models[arm]
    if harness.envelopes[arm].table_z is None or abs(harness.envelopes[arm].table_z - limit_z) > 1e-9:
        raise CalibrationError(f"{arm}: the harness's table must stand at the probe limit z = {limit_z:.3f} m")
    state = harness.backend.read()[arm]
    if state.fault:
        raise CalibrationError(f"{arm}: {state.fault}")
    if state.rest_q is None or float(np.max(np.abs(state.q - state.rest_q))) > math.radians(REST_TOLERANCE_DEG):
        raise CalibrationError(f"{arm}: the powered calibration starts only at the recorded rest pose (record it "
                               "with `llm-harness calibrate --rest`, or bring the arm to rest)")
    if state.temperature is not None and state.temperature >= MAX_START_C:
        raise CalibrationError(f"{arm}: a motor is at {state.temperature:.0f} C; let it cool below "
                               f"{MAX_START_C:.0f} C first")
    found = []
    try:
        _expect_done(harness.step({"action": "home", "arm": arm}), "home")
        _expect_done(harness.step({"action": "gripper", "opening": 0.0, "arm": arm}), "close the jaws")
        for x, y in points:
            found.append(_probe_point(harness, arm, float(x), float(y), expect_z, limit_z, say))
    finally:
        rested = harness.step({"action": "rest", "arm": arm}).result
        if not rested.ok:
            say(f"{arm}: rest after the probe: {rested.status}: {rested.message}")
    return found


def _probe_point(harness: Harness, arm: str, x: float, y: float, expect_z: float, limit_z: float,
                 say: Callable[[str], None]) -> tuple[float, float, list]:
    model = harness.models[arm]
    base = model.world_from_base[:2, 3]
    reach = float(math.hypot(x - base[0], y - base[1]))
    yaw = math.degrees(math.atan2(y - base[1], x - base[0]))
    pitch = _pitch(harness, arm, x, y, yaw, expect_z, limit_z)
    say(f"{arm}: probing at ({x:.3f}, {y:.3f}), {reach:.3f} m from the base, gripper pitched {pitch:.0f} deg")
    _reach_above(harness, arm, x, y, yaw, pitch, expect_z + ABOVE_M)
    target = expect_z + ABOVE_M
    lowest = _lowest_target(model, limit_z)
    rows: list[tuple[float, float, bool]] = []          # (commanded, measured fingertip z, standing still)
    while True:
        target -= STEP_M
        if target < lowest:
            raise CalibrationError(f"{arm}: no table found at ({x:.3f}, {y:.3f}) down to fingertip z = "
                                   f"{limit_z:.3f} m; give a lower --expect if the table is lower")
        result = harness.step(_move(model, x, y, target, pitch, yaw, arm, PROBE_SPEED_DEG_S)).result
        if result.status in ("fault", "rejected"):
            raise CalibrationError(f"{arm}: probe step at fingertip z = {target:.3f} m: {result.status}: "
                                   f"{result.message}")
        rows.append(_fingertips(harness, arm))
        offsets = [m - c for c, m, _ in rows]
        if len(rows) == FREE_STEPS:
            went = rows[0][1] - rows[-1][1]
            if went < FREE_FRACTION * STEP_M * (FREE_STEPS - 1):
                raise CalibrationError(f"{arm}: the fingertips did not descend freely at the start of the probe at "
                                       f"({x:.3f}, {y:.3f}) (they stand near z = {rows[-1][1]:.3f} m): the table is "
                                       "higher than expected; give a higher --expect")
        if len(rows) < FREE_STEPS + STILL_STEPS:
            continue
        free = float(np.median(offsets[:FREE_STEPS]))
        last = rows[-STILL_STEPS:]
        moved = max(m for _, m, _ in last) - min(m for _, m, _ in last)
        if moved < STILL_M and all(still for _, _, still in last) and offsets[-1] - free > CONTACT_M:
            z = float(np.mean([m for _, m, _ in last]))
            tip = harness.models[arm].fingertip(harness.backend.read()[arm].q)
            say(f"{arm}: table at z = {z:.4f} m, {reach:.3f} m from the base (the fingertips stopped "
                f"{(offsets[-1] - free) * 1000:.1f} mm above their command, free offset {free * 1000:.1f} mm)")
            _expect_done(harness.step(_move(model, x, y, expect_z + ABOVE_M, pitch, yaw, arm)), "lift")
            return reach, z, [round(float(v), 4) for v in tip]
        if result.status == "stopped":
            raise CalibrationError(f"{arm}: probe step at fingertip z = {target:.3f} m stopped without a clear "
                                   f"contact: {result.message}")


def _lowest_target(model, limit_z: float) -> float:
    """The lowest fingertip target: the envelope keeps the fingertips plus the gripper's margin above ``limit_z``."""
    return limit_z + model.gripper.margin_m + 0.002


def _fingertips(harness: Harness, arm: str) -> tuple[float, float, bool]:
    """(commanded, measured fingertip z, whether they stood still over STILL_S) once the step's command ended."""
    model = harness.models[arm]
    first = harness.backend.read()[arm]
    time.sleep(STILL_S)
    second = harness.backend.read()[arm]
    z1, z2 = float(model.fingertip(first.q)[2]), float(model.fingertip(second.q)[2])
    return float(model.fingertip(second.q_target)[2]), z2, abs(z2 - z1) < STILL_M / 2


def _move(model, x, y, tip_z, pitch, yaw, arm, speed: Optional[float] = None) -> dict:
    """move_tool putting the fingertip centre at (x, y, tip_z)."""
    p, w = math.radians(pitch), math.radians(yaw)
    pointing = np.array([math.cos(p) * math.cos(w), math.cos(p) * math.sin(w), math.sin(p)])
    tool = np.array([x, y, tip_z]) - model.gripper.finger_length_m * pointing
    action = {"action": "move_tool", "xyz": [float(v) for v in tool], "pitch_deg": pitch, "yaw_deg": yaw,
              "roll_deg": 0.0, "arm": arm}
    if speed is not None:
        action["speed_deg_s"] = speed
    return action


def _pitch(harness: Harness, arm: str, x, y, yaw, expect_z, limit_z) -> float:
    """The steepest pitch whose probe poses, from ABOVE_M down to ``limit_z``, are reachable with the fingertips the
    lowest point of the arm."""
    model = harness.models[arm]
    for pitch in PITCHES_DEG:
        top, bottom = (_move(model, x, y, z, pitch, yaw, arm)
                       for z in (expect_z + ABOVE_M, _lowest_target(model, limit_z)))
        try:
            q = model.solve_pose(model.home_q, model.pose_from_angles(bottom["xyz"], math.radians(pitch),
                                                                       math.radians(yaw), 0.0))
        except Unreachable:
            continue
        if float(model.lowest_z(q)) < float(model.fingertip(q)[2]) - model.gripper.margin_m - 1e-3:
            continue
        checks = harness.reachable([_pose(top), _pose(bottom)], arm=arm, chain=True, start=_pose(top))
        if all(c["ok"] or "per-action" in (c["why"] or "") for c in checks):
            return pitch
    raise CalibrationError(f"{arm}: no gripper pitch reaches down to the table at ({x:.3f}, {y:.3f}) with the "
                           "fingertips lowest; choose another point")


def _reach_above(harness: Harness, arm: str, x, y, yaw, pitch, tip_z) -> None:
    """From where the tool is to the fingertips at (x, y, tip_z), in as few moves as the per-action limit allows."""
    model = harness.models[arm]
    goal = _move(model, x, y, tip_z, pitch, yaw, arm)
    start = harness.state().arms[arm].tool
    for parts in range(1, 7):
        poses = []
        for k in range(1, parts + 1):
            f = k / parts
            poses.append({"xyz": [s + f * (g - s) for s, g in zip(start.xyz, goal["xyz"])],
                          "pitch": start.pitch_deg + f * (pitch - start.pitch_deg),
                          "yaw": start.yaw_deg + f * (yaw - start.yaw_deg)})
        if all(c["ok"] for c in harness.reachable(poses, arm=arm, chain=True, line=False)):
            for pose in poses:
                _expect_done(harness.step({"action": "move_tool", "xyz": pose["xyz"], "pitch_deg": pose["pitch"],
                                           "yaw_deg": pose["yaw"], "roll_deg": 0.0, "line": False, "arm": arm}),
                             "the move above the probe point")
            return
    raise CalibrationError(f"{arm}: no way found from home to above the probe point ({x:.3f}, {y:.3f})")


def _pose(action: dict) -> dict:
    return {"xyz": action["xyz"], "pitch": action["pitch_deg"], "yaw": action["yaw_deg"], "roll": 0.0}


def _expect_done(obs, what: str) -> None:
    if not obs.result.ok:
        raise CalibrationError(f"{what}: {obs.result.status}: {obs.result.message}")
