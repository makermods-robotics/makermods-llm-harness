"""Real arms and cameras through the arm SDK (``metal_arm`` or ``maker_arm``).

Opening the backend connects every arm (motors found, torque left as it is) and opens the cameras, each in a helper
process (``camera_process``); an arm gets torque only from ``enable``, or with ``attach`` an arm found holding with
torque on (left by an earlier session) is taken over where it holds. A camera that fails to open is left out and its error kept in
``camera_errors``; the arms stay usable.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from typing import Mapping, Optional

import numpy as np

from .base import ArmInfo, ArmState, ArmTarget, Backend, Frame
from .camera_process import CameraProcess

log = logging.getLogger(__name__)
ENABLE_SPEED = math.radians(5.0)   # rad/s until the first target: a joint resting past a limit comes inside slowly
REFRESH_S = 0.1                    # an arm that is off is asked for fresh feedback at most this often
HELD_TORQUE_NM = 0.1               # a motor that cannot report torque state but measures this much load is energized


class RealBackend(Backend):
    """``arm_specs`` maps arm names to the fields of an SDK rig arm entry (``backend``, ``port``, ...),
    ``camera_ids`` camera names to SDK camera ids, ``rest_q`` arm names to their recorded rest pose (rad).
    ``arms`` and ``cameras`` replace what would be built from the specs (tests pass simulated arms and fake
    cameras); they are connected and opened all the same. ``ask_operator``, when set, is asked at close about
    arms still holding: ``ask_operator({arm: recovery text})`` returns True to turn their torque off there."""

    def __init__(self, arm_type: str, arm_specs: Mapping[str, dict], camera_ids: Mapping[str, str],
                 infos: list[ArmInfo], *, rest_q: Optional[Mapping[str, object]] = None,
                 arms: Optional[Mapping[str, object]] = None, cameras: Optional[Mapping[str, object]] = None,
                 camera_max_age_s: float = 1.0, attach: bool = False):
        rig = importlib.import_module(f"{arm_type}_arm.rig")
        self.cli = f"{arm_type}-arm"
        self._infos = list(infos)
        names = [info.name for info in self._infos]
        parallel = True
        if arms is None:
            specs = {name: rig.ArmSpec(**arm_specs[name]) for name in names}
            arms = {name: spec.build() for name, spec in specs.items()}
            parallel = len({(s.backend, s.address) for s in specs.values()}) == len(specs)
        missing = [name for name in names if name not in arms]
        if missing:
            raise ValueError(f"no arm given for {', '.join(missing)}")
        if cameras is None:
            cameras = {name: CameraProcess(f"{arm_type}_arm.cameras", cid) for name, cid in camera_ids.items()}
        self.group = rig.ArmGroup({name: arms[name] for name in names}, parallel=parallel)
        self.rest_q = {name: np.array(q, float) for name, q in (rest_q or {}).items()}
        self.camera_max_age_s = camera_max_age_s
        self.camera_errors: dict[str, str] = {}
        self._camera_names = list(cameras)
        self._cameras: dict[str, object] = {}
        self._opening: dict[str, float] = {}       # last commanded gripper opening since enable
        self._refused: dict[str, str] = {}         # a refused target, reported until a hold or an accepted target
        self._off: dict[str, tuple[float, Optional[bool], list[str]]] = {}   # arm off: (read at, torque, stale)
        self._released: set[str] = set()           # torque off confirmed by this session's own release
        self.closed_as: dict[str, str] = {}        # per arm after close()
        self._group_of = rig.ArmGroup
        self._deadline: Optional[float] = None
        self._closed = False
        self._attach_errors: dict[str, str] = {}
        self._loaded: dict[str, list[str]] = {}   # arm off whose motors cannot report torque state but carry load
        self.ask_operator = None
        self.group.connect(clear_released=True)
        if attach:
            self._attach_energized()
        for name, camera in cameras.items():
            try:
                camera.open()
                self._cameras[name] = camera
            except Exception as e:
                self.camera_errors[name] = f"camera {name} did not open: {e}"

    @classmethod
    def from_robot(cls, robot, **kwargs) -> "RealBackend":
        """The backend for a robot description with ``arm``, ``arms``, ``cameras``, ``rest_q`` (degrees) and
        ``arm_infos()``."""
        rest = {name: np.radians(q) for name, q in (getattr(robot, "rest_q", None) or {}).items()}
        return cls(robot.arm, robot.arms, robot.cameras or {}, robot.arm_infos(), rest_q=rest, **kwargs)

    def _attach_energized(self) -> None:
        """Take over every arm whose motors report torque on, or that cannot report it but carry load, where it
        holds."""
        for name, arm in self.group:
            arm.refresh()
            joints = arm.status()["joints"]
            if not any(j["enabled"] for j in joints.values()) and not _loaded(joints):
                continue
            try:
                arm.attach(speed=ENABLE_SPEED)
            except Exception as e:
                self._attach_errors[name] = f"could not be taken over ({e}); it holds as it was. {self.recovery(name)}"
                log.error("%s: %s", name, self._attach_errors[name])

    def recovery(self, name: str) -> str:
        """How the operator brings an arm left holding down to its rest pose and turns its torque off."""
        pose = ""
        if name in self.rest_q:
            joints = self.group[name].joint_names
            pose = " --pose " + ",".join(f"{j}={math.degrees(v):.1f}" for j, v in zip(joints, self.rest_q[name]))
        return (f"Bring it down with `{self.cli} rest --arm {name}{pose}` once this session has ended "
                f"(`{self.cli} release` turns torque off where it is: the arm drops).")

    @property
    def arms(self) -> list[ArmInfo]:
        return list(self._infos)

    @property
    def cameras(self) -> list[str]:
        return list(self._camera_names)

    def enable(self, arm: str, cancel: Optional[threading.Event] = None, speed: Optional[float] = None) -> None:
        a = self.group[arm]
        if a.state.name != "ENABLED":
            try:
                a.enable(cancel=cancel, speed=ENABLE_SPEED if speed is None else min(ENABLE_SPEED, speed))
            except Exception as e:
                if type(e).__name__ != "HeldByAnotherProgram":
                    raise
                raise RuntimeError(f"another program holds this arm ({e}). Nothing was sent to it: restart the "
                                   f"session with `llm-harness serve --attach` to take it over and rest it. "
                                   f"{self.recovery(arm)}") from e
        if arm in self.rest_q:
            a.rest_pose = list(self.rest_q[arm]) + list((a.rest_pose or a.get_joint_positions())[6:])   # the SDK's own return goes there too
        self._released.discard(arm)
        self._off.pop(arm, None)
        self._opening.pop(arm, None)
        self._refused.pop(arm, None)

    def read(self) -> dict[str, ArmState]:
        return {name: self._read(name, arm) for name, arm in self.group}

    def _read(self, name: str, arm) -> ArmState:
        state = arm.state.name
        torque: Optional[bool] = state in ("ENABLED", "FAULT")
        stale: list[str] = []
        if state == "CONNECTED":
            torque, stale = self._torque_off(name, arm)
        positions = arm.get_joint_positions()
        q, gripper = np.array(positions[:6], float), float(arm.gripper_opening())
        holding = state in ("ENABLED", "FAULT")
        commanded = arm.get_commanded_positions() if holding else None
        q_target = np.array(commanded[:6], float) if holding else q.copy()
        temps = [t for t in arm.get_temperatures() if math.isfinite(t)]
        no_feedback = [label for label, p in zip(arm.joint_names, positions) if not math.isfinite(p)]
        fault = (arm.fault_reason or (f"arm is in {state}" if state == "FAULT" else None)
                 or (f"no fresh feedback from {', '.join(stale or no_feedback)} (no reply, or a motor fault)"
                     if stale or no_feedback else None)
                 or self._attach_errors.get(name)
                 or ((f"its motors carry load ({', '.join(self._loaded[name])}) and their firmware does not report "
                      "torque state: another program probably holds it. " if name in self._loaded else
                      "its motors report torque on, but not from this session (left on by another program): ")
                     + "restart the session with `llm-harness serve --attach` to take it over and rest it. "
                     + self.recovery(name) if state == "CONNECTED" and torque else None)
                 or self._refused.get(name))
        rest = self.rest_q.get(name)
        return ArmState(q=q, gripper=gripper, q_target=q_target,
                        gripper_target=self._commanded_opening(arm, commanded[6]) if holding else gripper,
                        enabled=torque, fault=fault, temperature=max(temps) if temps else None,
                        rest_q=None if rest is None else rest.copy())

    @staticmethod
    def _commanded_opening(arm, position: float) -> float:
        """The opening (0 closed .. 1 open) the SDK's commanded gripper position stands for, by the SDK's own
        linear mapping (``gripper_target``)."""
        closed, wide = arm.gripper_target(0.0), arm.gripper_target(1.0)
        return float((position - closed) / (wide - closed)) if wide != closed else 0.0

    def _torque_off(self, name: str, arm) -> tuple[Optional[bool], list[str]]:
        """(torque on: True / False / None when not known, joints without fresh feedback) of an arm that is
        off, from fresh replies at most REFRESH_S old."""
        now = time.monotonic()
        cached = self._off.get(name)
        if cached is None or now - cached[0] >= REFRESH_S:
            replied = arm.refresh()
            stale = [label for label, ok in zip(arm.joint_names, replied) if not ok]
            joints = arm.status()["joints"]
            reported = [j["enabled"] for j in joints.values()]
            torque = True if any(on is True for on in reported) else (
                False if not stale and all(on is False for on in reported) else None)
            if torque is None and not stale and name in self._released:
                torque = False
            loaded = _loaded(joints) if torque is None else []
            self._loaded.pop(name, None)
            if loaded:
                torque, self._loaded[name] = True, loaded
            cached = self._off[name] = (now, torque, stale)
        return cached[1], cached[2]

    def step(self, targets: Mapping[str, ArmTarget], dt: float) -> None:
        for name, target in targets.items():
            arm = self.group[name]
            command = np.asarray(target.q, float).tolist() + [arm.gripper_target(target.gripper)]
            if arm.move_to(command, speed=target.speed, wait=False):
                self._opening[name] = min(1.0, max(0.0, float(target.gripper)))
                self._refused.pop(name, None)
            else:
                self._refused[name] = f"targets refused: arm is {arm.state.name}"
        self._sleep(dt)

    def feedback_ages(self, arm: str) -> list[float]:
        """Per joint, from the SDK's public status: seconds since its last feedback (inf when it has none)."""
        joints = self.group[arm].status()["joints"].values()
        return [math.inf if j.get("feedback_age") is None else float(j["feedback_age"]) for j in joints]

    @property
    def hold_settle_s(self) -> float:
        """Two control periods of the slowest arm: the SDK's public API shows no tick counter, so a hold counts
        as sent once two periods have passed."""
        return 2.0 / min(float(a.config.control_rate_hz) for _, a in self.group)

    def hold(self, arm: str) -> tuple[np.ndarray, float]:
        """Freeze the arm at its measured pose; RuntimeError when the SDK refuses (the arm is not on). Returns the
        setpoint the SDK took (its commanded positions right after)."""
        a = self.group[arm]
        if not a.hold_current_position():
            raise RuntimeError(f"{arm}: the SDK refused the hold (arm is {a.state.name})")
        commanded = a.get_commanded_positions()
        self._opening[arm] = min(1.0, max(0.0, float(a.gripper_opening())))
        self._refused.pop(arm, None)
        return np.array(commanded[:6], float), self._commanded_opening(a, commanded[6])

    def _sleep(self, dt: float) -> None:
        """Sleep until one period after the previous deadline; restart the schedule when more than a period late."""
        if dt <= 0:
            return
        now = time.monotonic()
        if self._deadline is None or now - self._deadline > dt:
            self._deadline = now
        self._deadline += dt
        time.sleep(max(0.0, self._deadline - time.monotonic()))

    def images(self) -> dict[str, Frame]:
        frames = {}
        for name, camera in self._cameras.items():
            newest = camera.newest(self.camera_max_age_s)
            if newest is not None:
                image, stamp = newest
                frames[name] = Frame(np.ascontiguousarray(image[:, :, ::-1]), float(stamp),
                                     getattr(newest, "captured", None))
        return frames

    def rest(self, arm: str) -> str:
        """Torque off where the arm stands (the harness brought it to rest): "released" once every motor
        confirmed it, "holding" for an arm in a fault (not released), "unknown" when a motor did not confirm."""
        a = self.group[arm]
        if a.state.name == "FAULT":
            a.hold_current_position()
            return "holding"
        if a.state.name != "ENABLED":
            return "released" if arm in self._released else "unknown"
        a.disable()
        self._opening.pop(arm, None)
        self._off.pop(arm, None)
        if a.unconfirmed_disable:
            return "unknown"
        self._released.add(arm)
        return "released"

    def close(self) -> dict[str, str]:
        """Disconnect without moving anything; returns each arm's torque outcome. An arm still on is left
        holding its pose: "detached" (bus closed, motors holding their last command) or "holding" (a stiff
        joint would drop on its motor watchdog, so its control loop keeps running while this process does)."""
        if self._closed:
            return dict(self.closed_as)
        self._closed = True
        on = [name for name, arm in self.group if arm.state.name in ("ENABLED", "FAULT")]
        release = False
        if on and self.ask_operator is not None:
            for name in on:
                self.group[name].hold_current_position()
            try:
                release = bool(self.ask_operator({name: self.recovery(name) for name in on}))
            except Exception as e:
                log.error("asking the operator failed: %s", e)
        off = {}
        for name, arm in self.group:
            if arm.state.name not in ("ENABLED", "FAULT"):
                off[name] = arm
                continue
            if release:
                arm.disconnect()
                self.closed_as[name] = "unknown" if arm.unconfirmed_disable else "released_unrested"
                continue
            try:
                arm.hold_current_position()
                if arm.detach_unsafe():
                    self.closed_as[name] = "holding"
                else:
                    arm.detach()
                    self.closed_as[name] = "detached"
            except Exception as e:
                self.closed_as[name] = f"error: {e}"
            log.error("%s: left holding its pose with torque on (%s). %s", name, self.closed_as[name],
                      self.recovery(name))
        outcomes = self._group_of(off).close()
        for name, outcome in outcomes.items():
            self.closed_as[name] = "released" if outcome == "unknown" and name in self._released else outcome
        for camera in self._cameras.values():
            camera.close()
        return dict(self.closed_as)


def _loaded(joints: dict) -> list[str]:
    """The joints (from an SDK status) whose motors cannot report torque state but measure more than
    HELD_TORQUE_NM: energized, held by whatever program last enabled them."""
    return [f"{name} {j['torque']:+.2f} N*m" for name, j in joints.items()
            if j.get("enabled") is None and j.get("torque") is not None and abs(j["torque"]) > HELD_TORQUE_NM]
