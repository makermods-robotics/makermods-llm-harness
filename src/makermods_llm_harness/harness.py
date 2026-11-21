"""The harness: actions in, observations out, with every arm motion planned and checked by the safety
envelope before any of it is sent, and watched tick by tick while it runs.

    with Harness.open("my-robot") as h:
        print(h.describe())
        obs = h.step({"action": "move_tool", "arm": "left", "xyz": [0.3, 0.2, 0.1], "pitch_deg": -90})
        print(obs.text())
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from .actions import (Action, ActionError, ArmObservation, CameraImage, Gripper, Home, MoveJoints, MoveTool, Nudge,
                      Observation, Observe, Rest, Result, ToolPose, Wait, action_to_dict, parse_action)
from .arm_model import ArmModel, Unreachable, pose_error
from .backends.base import OFF, ArmState, ArmTarget, Backend
from .log import EpisodeLog
from .robot import Robot, load as load_robot
from .safety import REST_TOLERANCE_DEG, Envelope, KeepOut, MoveRejected, SafetyConfig, load_calibration

SETTLE_S = 1.0                     # longest wait for the joints to reach the goal after the last waypoint
SETTLED = math.radians(1.0)
ARRIVAL_TOL_M = 0.010              # a tool move is done only with the measured tool pose this close to the target ...
ARRIVAL_TOL_DEG = 3.0              # ... and within this angle of it
COMMAND_MATCH = 1e-3               # rad: the arm still follows the last path's end when its command is this close
GRIPPER_S = 2.0                    # longest gripper action
GRIPPER_STILL = 0.005              # opening change per tick that counts as stopped
NEAR_BASE_AXIS_M = 0.02            # closer than this to the base axis, move_tool keeps the current yaw
FRESH_FRAME_S = 0.5                # longest wait after an action for camera frames captured after it ended
JPEG_QUALITY = 85
SEVERITY = ("done", "rejected", "stopped", "fault")


class HarnessClosed(RuntimeError):
    """The harness is closed: it takes no more actions or observations."""


class _Outcome(Exception):
    """Ends an action early with this result."""

    def __init__(self, result: Result):
        super().__init__(result.message)
        self.result = result


class Harness:
    """Drives a backend from actions. ``step`` and ``observe`` may be called from several threads (they run
    one at a time); ``stop`` may be called at any time to end the action in progress."""

    def __init__(self, backend: Backend, *, robot_name: str = "robot", table_z: Optional[dict] = None,
                 safety: Optional[SafetyConfig] = None, log: Optional[EpisodeLog] = None,
                 image_dir: Union[str, Path, None] = None, read_only: bool = False,
                 keep_out: Sequence[KeepOut] = ()):
        self.backend = backend
        self.robot_name = robot_name
        self.config = safety or SafetyConfig()
        self.models = {info.name: ArmModel(info, self.config.gripper(info.type)) for info in backend.arms}
        table_z = table_z or {}
        self.keep_out = list(keep_out)
        self.envelopes = {n: Envelope(m, self.config, table_z.get(n), self.keep_out) for n, m in self.models.items()}
        self.cameras = list(backend.cameras)
        self.log = log
        self.image_dir = Path(image_dir).expanduser().resolve() if image_dir is not None \
            else (log.image_dir if log else None)
        if self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.steps = log.steps if log else 0
        self._looks = len(list(self.image_dir.glob("look-*"))) if self.image_dir is not None else 0
        self._stops = 0
        self._dt = 1.0 / self.config.rate_hz
        self._lock = threading.Lock()
        self._abort: Optional[threading.Event] = None     # set to end the action in progress
        self._released: dict[str, str] = {}               # backend.rest() outcome per arm, read by close()
        self._commanded: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # tool pose and joints of the last done motion
        self._closed = False
        if log:
            log.start(robot_name, list(self.models), self.cameras,
                      {n: e.table_z for n, e in self.envelopes.items()}, self.config.to_dict())

    @classmethod
    def open(cls, robot: Union[str, Path, Robot, None] = None, *, backend: Optional[Backend] = None,
             table_z: Union[float, dict, None] = None, safety: Optional[SafetyConfig] = None,
             log: Optional[EpisodeLog] = None, image_dir: Union[str, Path, None] = None,
             read_only: bool = False, backend_options: Optional[dict] = None,
             keep_out: Optional[Sequence[KeepOut]] = None) -> "Harness":
        """A harness for a robot (name, file, Robot, or the only robot defined) on its own backend, or on
        ``backend``. Table heights come from ``table_z`` (one for all arms or per arm), else the backend
        (simulators), else the robot's stored calibration when it was measured on the same setup.
        ``backend_options`` go to the robot's backend when it is opened here. Keep-out zones come from
        ``keep_out``, else the robot's."""
        if (robot is not None and not isinstance(robot, Robot)) or (robot is None and backend is None):
            robot = load_robot(robot)
        if safety is None:
            safety = SafetyConfig.from_dict(robot.safety) if robot else SafetyConfig()
        opened = backend is None
        backend = backend or robot.open_backend(**(backend_options or {}))
        try:
            names = [info.name for info in backend.arms]
            if backend.table_z is not None:
                heights = {n: backend.table_z for n in names}
            else:
                stored = load_calibration(robot.name) if robot else {}
                heights = {n: stored[n]["table_z"] for n in names
                           if n in stored and stored[n]["setup"] == robot.setup(n, safety)}
            if isinstance(table_z, dict):
                heights.update(table_z)
            elif table_z is not None:
                heights = {n: float(table_z) for n in names}
            if keep_out is None:
                keep_out = robot.zones() if robot else ()
            return cls(backend, robot_name=robot.name if robot else "robot", table_z=heights, safety=safety,
                       log=log, image_dir=image_dir, read_only=read_only, keep_out=keep_out)
        except BaseException:
            if opened:
                backend.close()
            raise

    # ── public API ──
    def observe(self) -> Observation:
        with self._lock:
            if self._closed:
                raise HarnessClosed("the harness is closed")
            self._looks += 1
            return self._observe(Result(), f"look-{self._looks:04d}")

    def step(self, action: Union[Action, dict, str], cancel: Optional[threading.Event] = None) -> Observation:
        """Run one action and observe. A malformed, refused or failed action comes back in ``result``. Setting
        ``cancel`` ends this action (the arm holds where it is), as ``stop()`` does."""
        stops = self._stops
        with self._lock:
            if self._closed:
                raise HarnessClosed("the harness is closed")
            self._abort = abort = cancel if cancel is not None else threading.Event()
            if stops != self._stops:            # stop() also ends the actions that were waiting
                abort.set()
            try:
                try:
                    act = parse_action(action)
                except ActionError as e:
                    result = Result("rejected", str(e), action if isinstance(action, dict) else None)
                else:
                    result = Result("stopped", "stopped on request before it started.") if abort.is_set() \
                        else self._logged(act)
                    result.action = action_to_dict(act)
                self.steps += 1
                obs = self._observe(result, f"step-{self.steps:04d}", fresh_after=time.monotonic())
                if self.log:
                    try:
                        self.log.step(result.action, obs.to_dict())
                    except (OSError, ValueError) as e:
                        result.message = f"{result.message} (not recorded in the episode log: {e})".lstrip()
                return obs
            finally:
                self._abort = None

    def rest(self, arm: Optional[str] = None) -> Observation:
        return self.step(Rest(arm))

    def stop(self) -> None:
        """End the action in progress and the ones waiting behind it; the arm holds where it is."""
        self._stops += 1
        abort = self._abort
        if abort is not None:
            abort.set()

    def close(self) -> dict[str, str]:
        """Rest every arm that is on (the same checked motion as the rest action), free the backend and close
        the log. Returns each arm's torque outcome: "released" or "always_on" when nothing holds torque any more,
        anything else (with the reason) when the arm may still be on. Later calls return {}."""
        with self._lock:
            if self._closed:
                return {}
            self._closed = True
            self._abort = threading.Event()
            words, reasons = {}, {}
            try:
                for name in self.models:
                    result, words[name] = self._rest(name)
                    reasons[name] = "" if result.ok else result.message
            finally:
                try:
                    try:
                        closed = self.backend.close() or {}
                    except Exception as e:
                        closed = {name: f"error: {e}" for name in self.models}
                    words.update({n: w for n, w in closed.items() if n in self.models})
                finally:
                    if self.log:
                        self.log.close()
            return {n: w if w in OFF or not reasons.get(n) else f"{w}: {reasons[n]}" for n, w in words.items()}

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── actions ──
    def _logged(self, act: Action) -> Result:
        """The action, recorded in the episode log before anything moves."""
        if self.log:
            try:
                self.log.write("action", step=self.steps + 1, action=action_to_dict(act))
            except (OSError, ValueError) as e:
                return Result("rejected", f"the episode log could not be written ({e}); nothing moved.")
        return self._guarded(lambda: self._dispatch(act))

    @staticmethod
    def _guarded(run) -> Result:
        try:
            return run()
        except (MoveRejected, Unreachable) as e:
            return Result("rejected", str(e))
        except _Outcome as e:
            return e.result
        except Exception as e:   # a backend failure mid-action
            return Result("fault", f"{type(e).__name__}: {e}")

    def _dispatch(self, act: Action) -> Result:
        if isinstance(act, Observe):
            return Result()
        if isinstance(act, Wait):
            return self._wait(act.seconds)
        if self.read_only:
            return Result("rejected", "this session is read-only: only observe and wait are allowed.")
        names = self._arm_names(act)
        if isinstance(act, Rest):
            return _combine({n: self._rest(n)[0] for n in names})
        name = names[0]
        model, env = self.models[name], self.envelopes[name]
        env.require_table()
        state = self._usable(name)
        if isinstance(act, Gripper):
            return self._gripper(name, self._ready(name, state), act.opening)
        if not state.enabled:
            # Planned and checked first (from where enabling leaves it), so a refused request never turns it on.
            self._checked_path(name, act, np.clip(state.q, model.lo, model.hi))
            state = self._ready(name, state)
        path, speed, goal = self._checked_path(name, act, state.q, self._commanded_pose(name, state))
        self._commanded.pop(name, None)
        result = self._execute(name, state.q, path, speed, state.gripper_target)
        if goal is not None:
            result = self._arrival(name, goal, result)
        if result.ok:
            self._commanded[name] = (model.tool_pose(path[-1]) if goal is None else goal, path[-1])
        return result

    def _commanded_pose(self, name: str, state: ArmState) -> Optional[np.ndarray]:
        """The tool pose the last done motion ended on, while the arm still follows that command and stands
        within the arrival tolerance of it; else None."""
        held = self._commanded.get(name)
        if held is None or not np.allclose(state.q_target, held[1], atol=COMMAND_MATCH):
            return None
        position, angle = pose_error(self.models[name].tool_pose(state.q), held[0])
        return held[0] if position <= ARRIVAL_TOL_M and math.degrees(angle) <= ARRIVAL_TOL_DEG else None

    def _arrival(self, name: str, goal: np.ndarray, result: Result) -> Result:
        """A tool move's result with how far the measured tool pose ended from ``goal``; done only within the
        arrival tolerance."""
        position, angle = pose_error(self.models[name].tool_pose(self.backend.read()[name].q), goal)
        result.error = {"position_m": round(position, 4), "angle_deg": round(math.degrees(angle), 1)}
        if result.ok and (position > ARRIVAL_TOL_M or math.degrees(angle) > ARRIVAL_TOL_DEG):
            return Result("stopped", f"{name}: ended {position * 1000:.0f} mm and {math.degrees(angle):.1f} deg from "
                                     "the target. The arm holds where it is; look at the cameras.", error=result.error)
        return result

    def _arm_names(self, act: Action) -> list[str]:
        names = list(self.models)
        if act.arm is None:
            if len(names) == 1 or isinstance(act, Rest):
                return names
            raise MoveRejected(f"name the arm: {' or '.join(names)}.")
        if act.arm not in self.models:
            raise MoveRejected(f"no arm named {act.arm!r}; the arms are {', '.join(names)}.")
        return [act.arm]

    def _usable(self, name: str) -> ArmState:
        """The arm's state; refuses it while any arm reports a fault, or when it is hot or has no feedback."""
        states = self.backend.read()
        for other in [name] + [n for n in states if n != name]:
            if states[other].fault:
                also = "" if other == name else f" No arm moves while another reports a fault; rest {name} if needed."
                raise _Outcome(Result("fault", f"{other}: {states[other].fault}.{also}"))
        state = states[name]
        if not np.all(np.isfinite(state.q)) or not np.all(np.isfinite(state.q_target)):
            raise _Outcome(Result("fault", f"{name}: no joint feedback."))
        if state.temperature is not None and state.temperature > self.config.max_temp_c:
            raise MoveRejected(f"{name}: a motor is at {state.temperature:.0f} C, over the "
                               f"{self.config.max_temp_c:.0f} C limit. Wait for it to cool, then try again.")
        return state

    def _ready(self, name: str, state: ArmState) -> ArmState:
        """The arm's state with its motors on. An arm that is off is turned on only at its rest pose; a joint
        resting past a limit is then brought inside it along a checked path."""
        if state.enabled:
            return state
        model, env = self.models[name], self.envelopes[name]
        if state.rest_q is None:
            raise MoveRejected(f"{name}: no rest pose is recorded for this arm, so its motors stay off. The operator "
                               "records it with `llm-harness calibrate --rest`.")
        off = np.abs(state.q - state.rest_q)
        j = int(np.argmax(off))
        if off[j] > math.radians(REST_TOLERANCE_DEG):
            raise MoveRejected(f"{name}: the motors turn on only at the rest pose, and {model.joint_names[j]} is "
                               f"{math.degrees(off[j]):.0f} deg from it (limit {REST_TOLERANCE_DEG:g} deg). The "
                               "operator must fold the arm to its rest pose.")
        inside = np.clip(state.q, model.lo, model.hi)
        ramp = None
        if not np.array_equal(inside, state.q):
            ramp = model.plan_joint_path(state.q, inside, env.slow_speed, self.config.rate_hz)
            env.check(state.q, ramp, capped=False)
        if self._abort.is_set():
            raise _Outcome(Result("stopped", f"{name}: stopped on request before its motors came on."))
        try:
            self.backend.enable(name, cancel=self._abort)
        except Exception as e:
            if self._abort.is_set():
                raise _Outcome(Result("stopped", f"{name}: stopped on request while its motors came on.")) from e
            raise _Outcome(Result("fault", f"{name}: the motors could not be enabled: {e}")) from e
        if ramp is not None:
            result = self._execute(name, state.q, ramp, env.slow_speed, state.gripper)
            if not result.ok:
                raise _Outcome(result)
        return self._usable(name)

    def _checked_path(self, name: str, act: Action, q0: np.ndarray, commanded: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, float, Optional[np.ndarray]]:
        """The joint path of a motion action from ``q0``, passed by the envelope, its speed limit and, for a tool
        move, the requested tool pose. A nudge starts from the ``commanded`` tool pose when there is one."""
        model, env = self.models[name], self.envelopes[name]
        goal = None
        if isinstance(act, (MoveTool, Nudge)):
            goal = self._tool_goal(model, q0, act, commanded)
            path = self._plan(env, model.plan_tool_path, q0, goal)
        else:
            q1 = model.home_q if isinstance(act, Home) else self._joint_goal(model, q0, act)
            path = self._plan(env, model.plan_joint_path, q0, q1)
        env.check(q0, path, capped=not isinstance(act, Home))
        return path, env.speed_for(q0, path), goal

    def _plan(self, env: Envelope, plan, q0: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """``plan(q0, goal, speed, rate)`` at full speed, again at the slow speed when it comes near the table."""
        path = plan(q0, goal, env.max_speed, self.config.rate_hz)
        if env.speed_for(q0, path) < env.max_speed:
            path = plan(q0, goal, env.slow_speed, self.config.rate_hz)
        return path

    @staticmethod
    def _tool_goal(model: ArmModel, q0: np.ndarray, act: Union[MoveTool, Nudge],
                   commanded: Optional[np.ndarray] = None) -> np.ndarray:
        pose = model.tool_pose(q0)
        if isinstance(act, Nudge):
            pose = pose if commanded is None else commanded.copy()
            axes = pose[:3, :3] if act.frame == "tool" else model.world_from_base[:3, :3]
            pose[:3, 3] += axes @ np.asarray(act.dxyz, dtype=float)
            return pose
        xyz = np.asarray(act.xyz, dtype=float)
        _, pitch, yaw, roll = model.tool_angles(q0)
        away = xyz[:2] - model.world_from_base[:2, 3]
        heading = yaw
        if act.yaw_deg is not None:
            heading = math.radians(act.yaw_deg)
        elif math.hypot(*away) >= NEAR_BASE_AXIS_M:
            heading = math.atan2(away[1], away[0])
        if act.pitch_deg is None and act.roll_deg is None:
            # The gripper keeps its exact tilt and roll, turned about the vertical by the change of heading.
            c, s = math.cos(heading - yaw), math.sin(heading - yaw)
            pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ pose[:3, :3]
            pose[:3, 3] = xyz
            return pose
        pitch = pitch if act.pitch_deg is None else math.radians(act.pitch_deg)
        roll = roll if act.roll_deg is None else math.radians(act.roll_deg)
        return model.pose_from_angles(xyz, pitch, heading, roll)

    @staticmethod
    def _joint_goal(model: ArmModel, q0: np.ndarray, act: MoveJoints) -> np.ndarray:
        q1 = q0.copy()
        for joint, deg in act.joints_deg.items():
            if joint == model.gripper_name:
                raise MoveRejected(f"{joint} is the gripper: use the gripper action (opening 0 closed .. 1 open).")
            if joint not in model.joint_names:
                raise MoveRejected(f"{model.name} has no joint {joint!r}; its joints are "
                                   f"{', '.join(model.joint_names)}.")
            i = model.joint_names.index(joint)
            value = q0[i] + math.radians(deg) if act.relative else math.radians(deg)
            if not model.lo[i] <= value <= model.hi[i]:
                raise MoveRejected(f"{model.name}: {joint} at {math.degrees(value):.1f} deg is outside its range "
                                   f"[{math.degrees(model.lo[i]):.0f}, {math.degrees(model.hi[i]):.0f}] deg.")
            q1[i] = value
        return q1

    def _execute(self, name: str, q0: np.ndarray, path: np.ndarray, speed: float, gripper: float,
                 rest: bool = False, floor: Optional[float] = None) -> Result:
        """Stream a checked path from the measured ``q0``, watching every tick; then wait for the joints to
        arrive. Anything that ends it early holds the arm where it is. ``rest`` lets a hot arm go on; ``floor``
        lowers the measured clearance it is held under (the way down to a rest pose under the floor)."""
        floor = min(self.envelopes[name].hold_floor(q0), math.inf if floor is None else floor)
        zones = self.envelopes[name].keep_out_hold(q0, path[-1] if rest else None)
        ext = np.vstack([q0, path])
        state = None
        for i, q in enumerate(path):
            if self._abort.is_set():
                return self._hold(name, f"{name}: stopped on request; the arm holds where it is.")
            self.backend.step({name: ArmTarget(q, gripper, speed)}, self._dt)
            state = self._watch(name, ext[max(0, i - 1):i + 2], floor, rest, zones)
        goal = path[-1]
        for _ in range(int(SETTLE_S * self.config.rate_hz)):
            if np.max(np.abs(state.q - goal)) <= SETTLED:
                return Result()
            if self._abort.is_set():
                return self._hold(name, f"{name}: stopped on request; the arm holds where it is.")
            self.backend.step({name: ArmTarget(goal, gripper, speed)}, self._dt)
            state = self._watch(name, goal[None], floor, rest, zones)
        off = np.abs(state.q - goal)
        j = int(np.argmax(off))
        if off[j] <= self.envelopes[name].tolerance:
            return Result()
        return self._hold(name, f"{name}: {self.models[name].joint_names[j]} stopped {math.degrees(off[j]):.1f} deg "
                                "short of the goal. The arm holds where it is; look at the cameras.")

    def _watch(self, name: str, window: Optional[np.ndarray] = None, floor: Optional[float] = None,
               rest: bool = False, zones: Optional[np.ndarray] = None) -> ArmState:
        """The state of the moving arm, after checking every arm for faults and the moving one for heat, its
        distance from the planned ``window`` of waypoints, its measured clearance and its distance from each
        keep-out zone (held under ``zones``, per zone); the arm holds on any."""
        states = self.backend.read()
        for other, s in states.items():
            if s.fault:
                raise _Outcome(self._hold(name, f"{other}: {s.fault}", "fault"))
        state = states[name]
        if not np.all(np.isfinite(state.q)):
            raise _Outcome(self._hold(name, f"{name}: no joint feedback.", "fault"))
        cfg, env = self.config, self.envelopes[name]
        if not rest and state.temperature is not None and state.temperature > cfg.max_temp_c:
            raise _Outcome(self._hold(name, f"{name}: a motor reached {state.temperature:.0f} C, over the "
                                            f"{cfg.max_temp_c:.0f} C limit. The arm holds where it is; let it cool."))
        if window is not None:
            model = self.models[name]
            window = np.clip(window, model.lo, model.hi)     # backends hold targets inside the joint limits
            lag = np.abs(state.q - window[-1])
            if np.min(np.max(np.abs(state.q - window), axis=1)) > env.tolerance:
                j = int(np.argmax(lag))
                raise _Outcome(self._hold(name, f"{name}: blocked: {self.models[name].joint_names[j]} is "
                                                f"{math.degrees(lag[j]):.1f} deg off its path (tolerance "
                                                f"{cfg.tracking_tolerance_deg:g} deg). The arm holds where it is; "
                                                "look at the cameras."))
            clearance = float(env.clearance(state.q))
            if clearance < floor:
                raise _Outcome(self._hold(name, f"{name}: the measured clearance fell to {clearance * 1000:.0f} mm "
                                                f"above the table. The arm holds where it is; lift it."))
            if zones is not None and len(zones):
                dist, part = env.zone_distances(state.q)
                z = int(np.argmin(dist - zones))
                if dist[z] < zones[z]:
                    raise _Outcome(self._hold(name, f"{name}: the {self.models[name].body_names[part[z]]} entered "
                                                    f"keep-out zone {env.zones[z].name} ({-dist[z] * 1000:.0f} mm "
                                                    "inside). The arm holds where it is; move it away from the zone."))
        return state

    def _hold(self, name: str, message: str, status: str = "stopped") -> Result:
        try:
            self.backend.hold(name)
        except Exception as e:
            message += f" (the hold command failed: {type(e).__name__}: {e})"
        return Result(status, message)

    def _gripper(self, name: str, state: ArmState, opening: float) -> Result:
        """Move the gripper; the arm joints keep their command."""
        last, still = state.gripper, 0
        for i in range(int(GRIPPER_S * self.config.rate_hz)):
            if self._abort.is_set():
                return self._hold(name, f"{name}: gripper stopped on request; it holds its opening.")
            self.backend.step({name: ArmTarget(state.q_target, opening)}, self._dt)
            now = self._watch(name)
            still = still + 1 if abs(now.gripper - last) < GRIPPER_STILL else 0
            last = now.gripper
            if abs(now.gripper - opening) < 0.02 or (still >= 3 and i * self._dt >= 0.3):
                break
        return Result()

    def _wait(self, seconds: float) -> Result:
        for _ in range(int(round(seconds * self.config.rate_hz))):
            if self._abort.is_set():
                return Result("stopped", "the wait ended early on request.")
            self.backend.step({}, self._dt)
        return Result()

    def _rest(self, name: str) -> tuple[Result, str]:
        """The rest action for one arm, and its torque outcome afterwards."""
        self._released.pop(name, None)
        result = self._guarded(lambda: self._rest_arm(name))
        outcome = self._released.pop(name, None)
        if outcome is None:
            try:
                enabled = self.backend.read()[name].enabled
                outcome = {True: "holding", False: "released"}.get(enabled, "unknown")
            except Exception as e:
                outcome = f"error: {e}"
        return result, outcome

    def _rest_arm(self, name: str) -> Result:
        """Back to the rest pose along a checked path from where the arm is (straight there, else by way of
        home), then released there. A faulted arm, or one that does not arrive, is not released."""
        state = self.backend.read()[name]
        if state.fault:
            return Result("fault", f"{name}: {state.fault}. The arm is not moved; it holds its pose. The operator "
                                   "must support it and release it.")
        if state.enabled is False:
            return Result()
        if state.enabled is None:
            return Result("stopped", f"{name}: its torque state cannot be read and it was not turned on here, so "
                                     "nothing was moved.")
        model, env = self.models[name], self.envelopes[name]
        self._commanded.pop(name, None)
        if state.rest_q is None:
            return Result("rejected", f"{name}: no rest pose is recorded, so the arm is not moved or released; it "
                                      "holds its pose.")
        goal = np.clip(state.rest_q, model.lo, model.hi)
        legs = self._rest_legs(env, model, state.q, goal)
        start = state.q
        for k, path in enumerate(legs):
            if k:
                start = self.backend.read()[name].q
                try:
                    path = self._plan(env, model.plan_joint_path, start, goal)
                    env.check(start, path, capped=False, rest_q=goal)
                except MoveRejected as e:
                    return self._hold(name, f"{name}: the way down from home to rest is refused from where the arm "
                                            f"stopped ({e}); it holds there.")
            result = self._execute(name, start, path, env.speed_for(start, path), state.gripper_target, rest=True,
                                   floor=env.rest_floor(goal))
            if not result.ok:
                return result
        off = np.abs(self.backend.read()[name].q - state.rest_q)
        j = int(np.argmax(off))
        if off[j] > math.radians(REST_TOLERANCE_DEG):
            return self._hold(name, f"{name}: {model.joint_names[j]} stopped {math.degrees(off[j]):.0f} deg from the "
                                    "rest pose; the arm holds its pose and was not released.")
        outcome = self._released[name] = self.backend.rest(name)
        if outcome in OFF:
            return Result()
        return Result("stopped", f"{name}: at the rest pose but not released ({outcome}); the arm keeps holding. "
                                 "The operator should check it.")

    def _rest_legs(self, env: Envelope, model: ArmModel, q0: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Checked joint paths from ``q0`` to the rest pose: one straight there, or two by way of home (the
        second planned from home; it is planned again from where the first ends)."""
        env.require_table()
        try:
            direct = self._plan(env, model.plan_joint_path, q0, goal)
            env.check(q0, direct, capped=False, rest_q=goal)
            return [direct]
        except MoveRejected as refusal:
            try:
                up = self._plan(env, model.plan_joint_path, q0, model.home_q)
                env.check(q0, up, capped=False)
                down = self._plan(env, model.plan_joint_path, model.home_q, goal)
                env.check(model.home_q, down, capped=False, rest_q=goal)
                return [up, down]
            except MoveRejected:
                raise MoveRejected(f"{refusal} Nothing moved: lift the arm, then rest again.") from None

    # ── observation ──
    def _observe(self, result: Result, label: str, fresh_after: Optional[float] = None) -> Observation:
        """The arms and one image per camera. With ``fresh_after`` (a monotonic time), waits up to FRESH_FRAME_S
        for frames captured after it; an older frame is marked stale. Camera and disk errors go in the cameras."""
        try:
            states = self.backend.read()
        except Exception as e:
            states = {}
            result.message = f"{result.message} (the arm state could not be read: {type(e).__name__}: {e})".lstrip()
        arms = {name: self._arm_observation(name, states[name]) for name in self.models if name in states}
        frames, error = self._frames(fresh_after)
        cameras = {}
        now = time.monotonic()
        for cam in self.cameras:
            frame = frames.get(cam)
            if frame is None:
                cameras[cam] = CameraImage(error=error or "no recent image from this camera")
                continue
            age = None if frame.captured is None else round(max(0.0, now - frame.captured), 3)
            stale = fresh_after is not None and frame.captured is not None and frame.captured < fresh_after
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                cameras[cam] = CameraImage(stamp=frame.stamp, error="the image could not be encoded")
                continue
            jpeg, path, problem = buf.tobytes(), None, None
            if self.image_dir is not None:
                try:
                    (self.image_dir / f"{label}-{cam}.jpg").write_bytes(jpeg)
                    path = str(self.image_dir / f"{label}-{cam}.jpg")
                except OSError as e:
                    problem = f"the image could not be saved: {e}"
            height, width = frame.image.shape[:2]
            cameras[cam] = CameraImage(frame.stamp, width, height, path, jpeg, problem, age, stale)
        return Observation(self.steps, round(time.time(), 3), arms, cameras, result)

    def _frames(self, after: Optional[float]) -> tuple[dict, Optional[str]]:
        try:
            frames = self.backend.images()
            while after is not None and time.monotonic() < after + FRESH_FRAME_S and \
                    any(f.captured is not None and f.captured < after for f in frames.values()):
                time.sleep(0.02)
                frames = self.backend.images()
            return frames, None
        except Exception as e:
            return {}, f"the cameras could not be read: {type(e).__name__}: {e}"

    def _arm_observation(self, name: str, state: ArmState) -> ArmObservation:
        model = self.models[name]
        xyz, pitch, yaw, roll = model.tool_angles(state.q)
        clearance = self.envelopes[name].clearance(state.q)
        nearest = self.envelopes[name].nearest_keep_out(state.q)
        return ArmObservation(
            {j: round(math.degrees(v), 1) for j, v in zip(model.joint_names, state.q)},
            round(float(state.gripper), 3),
            ToolPose(tuple(round(float(v), 4) for v in xyz), round(math.degrees(pitch), 1),
                     round(math.degrees(yaw), 1), round(math.degrees(roll), 1)),
            None if clearance is None else round(float(clearance), 4),
            state.enabled, state.fault, None if state.temperature is None else round(float(state.temperature), 1),
            None if nearest is None else {"name": nearest[0], "distance_m": round(nearest[1], 4)})

    # ── prompt material ──
    def describe(self) -> str:
        """The robot, frames, envelope and result meanings in plain words, for a model's prompt."""
        cfg, names = self.config, list(self.models)
        lines = [f"Robot {self.robot_name}: {len(names)} arm{'s' if len(names) > 1 else ''} ({', '.join(names)}); "
                 f"cameras: {', '.join(self.cameras) or 'none'}.",
                 "Units: metres and degrees; the gripper opening runs from 0 (closed) to 1 (fully open).",
                 "World frame: z up; positions and tool poses are in it. The tool point sits between the gripper "
                 "fingers. pitch_deg tilts its pointing direction (-90 straight down, 0 horizontal), yaw_deg is the "
                 "pointing direction's heading about the vertical (0 = +x, 90 = +y), roll_deg turns the gripper about "
                 "its pointing direction. Pointing straight down, roll reads 0 and yaw carries the turn."]
        for name, model in self.models.items():
            env, base, g = self.envelopes[name], model.info, model.gripper
            table = "not measured (the arm cannot move until the operator runs `llm-harness calibrate`)" \
                if env.table_z is None else f"z = {env.table_z:.3f} m"
            lines.append(f"Arm {name} ({base.type}): base at [{', '.join(f'{v:.2f}' for v in base.base_xyz)}] m facing "
                         f"yaw {math.degrees(base.base_yaw):.0f} deg (its x forward, y left); table {table}.")
            lines.append("  joints (deg): " + ", ".join(f"{j} [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]"
                                                        for j, lo, hi in zip(model.joint_names, model.lo, model.hi)))
            x, y, z = model.tool_pose(model.home_q)[:3, 3]
            lines.append(f"  home puts the tool point at [{x:.2f}, {y:.2f}, {z:.2f}] m; the fingertips are taken "
                         f"{(g.finger_length_m + g.margin_m) * 1000:.0f} mm past it and the open jaws "
                         f"{(g.jaw_half_width_m + g.margin_m) * 1000:.0f} mm to each side.")
        lines += ["Safety envelope (each motion is planned and checked before anything moves):",
                  f"- joints turn at most {cfg.max_speed_deg_s:g} deg/s, {cfg.slow_speed_deg_s:g} deg/s when the "
                  f"elbow, wrist, tool point or fingertips are within {cfg.slow_zone_m:.3f} m of the table;",
                  f"- hard floor: the elbow, wrist, tool point and fingertips (jaws taken as open) stay "
                  f"{cfg.floor_margin_m * 1000:.0f} mm above the table even with every joint "
                  f"{cfg.floor_tolerance_deg:g} deg off and {cfg.sag_allowance_m * 1000:.0f} mm of sag (0 mm: they "
                  "may touch it); an arm already below that may only keep gaining height; a measured point more than "
                  f"{cfg.contact_allowance_m * 1000:.0f} mm under that stops the motion;",
                  f"- one action turns no joint more than {cfg.max_excursion_deg:g} deg (home and rest are exempt): "
                  "split long motions and look in between;",
                  f"- no motion while a motor is above {cfg.max_temp_c:g} C; a joint more than "
                  f"{cfg.tracking_tolerance_deg:g} deg off its path stops the motion as blocked;",
                  "- the motors turn on only at the rest pose; opening and closing the gripper is not checked "
                  "against the table;"]
        if self.keep_out:
            lines.append(f"- keep-out zones (world frame): no point along an arm, from its base to the open jaw tips, "
                         f"comes within {cfg.keep_out_margin_m * 1000:.0f} mm of them; a path that would is rejected, "
                         "so go around; an arm already inside may only move out; each arm's observation gives the "
                         "nearest zone and its distance: " + "; ".join(z.text() for z in self.keep_out) + ";")
        if len(names) > 1:
            lines.append("- the arms are not checked against each other: keep them apart.")
        lines.append("Results: done; rejected = refused before anything moved, change the request; stopped = ended "
                     "early, the arm holds where it is, look before going on; fault = the arm reports a fault and "
                     "needs the operator.")
        return "\n".join(lines)


def _combine(results: dict[str, Result]) -> Result:
    """One result for several arms: the most severe status, with every arm's message."""
    status = max((r.status for r in results.values()), key=SEVERITY.index)
    return Result(status, " ".join(r.message for r in results.values() if r.message))
