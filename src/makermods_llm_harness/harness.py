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

from .actions import (MIN_SPEED_DEG_S, Action, ActionError, ArmObservation, CameraImage, Gripper, Home, MoveJoints,
                      MoveTool, Nudge, Observation, Observe, Rest, Result, ToolPose, Wait, action_to_dict, parse_action)
from .arm_model import ArmModel, Unreachable, limit_ticks, pose_error, resample, rotation_from_angles
from .backends.base import OFF, ArmState, ArmTarget, Backend
from .log import EpisodeLog
from .robot import Robot, load as load_robot
from .safety import REST_TOLERANCE_DEG, Envelope, KeepOut, MoveRejected, SafetyConfig, load_calibration
from .safety import table as stored_table

SETTLED = math.radians(1.0)
NEAR_VERTICAL_COS = 0.2            # |cos(pitch)| under which the tool counts as pointing straight up or down
ARRIVAL_TOL_M = 0.010              # a tool move is done only with the measured tool pose this close to the target ...
ARRIVAL_TOL_DEG = 3.0              # ... and within this angle of it
SAG_ARRIVAL_TOL_M = 0.03           # a miss this small still counts as arrived when the standing error the joints
SAG_ARRIVAL_TOL_DEG = 6.0          # held before the move explains it; the result says how far off it ended
COMMAND_MATCH = 1e-3               # rad: the arm still follows the last path's end when its command is this close
GRIPPER_S = 2.5                    # longest gripper action
GRIP_STILL = 0.005                 # opening change over GRIP_STILL_S under which the jaws count as settled
GRIP_STILL_S = 0.5
GRIP_REACHED = 0.02                # the jaws reached the request this close ...
GRIP_BAND = 0.04                   # ... or settled this close to it: done, not holding
GRIP_OPEN_BAND = 0.15              # opening: the jaws may settle this far short (their mechanical end)
GRIP_HOLD_GAP = 0.06               # a grasp: stalled further than this from the request ...
GRIP_HOLD_STILL = 0.003            # ... changing less than this over GRIP_HOLD_S ...
GRIP_HOLD_S = 1.0
GRIP_PROGRESS = 0.05               # ... after closing at least this far
GRIP_TOL = 0.05                    # a gripper that stops further than this from the requested opening was blocked
HOLD_EPS = math.radians(0.2)       # a hold is confirmed with each commanded joint this close to where it stood ...
GRIP_HOLD_EPS = 0.01               # ... and the commanded gripper opening this close to the measured one
FRESH_HOLD_S = 0.25                # ... from feedback no older than this
HOLD_WAIT_S = 10.0                 # longest wait for the harness before a hold after a failed queued action
STILL_S = 0.3                      # elapsed window over which standing still is judged
CONTACT_LATERAL_M = 0.010          # a contact ending lies this close to the planned tool path, or at the table
COMMAND_MOVED_M = 0.001            # contact: while the command moves further than this over STILL_S (one lag
CONTACT_RATIO = 0.25               # earlier), the tool point advances less than this fraction as far along it, for
                                   # STILL_S more; the shortfall then exceeds what it was while advancing freely
FREE_RATIO = 0.5                   # (advancing at least this fraction) by the contact tolerance
FOLLOW_DEG = 1.0                   # the arm follows its command once a joint has moved this far and half as far as
                                   # its command (gear take-up and lag before that are never contact)
END_ERROR_DEG = 2.0                # a joint ends a move within this of its goal, its standing error before the move
PROGRESS_MIN_DEG = 4.0             # commanded travel from which a joint must also be seen to cover half of it
STANDING_MARGIN_DEG = 0.5          # plus this, or this fraction of its travel ...
END_FRACTION = 0.25                # ... whichever is most (never more than the tracking tolerance)
NOISE_DEG = 0.5                    # a joint travel under which feedback cannot tell a move from a block
JOINT_STILL_DEG = 0.1              # joint displacement over STILL_S under which the joints count as standing still
FRESH_FEEDBACK_S = 1.0             # longest wait at the end of rest for every joint to report a newer sample
NEAR_BASE_AXIS_M = 0.02            # closer than this to the base axis, move_tool keeps the current yaw
FRESH_FRAME_S = 0.5                # longest wait after an action for camera frames captured after it ended
JPEG_QUALITY = 85
GRID_HEIGHT = 240                  # px: each camera's height in a look() grid of more than FULL_SIZE_CAMERAS ...
FULL_SIZE_CAMERAS = 2              # ... up to which each camera keeps its own size, scaled down only to TILE_MAX
TILE_MAX = (640, 480)
GRID_COLUMNS = 3
SEVERITY = ("done", "rejected", "stopped", "fault")
READ_ONLY = "this session is read-only: only observe and wait are allowed."


class HarnessClosed(RuntimeError):
    """The harness is closed: it takes no more actions or observations."""


class _BlockedShort(Result):
    """A move with until contact that ended with a joint blocked short of its goal, not yet held: ``_arrival``
    reports contact when the tool point stands still short along the approach, else holds it as stopped."""


class _Outcome(Exception):
    """Ends an action early with this result."""

    def __init__(self, result: Result):
        super().__init__(result.message)
        self.result = result


class _Serialized:
    """The backend with each method call made under ``lock``, so reads from other threads fall between ticks."""

    def __init__(self, backend: Backend, lock):
        self._backend, self._lock = backend, lock

    def __getattr__(self, name):
        value = getattr(self._backend, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                return value(*args, **kwargs)
        return call


class Harness:
    """Drives a backend from actions. ``step`` and ``observe`` may be called from several threads (they run
    one at a time); ``stop`` may be called at any time to end the action in progress. ``state``, ``look``,
    ``check`` and ``reachable`` move nothing and do not wait for the action in progress."""

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
        self._io = threading.RLock()                 # every backend call; a target is sent only after a stop check
        self._gate = threading.Lock()                # orders stop against the release at the end of rest
        self._closing = False                        # close() owns the arms from here on
        self._planning = threading.RLock()        # one plan at a time: a step's and a check's may overlap
        self._dev = _Serialized(backend, self._io)
        self.log = log
        self.image_dir = Path(image_dir).expanduser().resolve() if image_dir is not None \
            else (log.image_dir if log else None)
        if self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.steps = log.steps if log else 0
        self._looks = len(list(self.image_dir.glob("look-*"))) if self.image_dir is not None else 0
        self._grids = len(list(self.image_dir.glob("grid-*"))) if self.image_dir is not None else 0
        self._stops = 0
        self._dt = 1.0 / self.config.rate_hz
        self._lock = threading.Lock()
        self._abort: Optional[threading.Event] = None     # set to end the action in progress
        self._released: dict[str, str] = {}               # backend.rest() outcome per arm, read by close()
        self._commanded: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # tool pose and joints of the last done motion
        self._heading: dict[str, tuple[np.ndarray, Optional[np.ndarray]]] = {}   # end joints and pose of a motion
        self._ending: dict[str, tuple] = {}     # a contact move's end: (stillness window, last q, planned tool path)
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
        ``backend``. Table heights (numbers or ``safety.Table``) come from ``table_z`` (one for all arms or per arm),
        else the backend
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
                heights = {n: stored_table(stored[n]) for n in names
                           if n in stored and stored[n]["setup"] == robot.setup(n, safety)}
            if isinstance(table_z, dict):
                heights.update(table_z)
            elif table_z is not None:
                heights = {n: float(table_z) for n in names}
            if keep_out is None:
                keep_out = robot.zones() if robot else ()
            harness = cls(backend, robot_name=robot.name if robot else "robot", table_z=heights, safety=safety,
                          log=log, image_dir=image_dir, read_only=read_only, keep_out=keep_out)
            if robot and robot.home:
                for model in harness.models.values():
                    model.set_home(robot.home["xyz"], float(robot.home.get("pitch", 0.0)))
            for name in getattr(backend, "attach_candidates", ()):
                harness._take_over(name)
            return harness
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

    def step(self, action: Union[Action, dict, str], cancel: Optional[threading.Event] = None,
             images: bool = True) -> Observation:
        """Run one action and observe. A malformed, refused or failed action comes back in ``result``. Setting
        ``cancel`` ends this action (the arm holds where it is), as ``stop()`` does. Without ``images`` the
        observation has the arms only, and no camera is read."""
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
                obs = self._observe(result, f"step-{self.steps:04d}", fresh_after=time.monotonic(), images=images)
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

    def stop(self) -> Optional[str]:
        """End the action in progress and the ones waiting behind it, then hold every arm that may be on where it
        is and confirm it (``confirm_hold``). Returns None once confirmed, else what could not be confirmed. Once
        the harness closes, its rest owns the arms: this changes nothing and says so."""
        with self._gate:
            if self._closing:
                return "the harness is closing: its rest owns the arms"
            self._stops += 1
            abort = self._abort
            if abort is not None:
                abort.set()
            return self.confirm_hold()

    def hold(self, timeout_s: float = HOLD_WAIT_S) -> Optional[str]:
        """``confirm_hold`` for every arm once no action or observation holds the harness, waiting up to
        ``timeout_s`` for it. None when held and confirmed (or the harness closes: its rest owns the arms), else
        what could not be done."""
        if not self._lock.acquire(timeout=timeout_s):
            return f"the arms could not be held: an action kept the harness for more than {timeout_s:g} s"
        try:
            return None if self._closing else self.confirm_hold()
        finally:
            self._lock.release()

    def confirm_hold(self, names: Optional[Sequence[str]] = None) -> Optional[str]:
        """Hold every arm that may be on (or ``names``) where it is and confirm it. The backend's ``hold`` replaces
        the joint and gripper targets with the measured pose and returns that setpoint; after ``hold_settle_s``
        (control ticks already under way have sent their commands) the arm is read again and the hold is
        confirmed only with no fault, finite feedback, torque known, every joint's sample at most FRESH_HOLD_S old,
        every joint measured inside its limits (a joint past one cannot be held where it is), the setpoint within
        HOLD_EPS of a measurement around the hold, and the commands the backend reports equal to that setpoint
        (HOLD_EPS, gripper GRIP_HOLD_EPS). A backend that returns no setpoint is held to the measurement read just
        before. None when confirmed, else what was not, joint by joint. Arms reported off are left alone."""
        problems = []
        with self._io:
            try:
                before = self._dev.read()
            except Exception as e:
                return f"the arms could not be read to hold them ({type(e).__name__}: {e})"
            arms, setpoints = [], {}
            for name in (names or list(self.models)):
                state = before.get(name)
                if state is None or state.enabled is False:
                    continue
                try:
                    setpoints[name] = self._dev.hold(name)
                    arms.append(name)
                except Exception as e:
                    problems.append(f"{name}: the hold command failed ({type(e).__name__}: {e})")
            settle = float(getattr(self.backend, "hold_settle_s", 0.0) or 0.0)
            if arms and settle > 0:
                time.sleep(settle)
            try:
                after = self._dev.read() if arms else {}
                ages = {name: self._dev.feedback_ages(name) for name in arms}
            except Exception as e:
                return f"the hold could not be confirmed ({type(e).__name__}: {e})"
        for name in arms:
            problem = self._hold_problem(name, before[name], after[name], setpoints[name], ages[name])
            if problem:
                problems.append(f"{name}: the hold is not confirmed: {problem}")
        return "; ".join(problems) or None

    def _hold_problem(self, name: str, b: ArmState, a: ArmState, setpoint, ages) -> Optional[str]:
        model = self.models[name]
        if a.fault:
            return a.fault
        if not (np.all(np.isfinite(a.q)) and np.all(np.isfinite(a.q_target)) and math.isfinite(a.gripper)
                and math.isfinite(a.gripper_target)):
            return "no joint feedback"
        if a.enabled is None:
            return "its torque state is not known"
        if ages is not None:
            stale = [i for i, age in enumerate(ages) if not (age is not None and math.isfinite(age)
                                                             and age <= FRESH_HOLD_S)]
            if stale:
                return "no fresh feedback from " + ", ".join(self._joint_label(name, i) for i in stale)
        if not a.enabled:
            return None
        past = np.flatnonzero((b.q < model.lo - HOLD_EPS) | (b.q > model.hi + HOLD_EPS))
        if len(past):
            j = int(past[0])
            return (f"{model.joint_names[j]} is measured at {math.degrees(b.q[j]):.1f} deg, past its range "
                    f"[{math.degrees(model.lo[j]):.0f}, {math.degrees(model.hi[j]):.0f}] deg, so it cannot be held "
                    "where it is")
        if setpoint is None:
            q_set, g_set = b.q, float(b.gripper)
        else:
            q_set, g_set = np.asarray(setpoint[0], dtype=float), float(setpoint[1])
            measured = np.minimum(np.abs(q_set - b.q), np.abs(q_set - a.q))
            if np.any(measured > HOLD_EPS):
                j = int(np.argmax(measured))
                return (f"{model.joint_names[j]} was held at {math.degrees(q_set[j]):.2f} deg, not where it was "
                        "measured")
        off = np.abs(a.q_target - q_set)
        if np.any(off > HOLD_EPS):
            j = int(np.argmax(off))
            return (f"{model.joint_names[j]} is commanded {math.degrees(a.q_target[j]):.2f} deg, not its hold "
                    f"setpoint {math.degrees(q_set[j]):.2f} deg")
        if abs(float(a.gripper_target) - g_set) > GRIP_HOLD_EPS:
            return f"its gripper is commanded to {a.gripper_target:.3f}, not its hold setpoint {g_set:.3f}"
        return None

    def _joint_label(self, name: str, i: int) -> str:
        names = self.models[name].joint_names + [self.models[name].gripper_name]
        return names[i] if i < len(names) else f"joint {i}"

    def state(self) -> Observation:
        """The arms now, without images; does not wait for the action in progress."""
        if self._closed:
            raise HarnessClosed("the harness is closed")
        return self._observe(Result(), "", images=False)

    def look(self) -> dict:
        """One labelled grid of every camera's newest image, captured after this call: ``{"path", "jpeg",
        "errors"}``. ``path`` is the saved JPEG (None when the harness saves no images), ``errors`` maps a camera
        without an image to the reason. Does not wait for the action in progress."""
        if self._closed:
            raise HarnessClosed("the harness is closed")
        frames, error = self._frames(time.monotonic())
        tiles, errors = [], {}
        height = None if len(self.cameras) <= FULL_SIZE_CAMERAS else GRID_HEIGHT
        now = time.monotonic()
        for cam in self.cameras:
            frame = frames.get(cam)
            if frame is None:
                errors[cam] = error or "no recent image from this camera"
                tiles.append(_tile(None, f"{cam}: {errors[cam]}", height))
                continue
            age = "" if frame.captured is None else f" ({max(0.0, now - frame.captured):.1f} s old)"
            tiles.append(_tile(frame.image, cam + age, height))
        grid = _grid(tiles) if tiles else _tile(None, "no cameras")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg = buf.tobytes() if ok else None
        path = None
        if jpeg is not None and self.image_dir is not None:
            self._grids += 1
            target = self.image_dir / f"grid-{self._grids:04d}.jpg"
            try:
                target.write_bytes(jpeg)
                path = str(target)
            except OSError as e:
                errors["grid"] = f"the grid could not be saved: {e}"
        return {"path": path, "jpeg": jpeg, "errors": errors}

    def check(self, action: Union[Action, dict, str]) -> Result:
        """Plan and check an action exactly as ``step`` would, moving nothing: done when it would be sent, else
        why not. A motion of an arm that is moving is checked from where that motion ends."""
        try:
            act = parse_action(action)
        except ActionError as e:
            return Result("rejected", str(e), action if isinstance(action, dict) else None)
        result = self._guarded(lambda: self._check(act))
        result.action = action_to_dict(act)
        return result

    def reachable(self, poses: Sequence, arm: Optional[str] = None, line: bool = True, chain: bool = False,
                  start=None) -> list[dict]:
        """Check tool poses as ``move_tool`` would, moving nothing. Each pose is ``[x, y, z]`` or ``{"xyz",
        "pitch", "yaw", "roll"}`` (degrees; left out: as move_tool defaults them). Each is checked from where
        the arm is (or from where its running motion ends), from ``start`` when given (a pose given the same way,
        reached by the joints the IK finds from where the arm is, or "home"), and with ``chain`` from the pose
        before it. Per pose: ``{"ok", "why", "margin_deg", "margin_joint",
        "joints_deg", "seconds"}``: the refusal a move would get, the distance of the goal's nearest joint from
        its limit, the goal joints and the planned duration."""
        if self._closed:
            raise HarnessClosed("the harness is closed")
        out = []
        if self.read_only:
            return [_verdict(False, READ_ONLY) for _ in poses]
        try:
            act0 = MoveTool((0.0, 0.0, 0.0), arm=arm)
            name = self._arm_names(act0)[0]
            self.envelopes[name].require_table()
            state = self._usable(name)
            if not state.enabled:
                self._enable_ramp(name, state)
            q0, commanded = self._check_start(name, state)
            if isinstance(start, str) and start == "home":
                q0, commanded = self.models[name].home_q, None
            elif start is not None:
                commanded = self._tool_goal(self.models[name], q0, _pose_action(start, arm, line))
                q0 = self.models[name].solve_pose(q0, commanded)
        except (MoveRejected, Unreachable, ActionError, _Outcome) as e:
            why = e.result.message if isinstance(e, _Outcome) else str(e)
            return [_verdict(False, why) for _ in poses]
        model = self.models[name]
        broken = None
        for pose in poses:
            if broken is not None:
                out.append(_verdict(False, broken))
                continue
            try:
                act = _pose_action(pose, arm, line)
                path, _, goal = self._checked_path(name, act, q0, commanded)
            except (MoveRejected, Unreachable, ActionError) as e:
                out.append(_verdict(False, str(e)))
                if chain:
                    broken = "not checked: the pose before it is refused"
                continue
            end = path[-1]
            room = np.minimum(end - model.lo, model.hi - end)
            j = int(np.argmin(room))
            out.append({"ok": True, "why": None, "margin_deg": round(math.degrees(room[j]), 1),
                        "margin_joint": model.joint_names[j],
                        "joints_deg": {n: round(math.degrees(v), 1) for n, v in zip(model.joint_names, end)},
                        "seconds": round(len(path) / self.config.rate_hz, 2)})
            if chain:
                q0, commanded = end, goal
        return out

    def close(self) -> dict[str, str]:
        """Rest every arm that is on (the same checked motion as the rest action), free the backend and close
        the log. Returns each arm's torque outcome: "released" or "always_on" when nothing holds torque any more,
        anything else (with the reason) when the arm may still be on. Later calls return {}."""
        with self._lock:
            if self._closed:
                return {}
            with self._gate:
                self._closing = True
            self._closed = True
            self._abort = threading.Event()
            words, reasons = {}, {}
            if self.log:
                try:
                    self.log.write("closing")
                except (OSError, ValueError):
                    pass
            try:
                for name in self.models:
                    result, words[name] = self._rest(name)
                    reasons[name] = "" if result.ok else result.message
            finally:
                try:
                    try:
                        closed = self._dev.close() or {}
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
        return self._guarded(lambda: self._dispatch(act), hold=True)

    def _guarded(self, run, hold: bool = False) -> Result:
        """``run()``'s result, a refusal as rejected, and any other error as a fault; with ``hold`` (something may
        have moved) that fault comes after a confirmed hold of every arm that is on, or says it was not."""
        try:
            return run()
        except (MoveRejected, Unreachable) as e:
            return Result("rejected", str(e))
        except _Outcome as e:
            return e.result
        except Exception as e:   # a backend failure mid-action
            message = f"{type(e).__name__}: {e}"
            if hold:
                problem = self.confirm_hold()
                message += f". {problem}." if problem else ". Every arm that is on holds where it is."
            return Result("fault", message)

    def _dispatch(self, act: Action) -> Result:
        if isinstance(act, Observe):
            return Result()
        if isinstance(act, Wait):
            return self._wait(act.seconds)
        if self.read_only:
            return Result("rejected", READ_ONLY)
        names = self._arm_names(act)
        if isinstance(act, Rest):
            return _combine({n: self._rest(n)[0] for n in names})
        name = names[0]
        model, env = self.models[name], self.envelopes[name]
        env.require_table()
        state = self._usable(name)
        if isinstance(act, Gripper):
            self._gripper_check(name, state, act.opening)
            return self._gripper(name, self._ready(name, state), act.opening)
        if not state.enabled:
            # Planned and checked first (from where enabling leaves it), so a refused request never turns it on.
            self._checked_path(name, act, np.clip(state.q, model.lo, model.hi))
            speed = getattr(act, "speed_deg_s", None)
            state = self._ready(name, state, None if speed is None else math.radians(speed))
        held = requested = None
        start = state.q
        if isinstance(act, MoveJoints):
            # Joints not named keep the command they hold; the path starts there for them.
            requested = np.array([j in act.joints_deg for j in model.joint_names])
            held = ~requested
            start = np.where(held, state.q_target, state.q)
            off = np.abs(state.q - start)
            j = int(np.argmax(off))
            if off[j] > env.tolerance:
                raise MoveRejected(f"{name}: {model.joint_names[j]} stands {math.degrees(off[j]):.1f} deg from its "
                                   "command, more than the tracking tolerance; move it (or home) first.")
        path, speed, goal = self._checked_path(name, act, start, self._commanded_pose(name, state),
                                               measured=state.q if held is not None else None)
        self._commanded.pop(name, None)
        contact = getattr(act, "until", None) == "contact"
        self._heading[name] = (path[-1], goal)
        try:
            result = self._execute(name, state.q, path, speed, state.gripper_target,
                                   contact_goal=goal if contact else None, start_command=state.q_target,
                                   held=held, requested=requested)
        finally:
            self._heading.pop(name, None)
        if result.contact is not None or result.status == "fault":
            return result
        if goal is not None:
            result = self._arrival(name, goal, result, contact, state.q_target - state.q,
                                   model.tool_points(state.q))
        if result.ok and result.contact is None:
            self._commanded[name] = (model.tool_pose(path[-1]) if goal is None else goal, path[-1])
        return result

    def _check(self, act: Action) -> Result:
        """``_dispatch`` without moving: every check a step makes before anything is sent. What only feedback
        during the motion can show (blocking, contact, arrival, heat or faults that start later) is not known."""
        if isinstance(act, (Observe, Wait)):
            return Result()
        if self.read_only:
            return Result("rejected", READ_ONLY)
        names = self._arm_names(act)
        if isinstance(act, Rest):
            results = {}
            for name in names:
                state = self._dev.read()[name]
                early = self._rest_precheck(name, state)
                if early is None:
                    model, env = self.models[name], self.envelopes[name]
                    try:
                        self._rest_legs(env, model, state.q, np.clip(state.rest_q, model.lo, model.hi))
                        early = Result(message=f"{name}: dry run: it would pass the checks; nothing moved.")
                    except MoveRejected as e:
                        early = Result("rejected", str(e))
                results[name] = early
            return _combine(results)
        name = names[0]
        self.envelopes[name].require_table()
        state = self._usable(name)
        if not state.enabled:
            self._enable_ramp(name, state)
        if isinstance(act, Gripper):
            self._gripper_check(name, state, act.opening)
            return Result(message="dry run: it would pass the checks; nothing moved.")
        q0, commanded = self._check_start(name, state)
        path, _, _ = self._checked_path(name, act, q0, commanded)
        return Result(message=f"{name}: dry run: it would pass the checks and take about "
                              f"{len(path) / self.config.rate_hz:.1f} s; nothing moved.")

    def _check_start(self, name: str, state: Optional[ArmState] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Joints and commanded tool pose a check of the next motion starts from: where the arm's running
        motion ends, else where it is (inside its limits while its motors are off)."""
        state = state or self._usable(name)
        heading = self._heading.get(name)
        if heading is not None:
            q, goal = heading
            return q, self.models[name].tool_pose(q) if goal is None else goal
        if not state.enabled:
            model = self.models[name]
            return np.clip(state.q, model.lo, model.hi), None
        return state.q, self._commanded_pose(name, state)

    def _commanded_pose(self, name: str, state: ArmState) -> Optional[np.ndarray]:
        """The tool pose the last done motion ended on, while the arm still follows that command and stands
        within the arrival tolerance of it; else None."""
        held = self._commanded.get(name)
        if held is None or not np.allclose(state.q_target, held[1], atol=COMMAND_MATCH):
            return None
        position, angle = pose_error(self.models[name].tool_pose(state.q), held[0])
        return held[0] if position <= ARRIVAL_TOL_M and math.degrees(angle) <= ARRIVAL_TOL_DEG else None

    def _arrival(self, name: str, goal: np.ndarray, result: Result, contact: bool = False,
                 standing: Optional[np.ndarray] = None, start: Optional[np.ndarray] = None) -> Result:
        """A tool move's result with how far the measured tool pose ended from ``goal`` (``error``; with ``start``,
        the tool point the move started from, also ``short_m``: how far it ended short of the goal along the
        approach; ``ended_short`` True when that alone stopped it). Done only from valid, fresh final feedback (no fault, finite joints, torque on, every joint's
        sample at most FRESH_HOLD_S old; a fault otherwise) with the measured tool pose within the arrival
        tolerance. A miss that the joint ``standing`` error the arm held before the move (command minus measurement)
        explains, up to SAG_ARRIVAL_TOL_M, is done with the miss in the message; any other miss is ``stopped``, and so
        is a move that ended more than the contact tolerance short along the approach, whatever the arrival
        tolerance. ``contact``: the move asked for contact; one that ended more than the contact tolerance short at
        its last sample, or with a joint blocked short of its goal (``_BlockedShort``), is judged by
        ``_contact_ending``."""
        model = self.models[name]
        blocked = isinstance(result, _BlockedShort)
        if contact and (result.ok or blocked):
            ending = self._ending.pop(name, None)
            last = None if ending is None or start is None else _shortfall(goal[:3, 3], start,
                                                                          model.tool_points(ending[1]))
            if blocked or (last is not None and last > self.config.contact_tolerance_m):
                return self._contact_ending(name, goal, result, ending, start)
        with self._io:
            state = self._dev.read()[name]
            ages = self._dev.feedback_ages(name)
        finite = bool(np.all(np.isfinite(state.q)))
        stale = ages is not None and not all(age is not None and math.isfinite(age) and age <= FRESH_HOLD_S
                                             for age in ages)
        if state.fault or state.enabled is not True or not finite or stale:
            why = state.fault or ("no joint feedback" if not finite else
                                  "its torque state is not known" if state.enabled is not True else
                                  "its feedback is not fresh")
            return self._hold(name, f"{name}: the feedback at the end of the move is not valid ({why}).", "fault")
        position, angle = pose_error(model.tool_pose(state.q), goal)
        if not (math.isfinite(position) and math.isfinite(angle)):
            return self._hold(name, f"{name}: the tool pose at the end of the move is not a finite number.", "fault")
        result.error = {"position_m": round(position, 4), "angle_deg": round(math.degrees(angle), 1)}
        short = None if start is None else _shortfall(goal[:3, 3], start, model.tool_points(state.q))
        if short is not None:
            result.error["short_m"] = round(short, 4)
        tolerance = self.config.contact_tolerance_m
        if result.ok and not contact and short is not None and short > tolerance:
            held = self._hold(name, f"{name}: ended {short * 1000:.0f} mm short of the target along the approach "
                                    f"({position * 1000:.0f} mm and {math.degrees(angle):.1f} deg from it). It may "
                                    "rest on something, or the arm sags there: the arm holds where it is. Aim "
                                    f"{short * 1000:.0f} mm further, or move with until contact.")
            held.error = dict(result.error, ended_short=True) if held.status == "stopped" else result.error
            return held
        if result.ok and (position > ARRIVAL_TOL_M or math.degrees(angle) > ARRIVAL_TOL_DEG):
            explained = False
            if standing is not None:
                p2, a2 = pose_error(model.tool_pose(np.clip(state.q + standing, model.lo, model.hi)), goal)
                explained = p2 <= ARRIVAL_TOL_M and math.degrees(a2) <= ARRIVAL_TOL_DEG
            if explained and position <= SAG_ARRIVAL_TOL_M and math.degrees(angle) <= SAG_ARRIVAL_TOL_DEG:
                result.message = (f"{name}: ended {position * 1000:.0f} mm and {math.degrees(angle):.1f} deg from "
                                  "the target, which the standing error its joints held before the move explains.")
                return result
            held = self._hold(name, f"{name}: ended {position * 1000:.0f} mm and {math.degrees(angle):.1f} deg "
                                    "from the target. The arm holds where it is; look at the cameras.")
            held.error = result.error
            return held
        if result.ok and contact:
            result.message = f"{name}: reached the target without contact."
        return result

    def _contact_ending(self, name: str, goal: np.ndarray, result: Result, ending: Optional[tuple],
                        start: Optional[np.ndarray]) -> Result:
        """A contact move's end, held first: under the stop gate (a stop either came first, and wins, or waits for
        this result) the arm is held, read, and judged. Contact (done) when the samples taken under the watch show
        the tool point still over STILL_S (moving less than COMMAND_MOVED_M, also up to the held reading), the held
        reading is more than the contact tolerance short along the approach, and the tool point is within
        CONTACT_LATERAL_M of the planned tool path or the arm's lowest point is at the measured table (within the
        contact tolerance above it, no outside-reach margin: resting on the known surface explains a slide along
        it). Anything else is ``stopped``;
        too few samples to tell, too."""
        model, env, tol = self.models[name], self.envelopes[name], self.config.contact_tolerance_m
        blocked = isinstance(result, _BlockedShort)
        window, _, planned = ending if ending is not None else (None, None, None)
        with self._gate:
            if self._abort is not None and self._abort.is_set():
                return self._hold(name, f"{name}: stopped on request; the arm holds where it is.")
            held = self._hold(name, result.message if blocked else f"{name}: the move ended.")
            if held.status != "stopped":
                return held
            with self._io:
                state = self._dev.read()[name]
                ages = self._dev.feedback_ages(name)
            fresh = ages is None or all(age is not None and math.isfinite(age) and age <= FRESH_HOLD_S for age in ages)
            if state.fault or state.enabled is not True or not np.all(np.isfinite(state.q)) or not fresh:
                return Result("fault", f"{name}: the feedback after the hold is not valid ("
                                       f"{state.fault or 'torque state or fresh feedback missing'}).")
            xyz = model.tool_points(state.q)
            position, angle = pose_error(model.tool_pose(state.q), goal)
            short = None if start is None else _shortfall(goal[:3, 3], start, xyz)
            error = {"position_m": round(position, 4), "angle_deg": round(math.degrees(angle), 1)}
            if short is not None:
                error["short_m"] = round(short, 4)
            lateral = math.inf if planned is None else _path_distance(planned, xyz)
            clearance = env.measured_clearance(state.q)
            if window is None:
                why = f"too few samples after the command ended to tell whether it stood still over {STILL_S:g} s"
            else:
                q_past, q_last, span = window
                moved = max(float(np.linalg.norm(model.tool_points(q_last) - model.tool_points(q_past))),
                            float(np.linalg.norm(xyz - model.tool_points(q_last))))
                if moved >= COMMAND_MOVED_M:
                    why = f"was still moving ({moved * 1000:.1f} mm over {span:.2f} s)"
                elif short is None or short <= tol:
                    why = (f"ended {0.0 if short is None else short * 1000:.1f} mm short of the target, within the "
                           f"{tol * 1000:.0f} mm contact tolerance")
                elif lateral > CONTACT_LATERAL_M and clearance > tol:
                    why = (f"ended {lateral * 1000:.0f} mm to the side of its path and {clearance * 1000:.0f} mm above "
                           "the table: not a contact along its approach")
                else:
                    why = None
            if why is None:
                reach = float(np.linalg.norm(goal[:3, 3] - xyz))
                where = ", ".join(f"{v:.3f}" for v in xyz)
                out = Result("done", f"{name}: stopped on contact at the end of the move {reach * 1000:.0f} mm short "
                                     f"of the target, tool point at [{where}] m; the arm holds there.")
                out.contact = {"xyz": [round(float(v), 4) for v in xyz], "short_m": round(reach, 4)}
            elif blocked:
                out = Result("stopped", f"{result.message} Not a contact: it {why}.")
            else:
                out = Result("stopped", f"{name}: the move ended short of its target and {why}. The arm holds where "
                                        "it is; look at the cameras.")
            out.error = error
            return out

    def _on_contact(self, name: str, goal: np.ndarray, q: np.ndarray, error: Optional[dict] = None,
                    at_end: bool = False) -> Result:
        """Hold the arm where a move with until contact stopped following its command (``at_end``: where it stood
        short of the target once the command had ended), and say where."""
        xyz = self.models[name].tool_points(q)
        short = float(np.linalg.norm(goal[:3, 3] - xyz))
        where = ", ".join(f"{v:.3f}" for v in xyz)
        when = " at the end of the move" if at_end else ""
        result = self._hold(name, f"{name}: stopped on contact{when} {short * 1000:.0f} mm short of the target, tool "
                                  f"point at [{where}] m; the arm holds there.", "done")
        if result.ok:
            result.contact = {"xyz": [round(float(v), 4) for v in xyz], "short_m": round(short, 4)}
        result.error = error
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
        states = self._dev.read()
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

    def _gripper_check(self, name: str, state: ArmState, opening: float) -> None:
        """Refuse opening jaws that cannot open at the rest pose while the arm stands there."""
        model = self.models[name]
        if model.gripper.opens_at_rest or state.rest_q is None or opening <= state.gripper + GRIP_STILL:
            return
        if float(np.max(np.abs(state.q - state.rest_q))) <= math.radians(REST_TOLERANCE_DEG):
            raise MoveRejected(f"{name}: the jaws cannot open at the rest pose: they meet the arm there. Move it out "
                               "first (home), then open them.")

    def _can_enable(self, name: str, state: ArmState) -> None:
        """Refuse to turn on an arm that is not at its recorded rest pose."""
        model = self.models[name]
        if state.rest_q is None:
            raise MoveRejected(f"{name}: no rest pose is recorded for this arm, so its motors stay off. The operator "
                               "records it with `llm-harness calibrate --rest`.")
        off = np.abs(state.q - state.rest_q)
        j = int(np.argmax(off))
        if off[j] > math.radians(REST_TOLERANCE_DEG):
            raise MoveRejected(f"{name}: the motors turn on only at the rest pose, and {model.joint_names[j]} is "
                               f"{math.degrees(off[j]):.0f} deg from it (limit {REST_TOLERANCE_DEG:g} deg). The "
                               "operator must fold the arm to its rest pose.")

    def _enable_ramp(self, name: str, state: ArmState, speed: Optional[float] = None) -> Optional[np.ndarray]:
        """For an arm that is off: refuse unless it is at its rest pose, and the checked path that brings a joint
        resting past a limit inside it at the slow speed, or ``speed`` (rad/s) when lower (None when every joint is
        inside)."""
        model, env = self.models[name], self.envelopes[name]
        self._can_enable(name, state)
        inside = np.clip(state.q, model.lo, model.hi)
        if np.array_equal(inside, state.q):
            return None
        ramp = model.plan_joint_path(state.q, inside, self._ramp_speed(env, speed), self.config.rate_hz)
        env.check(state.q, ramp, capped=False)
        return ramp

    @staticmethod
    def _ramp_speed(env: Envelope, speed: Optional[float]) -> float:
        return env.slow_speed if speed is None else min(env.slow_speed, speed)

    def _ready(self, name: str, state: ArmState, speed: Optional[float] = None) -> ArmState:
        """The arm's state with its motors on. An arm that is off is turned on only at its rest pose; a joint
        resting past a limit is then brought inside it along a checked path. ``speed`` (rad/s) caps both the
        backend's enable and that path below the slow speed."""
        if state.enabled:
            return state
        env = self.envelopes[name]
        ramp = self._enable_ramp(name, state, speed)
        if self._abort.is_set():
            raise _Outcome(Result("stopped", f"{name}: stopped on request before its motors came on."))
        try:
            if speed is None:
                self._dev.enable(name, cancel=self._abort)
            else:
                self._dev.enable(name, cancel=self._abort, speed=self._ramp_speed(env, speed))
        except Exception as e:
            if self._abort.is_set():
                raise _Outcome(Result("stopped", f"{name}: stopped on request while its motors came on.")) from e
            raise _Outcome(Result("fault", f"{name}: the motors could not be enabled: {e}")) from e
        if ramp is not None:
            result = self._execute(name, state.q, ramp, self._ramp_speed(env, speed), state.gripper)
            if not result.ok:
                raise _Outcome(result)
        return self._usable(name)

    def _checked_path(self, name: str, act: Action, q0: np.ndarray, commanded: Optional[np.ndarray] = None,
                      measured: Optional[np.ndarray] = None) -> tuple[np.ndarray, float, Optional[np.ndarray]]:
        """The joint path of a motion action from ``q0``, passed by the envelope, its speed limit and, for a tool
        move, the requested tool pose. A nudge starts from the ``commanded`` tool pose when there is one. When the
        path starts from a held command, ``measured`` (where the arm stands) also sets the speed: the lower of the
        two near the table."""
        with self._planning:
            return self._planned(name, act, q0, commanded, measured)

    def _planned(self, name: str, act: Action, q0: np.ndarray, commanded: Optional[np.ndarray],
                 measured: Optional[np.ndarray] = None) -> tuple[np.ndarray, float, Optional[np.ndarray]]:
        model, env = self.models[name], self.envelopes[name]
        goal = None
        speed = getattr(act, "speed_deg_s", None)
        speed = None if speed is None else math.radians(speed)
        duration = getattr(act, "duration_s", None)
        timing = {"speed": speed, "duration": duration, "also": measured}
        if isinstance(act, (MoveTool, Nudge)):
            goal = self._tool_goal(model, q0, act, commanded)
            if isinstance(act, MoveTool) and act.line is False:
                path = self._plan(env, model.plan_joint_path, q0, model.solve_pose(q0, goal), **timing)
            else:
                path = self._plan(env, model.plan_tool_path, q0, goal, **timing)
        else:
            q1 = model.home_q if isinstance(act, Home) else self._joint_goal(model, q0, act)
            path = self._plan(env, model.plan_joint_path, q0, q1, **timing)
        env.check(q0, path, capped=not isinstance(act, Home), contact=getattr(act, "until", None) == "contact")
        cap = env.speed_for(q0, path) if measured is None else min(env.speed_for(q0, path),
                                                                  env.speed_for(measured, path))
        if speed is not None:
            cap = min(cap, speed)
        if duration is not None:
            peak = float(np.max(np.abs(np.diff(np.vstack([q0, path]), axis=0)))) * self.config.rate_hz
            cap = min(cap, max(1.25 * peak, math.radians(MIN_SPEED_DEG_S)))
        return path, cap, goal

    def _plan(self, env: Envelope, plan, q0: np.ndarray, goal: np.ndarray, speed: Optional[float] = None,
              duration: Optional[float] = None, also: Optional[np.ndarray] = None) -> np.ndarray:
        """``plan(q0, goal, speed, rate)`` at full speed (or ``speed``, rad/s, when lower), again at the slow speed
        when it, or ``also`` (another start: where the arm is measured), comes near the table; a path shorter than
        ``duration`` seconds is then stretched in time to it."""
        rate = self.config.rate_hz
        fast = env.max_speed if speed is None else min(speed, env.max_speed)
        path = plan(q0, goal, fast, rate)
        limit = env.speed_for(q0, path) if also is None else min(env.speed_for(q0, path), env.speed_for(also, path))
        if limit < fast:
            path = plan(q0, goal, limit, rate)
        if duration is not None and len(path) < duration * rate:
            ticks = int(math.ceil(duration * rate))
            limit_ticks(env.model.name, ticks, rate)
            path = resample(q0, path, rate * len(path) / ticks, rate)
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
        pitch = pitch if act.pitch_deg is None else math.radians(act.pitch_deg)
        if act.roll_deg is not None:
            return model.pose_from_angles(xyz, pitch, heading, math.radians(act.roll_deg))
        if act.pitch_deg is None and abs(math.cos(pitch)) > NEAR_VERTICAL_COS:
            # The gripper keeps its exact tilt and roll, turned about the vertical by the change of heading.
            c, s = math.cos(heading - yaw), math.sin(heading - yaw)
            pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ pose[:3, :3]
            pose[:3, 3] = xyz
            return pose
        if act.yaw_deg is not None or min(abs(math.cos(pitch)),
                                          abs(math.cos(model.tool_angles(q0)[1]))) > NEAR_VERTICAL_COS:
            return model.pose_from_angles(xyz, pitch, heading, roll)
        # Near straight up or down, yaw and roll are one turn about the vertical and a roll copied from the
        # reading could ask for half a turn: take the roll that turns the gripper least from where it is.
        turn = rotation_from_angles(pitch, heading, 0.0).T @ pose[:3, :3]
        return model.pose_from_angles(xyz, pitch, heading,
                                      math.atan2(turn[1, 0] - turn[0, 1], turn[0, 0] + turn[1, 1]))

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
                 rest: bool = False, floor: Optional[float] = None,
                 contact_goal: Optional[np.ndarray] = None, start_command: Optional[np.ndarray] = None,
                 held: Optional[np.ndarray] = None, requested: Optional[np.ndarray] = None) -> Result:
        """Stream a checked path from the measured ``q0``, watching every tick; then send the goal again for up to
        ``settle_s`` while the joints arrive (sooner once they are within SETTLED of it or stand still). Each
        target goes out only after a stop check made under the backend lock, so nothing is sent after a stop has
        held the arm. Time is elapsed time: each tick counts from the end of the one before (sending, reading,
        everything between) and at least ``dt``.

        Each tick the measured joints are compared with the waypoints sent within the tracking lag. At the end
        ``judge_motion`` decides from the start (``q0``, and ``start_command``, the command the arm held before),
        the goal and the end, with ``held`` joints (kept at their command) judged on drift only and ``requested``
        joints (asked for by name) held to the small-move rule. Anything but done holds the arm and confirms it
        (``_hold``), except a contact move that ended with a joint blocked short of its goal: that comes back unheld
        as ``_BlockedShort`` for ``_arrival`` to judge. ``rest`` lets a hot arm go on; ``floor`` lowers the measured clearance it is held under (the
        way down to a rest pose under the floor).

        With ``contact_goal`` (until contact, the requested tool pose) the motion ends, done, on contact: once the
        arm has followed its command (FOLLOW_DEG), its tool point has advanced along the approach less than
        CONTACT_RATIO as far as the command one lag earlier, for longer than STILL_S, while the command was still
        moving (more than COMMAND_MOVED_M), and it is further than ``contact_tolerance_m`` from every tool point
        sent within the lag beyond both its standing offset at the start and the largest shortfall it had while
        advancing freely (load changes and lag). A shortfall still there once the command has ended is judged by
        ``_arrival``."""
        model, env, cfg = self.models[name], self.envelopes[name], self.config
        self._ending.pop(name, None)
        start_command = q0 if start_command is None else np.asarray(start_command, dtype=float)
        floor = min(env.hold_floor(q0), math.inf if floor is None else floor)
        zones = env.keep_out_hold(q0, path[-1] if rest else None)
        goal = path[-1]
        settle = int(round(cfg.settle_s * cfg.rate_hz))
        ext = np.vstack([q0, path, np.repeat(goal[None], settle, axis=0)])
        lim = np.clip(ext, model.lo, model.hi)       # backends hold targets inside the joint limits
        direction = np.sign(lim[-1] - q0)
        tool = None if contact_goal is None else model.tool_points(lim)
        baseline = 0.0 if tool is None else float(np.linalg.norm(
            model.tool_points(q0) - model.tool_points(np.clip(start_command, model.lo, model.hi))))
        sent = np.zeros(len(ext))      # elapsed time each row was sent at (row 0: the measured start)
        seen = np.zeros(len(ext))      # elapsed time after each tick, and what was measured then
        measured = np.empty_like(ext)
        measured[0] = q0
        tools = None if tool is None else np.empty((len(ext), 3))
        if tools is not None:
            tools[0] = model.tool_points(q0)
        clock, first, following, state, settling, last = 0.0, 0, False, None, None, 0
        slow_since, free_short = None, 0.0
        tick_end = time.monotonic()
        for t in range(len(ext) - 1):
            if t >= len(path):
                settling = clock if settling is None else settling
                arrived = np.max(np.abs(state.q - goal)) <= SETTLED and judge_motion(
                    goal, q0, start_command, state.q, env.tolerance, held=held, requested=requested) is None
                if arrived or (clock - settling >= STILL_S and self._joints_still(measured, seen, t, clock)):
                    break
            with self._io:
                if self._abort.is_set():
                    return self._hold(name, f"{name}: stopped on request; the arm holds where it is.")
                sent[t + 1] = clock
                self._dev.step({name: ArmTarget(ext[t + 1], gripper, speed)}, self._dt)
            while sent[first + 1] < sent[t + 1] - cfg.tracking_lag_s - 1e-9:
                first += 1
            state = self._watch(name, ext[first + 1:t + 2], floor, rest, zones, track=tools is None)
            now = time.monotonic()
            clock += max(self._dt, now - tick_end)
            tick_end = now
            measured[t + 1], seen[t + 1] = state.q, clock
            last = t + 1
            if tools is None:
                continue
            tools[t + 1] = model.tool_points(state.q)
            lagged = lim[max(0, int(np.searchsorted(sent[:t + 2], clock - cfg.tracking_lag_s, side="right")) - 1)]
            commanded = np.abs(lagged - q0)
            progress = (state.q - q0) * direction
            following = following or bool(np.any((commanded >= math.radians(FOLLOW_DEG))
                                                  & (progress >= np.maximum(math.radians(FOLLOW_DEG),
                                                                            0.5 * commanded))))
            short = float(np.min(np.linalg.norm(tool[first + 1:t + 2] - tools[t + 1], axis=1)))
            ratio = self._progress_ratio(tool, sent, tools, seen, t + 1, clock, cfg.tracking_lag_s)
            if not following or ratio is None:
                slow_since = None
            elif ratio < CONTACT_RATIO:
                slow_since = clock if slow_since is None else slow_since
            else:
                slow_since = None
                if ratio >= FREE_RATIO:
                    free_short = max(free_short, short)
            if slow_since is not None and clock - slow_since >= STILL_S and t + 1 < len(path) and \
                    short > max(baseline, free_short) + cfg.contact_tolerance_m:
                return self._on_contact(name, contact_goal, state.q)
            window = np.clip(ext[first + 1:t + 2], model.lo, model.hi)
            off_path = float(np.min(np.max(np.abs(state.q - window), axis=1)))
            if off_path > env.tolerance and (not following or off_path > 2 * env.tolerance):
                lag = np.abs(state.q - window[-1])
                j = int(np.argmax(lag))
                return self._hold(name, f"{name}: blocked: {model.joint_names[j]} is {math.degrees(lag[j]):.1f} deg "
                                        f"off its path (tolerance {cfg.tracking_tolerance_deg:g} deg, lag "
                                        f"{cfg.tracking_lag_s:g} s"
                                        + (", and the arm never followed its command" if not following else "")
                                        + "). The arm holds where it is; look at the cameras.")
        if contact_goal is not None:
            self._ending[name] = (self._still_window(measured, seen, last), measured[last].copy(),
                                  tool[:len(path) + 1].copy())
        verdict = judge_motion(goal, q0, start_command, state.q, env.tolerance,
                               held=held, requested=requested)
        if verdict is None:
            return Result()
        j, kind = verdict
        joint = model.joint_names[j]
        what = {"drift": f"{joint} drifted {math.degrees(abs(state.q[j] - q0[j])):.1f} deg though it was not asked to "
                         "move",
                "short": f"{joint} ended {math.degrees(abs(state.q[j] - goal[j])):.1f} deg from its goal on a "
                         f"{math.degrees(abs(goal[j] - start_command[j])):.1f} deg move (blocked, or a standing error "
                         "larger than the motion allows)",
                "undetermined": f"undetermined: {joint} was asked to move "
                                f"{math.degrees(abs(goal[j] - start_command[j])):.2f} "
                                f"deg and moved {math.degrees(abs(state.q[j] - q0[j])):.2f} deg, too little to tell a "
                                "move from a block"}[kind]
        if kind == "short" and contact_goal is not None:
            return _BlockedShort("stopped", f"{name}: {what}. The arm holds where it is; look at the cameras.")
        return self._hold(name, f"{name}: {what}. The arm holds where it is; look at the cameras.")

    @staticmethod
    def _still_window(measured: np.ndarray, seen: np.ndarray, last: int) -> Optional[tuple]:
        """The joints measured at the last tick and the latest earlier tick at least STILL_S before it, and the
        elapsed time between: ``(q_past, q_last, span)``; None when the samples span less than STILL_S."""
        past = int(np.searchsorted(seen[:last + 1], seen[last] - STILL_S, side="right")) - 1
        if past < 0 or seen[last] - seen[past] < STILL_S:
            return None
        return measured[past].copy(), measured[last].copy(), float(seen[last] - seen[past])

    @staticmethod
    def _joints_still(measured: np.ndarray, seen: np.ndarray, last: int, now: float) -> bool:
        """Whether no measured joint moved more than JOINT_STILL_DEG over the last STILL_S of elapsed time."""
        if now < STILL_S:
            return False
        past = max(0, int(np.searchsorted(seen[:last + 1], now - STILL_S, side="right")) - 1)
        return float(np.max(np.abs(measured[last] - measured[past]))) < math.radians(JOINT_STILL_DEG)

    @staticmethod
    def _progress_ratio(tool: np.ndarray, sent: np.ndarray, tools: np.ndarray, seen: np.ndarray, last: int,
                        now: float, lag: float) -> Optional[float]:
        """How far the measured tool point advanced along its approach over the last STILL_S, as a fraction of how
        far the command advanced over the same span one tracking lag earlier; None while the command moved less
        than COMMAND_MOVED_M then, or before STILL_S plus the lag has passed."""
        if now < STILL_S + lag:
            return None

        def at(times, rows, when, upto):
            return rows[max(0, int(np.searchsorted(times[:upto + 1], when, side="right")) - 1)]

        command = at(sent, tool, now - lag, last) - at(sent, tool, now - lag - STILL_S, last)
        travel = float(np.linalg.norm(command))
        if travel <= COMMAND_MOVED_M:
            return None
        arm = tools[last] - at(seen, tools, now - STILL_S, last)
        return float(np.dot(arm, command / travel)) / travel

    def _watch(self, name: str, window: Optional[np.ndarray] = None, floor: Optional[float] = None,
               rest: bool = False, zones: Optional[np.ndarray] = None, track: bool = True) -> ArmState:
        """The state of the moving arm, after checking every arm for faults and the moving one for heat, its
        distance from the planned ``window`` of waypoints (unless ``track`` is false: the caller judges it), its
        measured clearance and its distance from each keep-out zone (held under ``zones``, per zone); the arm holds
        on any."""
        states = self._dev.read()
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
            if track and np.min(np.max(np.abs(state.q - window), axis=1)) > env.tolerance:
                j = int(np.argmax(lag))
                raise _Outcome(self._hold(name, f"{name}: blocked: {self.models[name].joint_names[j]} is "
                                                f"{math.degrees(lag[j]):.1f} deg off its path (tolerance "
                                                f"{cfg.tracking_tolerance_deg:g} deg, lag {cfg.tracking_lag_s:g} s). "
                                                "The arm holds where it is; look at the cameras."))
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
        """End the action with ``status`` and ``message`` after a confirmed hold of every arm that is on (``name``
        must be confirmed even when its torque state is unknown); a fault saying what was not confirmed
        otherwise."""
        problem = self.confirm_hold([name])
        others = self.confirm_hold([n for n in self.models if n != name]) if len(self.models) > 1 else None
        problem = "; ".join(p for p in (problem, others) if p)
        if problem:
            return Result("fault", f"{message} {problem}.")
        return Result(status, message)

    def _gripper(self, name: str, state: ArmState, opening: float) -> Result:
        """Move the gripper; the arm joints keep their command. The jaws have settled when their opening changed
        less than GRIP_STILL over GRIP_STILL_S of elapsed time.

        ==================================================  ===============================================
        evidence                                            outcome
        ==================================================  ===============================================
        within GRIP_REACHED of the request, or settled      done, not holding
        within GRIP_BAND of it
        closing, stalled (less than GRIP_HOLD_STILL over    done, holding
        GRIP_HOLD_S) more than GRIP_HOLD_GAP short of the
        request after closing GRIP_PROGRESS, or already
        closed on something
        anything else at GRIPPER_S (still moving or         ``stopped`` (undetermined, or blocked), the
        creeping, stalled without that progress, an         gripper held where it is
        opening that does not open)
        ==================================================  ===============================================
        """
        start = last = float(state.gripper)
        openings, times = [start], [0.0]
        clock = 0.0
        tick_end = time.monotonic()
        closing = opening < start - GRIP_STILL
        already = state.gripper_target <= opening + GRIP_REACHED and start > opening + GRIP_HOLD_GAP

        def change(span: float) -> Optional[float]:
            if clock < span:
                return None
            past = max(0, int(np.searchsorted(times, clock - span, side="right")) - 1)
            return abs(last - openings[past])

        verdict = None
        while clock < GRIPPER_S:
            with self._io:
                if self._abort.is_set():
                    return self._hold(name, f"{name}: gripper stopped on request; it holds its opening.")
                self._dev.step({name: ArmTarget(state.q_target, opening)}, self._dt)
            now_state = self._watch(name)
            now = time.monotonic()
            clock += max(self._dt, now - tick_end)
            tick_end = now
            last = float(now_state.gripper)
            openings.append(last)
            times.append(clock)
            settled, stalled = change(GRIP_STILL_S), change(GRIP_HOLD_S)
            if abs(last - opening) < GRIP_REACHED or (settled is not None and settled < GRIP_STILL
                                                      and abs(last - opening) <= (GRIP_BAND if closing else GRIP_OPEN_BAND)):
                verdict = "reached"
                break
            if stalled is not None and stalled < GRIP_HOLD_STILL and last > opening + GRIP_HOLD_GAP and \
                    ((closing and start - last >= GRIP_PROGRESS) or already):
                verdict = "holding"
                break
        grip = {"opening": round(last, 3), "target": opening, "holding": verdict == "holding",
                "settled": verdict is not None}
        if verdict == "reached":
            return Result(message=f"{name}: the jaws closed to {last:.2f} on nothing." if closing else "", grip=grip)
        if verdict == "holding":
            return Result(message=f"{name}: the jaws stopped at {last:.2f} (asked {opening:.2f}): they hold "
                                  "something.", grip=grip)
        settled = change(GRIP_STILL_S)
        if settled is None or settled >= GRIP_STILL:
            why = f"were still moving at the {GRIPPER_S:g} s limit (undetermined)"
        elif closing:
            why = "stalled before they had closed far enough to tell a grasp (undetermined)"
        else:
            why = "did not move as asked (blocked)"
        result = self._hold(name, f"{name}: the jaws {why} (at {last:.2f}, asked {opening:.2f}); they hold where "
                                  "they are. Look before relying on them.")
        result.grip = grip
        return result

    def _wait(self, seconds: float) -> Result:
        for _ in range(int(round(seconds * self.config.rate_hz))):
            with self._io:
                if self._abort.is_set():
                    return Result("stopped", "the wait ended early on request.")
                self._dev.step({}, self._dt)
        return Result()

    def _rest(self, name: str) -> tuple[Result, str]:
        """The rest action for one arm, and its torque outcome afterwards."""
        self._released.pop(name, None)
        result = self._guarded(lambda: self._rest_arm(name), hold=True)
        outcome = self._released.pop(name, None)
        if outcome is None:
            try:
                enabled = self._dev.read()[name].enabled
                outcome = {True: "holding", False: "released"}.get(enabled, "unknown")
            except Exception as e:
                outcome = f"error: {e}"
        return result, outcome

    def _rest_arm(self, name: str) -> Result:
        """Back to the rest pose along a checked path from where the arm is (straight there, else by way of
        home), then released there. A faulted arm, or one that does not arrive, is not released."""
        state = self._dev.read()[name]
        early = self._rest_precheck(name, state)
        if early is not None:
            if early.status == "rejected" and state.enabled:
                return self._hold(name, f"{early.message} It holds where it is.", "rejected")
            return early
        model, env = self.models[name], self.envelopes[name]
        self._commanded.pop(name, None)
        goal = np.clip(state.rest_q, model.lo, model.hi)
        legs = self._rest_legs(env, model, state.q, goal)
        start, command = state.q, state.q_target
        for k, path in enumerate(legs):
            if k:
                now = self._dev.read()[name]
                start, command = now.q, now.q_target
                try:
                    path = self._plan(env, model.plan_joint_path, start, goal)
                    env.check(start, path, capped=False, rest_q=goal)
                except MoveRejected as e:
                    return self._hold(name, f"{name}: the way down from home to rest is refused from where the arm "
                                            f"stopped ({e}); it holds there.")
            result = self._execute(name, start, path, env.speed_for(start, path), state.gripper_target, rest=True,
                                   floor=env.rest_floor(goal), start_command=command)
            if not result.ok:
                return result
        ended = time.monotonic()
        stale = self._wait_fresh(name, ended)
        if stale:
            return self._hold(name, f"{name}: {stale} after the last rest waypoint; the arm holds its pose and was not "
                                    "released.", "fault")
        with self._gate:                         # a stop either comes first (nothing is released) or finds it off
            if self._abort.is_set():
                return self._hold(name, f"{name}: stopped on request at the rest pose; the arm holds there and was "
                                        "not released.")
            final = self._dev.read()[name]
            if final.fault or not np.all(np.isfinite(final.q)):
                return self._hold(name, f"{name}: {final.fault or 'no joint feedback'} at the end of the way to rest; "
                                        "the arm holds its pose and was not released.", "fault")
            off = np.abs(final.q - state.rest_q)
            j = int(np.argmax(off))
            if not np.all(off <= math.radians(REST_TOLERANCE_DEG)):
                return self._hold(name, f"{name}: {model.joint_names[j]} stopped {math.degrees(off[j]):.0f} deg from "
                                        "the rest pose; the arm holds its pose and was not released.")
            outcome = self._released[name] = self._dev.rest(name)
        if outcome in OFF:
            return Result()
        return self._hold(name, f"{name}: at the rest pose but not released ({outcome}); the arm keeps holding. "
                                "The operator should check it.")

    def _wait_fresh(self, name: str, since: float) -> Optional[str]:
        """Wait up to FRESH_FEEDBACK_S for every joint of the arm to report a sample taken after ``since`` (a
        monotonic time), by the backend's ``feedback_ages``; None once they have (or when the backend's every read
        is fresh), else which joints have not."""
        deadline = since + FRESH_FEEDBACK_S
        while True:
            now = time.monotonic()                # before the query: a sample is no newer than now - its age
            ages = self._dev.feedback_ages(name)
            if ages is None:
                return None
            stale = [i for i, age in enumerate(ages) if not (age is not None and math.isfinite(age)
                                                            and now - age > since)]
            if not stale:
                return None
            if now > deadline:
                return "no fresh feedback from " + ", ".join(self._joint_label(name, i) for i in stale)
            time.sleep(0.01)

    def _rest_precheck(self, name: str, state: ArmState) -> Optional[Result]:
        """How a rest of this arm ends before it moves (a fault, already off, torque unknown, no rest pose), or
        None when it would move."""
        if state.fault:
            return Result("fault", f"{name}: {state.fault}. The arm is not moved; only its own controller holds it. "
                                   "The operator must support it and release it.")
        if state.enabled is False:
            return Result()
        if state.enabled is None:
            return Result("stopped", f"{name}: its torque state cannot be read and it was not turned on here, so "
                                     "nothing was moved.")
        if state.rest_q is None:
            return Result("rejected", f"{name}: no rest pose is recorded, so the arm is not moved or released.")
        return None

    def _take_over(self, name: str) -> None:
        """Attach an arm found holding (``serve --attach``) only with a recorded rest pose, a table calibration and
        a checked way from where it holds to rest; else the backend leaves it as it is and reports why."""
        state = self.backend.read()[name]
        env, model = self.envelopes[name], self.models[name]
        why = None
        if state.rest_q is None:
            why = "no rest pose is recorded for it"
        elif env.table is None:
            why = "no table calibration matches its setup"
        elif not np.all(np.isfinite(state.q)):
            why = "its joints report no position"
        else:
            try:
                self._rest_legs(env, model, state.q, state.rest_q)
            except MoveRejected as e:
                why = f"no checked way from where it holds to rest ({e})"
        if why is None:
            self.backend.attach(name)
        else:
            self.backend.refuse_attach(name, why)

    def _rest_legs(self, env: Envelope, model: ArmModel, q0: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Checked joint paths from ``q0`` to the rest pose: one straight there, or two by way of home (the
        second planned from home; it is planned again from where the first ends)."""
        with self._planning:
            return self._rest_plan(env, model, q0, goal)

    def _rest_plan(self, env: Envelope, model: ArmModel, q0: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        env.require_table()
        try:
            direct = self._plan(env, model.plan_joint_path, q0, goal)
            env.check(q0, direct, capped=False, rest_q=goal)
            return [direct]
        except MoveRejected as refusal:
            try:
                up = self._plan(env, model.plan_joint_path, q0, model.raised_q)
                env.check(q0, up, capped=False)
                down = self._plan(env, model.plan_joint_path, model.raised_q, goal)
                env.check(model.raised_q, down, capped=False, rest_q=goal)
                return [up, down]
            except MoveRejected:
                raise MoveRejected(f"{refusal} Nothing moved: lift the arm, then rest again.") from None

    # ── observation ──
    def _observe(self, result: Result, label: str, fresh_after: Optional[float] = None,
                 images: bool = True) -> Observation:
        """The arms and (with ``images``) one image per camera. With ``fresh_after`` (a monotonic time), waits up
        to FRESH_FRAME_S for frames captured after it; an older frame is marked stale. Camera and disk errors go
        in the cameras."""
        try:
            states = self._dev.read()
        except Exception as e:
            states = {}
            result.message = f"{result.message} (the arm state could not be read: {type(e).__name__}: {e})".lstrip()
        arms = {name: self._arm_observation(name, states[name]) for name in self.models if name in states}
        if not images:
            return Observation(self.steps, round(time.time(), 3), arms, {}, result)
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
            frames = self._dev.images()
            while after is not None and time.monotonic() < after + FRESH_FRAME_S and \
                    any(f.captured is not None and f.captured < after for f in frames.values()):
                time.sleep(0.02)
                frames = self._dev.images()
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
                if env.table is None else env.table.text()
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
                  f"{cfg.contact_allowance_m * 1000:.0f} mm under that stops the motion; a move with until contact "
                  f"may aim up to {cfg.contact_depth_m * 1000:.0f} mm under the floor, to end on contact;",
                  f"- one action turns no joint more than {cfg.max_excursion_deg:g} deg (home and rest are exempt): "
                  "split long motions;",
                  f"- no motion while a motor is above {cfg.max_temp_c:g} C; a joint more than "
                  f"{cfg.tracking_tolerance_deg:g} deg off its path (trailing its command by up to "
                  f"{cfg.tracking_lag_s:g} s) stops the motion as blocked;",
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


def _tile(image: Optional[np.ndarray], label: str, height: Optional[int] = None) -> np.ndarray:
    """An RGB image with ``label`` on a dark band at its top: scaled to ``height``, or without it at its own size
    scaled down only to fit TILE_MAX; a grey tile without an image."""
    if image is None:
        h = height or GRID_HEIGHT
        tile = np.full((h, h * 4 // 3, 3), 70, np.uint8)
    else:
        h, w = image.shape[:2]
        scale = height / h if height else min(1.0, TILE_MAX[0] / w, TILE_MAX[1] / h)
        tile = image if scale == 1.0 else cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                                                      interpolation=cv2.INTER_AREA)
    tile = np.ascontiguousarray(tile).copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(tile, label[:70], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _grid(tiles: list[np.ndarray]) -> np.ndarray:
    """Tiles in rows of GRID_COLUMNS, each row padded on the right to the widest and at the bottom to its
    tallest."""
    rows = []
    for i in range(0, len(tiles), GRID_COLUMNS):
        row = tiles[i:i + GRID_COLUMNS]
        tall = max(t.shape[0] for t in row)
        rows.append(np.hstack([np.pad(t, ((0, tall - t.shape[0]), (0, 0), (0, 0))) for t in row]))
    width = max(row.shape[1] for row in rows)
    return np.vstack([np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) for row in rows])


def _pose_action(pose, arm: Optional[str], line: bool) -> MoveTool:
    """The move_tool action for a pose given as ``[x, y, z]`` or ``{"xyz", "pitch", "yaw", "roll"}`` (degrees)."""
    if isinstance(pose, dict):
        unknown = sorted(str(k) for k in set(pose) - {"xyz", "pitch", "yaw", "roll"})
        if unknown:
            raise ActionError(f"a pose is [x, y, z] or {{xyz, pitch, yaw, roll}}; unknown keys {unknown}")
        raw = {"xyz": pose.get("xyz"), "pitch_deg": pose.get("pitch"), "yaw_deg": pose.get("yaw"),
               "roll_deg": pose.get("roll")}
    else:
        raw = {"xyz": pose.tolist() if isinstance(pose, np.ndarray) else pose}
    return parse_action({"action": "move_tool", **raw, "arm": arm, "line": bool(line)})


def _path_distance(path: np.ndarray, point: np.ndarray) -> float:
    """How far ``point`` lies from the polyline ``path`` (N, 3)."""
    path, point = np.asarray(path, dtype=float), np.asarray(point, dtype=float)
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    a, b = path[:-1], path[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    return float(np.min(np.linalg.norm(point - (a + t[:, None] * ab), axis=1)))


def _shortfall(goal: np.ndarray, start: np.ndarray, end: np.ndarray) -> Optional[float]:
    """How far ``end`` stopped short of ``goal`` along the approach from ``start`` (m; negative past it), or None for
    a move of the tool point shorter than COMMAND_MOVED_M."""
    approach = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(approach))
    if length < COMMAND_MOVED_M:
        return None
    return float(np.dot(np.asarray(goal, dtype=float) - np.asarray(end, dtype=float), approach / length))


def _verdict(ok: bool, why: Optional[str]) -> dict:
    return {"ok": ok, "why": why, "margin_deg": None, "margin_joint": None, "joints_deg": None, "seconds": None}


def judge_motion(goal: np.ndarray, start: np.ndarray, start_command: np.ndarray, end: np.ndarray, tolerance: float,
                 held: Optional[np.ndarray] = None, requested: Optional[np.ndarray] = None
                 ) -> Optional[tuple[int, str]]:
    """How a motion ended, from per joint evidence: None when it arrived, else (joint, kind) for the worst joint.

    For joint j: standing error before e0 = |start - start_command| (measured against the command it held) and
    after e1 = |end - goal|. A joint the caller named (``requested``) is judged from where it was measured: travel
    c = |goal - start|, measured travel m = (end - start) toward the goal. Any other moving joint is judged from the
    command it held: c = |goal - start_command|, m = (end - start) in that command's direction. allowed =
    max(END_ERROR_DEG, e0 + margin, END_FRACTION * c), at most ``tolerance``; noise = NOISE_DEG.

    ===================================================  ======================================================
    evidence                                             outcome
    ===================================================  ======================================================
    held (kept at its command), |end - start| > 2 deg     "drift"
    c >= noise, e1 <= allowed, and m >= c / 2 once        fine (under PROGRESS_MIN_DEG of travel a real arm's
    c >= PROGRESS_MIN_DEG                                 friction hides progress, so only the end error counts)
    c >= noise, named, |goal - start_command| < noise     "undetermined": the command barely changed (a request
                                                          that only asks to undo a standing error)
    c >= noise, otherwise                                 "short": it did not get there (blocked)
    c < noise, named, m >= c / 2 and e1 <= e0 + margin    fine
    c < noise, named, otherwise                           "undetermined": too small to tell a move from a block
    c < noise, not named, e1 <= allowed                   fine (a standing error that grew within the allowance)
    c < noise, not named, otherwise                       "undetermined"
    ===================================================  ======================================================
    """
    n = len(goal)
    held = np.zeros(n, bool) if held is None else np.asarray(held, bool)
    requested = np.zeros(n, bool) if requested is None else np.asarray(requested, bool)
    margin, noise, end_error = (math.radians(d) for d in (STANDING_MARGIN_DEG, NOISE_DEG, END_ERROR_DEG))
    progress = math.radians(PROGRESS_MIN_DEG)
    worst = None
    for j in range(n):
        e0, e1 = abs(start[j] - start_command[j]), abs(end[j] - goal[j])
        if held[j]:
            badness = abs(end[j] - start[j]) / end_error
            kind = "drift" if badness > 1 else None
        else:
            origin = start[j] if requested[j] else start_command[j]
            c = abs(goal[j] - origin)
            m = (end[j] - start[j]) * (1.0 if goal[j] >= origin else -1.0)
            allowed = min(tolerance, max(end_error, e0 + margin, END_FRACTION * c))
            if c >= noise:
                badness = e1 / allowed if c < progress else max(e1 / allowed, (0.5 * c) / max(m, 1e-12))
                kind = None if badness <= 1 else (
                    "undetermined" if requested[j] and abs(goal[j] - start_command[j]) < noise else "short")
            elif requested[j]:
                badness = e1 / (e0 + margin)
                if c > 1e-9:
                    badness = max(badness, (0.5 * c) / max(m, 1e-12))
                kind = "undetermined" if badness > 1 else None
            else:
                badness = e1 / allowed
                kind = "undetermined" if badness > 1 else None
        if kind and (worst is None or badness > worst[2]):
            worst = (j, kind, badness)
    return None if worst is None else (worst[0], worst[1])
