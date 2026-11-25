"""``llm-harness``: define a robot, measure its table, hold it open as a session and drive it one action
at a time, or hand it to an MCP client or a model loop.

    llm-harness robot init bench --arm metal --from-rig --arms left --cameras top left_wrist
    llm-harness calibrate --robot bench
    llm-harness serve --robot bench           # one terminal; `op quit` or Ctrl-C ends it and rests the arms
    llm-harness op home                       # another terminal: one action, then the observation
    llm-harness op move_tool 0.3 0 0.15 --pitch -90
    llm-harness py -c 'print(r.state().text())'   # code in the session's Python kernel
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml

from . import __version__
from . import robot as robots
from .actions import FRAMES, UNTIL, Observation
from .backends.base import ARM_TYPES, OFF, ArmState
from .harness import Harness
from .log import (EpisodeLog, latest_run, new_run_dir, profile, profile_text, read_episode, recorded_robot, runs_dir,
                  summary)
from .robot import RobotError, SIM_BACKENDS
from .safety import (REST_TOLERANCE_DEG, CalibrationError, Envelope, SafetyConfig, Table, load_calibration,
                     measure_table, save_calibration, table as stored_table, watch_for_touch)
from .session import AddressLock, Client, PrivateSession, Server, SessionError, default_address, ensure_free

ACTION_OPS = ("observe", "move_tool", "nudge", "gripper", "move_joints", "home", "rest", "wait")
CONTROL_OPS = ("stop", "status", "describe", "quit")
MAX_TOUCH_PITCH_DEG = -30.0   # a table touch counts only with the gripper pointing down at least this much
MAX_TABLE_OFFSET_M = 0.5      # a table height further than this from the arm base is taken for a typo
MAX_TOUCH_OPENING = 0.1       # a table touch counts only with the jaws closed this far
REST_GAP_S = 0.3              # the rest pose is read twice this far apart ...
REST_STILL_DEG = 0.5          # ... and both readings must agree this well
CAMERA_WARMUP_S = 5.0     # how long `robot check` waits for every camera's first image
QUIT_EXIT_S = 2.0         # after a quit has rested the arms of `mcp`, how long its loop gets to end before the exit

VOCABULARY = """\
actions (the same names as the MCP tools and the JSON actions):
  op observe                                   fresh camera images and arm state; moves nothing
  op move_tool X Y Z [--pitch D] [--yaw D] [--roll D] [--joint-path] [--until contact] [--arm A]
  op nudge DX DY DZ [--frame base|tool] [--until contact] [--arm A]
  op gripper open|close|0..1 [--arm A]
  op move_joints NAME=DEG ... [--relative] [--arm A]
  op home|rest [--arm A]
  op wait SECONDS
  op '{"action": "move_tool", "xyz": [0.3, 0, 0.1]}'   raw JSON action
  move_tool, nudge, move_joints and home also take [--speed DEG_S | --duration S]; those and gripper
  and rest take --dry-run (checked as they would be, nothing moves)
session control:
  op stop | status | describe | quit
Python kernel: llm-harness py -c CODE | llm-harness py < file.py | llm-harness py --wait [S]
Metres, degrees, world frame (z up). Exit status: 0 done; 1 rejected, stopped or fault (the text says why);
2 usage errors and no session running."""


class CommandError(Exception):
    """A usage mistake or a missing resource: printed as one line, exit status 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        command = self.prog.partition(" ")[2]
        raise CommandError(f"{command + ': ' if command else ''}{message} (see --help)")


def _user_errors() -> tuple[type, ...]:
    sdk_errors = tuple(importlib.import_module(f"{t}_arm.errors").ArmError for t in ARM_TYPES)
    return (CommandError, RobotError, SessionError, CalibrationError, OSError, *sdk_errors)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except SystemExit as e:      # --help
        return e.code if isinstance(e.code, int) else 0
    except _user_errors() as e:
        print(f"llm-harness: {e}", file=sys.stderr)
        return 2


# ── op: one action or session request ──
_CLIENT = argparse.ArgumentParser(add_help=False)
_CLIENT.add_argument("--address", help="session address (default: the one `llm-harness serve` uses)")
_CLIENT.add_argument("--json", action="store_true", help="print the observation (or reply) as JSON")
_ARM = argparse.ArgumentParser(add_help=False)
_ARM.add_argument("--arm", help="which arm; needed when the robot has two")


def _joint(text: str) -> tuple[str, float]:
    name, _, deg = text.partition("=")
    try:
        return name, float(deg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NAME=DEG, for example elbow_flex=30, got {text!r}") from None


def _opening(text: str):
    if text in ("open", "close"):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected open, close or a number 0..1, got {text!r}") from None


def _op_parser() -> _Parser:
    p = _Parser(prog="llm-harness op", description="One action or session request.", epilog=VOCABULARY,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="op", required=True, metavar="OP")

    def add(name: str, text: str, arm: bool = True, timing: bool = False) -> _Parser:
        q = sub.add_parser(name, help=text, description=text, parents=[_CLIENT, *([_ARM] if arm else [])])
        if timing:
            how = q.add_mutually_exclusive_group()
            how.add_argument("--speed", type=float, metavar="DEG_S", help="joint speed cap, below the envelope's")
            how.add_argument("--duration", type=float, metavar="S", help="take about this long")
        if arm:
            q.add_argument("--dry-run", action="store_true", help="check it as it would be checked; nothing moves")
        return q

    add("observe", "fresh camera images and arm state; moves nothing", arm=False)
    q = add("move_tool", "move the tool point in a straight line to X Y Z (metres, world frame)", timing=True)
    for axis in "xyz":
        q.add_argument(axis, metavar=axis.upper(), type=float)
    q.add_argument("--pitch", type=float, metavar="DEG", help="tilt: -90 straight down, 0 horizontal (default: keep)")
    q.add_argument("--yaw", type=float, metavar="DEG", help="heading about the vertical (default: away from the base)")
    q.add_argument("--roll", type=float, metavar="DEG", help="turn about the pointing direction (default: keep)")
    q.add_argument("--joint-path", action="store_true",
                   help="move the joints straight to the pose instead of the tool point in a straight line")
    q.add_argument("--until", choices=UNTIL, help="contact: end early, holding, where the arm stops following")
    q = add("nudge", "move the tool point by DX DY DZ metres, keeping its orientation", timing=True)
    for axis in "xyz":
        q.add_argument(axis, metavar=f"D{axis.upper()}", type=float)
    q.add_argument("--frame", choices=FRAMES, default="base", help="base: the arm's x forward, y left, z up; "
                   "tool: the gripper's axes, z its pointing direction (default: base)")
    q.add_argument("--until", choices=UNTIL, help="contact: end early, holding, where the arm stops following")
    q = add("gripper", "set the gripper opening: open, close, or 0 (closed) .. 1 (open)")
    q.add_argument("opening", type=_opening, metavar="open|close|0..1")
    q = add("move_joints", "move joints by name to angles in degrees; the others stay", timing=True)
    q.add_argument("joints", nargs="+", type=_joint, metavar="NAME=DEG")
    q.add_argument("--relative", action="store_true", help="add the angles to the current ones")
    add("home", "move to the ready pose, raised and clear of the table", timing=True)
    add("rest", "back to the rest pose, motors off (every arm unless --arm)")
    q = add("wait", "let SECONDS pass, then observe", arm=False)
    q.add_argument("seconds", type=float)
    add("stop", "end the action in progress; the arm holds where it is", arm=False)
    add("status", "robot, arms, cameras, step count, run directory, whether an action is running", arm=False)
    add("describe", "the robot, frames and safety envelope in words", arm=False)
    add("quit", "end the session; the server rests the arms", arm=False)
    return p


def _action(a: argparse.Namespace) -> dict:
    fields: dict = {"arm": getattr(a, "arm", None)}
    if a.op == "move_tool":
        fields.update(xyz=[a.x, a.y, a.z], pitch_deg=a.pitch, yaw_deg=a.yaw, roll_deg=a.roll, until=a.until,
                      line=False if a.joint_path else None)
    elif a.op == "nudge":
        fields.update(dxyz=[a.x, a.y, a.z], frame=a.frame, until=a.until)
    elif a.op == "gripper":
        fields.update(opening=a.opening)
    elif a.op == "move_joints":
        fields.update(joints_deg=dict(a.joints), relative=a.relative)
    elif a.op == "wait":
        fields.update(seconds=a.seconds)
    fields.update(speed_deg_s=getattr(a, "speed", None), duration_s=getattr(a, "duration", None))
    return {"action": a.op, **{k: v for k, v in fields.items() if v is not None}}


def parse_op(tokens: Sequence[str]) -> argparse.Namespace:
    """``op``'s arguments -> ``op``, ``action`` (a JSON object or text; None for session control),
    ``address``, ``json``."""
    tokens = list(tokens)
    if not tokens:
        raise CommandError("op: name an op: " + ", ".join(ACTION_OPS + CONTROL_OPS))
    if tokens[0].lstrip().startswith("{"):
        a = _Parser(prog="llm-harness op JSON", parents=[_CLIENT]).parse_args(tokens[1:])
        a.op, a.action = "step", tokens[0]
        return a
    a = _op_parser().parse_args(tokens)
    a.action = _action(a) if a.op in ACTION_OPS else None
    a.dry_run = getattr(a, "dry_run", False)
    return a


def _send(client: Client, op: argparse.Namespace, as_json: bool) -> int:
    """Run one parsed op on the session and print the outcome; the exit status."""
    if op.action is not None and getattr(op, "dry_run", False):
        result = client.check(op.action)
        print(json.dumps(dataclasses.asdict(result)) if as_json else
              f"{result.status}: {result.message}" if result.message else result.status)
        return 0 if result.ok else 1
    if op.action is not None:
        try:
            obs = client.step(op.action)
        except KeyboardInterrupt:
            client.stop()
            print("interrupted: stop sent; the arm holds where it is", file=sys.stderr)
            return 1
        print(json.dumps(obs.to_dict()) if as_json else obs.text())
        return 0 if obs.result.ok else 1
    reply = client.request(op.op)
    if as_json:
        print(json.dumps({"ok": True, **reply}))
    elif op.op == "describe":
        print(reply["text"])
    elif op.op == "status":
        print(f"robot {reply['robot']}: arms {', '.join(reply['arms'])}; "
              f"cameras {', '.join(reply['cameras']) or 'none'}; {reply['steps']} steps; run {reply['run_dir']}"
              + ("; read-only" if reply["read_only"] else "") + ("; an action is running" if reply["busy"] else ""))
    elif op.op == "quit":
        print("session ended")
        return report_outcomes(reply.get("arms", {})) if reply.get("arms") else 0
    else:
        print("stop sent")
    return 0


def cmd_op(args) -> int:
    op = parse_op(args.command)
    return _send(Client(op.address or args.address), op, op.json or args.json)


def cmd_ops(args) -> int:
    ops = [parse_op(shlex.split(command)) for command in args.commands]    # all parsed before any is sent
    client = Client(args.address)
    for command, op in zip(args.commands, ops):
        if not args.json:
            print(f"> {command}")
        code = _send(client, op, args.json or op.json)
        if code:
            return code
    return 0


def cmd_py(args) -> int:
    if args.code is not None:
        code = args.code
    elif args.wait is None:
        if sys.stdin.isatty():
            raise CommandError("py: give the code with -c CODE or on stdin, or --wait for the next report")
        code = sys.stdin.read()
    else:
        code = ""
    client = Client(args.address, timeout_s=args.timeout)
    try:
        reply = client.py(code, args.wait, timeout_s=args.timeout)
    except KeyboardInterrupt:
        client.stop()
        print("interrupted: stop sent; the arm holds where it is", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(reply))
    else:
        from .kernel import format_reply
        print(format_reply(reply))
    client.ack()                             # the reports printed here are not sent again
    return 1 if reply["error"] else 0


def cmd_rest(args) -> int:
    return _send(Client(args.address), argparse.Namespace(op="rest", action={"action": "rest"}), args.json)


# ── serve / sim ──
def _episode_log(robot_name: str, run_dir: Optional[str] = None, resume: Optional[str] = None) -> EpisodeLog:
    """A new run directory, ``run_dir``, or with ``resume`` an earlier run of this robot ("" = its latest)."""
    if resume is None:
        return EpisodeLog(Path(run_dir).expanduser() if run_dir else new_run_dir(robot_name))
    if resume:
        path = Path(resume).expanduser()
    else:
        runs = [p.parent for p in runs_dir().glob("*/episode.jsonl") if recorded_robot(p) == robot_name]
        if not runs:
            raise CommandError(f"no earlier run of {robot_name} in {runs_dir()} to resume")
        path = max(runs, key=lambda p: (p / "episode.jsonl").stat().st_mtime)
    if not (path / "episode.jsonl").is_file():
        raise CommandError(f"no episode to resume in {path}")
    if recorded_robot(path) != robot_name:
        raise CommandError(f"{path} is an episode of robot {recorded_robot(path)!r}, not {robot_name!r}")
    return EpisodeLog.resume(path)


def _open_logged(robot: robots.Robot, log: EpisodeLog, **options) -> Harness:
    try:
        return Harness.open(robot, log=log, **options)
    except BaseException:
        log.close()
        raise


class Lifecycle:
    """Owns a harness the CLI opened, and its session (``server``) when there is one. While inside it, SIGINT and
    SIGTERM raise KeyboardInterrupt in the main thread (motions run on other threads). Leaving it, by any way out
    (an exception, a signal, a quit, the end of the block), runs ``close_session``, ignoring further signals until
    it is done. A KeyboardInterrupt ends the block quietly; any other exception goes on after the arms are dealt
    with."""

    def __init__(self, harness: Harness, server: Optional[Server] = None, say: Callable[[str], None] = print):
        self.harness, self.server, self.say = harness, server, say
        self.outcomes: dict[str, str] = {}
        self.code = 1                          # 0 once every arm is confirmed off (or a simulator's)
        self.interrupted = False
        self._previous: dict = {}
        self._closing = threading.Lock()
        self._closed = False

    def close_now(self) -> None:
        """Run ``close_session`` once, from any thread (a quit's own, for an owner that cannot be woken)."""
        with self._closing:
            if not self._closed:
                self.outcomes, self.code = close_session(self.harness, self.server, self.say)
                self._closed = True

    def __enter__(self) -> "Lifecycle":
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM"):
                if hasattr(signal, name):
                    self._previous[name] = signal.signal(getattr(signal, name), _interrupt)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for name in self._previous:
            signal.signal(getattr(signal, name), signal.SIG_IGN)     # nothing cuts the rest short
        try:
            if exc_type is KeyboardInterrupt:
                self.interrupted = True
                _best_effort(self.say)("interrupted; resting the arms")
            self.close_now()
        finally:
            for name, handler in self._previous.items():
                signal.signal(getattr(signal, name), handler)
        return exc_type is KeyboardInterrupt


def close_session(harness: Harness, server: Optional[Server] = None,
                  say: Callable[[str], None] = print) -> tuple[dict[str, str], int]:
    """The one way a session the CLI owns ends. The session's part (shutdown phase, kernel ended) comes first and
    its failures are only reported; the arms' part always runs after it, whatever failed before: a confirmed hold,
    the requests still running drained, every arm rested and released (``Harness.close``: backend closed, episode
    log ended), each arm's outcome reported and handed to a waiting quit, and last the address given up. Every
    line it prints is best-effort (``say`` failing, a closed stdout, changes nothing). Returns the outcomes and the
    exit code (0 when every arm is off)."""
    say = _best_effort(say)
    outcomes: dict[str, str] = {}
    try:
        if server is not None:
            server.shutdown()
            for problem in server.shutdown_problems:
                say(f"shutdown: {problem}")
    except Exception as e:
        say(f"shutdown: the session did not close cleanly ({type(e).__name__}: {e}); resting the arms anyway")
    finally:
        try:
            try:
                problem = harness.stop()
                if problem:
                    say(f"shutdown: the hold before the rest is not confirmed: {problem}")
            except Exception as e:
                say(f"shutdown: the hold before the rest failed ({type(e).__name__}: {e})")
            if server is not None:
                try:
                    server.drain()
                except Exception as e:
                    say(f"shutdown: running requests were not drained ({type(e).__name__}: {e})")
            outcomes = harness.close()
            report_outcomes(outcomes, say, getattr(harness.backend, "recovery", None))
            if server is not None:
                server.finish(outcomes)
        finally:
            if server is not None:
                server.release()
    return outcomes, 0 if outcomes and all(w in OFF for w in outcomes.values()) else 1


def _best_effort(say: Callable[[str], None]) -> Callable[[str], None]:
    def write(line: str) -> None:
        try:
            say(line)
        except Exception:
            pass
    return write


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def report_outcomes(outcomes: dict[str, str], say: Callable[[str], None] = print,
                    recovery: Optional[Callable[[str], str]] = None) -> int:
    """Each arm's torque outcome in words, with ``recovery(arm)`` (how to bring it down) for an arm that may
    still hold torque; 0 when none may, else 1."""
    if outcomes and all(w == "released" for w in outcomes.values()):
        say("Every arm is at rest with its motors off.")
        return 0
    for name, word in outcomes.items():
        if word == "released":
            say(f"{name}: at rest, motors off.")
        elif word == "released_unrested":
            say(f"{name}: motors off where the operator supports it; it is NOT at its rest pose.")
        elif word == "always_on":
            say(f"{name}: at rest (simulated: its motors are always on).")
        else:
            how = recovery(name) if recovery else "Support it before you turn its motors off."
            say(f"{name} is NOT confirmed off ({word}). {how}")
    return 0 if all(w in OFF for w in outcomes.values()) else 1


def _say_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _ask_operator(texts: dict[str, str]) -> bool:
    """At exit, for arms that could not be brought to rest and still hold: True once the operator types RELEASE
    (torque off where they are), False on Enter (leave them holding)."""
    for name, text in texts.items():
        _say_stderr(f"{name} could not be brought to rest and holds its pose. {text}")
    _say_stderr("Press Enter to leave it holding and end the session, or type RELEASE once it is supported to turn "
                "its torque off here.")
    while True:
        try:
            answer = input().strip().upper()
        except EOFError:
            return False
        if answer in ("", "RELEASE"):
            return answer == "RELEASE"


def serve(robot: robots.Robot, address: Optional[str] = None, run_dir: Optional[str] = None,
          resume: Optional[str] = None, read_only: bool = False, attach: bool = False) -> int:
    """Hold ``robot`` open as a session until `op quit`, Ctrl-C or SIGTERM; then rest the arms and close.
    ``attach`` takes over arms found holding with torque on from an earlier session."""
    if attach and robot.backend != "real":
        raise CommandError("--attach is for real arms")
    address = address or default_address()
    lock = AddressLock(address)              # owned before any device opens, until the server has cleaned up
    try:
        ensure_free(address)
        harness = _open_logged(robot, _episode_log(robot.name, run_dir, resume), read_only=read_only,
                               backend_options={"attach": True} if attach else None)
    except BaseException:
        lock.release()
        raise
    if sys.stdin.isatty() and hasattr(harness.backend, "ask_operator"):
        harness.backend.ask_operator = _ask_operator
    try:
        server = Server(harness, address, say=_say_stderr, reports_close=True, lock=lock)
    except BaseException:
        report_outcomes(harness.close())
        raise
    print(harness.describe())
    print(f"\nSession at {server.address}; run directory {harness.log.run_dir} (step {harness.steps}).")
    print("Drive it with `llm-harness op ...` or `llm-harness py`; `llm-harness op quit` or Ctrl-C ends it and "
          "rests the arms.", flush=True)
    with Lifecycle(harness, server) as life:
        server.serve_forever()
    return life.code


def cmd_serve(args) -> int:
    return serve(robots.load(args.robot), args.address, args.run_dir, args.resume, args.read_only, args.attach)


def cmd_sim(args) -> int:
    robot = robots.simulated("mujoco" if args.mujoco else "sim", args.arm, args.arms)
    return serve(robot, args.address, args.run_dir)


# ── robot ──
def cmd_robot_init(args) -> int:
    if args.name in robots.robot_names() and not args.force:
        raise CommandError(f"robot {args.name!r} exists; pass --force to replace it")
    if args.from_rig is not None:
        if not args.arm:
            raise CommandError("--from-rig needs --arm metal or --arm maker")
        robot = robots.from_rig(args.name, args.arm, args.from_rig or None, args.arms, args.cameras)
    elif args.sim or args.mujoco:
        robot = robots.simulated("mujoco" if args.mujoco else "sim", args.arm or "metal", args.arms or ("arm",),
                                 args.name)
        robot.cameras = {c: "" for c in args.cameras or ()}
    else:
        raise CommandError("say where the robot comes from: --from-rig [RIG.yaml], --sim or --mujoco")
    path = robots.save(robot)
    print(f"wrote {path}\n{yaml.safe_dump(robot.to_dict(), sort_keys=False).rstrip()}")
    if robot.backend == "real" and len(robot.arms) > 1:
        print(f"\nTwo arms: edit `bases:` in {path} to say where each arm base sits in the world frame "
              "(xyz in metres, yaw_deg); both start at the origin.")
    if robot.backend == "real":
        print(f"\nNext: llm-harness robot check {robot.name}, then llm-harness calibrate --robot {robot.name} "
              "(table height and rest pose)")
    return 0


def cmd_robot_list(args) -> int:
    names = robots.robot_names()
    if not names:
        print(f"no robots in {robots.robots_dir()}; create one with `llm-harness robot init`")
    for name in names:
        try:
            r = robots.load(name)
            print(f"{name}: {r.backend} {r.arm}, arms {' '.join(r.arms)}, cameras {' '.join(r.cameras) or 'default'}")
        except RobotError as e:
            print(f"{name}: invalid: {e}")
    return 0


def cmd_robot_show(args) -> int:
    robot = robots.load(args.name)
    print(f"# {robot.path}\n{yaml.safe_dump(robot.to_dict(), sort_keys=False).rstrip()}")
    return 0


def _first_look(harness: Harness) -> Observation:
    """An observation, retried for a few seconds while a camera has not delivered its first image."""
    deadline = time.monotonic() + CAMERA_WARMUP_S
    while True:
        obs = harness.observe()
        if all(cam.error is None for cam in obs.cameras.values()) or time.monotonic() > deadline:
            return obs
        time.sleep(0.5)


def cmd_robot_check(args) -> int:
    robot = robots.load(args.name)
    out = Path(args.out).expanduser() if args.out else \
        robots.home() / "checks" / f"{time.strftime('%Y%m%d-%H%M%S')}-{robot.name}"
    stored = load_calibration(robot.name)
    with Harness.open(robot, read_only=True, image_dir=out) as harness:
        obs = _first_look(harness)
        print(harness.describe() + "\n\n" + obs.text())
        for name, arm in obs.arms.items():
            temperature = "unknown" if arm.temperature_c is None else f"{arm.temperature_c:.0f} C"
            table = harness.envelopes[name].table_z
            if table is None:
                calibration = f"not calibrated: run `llm-harness calibrate --robot {robot.name} --arm {name}`"
            elif harness.backend.table_z is not None:
                calibration = f"table z = {table:.4f} m from the simulator"
            else:
                measured = time.strftime("%Y-%m-%d %H:%M", time.localtime(stored[name]["measured_at"]))
                calibration = f"table {harness.envelopes[name].table.text()}, measured {measured}"
            state = harness.backend.read()[name]
            torque = {True: "on", False: "off"}.get(state.enabled, "not known")
            if state.rest_q is None:
                rest = f"rest pose not recorded: run `llm-harness calibrate --robot {robot.name} --arm {name} --rest`"
            else:
                rest = f"{math.degrees(np.max(np.abs(state.q - state.rest_q))):.1f} deg from its rest pose"
            print(f"arm {name}: hottest motor {temperature}; motors {torque}; {rest}; {calibration}")
            note = None if state.rest_q is None else harness.envelopes[name].rest_note(state.rest_q)
            if note:
                print(note)
            for text in harness.envelopes[name].keep_out_inside(state.q):
                print(f"arm {name}: {text}; it may only move out of it")
            for text in ([] if state.rest_q is None else harness.envelopes[name].keep_out_inside(state.rest_q)):
                print(f"arm {name}: at its recorded rest pose {text}")
        errors = dict(getattr(harness.backend, "camera_errors", {}))
    errors.update({n: cam.error for n, cam in obs.cameras.items() if cam.error and n not in errors})
    for name, error in errors.items():
        print(f"camera {name}: {error}")
    faults = {n: a.fault for n, a in obs.arms.items() if a.fault}
    print(f"images in {out}")
    print("OK" if not faults and not errors else
          "PROBLEMS: " + ", ".join([f"arm {n} fault" for n in faults] + [f"camera {n}" for n in errors]))
    return 1 if faults or errors else 0


def cmd_keep_out(args) -> int:
    robot = robots.load(args.robot)
    names = [zone.name for zone in robot.zones()]
    if args.keep_out_command == "add":
        if args.name in names:
            raise CommandError(f"{robot.name} already has a keep-out zone named {args.name!r}; remove it first")
        robot.keep_out = list(robot.keep_out) + [{"name": args.name, "min": args.min, "max": args.max}]
        path = robots.save(robot)
        print(f"added keep-out zone {robot.zones()[-1].text()} to {path}")
    elif args.keep_out_command == "remove":
        if args.name not in names:
            raise CommandError(f"{robot.name} has no keep-out zone named {args.name!r} (zones: "
                               f"{', '.join(names) or 'none'})")
        robot.keep_out = [zone for zone in robot.keep_out if zone["name"] != args.name]
        print(f"removed keep-out zone {args.name} from {robots.save(robot)}")
    else:
        for zone in robot.zones():
            print(zone.text())
        if not names:
            print(f"{robot.name} has no keep-out zones")
    return 0


# ── calibrate ──
def _sample(harness: Harness, arm: str, motors_off: bool = False) -> ArmState:
    """One fresh reading of an arm guided by hand: no fault, every joint read, torque confirmed off.
    ``motors_off`` is the operator's word that they are off, for motors that do not report it."""
    state = harness.backend.read()[arm]
    if state.fault:
        raise CalibrationError(f"{arm}: {state.fault}")
    if not np.all(np.isfinite(state.q)):
        raise CalibrationError(f"{arm}: no joint feedback")
    if state.enabled is not False and not (motors_off and state.enabled is None):
        recovery = getattr(harness.backend, "recovery", None)
        how = recovery(arm) if recovery else "Turn its motors off"
        raise CalibrationError(f"{arm}: its torque is {'on' if state.enabled else 'not confirmed off'}, and a reading "
                               f"taken by hand needs the motors off. {how}, then try again"
                               + ("" if state.enabled else " (motors that cannot report their torque: check by hand "
                                                           "that the arm moves freely, then add --motors-off)"))
    return state


def _touch_table(harness: Harness, robot: robots.Robot, arm: str, watch: bool, ask, say,
                 motors_off: bool = False) -> float:
    model = harness.models[arm]

    def read_z() -> float:
        return float(model.fingertip(_sample(harness, arm, motors_off).q)[2])

    def read_touch() -> float:
        """The fingertip height, from a sample that also shows the gripper pointing down with its jaws closed."""
        state = _sample(harness, arm, motors_off)
        pitch = math.degrees(model.tool_angles(state.q)[1])
        if pitch > MAX_TOUCH_PITCH_DEG:
            raise CalibrationError(f"{arm}: the gripper points {pitch:.0f} deg from horizontal; the fingertips are "
                                   f"the lowest part only when it points down (below {MAX_TOUCH_PITCH_DEG:.0f} deg). "
                                   "Nothing was stored; touch the table again with the gripper pointing down.")
        if not state.gripper <= MAX_TOUCH_OPENING:
            raise CalibrationError(f"{arm}: the gripper is open ({state.gripper:.2f}); close the jaws and touch "
                                   "again. Nothing was stored.")
        return float(model.fingertip(state.q)[2])

    say(f"Arm {arm}: point the gripper down at the table, jaws closed.")
    z = measure_table(read_touch, ask=watch_for_touch(read_z, say=say) if watch else ask, say=say)
    path = save_calibration(robot.name, arm, z, robot.setup(arm, harness.config))
    say(f"{arm}: stored table z = {z:.4f} m (world frame) in {path}.")
    return z


def _record_rest(harness: Harness, robot: robots.Robot, arm: str, yes: bool, ask, say,
                 motors_off: bool = False) -> list[float]:
    model = harness.models[arm]
    say(f"Arm {arm}: fold it into its rest pose, resting on its support, motors off.")
    if not yes and ask(f"Is {arm} folded in its rest pose and resting on its support? [y/N] ").strip().lower() \
            not in ("y", "yes"):
        raise CalibrationError(f"{arm}: no rest pose recorded")
    first = _sample(harness, arm, motors_off).q
    time.sleep(REST_GAP_S)
    q = _sample(harness, arm, motors_off).q
    if np.max(np.abs(q - first)) > math.radians(REST_STILL_DEG):
        raise CalibrationError(f"{arm}: the arm moved while it was read; let it rest still, then try again")
    past = np.maximum(model.lo - q, q - model.hi)
    j = int(np.argmax(past))
    if past[j] > math.radians(REST_TOLERANCE_DEG):
        raise CalibrationError(f"{arm}: {model.joint_names[j]} rests {math.degrees(past[j]):.1f} deg past its limit "
                               f"[{math.degrees(model.lo[j]):.0f}, {math.degrees(model.hi[j]):.0f}] deg; turned on, it "
                               f"would be brought inside by more than the {REST_TOLERANCE_DEG:g} deg rest tolerance. "
                               "Fold the arm within its limits, or correct the profile's limits. Nothing was recorded.")
    robot.rest_q[arm] = [round(math.degrees(v), 2) for v in q]
    path = robots.save(robot)
    say(f"{arm}: rest pose recorded in {path}: " + ", ".join(f"{n} {d:.1f}" for n, d in zip(model.joint_names,
                                                                                           robot.rest_q[arm])))
    return robot.rest_q[arm]


def calibrate(robot: robots.Robot, arms: Sequence[str], watch: bool = False, table: bool = True,
              rest: bool = True, yes: bool = False, motors_off: bool = False,
              ask: Callable[[str], str] = input, say: Callable[[str], None] = print) -> dict[str, float]:
    """For each arm: measure the table by touching it with the fingertips and store it (``table``), then
    record its folded rest pose (``rest``; ``yes`` skips the question whether it is folded and supported).
    The robot is opened read-only: the arms stay off, moved by hand, and every reading needs torque off
    confirmed by the motors, or stated by the operator (``motors_off``) for motors that do not report it."""
    heights = {}
    with Harness.open(robot, read_only=True) as harness:
        for arm in arms:
            if table:
                heights[arm] = _touch_table(harness, robot, arm, watch, ask, say, motors_off)
            if rest:
                _record_rest(harness, robot, arm, yes, ask, say, motors_off)
            table_z = heights.get(arm, harness.envelopes[arm].table_z)
            if arm in robot.rest_q:
                note = Envelope(harness.models[arm], harness.config, table_z).rest_note(np.radians(robot.rest_q[arm]))
                if note:
                    say(note)
    return heights


def cmd_calibrate(args) -> int:
    robot = robots.load(args.robot)
    if robot.backend in SIM_BACKENDS:
        raise CommandError(f"{robot.name} is a simulator robot: its table height is scene.table_z and its rest pose "
                           "the simulator's; there is nothing to calibrate")
    if args.arm and args.arm not in robot.arms:
        raise CommandError(f"{robot.name} has no arm named {args.arm!r}; its arms are {', '.join(robot.arms)}")
    arms = [args.arm] if args.arm else list(robot.arms)
    if args.table_z is not None:
        bases = {info.name: info.base_xyz[2] for info in robot.arm_infos()}
        for arm in arms:
            if abs(args.table_z - bases[arm]) > MAX_TABLE_OFFSET_M:
                raise CommandError(f"{arm}: table z {args.table_z} m is more than {MAX_TABLE_OFFSET_M} m from the "
                                   f"arm base (z = {bases[arm]} m); nothing was stored")
            path = save_calibration(robot.name, arm, args.table_z, robot.setup(arm))
            print(f"{arm}: stored table z = {args.table_z:.4f} m (world frame) in {path}")
        return 0
    if args.powered:
        calibrate_powered(robot, arms, points=args.points, expect=args.expect, yes=args.yes)
        return 0
    if args.points or args.expect is not None:
        raise CommandError("--points and --expect go with --powered")
    calibrate(robot, arms, watch=args.watch, table=not args.rest, yes=args.yes, motors_off=args.motors_off)
    return 0


def calibrate_powered(robot: robots.Robot, arms: Sequence[str], points: Optional[Sequence[tuple]] = None,
                      expect: Optional[float] = None, yes: bool = False, ask: Callable[[str], str] = input,
                      say: Callable[[str], None] = print) -> dict[str, Table]:
    """For each arm: the powered table calibration (``probe.probe_table``) at ``points`` (world x, y; default
    straight ahead at three reaches), stored as a table per reach. ``expect``: roughly where the table is (world z);
    without it, the table stored for the arm, which must have been measured with its present setup
    (``Robot.setup``), at each point's own reach. The fingertips are never sent lower than ``probe.BELOW_M`` under
    the expected height at the point. The arm must be at its recorded rest pose and ends there."""
    from . import probe
    stored = load_calibration(robot.name)
    tables = {}
    for arm in arms:
        if expect is not None:
            expected = Table(float(expect))
        elif arm in stored and stored[arm]["setup"] == robot.setup(arm):
            expected = stored_table(stored[arm])
        else:
            raise CalibrationError(f"{arm}: no table is stored for this arm's present setup (base pose, finger "
                                   "length, bus); give --expect Z, roughly where the table is")
        limit = expected.shifted(-probe.BELOW_M)
        count = len(points or probe.DEFAULT_REACHES_M)
        if not yes and ask(f"Arm {arm} will move on its own: down to the table at {count} points, slowly, the "
                           f"fingertips never more than {probe.BELOW_M * 100:.0f} cm under the expected table "
                           f"({expected.text()}). Keep the workspace clear and a hand on the emergency stop. Go on? "
                           "[y/N] ").strip().lower() not in ("y", "yes"):
            raise CalibrationError(f"{arm}: not calibrated")
        safety = dataclasses.replace(SafetyConfig.from_dict(robot.safety), contact_tolerance_m=probe.SHORT_M)
        with Harness.open(robot, table_z={arm: limit}, safety=safety) as harness:
            where = [tuple(p) for p in points] if points else probe.default_points(harness, arm)
            found = probe.probe_table(harness, arm, where, expected, say=say)
            measured = Table.measured([(reach, z) for reach, z, _ in found])
            path = save_calibration(robot.name, arm, measured, robot.setup(arm, harness.config))
        say(f"{arm}: stored table {measured.text()} in {path}")
        tables[arm] = measured
    return tables


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a probe point is X,Y in metres, got {text!r}") from None
    return x, y


# ── mcp / run ──
def _optional_module(name: str, extra: str):
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError as e:
        raise CommandError(f"this needs the {extra} extra: pip install 'makermods-llm-harness[{extra}]' ({e})") from e


def _target(args):
    """A session Client (--connect), or a ``PrivateSession`` on a robot opened here with a new episode log: the
    model then reaches it through a session of this process's own, as it would a served one."""
    if args.connect is not None:
        return Client(args.connect or None)
    if args.sim or args.mujoco:
        robot = robots.simulated("mujoco" if args.mujoco else "sim", args.arm, args.arms)
    else:
        robot = robots.load(args.robot)
    harness = _open_logged(robot, EpisodeLog(new_run_dir(robot.name)))
    print(f"run directory {harness.log.run_dir}", file=sys.stderr)
    try:
        session = PrivateSession(harness)
        print(f"session at {session.server.address}", file=sys.stderr, flush=True)
        return session
    except BaseException:
        report_outcomes(harness.close(), _say_stderr)
        raise


def cmd_mcp(args) -> int:
    mcp_server = _optional_module("mcp_server", "mcp")
    target = _target(args)
    if isinstance(target, Client):
        mcp_server.run(target)
        return 0
    stops = []

    def quit_now(life: Lifecycle) -> None:
        """A quit: rest and release here (its reply waits for the outcomes), then end serving. The stdio loop may
        be blocked reading an idle stdin, so the process ends once the arms are dealt with."""
        def close() -> None:
            life.close_now()
            for stop in stops:
                stop()
            time.sleep(QUIT_EXIT_S)
            os._exit(life.code)
        threading.Thread(target=close, daemon=True).start()

    try:
        with Lifecycle(target.harness, target.server, say=_say_stderr) as life:   # stdout carries the MCP protocol
            target.server.on_quit = lambda: quit_now(life)
            mcp_server.run(target.client, on_ready=stops.append)
    finally:
        target.cleanup()
    return life.code


def cmd_run(args) -> int:
    agent = _optional_module("agent", "agent")
    options = {"max_steps": args.max_steps, "timeout_s": args.timeout, **({"model": args.model} if args.model else {})}
    target = _target(args)
    if isinstance(target, Client):
        result = agent.run(target, args.task, **options)
        print(f"{result.status} after {result.steps} steps: {result.message}")
        return 0 if result.status == "done" else 1
    result = None
    try:
        with Lifecycle(target.harness, target.server) as life:
            target.harness.log.note(f"task: {args.task}")
            result = agent.run(target.client, args.task, **options)
            print(f"{result.status} after {result.steps} steps: {result.message}")
    finally:
        target.cleanup()
    return 0 if result is not None and result.status == "done" and life.code == 0 else 1


# ── log ──
def _run_dir(arg: Optional[str]) -> Path:
    path = Path(arg).expanduser() if arg else latest_run()
    if path is None:
        raise CommandError(f"no runs in {runs_dir()} yet")
    if not (path / "episode.jsonl").is_file():
        raise CommandError(f"no episode in {path}")
    return path


def _action_text(action: Optional[dict]) -> str:
    if not action:
        return "(unreadable action)"
    return " ".join([str(action.get("action")), *(f"{k}={json.dumps(v)}" for k, v in action.items() if k != "action")])


def cmd_log_show(args) -> int:
    path = _run_dir(args.run_dir)
    print(path)
    for e in read_episode(path):
        if e.get("event") == "start":
            print(f"start: robot {e.get('robot')}, arms {', '.join(e.get('arms', []))}")
        elif e.get("event") == "note":
            print(f"note: {e.get('text')}")
        elif e.get("event") == "step":
            result = e.get("observation", {}).get("result", {})
            message = f": {result['message']}" if result.get("message") else ""
            print(f"{e.get('step'):>4}  {_action_text(e.get('action'))} -> {result.get('status', '?')}{message}")
    return 0


def cmd_log_profile(args) -> int:
    path = _run_dir(args.run_dir)
    print(json.dumps(profile(path)) if args.json else profile_text(path, steps=not args.brief))
    return 0


def cmd_log_resume(args) -> int:
    print(summary(_run_dir(args.run_dir)))
    return 0


# ── parser ──
def build_parser() -> _Parser:
    p = _Parser(prog="llm-harness", description="Let an LLM drive Metal or Maker arms: tool targets in, camera "
                "images and arm state out, behind a table-floor safety envelope.", epilog=VOCABULARY,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"llm-harness {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sim_options = argparse.ArgumentParser(add_help=False)
    sim_options.add_argument("--arm", choices=ARM_TYPES, default="metal", help="arm type (default: metal)")
    sim_options.add_argument("--arms", nargs="+", default=["arm"], metavar="NAME", help="arm names (default: arm)")
    session = argparse.ArgumentParser(add_help=False)
    session.add_argument("--address", help="session address (default: <home>/session.sock)")
    session.add_argument("--run-dir", metavar="DIR", help="episode directory (default: a new one under <home>/runs)")

    robot = sub.add_parser("robot", help="define and inspect robots")
    rsub = robot.add_subparsers(dest="robot_command", required=True, metavar="ACTION")
    q = rsub.add_parser("init", help="write a robot file from an SDK rig file, or a simulator robot")
    q.add_argument("name")
    source = q.add_mutually_exclusive_group()
    source.add_argument("--from-rig", nargs="?", const="", metavar="RIG.yaml",
                        help="real arms and cameras from an SDK rig file (default: the SDK's rig file)")
    source.add_argument("--sim", action="store_true", help="the kinematic simulator")
    source.add_argument("--mujoco", action="store_true", help="the MuJoCo simulator")
    q.add_argument("--arm", choices=ARM_TYPES, help="arm type (required with --from-rig; default metal)")
    q.add_argument("--arms", nargs="+", metavar="NAME", help="arms to use (default: all in the rig, or one 'arm')")
    q.add_argument("--cameras", nargs="+", metavar="NAME", help="cameras to use (default: all in the rig)")
    q.add_argument("--force", action="store_true", help="replace an existing robot file")
    q.set_defaults(func=cmd_robot_init)
    rsub.add_parser("list", help="the robots defined").set_defaults(func=cmd_robot_list)
    q = rsub.add_parser("show", help="print a robot file")
    q.add_argument("name", nargs="?", help="robot name (default: the only robot)")
    q.set_defaults(func=cmd_robot_show)
    q = rsub.add_parser("check", help="read-only look: state, temperatures, one image per camera, calibration")
    q.add_argument("name", nargs="?", help="robot name (default: the only robot)")
    q.add_argument("--out", metavar="DIR", help="where the images go (default: <home>/checks/<stamp>-<robot>)")
    q.set_defaults(func=cmd_robot_check)
    q = rsub.add_parser("keep-out", help="boxes in the world frame no part of an arm may enter")
    ksub = q.add_subparsers(dest="keep_out_command", required=True, metavar="ACTION")
    which_robot = argparse.ArgumentParser(add_help=False)
    which_robot.add_argument("--robot", help="robot name (default: the only robot)")
    q = ksub.add_parser("add", parents=[which_robot], help="add a zone: its lowest and highest corner, metres")
    q.add_argument("name")
    q.add_argument("--min", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    q.add_argument("--max", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    q.set_defaults(func=cmd_keep_out)
    q = ksub.add_parser("remove", parents=[which_robot], help="remove a zone by name")
    q.add_argument("name")
    q.set_defaults(func=cmd_keep_out)
    ksub.add_parser("list", parents=[which_robot], help="the zones").set_defaults(func=cmd_keep_out)

    q = sub.add_parser("calibrate", help="measure the table under each arm and record its folded rest pose")
    q.add_argument("--robot", help="robot name (default: the only robot)")
    q.add_argument("--arm", help="one arm (default: each arm in turn)")
    how = q.add_mutually_exclusive_group()
    how.add_argument("--watch", action="store_true",
                     help="hands-free table touch: the reading is taken once the tip has been held still on it")
    how.add_argument("--table-z", type=float, metavar="Z",
                     help="store a known table height (world frame, metres) without opening the robot")
    how.add_argument("--rest", action="store_true", help="record only the folded rest pose, not the table")
    how.add_argument("--powered", action="store_true",
                     help="the arm finds the table itself: a slow guarded descent at a few reaches, from and back "
                          "to its rest pose; stored as a table height per reach")
    q.add_argument("--points", nargs="+", type=_point, metavar="X,Y",
                   help="with --powered: probe points, world metres (default: straight ahead at 0.25, 0.32, 0.39 m)")
    q.add_argument("--expect", type=float, metavar="Z",
                   help="with --powered: roughly where the table is, world z (default: the table stored with the "
                        "arm's present setup, at each point's reach; needed without one); the fingertips never go "
                        "more than 4 cm below it")
    q.add_argument("--yes", action="store_true",
                   help="record the rest pose without asking whether the arm is folded and supported (with "
                        "--powered: start without asking)")
    q.add_argument("--motors-off", action="store_true",
                   help="state that the motors are off, for an arm whose motors do not report their torque")
    q.set_defaults(func=cmd_calibrate)

    q = sub.add_parser("serve", parents=[session], help="hold a robot open as a session for `op` commands")
    q.add_argument("--robot", help="robot name (default: the only robot)")
    q.add_argument("--read-only", action="store_true", help="refuse every action except observe and wait")
    q.add_argument("--attach", action="store_true",
                   help="take over arms left holding with torque on by an earlier session, when a checked path "
                        "leads from where they hold to rest; rest them first")
    q.add_argument("--resume", nargs="?", const="", metavar="RUN_DIR",
                   help="continue an earlier episode (default: this robot's latest run)")
    q.set_defaults(func=cmd_serve)
    q = sub.add_parser("sim", parents=[session, sim_options], help="serve a simulator robot; no robot file needed")
    q.add_argument("--mujoco", action="store_true", help="MuJoCo physics instead of the kinematic simulator")
    q.set_defaults(func=cmd_sim)

    q = sub.add_parser("op", parents=[_CLIENT], help="send one action or session request; print the observation",
                       epilog=VOCABULARY, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("command", nargs=argparse.REMAINDER, metavar="OP ...", help="see below")
    q.set_defaults(func=cmd_op)
    q = sub.add_parser("ops", parents=[_CLIENT], help="send several ops in order; stop at the first not done")
    q.add_argument("commands", nargs="+", metavar="'OP ARGS'", help='each one quoted, e.g. "nudge 0 0 -0.02"')
    q.set_defaults(func=cmd_ops)
    q = sub.add_parser("rest", parents=[_CLIENT], help="rest every arm of the session (op rest)")
    q.set_defaults(func=cmd_rest)
    q = sub.add_parser("py", parents=[_CLIENT], help="run Python in the session's kernel (code on stdin or -c)",
                       description="Run Python in the session's persistent kernel and print what it printed, one "
                       "line of state per arm, the action queue and the reports of queued actions that finished "
                       "since the last call. The code comes from -c, else from stdin (not read with --wait). The "
                       "kernel runs code as the user who started the session; only that user can reach it.")
    q.add_argument("-c", dest="code", metavar="CODE", help="the code (default: read stdin)")
    q.add_argument("--wait", nargs="?", type=float, const=60.0, metavar="S",
                   help="then wait up to S seconds (default 60) for the next report, or for the queue to empty")
    q.add_argument("--timeout", type=float, default=300.0, metavar="S",
                   help="longest the code may run before this gives up waiting (default 300)")
    q.set_defaults(func=cmd_py)

    target = argparse.ArgumentParser(add_help=False, parents=[sim_options])
    which = target.add_mutually_exclusive_group()
    which.add_argument("--robot", help="robot name (default: the only robot)")
    which.add_argument("--sim", action="store_true", help="the kinematic simulator")
    which.add_argument("--mujoco", action="store_true", help="the MuJoCo simulator")
    which.add_argument("--connect", nargs="?", const="", metavar="ADDRESS",
                       help="drive a running session (default address: the one `serve` uses)")
    q = sub.add_parser("mcp", parents=[target], help="serve the actions as MCP tools on stdio")
    q.set_defaults(func=cmd_mcp)
    q = sub.add_parser("run", parents=[target], help="let a model carry out a task")
    q.add_argument("task")
    q.add_argument("--model", help="model id (default: the agent's)")
    q.add_argument("--max-steps", type=int, default=40, metavar="N")
    q.add_argument("--timeout", type=float, default=600.0, metavar="S", help="seconds (default 600)")
    q.set_defaults(func=cmd_run)

    q = sub.add_parser("log", help="read episode logs")
    lsub = q.add_subparsers(dest="log_command", required=True, metavar="ACTION")
    q = lsub.add_parser("show", help="the steps of a run: action and result")
    q.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help="default: the latest run")
    q.set_defaults(func=cmd_log_show)
    q = lsub.add_parser("profile", help="where a run's time went: steps, motion, idle gaps, kernel actions")
    q.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help="default: the latest run")
    q.add_argument("--brief", action="store_true", help="totals only, without the per-step table")
    q.add_argument("--json", action="store_true", help="print the numbers as JSON")
    q.set_defaults(func=cmd_log_profile)
    q = lsub.add_parser("resume", help="what a fresh agent needs to take over a run")
    q.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help="default: the latest run")
    q.set_defaults(func=cmd_log_resume)
    return p


if __name__ == "__main__":
    sys.exit(main())
