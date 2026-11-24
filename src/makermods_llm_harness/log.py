"""Episode logs: one JSON object per line in ``<run_dir>/episode.jsonl``, images in ``<run_dir>/images/``.

Events: ``start`` (robot, arms, cameras, table_z, safety), ``action`` (an action about to run, written
before anything moves), ``step`` (action and observation), ``note`` (free text such as the task), ``py`` (a call
to the session's Python kernel: seconds, chars, error), ``kernel_action`` (a queued action that ended: id, name,
status, seconds, message) and ``end``. One process writes an episode at a time: the log holds a lock on it while
open.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

from .actions import Observation
from .robot import home


class EpisodeBusy(OSError):
    """Another process is writing this episode."""


class EpisodeLog:
    """Appends events to ``run_dir/episode.jsonl`` (an absolute path), flushed line by line. ``steps`` counts
    step events. A torn last line left by a crash is cut off before the first new event."""

    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.image_dir = self.run_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "episode.jsonl"
        self._write_lock = threading.Lock()
        self._file = self.path.open("a+b")
        try:
            _lock(self._file)
        except OSError as e:
            self._file.close()
            raise EpisodeBusy(f"{self.run_dir} is being written by another session") from e
        self._file.seek(0)
        data = self._file.read()
        if data and not data.endswith(b"\n"):
            self._file.truncate(data.rfind(b"\n") + 1)
        self.steps = sum(1 for e in read_episode(self.path) if e.get("event") == "step")

    @classmethod
    def resume(cls, run_dir: Union[str, Path]) -> "EpisodeLog":
        """Continue an existing episode: new events are appended and the step count goes on."""
        if not (Path(run_dir) / "episode.jsonl").is_file():
            raise FileNotFoundError(f"no episode to resume in {run_dir}")
        return cls(run_dir)

    def write(self, event: str, **fields: Any) -> None:
        """Append one event (from any thread)."""
        with self._write_lock:
            if self._file is None:
                raise ValueError(f"the episode log {self.path} is closed")
            line = json.dumps({"event": event, "time": round(time.time(), 3), **fields}, default=str) + "\n"
            self._file.write(line.encode("utf-8"))
            self._file.flush()

    def start(self, robot: str, arms: list[str], cameras: list[str], table_z: dict, safety: dict) -> None:
        self.write("start", robot=robot, arms=arms, cameras=cameras, table_z=table_z, safety=safety)

    def step(self, action: Optional[dict], observation: dict) -> None:
        self.steps += 1
        self.write("step", step=self.steps, action=action, observation=observation)

    def note(self, text: str) -> None:
        self.write("note", text=text)

    def close(self) -> None:
        """Write the end event when the disk allows it, and release the episode."""
        if self._file is not None:
            try:
                self.write("end")
            except OSError:
                pass
            finally:
                self._file.close()
                self._file = None


WINDOWS_LOCK_OFFSET = 0x7FFFFFF0   # bytes; the C runtime's lock offsets are 32-bit


def _lock(file) -> None:
    """An exclusive, non-blocking lock on an open file, released when it closes."""
    if os.name == "nt":
        import msvcrt
        # Windows locks are mandatory: the locked byte lies past any episode's length so readers are not blocked.
        file.seek(WINDOWS_LOCK_OFFSET)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def recorded_robot(path: Union[str, Path]) -> Optional[str]:
    """The robot named by an episode's first start event, or None."""
    return next((e.get("robot") for e in read_episode(path) if e.get("event") == "start"), None)


def runs_dir() -> Path:
    return home() / "runs"


def new_run_dir(robot_name: str) -> Path:
    """A new, empty ``<home>/runs/<stamp>-<robot>/``."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i in range(1000):
        path = runs_dir() / (f"{stamp}-{robot_name}" + (f"-{i}" if i else ""))
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"no free run directory for {stamp}-{robot_name}")


def latest_run() -> Optional[Path]:
    """The most recently started run directory, or None."""
    runs = [p for p in runs_dir().glob("*") if (p / "episode.jsonl").is_file()] if runs_dir().is_dir() else []
    return max(runs, key=lambda p: (p / "episode.jsonl").stat().st_mtime, default=None)


def read_episode(path: Union[str, Path]) -> list[dict]:
    """The events of an episode (a run directory or its episode.jsonl). A torn last line is skipped."""
    path = Path(path)
    if path.is_dir():
        path = path / "episode.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def summary(path: Union[str, Path], last: int = 5) -> str:
    """What a fresh agent needs to take over an episode: the robot, task notes, step count, the last few
    actions with their results, and the last observation with its image paths."""
    events = read_episode(path)
    start = next((e for e in reversed(events) if e.get("event") == "start"), {})
    steps = [e for e in events if e.get("event") == "step"]
    lines = [f"Episode {Path(path)}: robot {start.get('robot', '?')}, arms {', '.join(start.get('arms', []))}, "
             f"cameras {', '.join(start.get('cameras', [])) or 'none'}; {len(steps)} steps"
             + ("; ended" if events and events[-1].get("event") == "end" else "") + "."]
    lines += [f"Note: {e['text']}" for e in events if e.get("event") == "note"]
    if steps:
        lines.append(f"Last {min(last, len(steps))} actions:")
        for e in steps[-last:]:
            result = e["observation"].get("result", {})
            outcome = result.get("status", "?") + (f" ({result['message']})" if result.get("message") else "")
            lines.append(f"  {e.get('step')}. {json.dumps(e.get('action'))} -> {outcome}")
        lines.append("Last observation:")
        lines.append(Observation.from_dict(steps[-1]["observation"]).text())
    return "\n".join(lines)


MOTION_ACTIONS = ("move_tool", "nudge", "move_joints", "home", "rest", "gripper")


def profile(path: Union[str, Path]) -> dict:
    """Where an episode's time went, from its log; the parts add up to ``wall_s``. Each session runs from its
    ``start`` event to its ``end`` event (or its last event). Each step runs from its ``action`` event to its
    ``step`` event (a step refused before it ran takes 0 s). ``between_s`` is the time inside sessions outside steps
    (the arm idle: before the first step, between steps and after the last); ``between_sessions_s`` is the time
    between one session's end and the next one's start. ``motion_s`` counts motion steps that were not rejected,
    ``refused_s`` the planning of rejected ones; ``shutdown_s`` runs from a ``closing`` event (the arms resting as
    the session ends) to the session's end. Returns ``{"wall_s", "steps": [{"step", "action", "status",
    "start_s", "seconds", "gap_s"}], "by_action": {name: {"count", "seconds"}}, "in_steps_s", "motion_s",
    "refused_s", "between_s", "after_last_s", "between_sessions_s", "shutdown_s", "sessions", "statuses", "py_calls", "py_s",
    "kernel_actions": [{"id", "name", "status", "seconds"}]}``; times in seconds from the first event."""
    events = read_episode(path)
    empty = {"wall_s": 0.0, "steps": [], "by_action": {}, "in_steps_s": 0.0, "motion_s": 0.0, "refused_s": 0.0,
             "between_s": 0.0, "after_last_s": 0.0, "between_sessions_s": 0.0, "shutdown_s": 0.0, "sessions": 0,
             "statuses": {},
             "py_calls": 0, "py_s": 0.0, "kernel_actions": []}
    if not events:
        return empty
    t0 = float(events[0].get("time", 0.0))
    started: dict = {}
    steps: list = []
    between = after_last = between_sessions = shutdown = 0.0
    sessions = 0
    last_end = t0
    session_open = False
    closing_at = None
    last_time = t0

    def close_session(at: float) -> None:
        nonlocal after_last, session_open, shutdown, closing_at
        if closing_at is None:
            after_last += max(0.0, at - last_end)
        else:
            shutdown += max(0.0, at - closing_at)
            closing_at = None
        session_open = False

    for e in events:
        kind, at = e.get("event"), float(e.get("time", t0))
        if kind == "start":
            if session_open:
                close_session(last_time)
            if sessions:
                between_sessions += max(0.0, at - last_time)
            sessions += 1
            last_end = at
            session_open = True
        elif kind == "closing" and session_open and closing_at is None:
            after_last += max(0.0, at - last_end)
            closing_at = at
        elif kind == "end" and session_open:
            close_session(at)
        elif kind == "action":
            started[e.get("step")] = at
        elif kind == "step":
            begin = min(started.pop(e.get("step"), at), at)
            gap = max(0.0, begin - last_end)
            between += gap
            action = e.get("action") or {}
            status = (e.get("observation") or {}).get("result", {}).get("status", "?")
            steps.append({"step": e.get("step"), "action": action.get("action", "?"), "status": status,
                          "start_s": round(begin - t0, 3), "seconds": round(at - begin, 3), "gap_s": round(gap, 3)})
            last_end = at
        last_time = max(last_time, at)
    if session_open:
        close_session(last_time)
    by_action: dict = {}
    statuses: dict = {}
    for st in steps:
        entry = by_action.setdefault(st["action"], {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] = round(entry["seconds"] + st["seconds"], 3)
        statuses[st["status"]] = statuses.get(st["status"], 0) + 1
    py = [e for e in events if e.get("event") == "py"]
    return {**empty, "wall_s": round(last_time - t0, 3), "steps": steps, "by_action": by_action,
            "in_steps_s": round(sum(st["seconds"] for st in steps), 3),
            "motion_s": round(sum(st["seconds"] for st in steps
                                  if st["action"] in MOTION_ACTIONS and st["status"] != "rejected"), 3),
            "refused_s": round(sum(st["seconds"] for st in steps if st["status"] == "rejected"), 3),
            "between_s": round(between + after_last, 3), "after_last_s": round(after_last, 3),
            "between_sessions_s": round(between_sessions, 3), "shutdown_s": round(shutdown, 3), "sessions": sessions, "statuses": statuses,
            "py_calls": len(py), "py_s": round(sum(float(e.get("seconds", 0.0)) for e in py), 3),
            "kernel_actions": [{k: e.get(k) for k in ("id", "name", "status", "seconds")}
                               for e in events if e.get("event") == "kernel_action"]}


def profile_text(path: Union[str, Path], steps: bool = True) -> str:
    """``profile`` in words: totals, time per action kind, the longest idle gaps and (``steps``) every step."""
    p = profile(path)
    wall = p["wall_s"] or 1e-9
    lines = [f"Episode {Path(path)}: {len(p['steps'])} steps over {p['wall_s']:.1f} s"
             + (f" in {p['sessions']} sessions" if p["sessions"] > 1 else ""),
             f"  in steps       {p['in_steps_s']:7.1f} s ({100 * p['in_steps_s'] / wall:.0f}%): moving "
             f"{p['motion_s']:.1f} s, refused {p['refused_s']:.1f} s",
             f"  outside steps  {p['between_s']:7.1f} s ({100 * p['between_s'] / wall:.0f}%): the arm idle (thinking, "
             f"looking, waiting for the next request), {p['after_last_s']:.1f} s of it after the last step"]
    if p["shutdown_s"]:
        lines.append(f"  shutdown rest  {p['shutdown_s']:7.1f} s ({100 * p['shutdown_s'] / wall:.0f}%)")
    if p["between_sessions_s"]:
        lines.append(f"  between sessions {p['between_sessions_s']:.1f} s ({100 * p['between_sessions_s'] / wall:.0f}%)")
    if p["py_calls"]:
        lines.append(f"  py calls     {p['py_calls']:4d}, {p['py_s']:.1f} s running code")
    lines.append("  results: " + ", ".join(f"{k} {v}" for k, v in sorted(p["statuses"].items())))
    for name, entry in sorted(p["by_action"].items(), key=lambda kv: -kv[1]["seconds"]):
        lines.append(f"  {name:<12} {entry['count']:4d} x  {entry['seconds']:7.1f} s")
    gaps = sorted(p["steps"], key=lambda s: -s["gap_s"])[:3]
    if gaps and gaps[0]["gap_s"] > 0:
        lines.append("  longest gaps: " + ", ".join(f"{s['gap_s']:.1f} s before step {s['step']}" for s in gaps
                                                    if s["gap_s"] > 0))
    if p["kernel_actions"]:
        lines.append("kernel actions:")
        lines += [f"  #{a['id']} {a['name']}: {a['status']} in {a['seconds']} s" for a in p["kernel_actions"]]
    if steps and p["steps"]:
        lines.append(" step   start    took     gap  action -> result")
        lines += [f"{s['step']:>5} {s['start_s']:7.1f} {s['seconds']:7.2f} {s['gap_s']:7.2f}  {s['action']} -> "
                  f"{s['status']}" for s in p["steps"]]
    return "\n".join(lines)
