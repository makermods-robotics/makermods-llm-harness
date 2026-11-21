"""Episode logs: one JSON object per line in ``<run_dir>/episode.jsonl``, images in ``<run_dir>/images/``.

Events: ``start`` (robot, arms, cameras, table_z, safety), ``action`` (an action about to run, written
before anything moves), ``step`` (action and observation), ``note`` (free text such as the task) and ``end``.
One process writes an episode at a time: the log holds a lock on it while open.
"""

from __future__ import annotations

import json
import os
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


def _lock(file) -> None:
    """An exclusive, non-blocking lock on an open file, released when it closes."""
    if os.name == "nt":
        import msvcrt
        file.seek(0)
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
