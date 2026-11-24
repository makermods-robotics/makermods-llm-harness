"""The program API: a small object an agent's code drives, on a ``Harness`` in this process or on a session
``Client``. Every motion is an action run through the harness and its safety envelope, without camera images;
pictures come from ``look()``.

    r = Program(harness)
    print(r.state().text())
    print(r.reachable([[0.28, 0, 0.12], {"xyz": [0.28, 0, 0.05], "pitch": -90}]))
    r.move([0.28, 0, 0.12], pitch=-90)
    r.nudge([0, 0, -0.08], until="contact")
    r.gripper("close")

Units: metres, degrees, seconds. ``speed`` is a joint speed cap in deg/s and ``duration`` the time a move
should take; either may be given, and neither makes a motion faster than the envelope allows.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .actions import Observation, Result


class MotionFailed(RuntimeError):
    """A motion did not end done. ``result`` says how it ended and why."""

    def __init__(self, result: Result):
        super().__init__(f"{result.status}: {result.message}")
        self.result = result


class Stopped(MotionFailed):
    """The motion, or the code that sent it, was stopped on request; the arm holds where it is."""


class Program:
    """Drives ``target`` (a ``Harness`` or a session ``Client``). With ``check`` (the default) a motion that does
    not end done raises MotionFailed; otherwise every motion returns its Result. A target may raise Stopped from
    ``step`` when the code sending the motion has been stopped."""

    def __init__(self, target, *, check: bool = True):
        self.target = target
        self.check = check

    # ── queries: nothing moves ──
    def state(self) -> Observation:
        """The arms now (``.text()`` for a summary, ``.arms[name]`` for the numbers); no images."""
        return self.target.state()

    def look(self) -> Optional[str]:
        """One labelled grid of every camera, saved in the run's image directory; its path (None when the
        harness saves no images)."""
        return self.target.look()["path"]

    def reachable(self, poses: Sequence, *, arm: Optional[str] = None, line: bool = True, chain: bool = False,
                  start=None) -> list[dict]:
        """Check tool poses (``[x, y, z]`` or ``{"xyz", "pitch", "yaw", "roll"}``) exactly as ``move`` would,
        moving nothing: per pose ``{"ok", "why", "margin_deg", "margin_joint", "joints_deg", "seconds"}``. Each is
        checked from where the arm is (where its running motion ends), or from ``start`` (a pose, or "home"); with
        ``chain`` each later one from the pose before it."""
        return self.target.reachable(list(poses), arm=arm, line=line, chain=chain, start=start)

    # ── motions ──
    def move(self, xyz, pitch: Optional[float] = None, yaw: Optional[float] = None, roll: Optional[float] = None, *,
             arm: Optional[str] = None, line: bool = True, speed: Optional[float] = None,
             duration: Optional[float] = None, until: Optional[str] = None, dry_run: bool = False) -> Result:
        """Move the tool point to ``xyz`` (world frame), in a straight line unless ``line`` is false. ``until=
        "contact"`` ends the move early, holding, where the arm stops following (``result.contact`` says where)."""
        return self._motion({"action": "move_tool", "xyz": _list(xyz), "pitch_deg": pitch, "yaw_deg": yaw,
                             "roll_deg": roll, "arm": arm, "line": None if line else False, "until": until,
                             "speed_deg_s": speed, "duration_s": duration}, dry_run)

    def nudge(self, dxyz, *, frame: str = "base", arm: Optional[str] = None, speed: Optional[float] = None,
              duration: Optional[float] = None, until: Optional[str] = None, dry_run: bool = False) -> Result:
        """Move the tool point by ``dxyz`` metres along the arm's base axes (or the gripper's, ``frame="tool"``)."""
        return self._motion({"action": "nudge", "dxyz": _list(dxyz), "frame": frame, "arm": arm, "until": until,
                             "speed_deg_s": speed, "duration_s": duration}, dry_run)

    def move_joints(self, joints: dict, *, relative: bool = False, arm: Optional[str] = None,
                    speed: Optional[float] = None, duration: Optional[float] = None, dry_run: bool = False) -> Result:
        """Move joints by name to angles in degrees (``relative``: by them)."""
        return self._motion({"action": "move_joints", "joints_deg": dict(joints), "relative": relative, "arm": arm,
                             "speed_deg_s": speed, "duration_s": duration}, dry_run)

    def gripper(self, opening, *, arm: Optional[str] = None, dry_run: bool = False) -> Result:
        """Set the opening, 0 closed .. 1 open (or "open" / "close"); ``result.grip["holding"]`` says whether the
        jaws stopped on something."""
        return self._motion({"action": "gripper", "opening": opening, "arm": arm}, dry_run)

    def home(self, *, arm: Optional[str] = None, speed: Optional[float] = None, duration: Optional[float] = None,
             dry_run: bool = False) -> Result:
        return self._motion({"action": "home", "arm": arm, "speed_deg_s": speed, "duration_s": duration}, dry_run)

    def rest(self, *, arm: Optional[str] = None, dry_run: bool = False) -> Result:
        """Back to the rest pose, motors off (every arm without ``arm``)."""
        return self._motion({"action": "rest", "arm": arm}, dry_run)

    def wait(self, seconds: float) -> Result:
        """Let time pass; a stop ends it early."""
        return self._motion({"action": "wait", "seconds": seconds}, False)

    def _motion(self, action: dict, dry_run: bool) -> Result:
        action = {k: v for k, v in action.items() if v is not None}
        if dry_run:
            return self.target.check(action)
        result = self.target.step(action, images=False).result
        if self.check and not result.ok:
            raise MotionFailed(result)
        return result


def _list(values) -> list:
    return values.tolist() if hasattr(values, "tolist") else list(values)
