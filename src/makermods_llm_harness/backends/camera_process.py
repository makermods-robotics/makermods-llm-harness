"""An SDK camera captured in a helper process.

A camera's receive and decode work (an MJPEG stream parsed and every frame decoded, 30 times a second) shares the
interpreter lock with the arms' control loops when it runs in their process, and slows them. Here the SDK camera
runs in a child Python process, on the computer's slowest CPUs when its CPUs differ (the control loops keep the fast
ones); this process asks it for the newest frame only when it needs one. Every reply has a deadline: a helper that
does not answer in time is killed (and reaped in the background), so no caller ever waits on a stalled camera.
"""

from __future__ import annotations

import pickle
import queue
import struct
import subprocess
import sys
import threading
from typing import Optional

_HEADER = struct.Struct("<I")
START_S = 10.0          # longest wait for the helper to open its camera
FRAME_S = 1.0           # longest wait for a frame
CLOSE_S = 1.0           # longest wait for the helper to end once asked to; then it is killed


def _send(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)) + data)
    stream.flush()


def _receive(stream):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError("the camera helper closed its pipe")
    size, = _HEADER.unpack(header)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError("the camera helper closed its pipe")
    return pickle.loads(data)


class CameraProcess:
    """``module.Camera(camera_id)`` (an SDK cameras module) opened and read in a helper process, with the SDK
    camera's ``open``, ``newest`` and ``close``. ``newest`` returns the module's ``Frame``. A thread reads the
    helper's replies; a reply later than START_S (opening) or FRAME_S (a frame) kills the helper."""

    def __init__(self, module: str, camera_id: str):
        self.module, self.camera_id = module, camera_id
        self._proc: Optional[subprocess.Popen] = None
        self._replies: Optional[queue.Queue] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", f"from {__package__} import camera_process; camera_process._serve()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        replies: queue.Queue = queue.Queue()
        threading.Thread(target=_read_replies, args=(proc.stdout, replies), daemon=True,
                         name=f"camera {self.camera_id} replies").start()
        self._proc, self._replies = proc, replies
        try:
            _send(proc.stdin, (self.module, self.camera_id))
            status, detail = self._reply(replies, START_S)
        except (EOFError, OSError, TimeoutError) as e:
            self.close()
            raise OSError(f"camera {self.camera_id}: its helper process ended ({e})") from e
        if status != "ok":
            self.close()
            raise OSError(detail)

    def newest(self, max_age_s: Optional[float] = None):
        """The helper camera's newest frame, or None (none, too old, the helper has ended, or it did not answer
        within FRAME_S: it is then killed)."""
        with self._lock:
            proc, replies = self._proc, self._replies
            if proc is None or proc.poll() is not None:
                return None
            try:
                _send(proc.stdin, max_age_s)
                reply = self._reply(replies, FRAME_S)
            except (EOFError, OSError, TimeoutError):
                self._end(proc)
                return None
        if reply is None:
            return None
        return _frame_type(self.module)(*reply)

    def close(self) -> None:
        """End the helper (killed after CLOSE_S); never waits for a frame request in progress, which then returns
        None."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=CLOSE_S)
        except subprocess.TimeoutExpired:
            self._end(proc)

    def _reply(self, replies: queue.Queue, timeout: float):
        try:
            kind, value = replies.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no reply within {timeout:g} s") from None
        if kind == "end":
            raise EOFError(str(value))
        return value

    def _end(self, proc: subprocess.Popen) -> None:
        """Kill the helper and reap it in the background."""
        if self._proc is proc:
            self._proc = None
        try:
            proc.kill()
        except OSError:
            pass
        threading.Thread(target=proc.wait, daemon=True, name=f"camera {self.camera_id} reaper").start()


def _read_replies(stream, replies: queue.Queue) -> None:
    """Put each reply of the helper on ``replies`` as ("reply", value); ("end", why) once its pipe closes."""
    try:
        while True:
            replies.put(("reply", _receive(stream)))
    except Exception as e:   # noqa: BLE001 - a closed or broken pipe ends the reader either way
        replies.put(("end", e))


def _frame_type(module: str):
    import importlib
    return importlib.import_module(module).Frame


def slow_cpus(sysfs: str = "/sys/devices/system/cpu") -> Optional[set[int]]:
    """The CPUs of lowest capacity (Linux ``cpu_capacity``) when the computer's CPUs differ; else None."""
    import os
    capacity = {}
    for cpu in range(os.cpu_count() or 0):
        try:
            with open(f"{sysfs}/cpu{cpu}/cpu_capacity") as f:
                capacity[cpu] = int(f.read().strip())
        except (OSError, ValueError):
            return None
    if len(set(capacity.values())) < 2:
        return None
    low = min(capacity.values())
    return {cpu for cpu, c in capacity.items() if c == low}


def _serve() -> None:
    """The helper: open the camera named on stdin, then answer each request (a max age) with the newest frame as
    (image, stamp, captured), or None."""
    import importlib
    import os
    cpus = slow_cpus()
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    module, camera_id = _receive(inp)
    try:
        camera = importlib.import_module(module).Camera(camera_id)
        camera.open()
    except Exception as e:
        _send(out, ("error", f"camera {camera_id} did not open: {e}"))
        return
    _send(out, ("ok", None))
    try:
        while True:
            try:
                max_age = _receive(inp)
            except EOFError:
                return
            frame = camera.newest(max_age)
            _send(out, None if frame is None else (frame[0], frame[1], getattr(frame, "captured", None)))
    finally:
        camera.close()
