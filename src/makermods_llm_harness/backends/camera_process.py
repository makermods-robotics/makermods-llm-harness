"""An SDK camera captured in a helper process.

A camera's receive and decode work (an MJPEG stream parsed and every frame decoded, 30 times a second) shares the
interpreter lock with the arms' control loops when it runs in their process, and slows them. Here the SDK camera
runs in a child Python process, on the computer's slowest CPUs when its CPUs differ (the control loops keep the fast
ones); this process asks it for the newest frame only when it needs one.
"""

from __future__ import annotations

import pickle
import struct
import subprocess
import sys
import threading
from typing import Optional

_HEADER = struct.Struct("<I")
START_S = 10.0          # longest wait for the helper to open its camera


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
    camera's ``open``, ``newest`` and ``close``. ``newest`` returns the module's ``Frame``."""

    def __init__(self, module: str, camera_id: str):
        self.module, self.camera_id = module, camera_id
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-c", f"from {__package__} import camera_process; camera_process._serve()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        timer = threading.Timer(START_S, self._proc.kill)
        timer.start()
        try:
            _send(self._proc.stdin, (self.module, self.camera_id))
            status, detail = _receive(self._proc.stdout)
        except (EOFError, OSError) as e:
            self.close()
            raise OSError(f"camera {self.camera_id}: its helper process ended ({e})") from e
        finally:
            timer.cancel()
        if status != "ok":
            self.close()
            raise OSError(detail)

    def newest(self, max_age_s: Optional[float] = None):
        """The helper camera's newest frame, or None (none, too old, or the helper has ended)."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            try:
                _send(self._proc.stdin, max_age_s)
                reply = _receive(self._proc.stdout)
            except (EOFError, OSError):
                return None
        if reply is None:
            return None
        return _frame_type(self.module)(*reply)

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


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
