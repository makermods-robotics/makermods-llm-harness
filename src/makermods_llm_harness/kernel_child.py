"""The kernel process a session starts (``python -m makermods_llm_harness.kernel_child ADDRESS``): it runs model
code and queued actions, and reaches the robot only as a client of the session's kernel channel at ADDRESS.

It reads commands from the session on stdin, one JSON object per line, the first being ``{"key"}`` (the channel's
key, hex), and answers on its original stdout (the process's file descriptor 1 is pointed at stderr, so nothing
else can write there):

    {"cmd": "cell", "id", "token", "code"}   ->  {"event": "cell_done", "id", "output", "error", "stopped", "motion",
                                                  "images"}   ("motion": the status of a motion that did not end done)
    {"cmd": "run", "id", "key", "token"}     ->  {"event": "action_done", "id", "status", "message", "output"}
    {"cmd": "interrupt", "token"}            (raise Interrupted in the code running under that token)
    {"cmd": "forget", "keys"}                (drop queued functions the session dropped)

Each cell and action runs on its own thread with its token, which every step and queue request it sends carries;
the session refuses them once the token is revoked. What the code prints is kept per thread, up to MAX_OUTPUT
characters.
"""

from __future__ import annotations

import ast
import ctypes
import io
import itertools
import json
import linecache
import math
import os
import sys
import threading
import traceback
import uuid
from typing import Optional

import numpy as np

from .actions import Observation, Result
from .program import MotionFailed, Program, Stopped
from .session import Client, SessionError

MAX_OUTPUT = 20000           # characters of printed output kept per call or action
ACTION_TIMEOUT_S = 120.0


class Interrupted(BaseException):
    """Raised in code whose authority the session revoked."""


class _Bounded(io.TextIOBase):
    """Keeps the first and last MAX_OUTPUT / 2 characters written, and counts the rest."""

    def __init__(self):
        self.head, self.tail, self.left_out = [], [], 0
        self._head_len = self._tail_len = 0

    def write(self, text: str) -> int:
        if self._head_len < MAX_OUTPUT // 2:
            room = MAX_OUTPUT // 2 - self._head_len
            self.head.append(text[:room])
            self._head_len += len(text[:room])
            text = text[room:]
        if text:
            self.tail.append(text)
            self._tail_len += len(text)
            while self._tail_len > MAX_OUTPUT // 2 and self.tail:
                extra = self._tail_len - MAX_OUTPUT // 2
                first = self.tail[0]
                cut = min(extra, len(first))
                self.tail[0] = first[cut:]
                self._tail_len -= cut
                self.left_out += cut
                if not self.tail[0]:
                    self.tail.pop(0)
        return len(text)

    def text(self) -> str:
        middle = f"\n... ({self.left_out} characters left out) ...\n" if self.left_out else ""
        return "".join(self.head) + middle + "".join(self.tail)


class _Router(io.TextIOBase):
    """sys.stdout and sys.stderr: each thread's writes go to the buffer registered for it, else to stderr."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.buffers: dict[int, _Bounded] = {}

    def write(self, text: str) -> int:
        buffer = self.buffers.get(threading.get_ident())
        return (buffer or self.fallback).write(text)

    def flush(self) -> None:
        if threading.get_ident() not in self.buffers:
            self.fallback.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self):
        return getattr(self.fallback, "encoding", "utf-8")


class _Context(threading.local):
    token = None
    images = None


_context = _Context()


class _Session:
    """The target of the namespace's ``r``: session requests, steps carrying the running code's token."""

    def __init__(self, client: Client):
        self.client = client

    def state(self) -> Observation:
        return self.client.state()

    def look(self) -> dict:
        look = self.client.look()
        if look["path"] and _context.images is not None:
            _context.images.append(look["path"])
        return look

    def check(self, action) -> Result:
        return self.client.check(action)

    def reachable(self, poses, **options) -> list[dict]:
        return self.client.reachable(poses, **options)

    def step(self, action, cancel=None, images: bool = True) -> Observation:
        token = _context.token
        if token is None:
            raise Stopped(Result("stopped", "no authority to move: motions run only from a py call or a queued "
                                            "action, not from threads they start"))
        reply = self.client.request("step", action=action, images=images, token=token)
        obs = Observation.from_dict(reply["observation"])
        if reply.get("revoked"):
            raise Stopped(obs.result)
        return obs


class KernelControl:
    """``kernel`` in the namespace: the session's action queue."""

    def __init__(self, child: "Child"):
        self._child = child

    def queue(self, actions, replace_from=None, timeout: float = ACTION_TIMEOUT_S):
        """Queue an action ``fn(r)`` or a list of them; returns its id (or their ids) at once. ``replace_from``
        (an id or name of a waiting action) replaces it and everything behind it. An action still running after
        ``timeout`` seconds is stopped."""
        fns = list(actions) if isinstance(actions, (list, tuple)) else [actions]
        if not fns or not all(callable(fn) for fn in fns):
            raise TypeError("an action is a function fn(r); queue one or a list of them")
        keys = [uuid.uuid4().hex for _ in fns]
        with self._child.lock:
            self._child.functions.update(zip(keys, fns))
        try:
            reply = self._child.client.request("kqueue", token=_context.token, keys=keys, names=[_name(f) for f in fns],
                                               replace_from=replace_from, timeout=timeout)
        except BaseException:
            with self._child.lock:
                for key in keys:
                    self._child.functions.pop(key, None)
            raise
        ids = reply["ids"]
        return ids if isinstance(actions, (list, tuple)) else ids[0]

    def flush(self) -> list[str]:
        """Drop the waiting actions; the running one goes on."""
        return self._child.client.request("kflush", token=_context.token)["dropped"]

    def stop(self) -> None:
        """Halt every arm where it is: the running action and this code are stopped, the queue is dropped. Raises
        Stopped once the session has held the arms, or MotionFailed with a fault when it could not confirm that."""
        try:
            self._child.client.request("stop", token=_context.token)
        except SessionError as e:
            if str(e).startswith("stopped:"):
                raise Stopped(Result("stopped", str(e))) from None
            raise MotionFailed(Result("fault", f"kernel.stop(): {e}")) from None
        raise Stopped(Result("stopped", "stopped by kernel.stop(); every arm that is on holds where it is"))

    def status(self) -> dict:
        """``{"running": {"id", "name", "seconds"} or None, "waiting": [{"id", "name"}], "reports_pending"}``."""
        return self._child.client.request("kstatus")

    def __repr__(self) -> str:
        return f"<kernel: {self.status()}>"


class Child:
    def __init__(self, address: str, key: bytes, out):
        self.out = out
        self.write_lock = threading.Lock()
        self.lock = threading.Lock()
        self.client = Client(address, timeout_s=600.0, key=key)
        self.functions: dict[str, object] = {}
        self.threads: dict[str, int] = {}         # token -> thread ident running under it
        self.codes = itertools.count(1)
        self.router = _Router(sys.stderr)
        sys.stdout = sys.stderr = self.router
        self.namespace = {"r": Program(_Session(self.client)), "np": np, "math": math, "kernel": KernelControl(self),
                          "MotionFailed": MotionFailed, "Stopped": Stopped}

    def send(self, message: dict) -> None:
        with self.write_lock:
            self.out.write(json.dumps(message, default=str) + "\n")
            self.out.flush()

    def serve(self, commands) -> None:
        self.send({"event": "ready"})
        for line in commands:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            cmd = message.get("cmd")
            if cmd == "cell":
                self._start(self._cell, message)
            elif cmd == "run":
                self._start(self._action, message)
            elif cmd == "interrupt":
                self._interrupt(message.get("token"))
            elif cmd == "forget":
                with self.lock:
                    for key in message.get("keys", []):
                        self.functions.pop(key, None)

    def _start(self, target, message: dict) -> None:
        threading.Thread(target=target, args=(message,), daemon=True).start()

    def _interrupt(self, token) -> None:
        with self.lock:
            ident = self.threads.get(token)
        if ident is not None and hasattr(ctypes, "pythonapi"):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), ctypes.py_object(Interrupted))

    def _run(self, message: dict, body, images: list) -> tuple[_Bounded, Optional[BaseException]]:
        """``body()`` under the message's token with this thread's prints captured and its looks in ``images``."""
        buffer = _Bounded()
        _context.token, _context.images = message.get("token"), images
        with self.lock:
            self.threads[_context.token] = threading.get_ident()
        self.router.buffers[threading.get_ident()] = buffer
        error = None
        try:
            body()
        except BaseException as e:      # reported to the session
            error = e
            if not isinstance(e, (Stopped, Interrupted, SystemExit)):
                buffer.write(_traceback())
        finally:
            self.router.buffers.pop(threading.get_ident(), None)
            with self.lock:
                self.threads.pop(_context.token, None)
            _context.token = None
        return buffer, error

    def _cell(self, message: dict) -> None:
        code = str(message.get("code", ""))

        def body():
            filename = f"<py-{next(self.codes)}>"
            linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
            tree = ast.parse(code, filename, "exec")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, filename, "exec"), self.namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), filename, "eval"), self.namespace)
                if value is not None:
                    print(repr(value))

        images: list = []
        buffer, error = self._run(message, body, images)
        stopped = isinstance(error, (Stopped, Interrupted))
        if isinstance(error, Stopped):
            buffer.write(f"stopped: {error.result.message}\n")
        elif isinstance(error, Interrupted):
            buffer.write("stopped: this code was stopped\n")
        elif isinstance(error, SystemExit):
            buffer.write(f"exit({error.code})\n")
        failed = error is not None and not (isinstance(error, SystemExit) and error.code in (None, 0))
        motion = error.result.status if isinstance(error, MotionFailed) else ("stopped" if stopped else None)
        self.send({"event": "cell_done", "id": message.get("id"), "output": buffer.text(), "error": failed,
                   "stopped": stopped, "motion": motion, "images": images})

    def _action(self, message: dict) -> None:
        with self.lock:
            fn = self.functions.pop(message.get("key"), None)
        if fn is None:
            self.send({"event": "action_done", "id": message.get("id"), "status": "error",
                       "message": "the kernel does not have that action (it restarted)", "output": ""})
            return
        buffer, error = self._run(message, lambda: fn(Program(_Session(self.client))), [])
        if error is None:
            status, text = "done", ""
        elif isinstance(error, Stopped):
            status, text = "stopped", error.result.message
        elif isinstance(error, Interrupted):
            status, text = "stopped", "stopped on request"
        elif isinstance(error, MotionFailed):
            status, text = "error", f"{error.result.status}: {error.result.message}"
        elif isinstance(error, SessionError) and str(error).startswith("stopped"):
            status, text = "stopped", str(error)
        else:
            status, text = "error", "".join(traceback.format_exception_only(type(error), error)).strip()
        self.send({"event": "action_done", "id": message.get("id"), "status": status, "message": text,
                   "output": buffer.text()})


def _name(fn) -> str:
    name = getattr(fn, "__name__", "") or "action"
    return "action" if name == "<lambda>" else name


def _traceback() -> str:
    """The exception being handled, with only the frames of code run in the kernel."""
    kind, error, tb = sys.exc_info()
    frames = [f for f in traceback.extract_tb(tb) if f.filename.startswith("<py-")]
    text = "".join(traceback.format_exception_only(kind, error))
    if frames:
        text = "Traceback (your code):\n" + "".join(traceback.format_list(frames)) + text
    return text


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    commands = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    key = bytes.fromhex(json.loads(commands.readline())["key"])
    child = Child(argv[0], key, out)
    child.serve(commands)
    return 0


if __name__ == "__main__":
    sys.exit(main())
