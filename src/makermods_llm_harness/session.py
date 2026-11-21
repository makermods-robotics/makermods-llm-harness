"""One robot held open by a long-lived server, driven by short-lived clients over a local socket.

``llm-harness serve`` runs a ``Server`` around a ``Harness``; each ``llm-harness op ...`` is a ``Client``
request. Requests and replies are JSON objects sent as bytes over ``multiprocessing.connection``
(a Unix socket on POSIX, inside a directory only its owner can enter; a named pipe on Windows), one
connection per request:

    {"op": "step", "action": {...}}  ->  {"ok": true, "observation": {...}}
    {"op": "observe"}                ->  {"ok": true, "observation": {...}}
    {"op": "describe"}               ->  {"ok": true, "text": "..."}
    {"op": "stop"}                   ->  {"ok": true}       (ends the action in progress and those waiting)
    {"op": "status"}                 ->  {"ok": true, "robot", "arms", "cameras", "steps", "run_dir",
                                          "read_only", "busy"}     (no device I/O)
    {"op": "quit"}                   ->  {"ok": true, "arms": {arm: torque outcome}}   (after the shutdown)

Observations carry image paths, plus the JPEG bytes (``jpeg_base64``) of any image without a path. A step's
motion is cancelled (the arm holds) when its client's connection closes before the reply. Anything that goes
wrong answers ``{"ok": false, "error": "..."}``; the server keeps running.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
import sys
import threading
import time
from multiprocessing.connection import Client as _connect, Connection, Listener
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .actions import ACTIONS, Action, Observation, action_to_dict
from .robot import home

FAMILY = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
MAX_SOCKET_PATH = 100     # bytes; Unix socket paths are limited to about this on Linux and macOS
MAX_REQUEST = 1 << 20     # bytes in one request
READ_S = 5.0              # a client must send its request within this
MAX_CLIENTS = 16          # connections handled at once; past that only stop and status are answered
MAX_QUEUED = 4            # step and observe requests running or waiting at once
QUIT_WAIT_S = 600.0       # how long quit waits for the arms to rest before it answers


class SessionError(RuntimeError):
    """No session to talk to, or the session refused a request. The message says what to do."""


def default_address() -> str:
    """``<home>/session/llm-harness.sock`` on POSIX; on Windows a named pipe derived from the home directory."""
    if FAMILY == "AF_PIPE":
        return r"\\.\pipe\llm-harness-" + hashlib.sha1(str(home()).encode()).hexdigest()[:12]
    return str(home() / "session" / "llm-harness.sock")


def _private_dir(address: str) -> None:
    """Create the socket's directory owner-only (0700), or refuse one that others can enter or that is not
    the user's."""
    folder = Path(address).parent
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = folder.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise SessionError(f"{folder} must belong to you and be closed to others (chmod 700 {folder}) before a "
                           "session socket goes in it")


def _open(address: str) -> Connection:
    try:
        return _connect(address, family=FAMILY)
    except OSError as e:
        raise SessionError(f"no session is running at {address}; start one with `llm-harness serve`") from e


def ensure_free(address: str) -> None:
    """Make ``address`` ready for a new server: refuse when a live session answers there, remove the socket
    file a dead one left behind."""
    if FAMILY == "AF_UNIX" and len(os.fsencode(address)) > MAX_SOCKET_PATH:
        raise SessionError(f"the session address {address} is too long for a Unix socket (at most "
                           f"{MAX_SOCKET_PATH} bytes); pass a shorter path with --address")
    try:
        _open(address).close()
    except SessionError:
        if FAMILY == "AF_UNIX" and os.path.lexists(address):
            if not stat.S_ISSOCK(os.lstat(address).st_mode):
                raise SessionError(f"{address} exists and is not a socket; pass another path with --address")
            os.unlink(address)
        return
    raise SessionError(f"a session is already running at {address}")


def _same_user(conn: Connection) -> bool:
    """The peer runs as this user, where the platform tells (Linux); True elsewhere."""
    if not hasattr(socket, "SO_PEERCRED"):
        return True
    try:
        sock = socket.socket(fileno=os.dup(conn.fileno()))
        try:
            _, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                            struct.calcsize("3i")))
        finally:
            sock.close()
    except OSError:
        return False
    return uid == os.getuid()


class Server:
    """Answers requests for one harness, each connection on its own thread, so ``stop`` and ``status`` are
    answered while an action runs. ``say`` gets one line per request.

    The owner ends it with ``shutdown()`` (no new requests), ``harness.stop()``, ``drain()`` (the requests
    already running finish), ``harness.close()`` and, with ``reports_close``, ``finish(outcomes)``, which a
    quit waits for and answers with; without it a quit answers once the server stops taking requests."""

    def __init__(self, harness, address: Optional[str] = None, say: Optional[Callable[[str], None]] = None,
                 reports_close: bool = False):
        self.harness = harness
        self.reports_close = reports_close
        self.address = str(address or default_address())
        self.say = say
        if FAMILY == "AF_UNIX":
            _private_dir(self.address)
        ensure_free(self.address)
        self._listener = Listener(self.address, family=FAMILY)
        if FAMILY == "AF_UNIX":
            os.chmod(self.address, 0o600)
        self._closing = threading.Event()
        self._finished = threading.Event()
        self._outcomes: dict[str, str] = {}
        self._serving = False
        self._clients = threading.BoundedSemaphore(MAX_CLIENTS)
        self._queued = threading.BoundedSemaphore(MAX_QUEUED)
        self._work: set[threading.Thread] = set()          # handling step or observe
        self._handlers: set[threading.Thread] = set()      # handling any request
        self._work_lock = threading.Lock()

    def serve_forever(self) -> None:
        """Accept connections until ``shutdown()``."""
        self._serving = True
        try:
            while not self._closing.is_set():
                try:
                    conn = self._listener.accept()
                except OSError:
                    if self._closing.is_set():
                        break
                    raise
                if self._closing.is_set():
                    conn.close()
                    break
                if not _same_user(conn):
                    conn.close()
                    continue
                if self._clients.acquire(blocking=False):
                    handler = threading.Thread(target=self._handle, args=(conn, True), daemon=True)
                    with self._work_lock:
                        self._handlers = {t for t in self._handlers if t.is_alive()} | {handler}
                    handler.start()
                else:
                    self._handle(conn, False, read_s=0.5)     # saturated: answers only stop and status
        finally:
            self._listener.close()

    def shutdown(self) -> None:
        """Take no new requests and end ``serve_forever`` (from any thread). Actions in progress go on."""
        if self._closing.is_set():
            return
        self._closing.set()
        if not self._serving:
            self._listener.close()
            return
        try:
            _connect(self.address, family=FAMILY).close()     # wakes the blocked accept()
        except OSError:
            pass

    def drain(self, timeout_s: float = 60.0) -> None:
        """Wait for the step and observe requests already running to finish."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._work_lock:
                work = [t for t in self._work if t is not threading.current_thread()]
            if not work or time.monotonic() > deadline:
                return
            work[0].join(max(0.0, deadline - time.monotonic()))

    def finish(self, outcomes: dict[str, str], timeout_s: float = 5.0) -> None:
        """The arms' torque outcomes after the harness closed: a waiting quit answers with them, and the
        replies still being sent get up to ``timeout_s`` to go out."""
        self._outcomes = dict(outcomes)
        self._finished.set()
        deadline = time.monotonic() + timeout_s
        with self._work_lock:
            handlers = [t for t in self._handlers if t is not threading.current_thread()]
        for handler in handlers:
            handler.join(max(0.0, deadline - time.monotonic()))

    def _handle(self, conn: Connection, threaded: bool, read_s: float = READ_S) -> None:
        op = None
        try:
            with conn:
                try:
                    if not conn.poll(read_s):
                        return
                    request = conn.recv_bytes(MAX_REQUEST)
                except (EOFError, OSError):
                    return
                if not threaded:
                    op, reply = self._reply(request, only=("stop", "status"))
                else:
                    op, reply = self._reply(request, conn=conn)
                try:
                    conn.send_bytes(json.dumps(reply, default=str).encode())
                except OSError:
                    pass
        finally:
            if threaded:
                self._clients.release()

    def _reply(self, raw: bytes, conn: Optional[Connection] = None,
               only: Optional[tuple[str, ...]] = None) -> tuple[str, dict[str, Any]]:
        op = "?"
        try:
            request = json.loads(raw)
            if not isinstance(request, dict) or not isinstance(request.get("op"), str):
                raise SessionError('a request is a JSON object with an "op"')
            op = request["op"]
            if only is not None and op not in only:
                raise SessionError(f"the session is busy ({MAX_CLIENTS} connections); try again")
            reply = {"ok": True, **self._run(op, request, conn)}
        except Exception as e:
            reply = {"ok": False, "error": str(e) if isinstance(e, SessionError) else f"{type(e).__name__}: {e}"}
        if self.say:
            what = op
            if "observation" in reply:
                action = reply["observation"]["result"].get("action") or {}
                what += f" {action.get('action', '?')}" if op == "step" else ""
                outcome = reply["observation"]["result"]["status"]
            else:
                outcome = "ok" if reply["ok"] else f"error: {reply['error']}"
            self.say(f"{time.strftime('%H:%M:%S')} {what} -> {outcome}")
        return op, reply

    def _run(self, op: str, request: dict, conn: Optional[Connection]) -> dict[str, Any]:
        h = self.harness
        if op in ("step", "observe"):
            if op == "step" and "action" not in request:
                raise SessionError('step needs an "action"')
            if self._closing.is_set():
                raise SessionError("the session is closing")
            if not self._queued.acquire(blocking=False):
                raise SessionError(f"{MAX_QUEUED} actions are already running or waiting; try again")
            with self._work_lock:
                self._work.add(threading.current_thread())
            try:
                if op == "observe":
                    obs = h.observe()
                else:
                    cancel = threading.Event()
                    watcher = threading.Thread(target=_watch_hangup, args=(conn, cancel), daemon=True)
                    watcher.start()
                    try:
                        obs = h.step(request["action"], cancel=cancel)
                    finally:
                        cancel.set()
            finally:
                with self._work_lock:
                    self._work.discard(threading.current_thread())
                self._queued.release()
            unsaved = any(c.jpeg is not None and c.path is None for c in obs.cameras.values())
            return {"observation": obs.to_dict(images=unsaved)}
        if op == "describe":
            return {"text": h.describe()}
        if op == "stop":
            h.stop()
            return {}
        if op == "status":
            with self._work_lock:
                busy = bool(self._work)
            return {"robot": h.robot_name, "arms": list(h.models), "cameras": list(h.cameras), "steps": h.steps,
                    "run_dir": str(h.log.run_dir) if h.log else None, "read_only": h.read_only, "busy": busy}
        if op == "quit":
            self.shutdown()
            if not self.reports_close:
                return {"arms": {}}
            if not self._finished.wait(QUIT_WAIT_S):
                raise SessionError("the session is shutting down but has not finished resting the arms; watch "
                                   "the terminal that runs it")
            return {"arms": self._outcomes}
        raise SessionError(f"unknown op {op!r}; expected step, observe, describe, stop, status or quit")


def _watch_hangup(conn: Optional[Connection], cancel: threading.Event) -> None:
    """Set ``cancel`` when the client closes its connection before the reply (``cancel`` set ends the watch)."""
    if conn is None:
        return
    while not cancel.is_set():
        try:
            if conn.poll(0.2):          # a client sends nothing more after its request: this is the hang-up
                cancel.set()
        except (EOFError, OSError, ValueError):
            cancel.set()


class Client:
    """Talks to a session with the harness's methods: ``observe``, ``step``, ``rest``, ``stop``,
    ``describe``. Raises SessionError when no session runs at ``address`` or it fails a request."""

    def __init__(self, address: Optional[str] = None, timeout_s: float = 300.0):
        self.address = str(address or default_address())
        self.timeout_s = timeout_s
        _open(self.address).close()

    def request(self, op: str, cancel: Optional[threading.Event] = None, **fields: Any) -> dict[str, Any]:
        """Send one request and return the reply's fields. Setting ``cancel`` closes the connection, which
        cancels this request's motion on the server, and raises SessionError."""
        conn = _open(self.address)
        try:
            conn.send_bytes(json.dumps({"op": op, **fields}).encode())
            deadline = time.monotonic() + self.timeout_s
            while not conn.poll(0.1):
                if cancel is not None and cancel.is_set():
                    raise SessionError("the request was cancelled; the arm holds where it is")
                if time.monotonic() > deadline:
                    raise SessionError(f"the session at {self.address} gave no answer within {self.timeout_s:g} s")
            reply = json.loads(conn.recv_bytes())
        except (EOFError, OSError) as e:
            raise SessionError(f"the session at {self.address} ended before it answered") from e
        finally:
            conn.close()
        if not reply.get("ok"):
            raise SessionError(reply.get("error", "the session refused the request"))
        reply.pop("ok")
        return reply

    def observe(self) -> Observation:
        return Observation.from_dict(self.request("observe")["observation"])

    def step(self, action: Union[Action, dict, str], cancel: Optional[threading.Event] = None) -> Observation:
        if isinstance(action, tuple(ACTIONS.values())):
            action = action_to_dict(action)
        return Observation.from_dict(self.request("step", cancel=cancel, action=action)["observation"])

    def rest(self, arm: Optional[str] = None) -> Observation:
        return self.step({"action": "rest", **({"arm": arm} if arm else {})})

    def stop(self) -> None:
        self.request("stop")

    def describe(self) -> str:
        return self.request("describe")["text"]

    def status(self) -> dict[str, Any]:
        return self.request("status")

    def quit(self) -> dict[str, str]:
        """End the session; returns each arm's torque outcome once the server has rested and closed them."""
        return self.request("quit").get("arms", {})

    def close(self) -> None:
        """Nothing to free: every request uses its own connection."""

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
