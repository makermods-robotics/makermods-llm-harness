"""One robot held open by a long-lived server, driven by short-lived clients over a local socket.

``llm-harness serve`` runs a ``Server`` around a ``Harness``; each ``llm-harness op ...`` is a ``Client``
request. Requests and replies are JSON objects sent as bytes over ``multiprocessing.connection``
(a Unix socket on POSIX, inside a directory only its owner can enter; a named pipe on Windows), one
connection per request:

    {"op": "step", "action": {...}}  ->  {"ok": true, "observation": {...}}   ("images": false: no cameras)
    {"op": "observe"}                ->  {"ok": true, "observation": {...}}
    {"op": "state"}                  ->  {"ok": true, "observation": {...}}    (the arms only; no device wait)
    {"op": "look"}                   ->  {"ok": true, "path", "errors"}        (one camera grid)
    {"op": "check", "action": {...}} ->  {"ok": true, "result": {...}}         (a dry run: nothing moves)
    {"op": "reachable", "poses": [...], "arm", "line", "chain", "start"}  ->  {"ok": true, "poses": [...]}
    {"op": "py", "code": "...", "wait_s", "run_s", "ack", "owner"}  ->  {"ok": true, "output", "error", "stopped",
                                     "motion", "arms", "faults", "queue", "busy", "reports", "images", "image_dir"}
    {"op": "kack", "ack": N}         ->  {"ok": true}       (reports up to N were shown)
    {"op": "kcancel", "owner": "..."}  ->  {"ok": true, "stopped": [...]}   (that owner's kernel work)
    {"op": "describe"}               ->  {"ok": true, "text": "..."}
    {"op": "stop"}                   ->  {"ok": true}       (after every arm that is on holds, confirmed)
    {"op": "status"}                 ->  {"ok": true, "robot", "arms", "cameras", "steps", "run_dir",
                                          "read_only", "busy", "queue", "kernel"}     (no device I/O)
    {"op": "quit"}                   ->  {"ok": true, "arms": {arm: torque outcome}}   (after the shutdown)

Two channels. The operator's socket (``address``) takes every request; its key is in an owner-only file next to
it (``key_path``). The kernel's socket (``address`` + ".k", POSIX only) is for the session's Python kernel process,
with a key the session hands that process directly: it takes queries, and ``step``, ``kqueue``, ``kflush`` and
``stop`` only with a live execution ``token`` (``kernel.py``). Every connection first answers a random challenge
with an HMAC under its channel's key, and the whole request frame must arrive within READ_S.

Admission: every request that can move or queue registers with the server's gate when it is accepted, before
any wait, and is cancelled by a stop that comes after; ``stop`` holds every arm that is on and confirms it before
it answers, or answers the failure. ``shutdown`` enters its phase under the same gate: from then on only the
shutdown's own rest commands the arms.

A step's motion is cancelled (the arm holds) when its client's connection closes before the reply, and so are a py
call's motions and the work it queued. A motion step not sent by a queued action stops the kernel's running action
and drops its queue first. Anything that goes wrong answers ``{"ok": false, "error": "..."}``; the server keeps
running.

The kernel process runs model code as the user who started the session, with that user's rights: it is not
sandboxed. Its socket only limits what it can ask of the session. It is not available on Windows, where the key file
cannot be shown to be private.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import os
import secrets
import shutil
import socket
import stat
import struct
import sys
import tempfile
import threading
import time
import uuid
from multiprocessing.connection import Client as _connect, Connection, Listener
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .actions import ACTIONS, Action, ActionError, Observation, Result, action_to_dict, parse_action
from .robot import home

FAMILY = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
MAX_SOCKET_PATH = 100     # bytes; Unix socket paths are limited to about this on Linux and macOS
MAX_REQUEST = 1 << 20     # bytes in one request
READ_S = 5.0              # a client must answer the challenge and send its whole request within this
MAX_CLIENTS = 16          # connections handled at once; past that only stop and status are answered ...
MAX_CONTROL = 4           # ... by up to this many more, and further connections are closed
MAX_QUEUED = 4            # step and observe requests running or waiting at once
QUIT_WAIT_S = 600.0       # how long quit waits for the arms to rest before it answers
MAX_PY_WAIT_S = 300.0     # longest wait_s and run_s of a py request
CHALLENGE = b"llm-harness challenge "
KEY_BYTES = 32
KERNEL_AVAILABLE = FAMILY != "AF_PIPE"    # the py op needs a key file shown to be private
KERNEL_OPS = ("state", "look", "check", "reachable", "describe", "status", "kstatus", "step", "kqueue", "kflush",
              "stop")
MUTATING = ("step", "kqueue", "kflush", "stop")


class SessionError(RuntimeError):
    """No session to talk to, or the session refused a request. The message says what to do."""


def default_address() -> str:
    """``<home>/session/llm-harness.sock`` on POSIX; on Windows a named pipe derived from the home directory."""
    if FAMILY == "AF_PIPE":
        return r"\\.\pipe\llm-harness-" + hashlib.sha1(str(home()).encode()).hexdigest()[:12]
    return str(home() / "session" / "llm-harness.sock")


def private_address() -> str:
    """A new address for a session only this process uses (``mcp`` and ``run`` without ``--connect``)."""
    if FAMILY == "AF_PIPE":
        return rf"\\.\pipe\llm-harness-{uuid.uuid4().hex[:16]}"
    return os.path.join(tempfile.mkdtemp(prefix="lh"), "s.sock")


def kernel_address(address: str) -> str:
    return address + ".k"


def _side_file(address: str, suffix: str) -> Path:
    if FAMILY == "AF_PIPE":
        return home() / "session" / (hashlib.sha1(address.encode()).hexdigest()[:16] + suffix)
    return Path(address + suffix)


def key_path(address: str) -> Path:
    """The file holding the operator key: next to the socket on POSIX, in ``<home>/session`` on Windows."""
    return _side_file(address, ".key")


def _private_dir(address: str) -> None:
    """Create the socket's directory owner-only (0700), or refuse one that others can enter or that is not
    the user's."""
    folder = Path(address).parent
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = folder.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise SessionError(f"{folder} must belong to you and be closed to others (chmod 700 {folder}) before a "
                           "session socket goes in it")


class AddressLock:
    """Exclusive ownership of a session address, held from before a server opens any device until it has
    cleaned up; a second server for the same address is refused while it is held."""

    def __init__(self, address: str):
        self.path = _side_file(address, ".lock")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._file = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._file.close()
            raise SessionError(f"a session is already running or starting at {address}") from e

    def release(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _publish_key(path: Path) -> bytes:
    """A new key, written owner-only to a temporary file and renamed over ``path`` in one step."""
    key = secrets.token_bytes(KEY_BYTES)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    os.replace(tmp, path)
    return key


def _read_key(address: str) -> bytes:
    path = key_path(address)
    try:
        key = path.read_bytes()
    except OSError as e:
        raise SessionError(f"no session is running at {address} (its key {path} cannot be read: {e}); start one "
                           "with `llm-harness serve`") from e
    if len(key) != KEY_BYTES:
        raise SessionError(f"the session key {path} is damaged; restart the session")
    return key


def _digest(key: bytes, challenge: bytes) -> bytes:
    return hmac.new(key, challenge, "sha256").digest()


def _connect_raw(address: str) -> Connection:
    try:
        return _connect(address, family=FAMILY)
    except OSError as e:
        raise SessionError(f"no session is running at {address}; start one with `llm-harness serve`") from e


def _open(address: str, key: bytes) -> Connection:
    """A connection that has answered the session's challenge."""
    conn = _connect_raw(address)
    try:
        if not conn.poll(READ_S):
            raise SessionError(f"the session at {address} sent no challenge")
        challenge = conn.recv_bytes(256)
        if not challenge.startswith(CHALLENGE):
            raise SessionError(f"the server at {address} is not an llm-harness session")
        conn.send_bytes(_digest(key, challenge))
    except (EOFError, OSError) as e:
        conn.close()
        raise SessionError(f"the session at {address} closed the connection") from e
    except BaseException:
        conn.close()
        raise
    return conn


def _recv_frame(conn: Connection, deadline: float, limit: int) -> Optional[bytes]:
    """One whole frame (``multiprocessing.connection`` framing) received by ``deadline`` (monotonic), at most
    ``limit`` bytes; None when it did not all arrive in time. OSError when it is larger."""
    if FAMILY != "AF_UNIX":
        if not conn.poll(max(0.0, deadline - time.monotonic())):
            return None
        return conn.recv_bytes(limit)
    sock = socket.socket(fileno=os.dup(conn.fileno()))
    try:
        header = _read_exact(sock, 4, deadline)
        if header is None:
            return None
        size, = struct.unpack("!i", header)
        if size < 0 or size > limit:
            raise OSError("request too large")
        return _read_exact(sock, size, deadline)
    finally:
        sock.setblocking(True)          # the descriptor's flags are shared with the connection's
        sock.close()


def _read_exact(sock: socket.socket, size: int, deadline: float) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        sock.settimeout(left)
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            return None
        if not chunk:
            raise EOFError
        data += chunk
    return bytes(data)


def ensure_free(address: str) -> None:
    """Make ``address`` ready for a new server: refuse when a live session answers there, remove the socket
    file a dead one left behind."""
    if FAMILY == "AF_UNIX" and len(os.fsencode(kernel_address(address))) > MAX_SOCKET_PATH:
        raise SessionError(f"the session address {address} is too long for a Unix socket (at most "
                           f"{MAX_SOCKET_PATH - 2} bytes); pass a shorter path with --address")
    for path in (address, kernel_address(address)) if FAMILY == "AF_UNIX" else (address,):
        try:
            _connect_raw(path).close()
        except SessionError:
            if FAMILY == "AF_UNIX" and os.path.lexists(path):
                if not stat.S_ISSOCK(os.lstat(path).st_mode):
                    raise SessionError(f"{path} exists and is not a socket; pass another path with --address")
                os.unlink(path)
            continue
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


class Ticket:
    """An admitted request: the stop generation it was accepted in and the event a later stop sets."""

    def __init__(self, server: "Server", generation: int):
        self.server, self.generation = server, generation
        self.cancel = threading.Event()

    def valid(self) -> bool:
        """Still admitted (call with the server's gate held)."""
        return self.server._generation == self.generation and self.server._phase == "open" \
            and not self.cancel.is_set()

    def done(self) -> None:
        with self.server._gate:
            self.server._admitted.discard(self)


class Server:
    """Answers requests for one harness, each connection on its own thread, so ``stop`` and ``status`` are
    answered while an action runs. ``say`` gets one line per request.

    The owner ends it with ``shutdown()`` (its phase starts: nothing admitted earlier moves any more, every arm
    that is on holds, the kernel's process and queue worker end; ``shutdown_problems`` lists what could not be
    confirmed), ``harness.stop()``, ``drain()`` (the requests already running finish), ``harness.close()``,
    ``finish(outcomes)`` (a quit waits for it and answers with them when ``reports_close``; without it a quit
    answers once the server stops taking requests) and last ``release()``, which gives up the address. ``on_quit`` is called once a quit has started the shutdown, for
    an owner that must be woken to rest the arms. ``lock`` is an ``AddressLock`` the caller already holds."""

    def __init__(self, harness, address: Optional[str] = None, say: Optional[Callable[[str], None]] = None,
                 reports_close: bool = False, on_quit: Optional[Callable[[], None]] = None,
                 lock: Optional[AddressLock] = None):
        self.harness = harness
        self.reports_close = reports_close
        self.on_quit = on_quit
        self.address = str(address or default_address())
        self.say = say
        if FAMILY == "AF_UNIX":
            _private_dir(self.address)
        owns_lock = lock is None
        self._lock = lock or AddressLock(self.address)
        self._listeners: list[tuple[Listener, str]] = []
        try:
            ensure_free(self.address)
            self._key_file = key_path(self.address)
            self._keys = {"operator": _publish_key(self._key_file), "kernel": secrets.token_bytes(KEY_BYTES)}
            # macOS refuses a Unix socket connection outright once the backlog is full (Linux makes it wait),
            # so the backlog covers a burst of clients while the accept loop is busy.
            self._listeners.append((Listener(self.address, family=FAMILY, backlog=MAX_CLIENTS), "operator"))
            if KERNEL_AVAILABLE:
                self._listeners.append((Listener(kernel_address(self.address), family=FAMILY,
                                                 backlog=MAX_CLIENTS), "kernel"))
            self._addresses = [self.address] + ([kernel_address(self.address)] if KERNEL_AVAILABLE else [])
        except BaseException:
            for listener, _ in self._listeners:
                listener.close()
            if owns_lock:                     # a caller's lock stays with it until its arms are dealt with
                self._lock.release()
            raise
        if FAMILY == "AF_UNIX":
            for path in self._addresses:
                os.chmod(path, 0o600)
        self._closing = threading.Event()
        self._finished = threading.Event()
        self._outcomes: dict[str, str] = {}
        self._serving = False
        self._clients = threading.BoundedSemaphore(MAX_CLIENTS)
        self._control = threading.BoundedSemaphore(MAX_CONTROL)
        self._queued = threading.BoundedSemaphore(MAX_QUEUED)
        self._work: set[threading.Thread] = set()          # handling step or observe
        self._moving: set[threading.Thread] = set()        # handling a step
        self._py_work: set[threading.Thread] = set()       # handling py
        self._handlers: set[threading.Thread] = set()      # handling any request
        self._work_lock = threading.Lock()
        self._gate = threading.Lock()                      # admission, stop and the shutdown phase
        self._generation = 0
        self._phase = "open"
        self._admitted: set[Ticket] = set()
        self._kernel = None
        self._loop_failure: Optional[BaseException] = None
        self.shutdown_problems: list[str] = []

    @property
    def kernel_key(self) -> bytes:
        return self._keys["kernel"]

    def serve_forever(self) -> None:
        """Accept connections on both channels until ``shutdown()``."""
        self._serving = True
        loops = [threading.Thread(target=self._accept_or_report, args=(listener, channel), daemon=True)
                 for listener, channel in self._listeners[1:]]
        for loop in loops:
            loop.start()
        try:
            self._accept(*self._listeners[0])
            for loop in loops:
                loop.join(10)
            if self._loop_failure is not None:
                raise self._loop_failure
        finally:
            self._close_listeners()

    def _accept_or_report(self, listener: Listener, channel: str) -> None:
        """A second accept loop: a failure ends serving, raised from ``serve_forever`` to its owner."""
        try:
            self._accept(listener, channel)
        except BaseException as e:
            if not self._closing.is_set():
                self._loop_failure = e
                self._closing.set()
                try:
                    _connect(self.address, family=FAMILY).close()     # wakes the main accept loop
                except OSError:
                    pass

    def _close_listeners(self) -> None:
        """Close both listeners; safe to call again."""
        for listener, _ in self._listeners:
            try:
                listener.close()
            except OSError:
                pass

    def release(self) -> None:
        """Give up the address: remove the key file and release the address lock. The owner calls it last, once
        the arms' outcomes are published; safe to call again."""
        self._close_listeners()
        try:
            self._key_file.unlink()
        except OSError:
            pass
        self._lock.release()

    def _accept(self, listener: Listener, channel: str) -> None:
        while not self._closing.is_set():
            try:
                conn = listener.accept()
            except OSError:
                if self._closing.is_set():
                    break
                raise
            if self._closing.is_set():
                conn.close()
                break
            self._handle_connection(conn, channel)

    def _handle_connection(self, conn: Connection, channel: str) -> None:
        """Hand an accepted connection to a handler thread of its own."""
        if not _same_user(conn):
            conn.close()
            return
        if self._clients.acquire(blocking=False):
            full = False
        elif self._control.acquire(blocking=False):
            full = True                               # saturated: answers only stop and status
        else:
            conn.close()
            return
        handler = threading.Thread(target=self._handle, args=(conn, channel, full), daemon=True)
        with self._work_lock:
            self._handlers = {t for t in self._handlers if t.is_alive()} | {handler}
        handler.start()

    # ── admission ──
    def admit(self) -> Ticket:
        """Register a request that can move or queue, before it waits for anything."""
        with self._gate:
            if self._phase != "open":
                raise SessionError("the session is shutting down; its rest owns the arms")
            ticket = Ticket(self, self._generation)
            self._admitted.add(ticket)
            return ticket

    def stop(self, token: Optional[str] = None) -> None:
        """Cancel everything admitted so far, revoke the kernel's work and queue, then hold every arm that is on
        and confirm it. Raises SessionError with what could not be confirmed. ``token`` (a stop from kernel code)
        must still be live when the gate is taken; that code is not interrupted, so it gets this answer."""
        with self._gate:
            if self._phase != "open":
                raise SessionError("the session is shutting down: its rest owns the arms")
            if token is not None and (self._kernel is None or not self._kernel.valid(token)):
                raise SessionError("stopped: this code may no longer stop the arm (it was stopped or has ended)")
            problem = self._halt_locked(spare=token)
        if problem:
            raise SessionError(f"stopped, but {problem}")

    def _halt_locked(self, spare: Optional[str] = None) -> Optional[str]:
        self._generation += 1
        for ticket in self._admitted:
            ticket.cancel.set()
        if self._kernel is not None:
            self._kernel.stop(spare=spare)
        return self.harness.stop()

    def shutdown(self) -> None:
        """Enter the shutdown phase (from any thread): take no new requests, cancel everything admitted, hold every
        arm that is on and confirm it, then end the kernel's process and queue worker and ``serve_forever``."""
        with self._gate:
            if self._phase != "open":
                return
            self._phase = "shutdown"
            self._closing.set()
            problem = self._halt_locked()
        if problem:
            self.shutdown_problems.append(f"the hold before the rest is not confirmed: {problem}")
        with self._work_lock:
            kernel = self._kernel
        if kernel is not None:
            problem = kernel.close()
            if problem:
                self.shutdown_problems.append(problem)
        if not self._serving:
            self._close_listeners()
            return
        for path in self._addresses:
            try:
                _connect(path, family=FAMILY).close()     # wakes a blocked accept(); refused once it is closed
            except OSError:
                pass

    def drain(self, timeout_s: float = 60.0) -> None:
        """Wait for the requests already running to finish."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._work_lock:
                work = [t for t in self._work | self._py_work if t is not threading.current_thread()]
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

    def kernel(self):
        """The session's Python kernel, made on first use."""
        if not KERNEL_AVAILABLE:
            raise SessionError("the Python kernel is not available on Windows: the session key file cannot be "
                               "shown to be private there; use `llm-harness op` or the MCP action tools")
        with self._work_lock:
            if self._closing.is_set():
                raise SessionError("the session is closing")
            if self._kernel is None:
                from .kernel import KernelHost
                self._kernel = KernelHost(self.harness, self)
            return self._kernel

    # ── connections ──
    def _authenticate(self, conn: Connection, channel: str, deadline: float) -> bool:
        challenge = CHALLENGE + secrets.token_bytes(32)
        try:
            conn.send_bytes(challenge)
            answer = _recv_frame(conn, deadline, 64)
        except (EOFError, OSError):
            return False
        return answer is not None and hmac.compare_digest(answer, _digest(self._keys[channel], challenge))

    def _handle(self, conn: Connection, channel: str, full: bool) -> None:
        try:
            with conn:
                deadline = time.monotonic() + (0.5 if full else READ_S)
                if not self._authenticate(conn, channel, deadline):
                    return
                try:
                    request = _recv_frame(conn, deadline, MAX_REQUEST)
                except (EOFError, OSError):
                    return
                if request is None:
                    return
                only = ("stop", "status") if full else (KERNEL_OPS if channel == "kernel" else None)
                _, reply = self._reply(request, conn, channel, only)
                delivered = reply.pop("_delivered", None)
                sent = False
                try:
                    conn.send_bytes(json.dumps(reply, default=str).encode())
                    sent = True
                except OSError:
                    pass
                finally:
                    if delivered is not None:
                        delivered(sent)       # anything but a sent reply revokes the work the call queued
        finally:
            (self._control if full else self._clients).release()

    def _reply(self, raw: bytes, conn: Optional[Connection], channel: str = "operator",
               only: Optional[tuple[str, ...]] = None) -> tuple[str, dict[str, Any]]:
        op = "?"
        try:
            request = json.loads(raw)
            if not isinstance(request, dict) or not isinstance(request.get("op"), str):
                raise SessionError('a request is a JSON object with an "op"')
            op = request["op"]
            if only is not None and op not in only:
                raise SessionError(f"{op!r} is not available here" if channel == "kernel" else
                                   f"the session is busy ({MAX_CLIENTS} connections); try again")
            if channel == "kernel" and op in MUTATING and op != "stop":
                token = request.get("token")
                kernel = self._kernel
                if not isinstance(token, str) or kernel is None or not kernel.valid(token):
                    if op == "step":
                        obs = self.harness.state()
                        obs.result = Result("stopped", "stopped on request: this code may no longer move the arm.")
                        return op, {"ok": True, "observation": obs.to_dict(), "revoked": True}
                    raise SessionError("stopped: this code may no longer move the arm or queue actions (it was "
                                       "stopped or has ended)")
            reply = {"ok": True, **self._run(op, request, conn, channel)}
        except Exception as e:
            reply = {"ok": False, "error": str(e) if isinstance(e, SessionError) else f"{type(e).__name__}: {e}"}
        if self.say:
            try:                          # the log line is best-effort: it never changes the reply
                what = op
                if "observation" in reply:
                    action = reply["observation"]["result"].get("action") or {}
                    what += f" {action.get('action', '?')}" if op == "step" else ""
                    outcome = reply["observation"]["result"]["status"]
                else:
                    outcome = "ok" if reply["ok"] else f"error: {reply['error']}"
                self.say(f"{time.strftime('%H:%M:%S')} {what} -> {outcome}")
            except Exception:
                pass
        return op, reply

    def _run(self, op: str, request: dict, conn: Optional[Connection], channel: str = "operator") -> dict[str, Any]:
        h = self.harness
        if op == "step":
            return self._step(request, conn, channel)
        if op == "observe":
            self._enter(self._work)
            try:
                obs = h.observe()
            finally:
                self._leave(self._work)
            unsaved = any(c.jpeg is not None and c.path is None for c in obs.cameras.values())
            return {"observation": obs.to_dict(images=unsaved)}
        if op == "state":
            return {"observation": h.state().to_dict()}
        if op == "look":
            look = h.look()
            if look["path"] and self._kernel is not None:
                self._kernel.issue(look["path"], look["jpeg"])
            reply = {"path": look["path"], "errors": look["errors"]}
            if look["path"] is None and look["jpeg"] is not None:
                reply["jpeg_base64"] = base64.b64encode(look["jpeg"]).decode("ascii")
            return reply
        if op == "check":
            if "action" not in request:
                raise SessionError('check needs an "action"')
            return {"result": dataclasses.asdict(h.check(request["action"]))}
        if op == "reachable":
            poses = request.get("poses")
            if not isinstance(poses, list):
                raise SessionError('reachable needs "poses", a list')
            return {"poses": h.reachable(poses, arm=request.get("arm"), line=request.get("line", True) is not False,
                                         chain=bool(request.get("chain")), start=request.get("start"))}
        if op == "py":
            return self._py(request, conn)
        if op in ("kqueue", "kflush"):
            ticket = self.admit()
            try:
                return self.kernel().request(op, request, ticket)
            finally:
                ticket.done()
        if op == "kstatus":
            return self.kernel().queue_status()
        if op == "kack":
            ack = request.get("ack")
            if self._kernel is not None and isinstance(ack, int) and not isinstance(ack, bool):
                self._kernel.ack(ack)
            return {}
        if op == "kcancel":
            owner = request.get("owner")
            if not isinstance(owner, str):
                raise SessionError('kcancel needs an "owner"')
            return {"stopped": self._kernel.cancel_owner(owner) if self._kernel is not None else []}
        if op == "describe":
            return {"text": h.describe()}
        if op == "stop":
            token = request.get("token") if channel == "kernel" else None
            if channel == "kernel" and not isinstance(token, str):
                raise SessionError("stopped: this code may no longer stop the arm (it was stopped or has ended)")
            self.stop(token)
            return {}
        if op == "status":
            with self._work_lock:
                moving = bool(self._moving)
                kernel = self._kernel
            queue = kernel.queue_status() if kernel is not None else {"running": None, "waiting": []}
            queue = {"running": queue["running"], "waiting": queue["waiting"]}
            return {"robot": h.robot_name, "arms": list(h.models), "cameras": list(h.cameras), "steps": h.steps,
                    "run_dir": str(h.log.run_dir) if h.log else None, "read_only": h.read_only,
                    "busy": moving or queue["running"] is not None or bool(queue["waiting"]), "queue": queue,
                    "kernel": KERNEL_AVAILABLE}
        if op == "quit":
            self.shutdown()
            if self.on_quit is not None:
                self.on_quit()
            if not self.reports_close:
                return {"arms": {}}
            if not self._finished.wait(QUIT_WAIT_S):
                raise SessionError("the session is shutting down but has not finished resting the arms; watch "
                                   "the terminal that runs it")
            return {"arms": self._outcomes}
        raise SessionError(f"unknown op {op!r}; expected step, observe, state, look, check, reachable, py, describe, "
                           "stop, status or quit")

    def _enter(self, *sets) -> None:
        with self._work_lock:
            for s in sets:
                s.add(threading.current_thread())

    def _leave(self, *sets) -> None:
        with self._work_lock:
            for s in sets:
                s.discard(threading.current_thread())

    def _step(self, request: dict, conn: Optional[Connection], channel: str) -> dict[str, Any]:
        """A step admitted at once (a stop from now on ends it); for the kernel's channel its token must be live
        then. Then the kernel's queue is interrupted for a motion it did not send, then the step runs."""
        if "action" not in request:
            raise SessionError('step needs an "action"')
        if not self._queued.acquire(blocking=False):
            raise SessionError(f"{MAX_QUEUED} actions are already running or waiting; try again")
        try:
            ticket = self.admit()
            token = request.get("token") if channel == "kernel" else None
            kernel = self._kernel
            admitted = True
            if token is not None:
                with self._gate:
                    admitted = ticket.valid() and kernel.admit(token, ticket.cancel)
            if not admitted:
                ticket.done()
                obs = self.harness.state()
                obs.result = Result("stopped", "stopped on request: this code may no longer move the arm.")
                return {"observation": obs.to_dict(), "revoked": True}
            moving = (self._work, self._moving) if _moves(request["action"]) else (self._work,)
            self._enter(*moving)
            watcher = threading.Thread(target=_watch_hangup, args=(conn, ticket.cancel), daemon=True)
            watcher.start()
            try:
                if kernel is not None and _moves(request["action"]) and (token is None or kernel.is_cell(token)):
                    kernel.interrupt()
                obs = self.harness.step(request["action"], cancel=ticket.cancel,
                                        images=request.get("images", True) is not False)
            finally:
                done = ticket.cancel.is_set()
                ticket.cancel.set()
                ticket.done()
                self._leave(*moving)
                if token is not None:
                    kernel.release(token, ticket.cancel)
        finally:
            self._queued.release()
        unsaved = any(c.jpeg is not None and c.path is None for c in obs.cameras.values())
        reply = {"observation": obs.to_dict(images=unsaved)}
        if token is not None and done and not kernel.valid(token):
            reply["revoked"] = True
        return reply

    def _py(self, request: dict, conn: Optional[Connection]) -> dict[str, Any]:
        code, wait_s = request.get("code", ""), request.get("wait_s")
        run_s, ack, owner = request.get("run_s", MAX_PY_WAIT_S), request.get("ack"), request.get("owner")
        if not isinstance(code, str):
            raise SessionError('py: "code" must be text')
        for name, value in (("wait_s", wait_s), ("run_s", run_s)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not 0 <= value <= MAX_PY_WAIT_S):
                raise SessionError(f'py: "{name}" must be a number from 0 to {MAX_PY_WAIT_S:g}')
        if ack is not None and (isinstance(ack, bool) or not isinstance(ack, int)):
            raise SessionError('py: "ack" must be a report number')
        if owner is not None and not isinstance(owner, str):
            raise SessionError('py: "owner" must be text')
        ticket = self.admit()                    # before anything that can wait
        try:
            kernel = self.kernel()
        except BaseException:
            ticket.done()
            raise
        self._enter(self._py_work)
        finished = threading.Event()
        watcher = threading.Thread(target=_watch_hangup, args=(conn, ticket.cancel, finished), daemon=True)
        watcher.start()
        try:
            reply, delivered = kernel.execute(code, wait_s, run_s or MAX_PY_WAIT_S, ticket, ack=ack, owner=owner)
            reply["_delivered"] = delivered
        finally:
            finished.set()
            ticket.done()
            self._leave(self._py_work)
        return reply


def _moves(action) -> bool:
    """Whether a step's action is anything but observe or wait (a malformed one counts as moving)."""
    try:
        return parse_action(action).name not in ("observe", "wait")
    except ActionError:
        return True


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _watch_hangup(conn: Optional[Connection], cancel: threading.Event,
                  finished: Optional[threading.Event] = None) -> None:
    """Set ``cancel`` when the client closes its connection before the reply. The watch ends when ``finished``
    (or else ``cancel``) is set."""
    if conn is None:
        return
    finished = finished or cancel
    while not finished.is_set():
        try:
            if conn.poll(0.2):          # a client sends nothing more after its request: this is the hang-up
                cancel.set()
                return
        except (EOFError, OSError, ValueError):
            cancel.set()
            return


class Client:
    """Talks to a session with the harness's methods: ``observe``, ``step``, ``rest``, ``stop``,
    ``describe``. Raises SessionError when no session runs at ``address`` or it fails a request. ``key`` is the
    channel key when it is not the operator's (the kernel process's own)."""

    def __init__(self, address: Optional[str] = None, timeout_s: float = 300.0, key: Optional[bytes] = None):
        self.address = str(address or default_address())
        self.timeout_s = timeout_s
        self._key = key if key is not None else _read_key(self.address)
        self._shown = 0            # the highest report number this client has returned
        _open(self.address, self._key).close()

    def request(self, op: str, cancel: Optional[threading.Event] = None, timeout_s: Optional[float] = None,
                **fields: Any) -> dict[str, Any]:
        """Send one request and return the reply's fields. Setting ``cancel`` closes the connection, which
        cancels this request's motion on the server, and raises SessionError. ``timeout_s`` overrides the
        client's timeout for this request."""
        conn = _open(self.address, self._key)
        try:
            conn.send_bytes(json.dumps({"op": op, **fields}, default=_jsonable).encode())
            timeout_s = self.timeout_s if timeout_s is None else timeout_s
            deadline = time.monotonic() + timeout_s
            while not conn.poll(0.1):
                if cancel is not None and cancel.is_set():
                    raise SessionError("the request was cancelled; the arm holds where it is")
                if time.monotonic() > deadline:
                    raise SessionError(f"the session at {self.address} gave no answer within {timeout_s:g} s")
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

    def step(self, action: Union[Action, dict, str], cancel: Optional[threading.Event] = None,
             images: bool = True) -> Observation:
        if isinstance(action, tuple(ACTIONS.values())):
            action = action_to_dict(action)
        extra = {} if images else {"images": False}
        return Observation.from_dict(self.request("step", cancel=cancel, action=action, **extra)["observation"])

    def state(self) -> Observation:
        return Observation.from_dict(self.request("state")["observation"])

    def look(self) -> dict:
        """``{"path", "jpeg", "errors"}`` as ``Harness.look`` gives them."""
        reply = self.request("look")
        encoded = reply.get("jpeg_base64")
        return {"path": reply.get("path"), "jpeg": base64.b64decode(encoded) if encoded else None,
                "errors": reply.get("errors", {})}

    def check(self, action: Union[Action, dict, str]) -> Result:
        if isinstance(action, tuple(ACTIONS.values())):
            action = action_to_dict(action)
        return Result(**self.request("check", action=action)["result"])

    def reachable(self, poses: list, arm: Optional[str] = None, line: bool = True, chain: bool = False,
                  start=None) -> list[dict]:
        return self.request("reachable", poses=poses, arm=arm, line=line, chain=chain, start=start)["poses"]

    def py(self, code: str = "", wait_s: Optional[float] = None, cancel: Optional[threading.Event] = None,
           timeout_s: Optional[float] = None, owner: Optional[str] = None) -> dict[str, Any]:
        """Run code in the session's kernel (``KernelHost.execute``); closing the request (``cancel``) stops its
        motions and the work it queued. ``timeout_s`` bounds how long the code may run (the kernel restarts after
        it). The request acknowledges the reports this client has already returned; the reply's reports are only
        those it has not."""
        run_s = min(MAX_PY_WAIT_S, self.timeout_s if timeout_s is None else timeout_s)
        fields: dict[str, Any] = {"code": code, "run_s": run_s, "ack": self._shown}
        if wait_s is not None:
            fields["wait_s"] = wait_s
        if owner is not None:
            fields["owner"] = owner
        reply = self.request("py", cancel=cancel, timeout_s=run_s + (wait_s or 0) + 30, **fields)
        reply["reports"] = [r for r in reply.get("reports", []) if r.get("seq", 0) > self._shown]
        self._shown = max([self._shown] + [r.get("seq", 0) for r in reply["reports"]])
        return reply

    def ack(self) -> None:
        """Acknowledge every report this client has returned (a client that makes no further call)."""
        if self._shown:
            self.request("kack", ack=self._shown)

    def cancel(self, owner: str) -> list[str]:
        """Stop the kernel work queued by calls made with ``owner``; returns what was stopped."""
        return self.request("kcancel", owner=owner)["stopped"]

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


class PrivateSession:
    """A session for one process's own use (``mcp`` and ``run`` without ``--connect``): a ``Server`` for
    ``harness`` on a new private address, served on a thread, and a ``client`` for it. ``server`` and
    ``harness`` go to a ``cli.Lifecycle`` as for ``serve``; a ``quit`` wakes the main thread (KeyboardInterrupt),
    so that lifecycle rests the arms, and answers with their outcomes. ``cleanup()`` removes the address
    afterwards."""

    def __init__(self, harness, say: Optional[Callable[[str], None]] = None):
        import _thread
        self.harness = harness
        address = private_address()
        self.server = Server(harness, address, say=say, reports_close=True, on_quit=_thread.interrupt_main)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        self.client = Client(address)

    def cleanup(self) -> None:
        self.server.shutdown()
        self._thread.join(10)
        self.server.release()
        if FAMILY == "AF_UNIX":
            shutil.rmtree(os.path.dirname(self.server.address), ignore_errors=True)
