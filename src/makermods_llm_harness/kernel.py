"""The Python kernel of a session: model code runs in a separate process, the action queue and the authority to
move stay in the session.

``llm-harness py``, the MCP ``run_python`` tool and the model loop send code with the session's ``py`` request. The
session (``KernelHost``) starts a child process (``kernel_child``) that keeps one namespace between calls, with
``r`` (a ``Program`` on a session client), ``np``, ``math`` and ``kernel``. The child holds no harness and no device.
It talks to the session over the kernel channel, with a key the session gives it: there every request that moves or
queues needs a live execution token, and the operator's requests (``py``, ``quit``, ...) are not available.

Authority: each call's code and each running queued action has an execution token, made under the server's gate
only while the request that asked for it is still admitted (no stop since it was accepted). A step from the child
is admitted only while its token is valid; a stop, a hang-up, the end of the call or action (as its message
arrives), a timeout or the session's shutdown revoke it, which also ends the motions it has in flight. Code whose
token is revoked can neither move nor queue; the child is told to interrupt it.

The queue lives in the session. ``kernel.queue(fn)`` in the child registers the function there under a key and asks
the session to queue it; each action records the call that queued it and that call's ``owner``. The session's worker
runs one action at a time: it gives the action a token, tells the child to run it, waits until it ends (or times
out, or is revoked), holds every arm that is on and confirms it if it did not end done, records its report (how it
ended, how long it took, what it printed, each arm's state, a camera grid taken then), and only then starts the
next. An action that did not end done drops everything waiting. At most MAX_WAITING actions wait. A motion not sent
by a queued action stops the running action and drops the queue first. Work a call queued persists once that call's
reply has reached its client; if the call is stopped, its client leaves or its reply cannot be sent, that work is
stopped too. ``kcancel`` stops the work of one owner (a model run).

Reports are numbered. Every reply carries each report its client has not acknowledged; a client acknowledges the
highest number it has shown with its next call (or ``kack``) and does not show a number twice. At most MAX_REPORTS
are kept. A call whose code runs longer than its timeout ends the child process, so the kernel restarts empty; so
does the session's shutdown, which revokes everything first and never waits for model code.

The child runs model code as the user who started the session, with all of that user's rights: it is not
sandboxed. Its channel only limits what it can ask of the session.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .session import SessionError, kernel_address

MAX_WAITING = 3              # actions waiting behind the running one
ACTION_TIMEOUT_S = 120.0     # default limit of one action; an action still running then is stopped
MAX_ACTION_TIMEOUT_S = 600.0
MAX_CODE = 200_000           # characters of code in one call
MAX_REPORTS = 100            # undelivered reports kept; older ones are dropped
CALL_WAIT_S = 2.0            # how long a call waits for another call's code to finish
GRACE_S = 2.0                # how long revoked code gets to end before it is left behind
START_S = 60.0               # longest start of the kernel process
KILL_S = 1.0                 # from asking the kernel process to end to killing it
MAX_LINE = 4 << 20           # bytes in one message from the kernel process
MAX_IMAGE_BYTES = 8 << 20    # largest image file a client reads from a reply

TOOL = {
    "name": "run_python",
    "description": ("Run Python in the robot's persistent kernel and get what it printed, each arm's state, the "
                    "action queue and the reports of queued actions that finished (with camera grid image paths). "
                    "The namespace keeps r (the robot), np, math and kernel between calls. wait_s: then wait up to "
                    "that long for the next report (or for the queue to empty)."),
    "input_schema": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python code; may be empty with wait_s."},
        "wait_s": {"type": "number", "minimum": 0, "maximum": 120}},
        "required": ["code"], "additionalProperties": False},
}

GUIDE = """Python kernel (the run_python tool): one namespace kept between calls, with r (the robot), np, math and \
kernel. Metres and degrees, world frame (z up).
- Queries move nothing and never disturb a running action: r.state().text(); r.look() (one camera grid, returns its \
path); r.reachable([p1, p2, ...]) where each pose is [x, y, z] or {"xyz": [...], "pitch": -90} (ok, why, margin_deg \
per pose, checked exactly as r.move would check it; chain=True checks each from the one before); any motion with \
dry_run=True.
- Motions: r.move(xyz, pitch=None, yaw=None, roll=None, line=True, speed=None, duration=None, until=None), \
r.nudge(dxyz, until="contact"), r.move_joints({name: deg}), r.gripper(0..1 or "open"/"close"), r.home(), r.rest(); \
arm="name" on each when there are two arms. A motion that does not end done raises MotionFailed; a stop raises \
Stopped. until="contact" ends a move where the arm stops on something (result.contact); result.grip["holding"] \
says whether the jaws stopped on something.
- Put motions in actions: def grasp(r): ...; kernel.queue(grasp) returns at once and the session runs it. \
kernel.queue([a, b]) queues several; at most 3 wait. kernel.queue(new, replace_from=id) replaces that waiting action \
and those behind it; kernel.flush() drops what waits; kernel.stop() halts the arm and drops the queue (and stops \
the code that called it); kernel.status(). A motion sent by your own call while an action runs stops that action \
first.
- Each finished action reports once: done/error/stopped, seconds, what it printed, the queue, each arm, a camera \
grid path. An error or stop drops what waits and the arm holds.
- The cycle: look, measure, check reach of the whole plan in one r.reachable call, queue the action, then check and \
queue the next while it runs. Define helpers once and reuse them. With nothing to prepare, wait for the next report \
(wait_s) instead of polling."""


@dataclass
class _Queued:
    id: int
    key: str
    name: str
    timeout: float
    token: Optional[str] = None
    status: str = "waiting"          # waiting, running, done, error, stopped
    message: str = ""
    output: str = ""
    started: float = 0.0
    ended: float = 0.0
    dropped: list = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)       # the child said it ended
    wake: threading.Event = field(default_factory=threading.Event)       # done, or revoked
    revoked: bool = False
    timed_out: bool = False
    cell: Optional[str] = None       # the token of the call that queued it
    owner: Optional[str] = None      # the owner that call named (a model run)
    confirmed: bool = True           # the call's reply reached its client

    def label(self) -> str:
        return f"#{self.id} {self.name}"


class KernelHost:
    """The kernel of one session: the child process, the execution tokens, the action queue and its reports.
    ``server`` is the session's ``Server``: its gate orders token creation and queueing against stops."""

    def __init__(self, harness, server):
        self.harness = harness
        self.server = server
        self._cond = threading.Condition()
        self._tokens: dict[str, set] = {}      # valid execution token -> cancel events of its motions in flight
        self._cells: dict[str, Optional[str]] = {}     # running call's token -> its owner
        self._cell_tokens: dict[int, str] = {}
        self._waiting: list[_Queued] = []
        self._running: Optional[_Queued] = None
        self._reports: list[dict] = []
        self._lost_reports = 0
        self._ids = itertools.count(1)
        self._seqs = itertools.count(1)
        self._cell_ids = itertools.count(1)
        self._calls = threading.Lock()
        self._cell_results: dict[int, dict] = {}
        self._cell_events: dict[int, threading.Event] = {}
        self._proc: Optional[subprocess.Popen] = None
        self._outbox: Optional[queue.Queue] = None
        self._ready = threading.Event()
        self._alive = False
        self._closed = False
        self._spared: set[str] = set()         # tokens whose own code asked for a stop
        self._issued: dict[str, str] = {}      # image file the session made -> SHA-256 of the bytes it wrote
        self._start_lock = threading.Lock()
        self._dispatcher = threading.Thread(target=self._dispatch, name="kernel-queue", daemon=True)
        self._dispatcher.start()

    # ── authority ──
    def admit(self, token: str, cancel: threading.Event) -> bool:
        """Admit a motion of ``token``: True, with ``cancel`` set on revocation, while the token is valid."""
        with self._cond:
            if self._closed or token not in self._tokens:
                return False
            self._tokens[token].add(cancel)
            return True

    def release(self, token: str, cancel: threading.Event) -> None:
        with self._cond:
            self._tokens.get(token, set()).discard(cancel)

    def valid(self, token: str) -> bool:
        with self._cond:
            return token in self._tokens

    def is_cell(self, token: str) -> bool:
        with self._cond:
            return token in self._cells

    def _new_token(self, kind: str, owner: Optional[str] = None) -> str:
        token = f"{kind}-{secrets.token_hex(8)}"
        self._tokens[token] = set()
        if kind == "cell":
            self._cells[token] = owner
        return token

    def _revoke(self, token: Optional[str], interrupt: bool = True) -> None:
        """Revoke ``token`` (the lock is held): its motions in flight end and, with ``interrupt``, the child
        interrupts its code."""
        if token is None or token not in self._tokens:
            return
        for cancel in self._tokens.pop(token):
            cancel.set()
        self._cells.pop(token, None)
        if interrupt:
            self._send({"cmd": "interrupt", "token": token})

    def _drop(self, which, why: str) -> list[str]:
        """Drop the waiting actions ``which(a)`` selects and revoke the running one if selected (the lock is held);
        their labels."""
        dropped = [a for a in self._waiting if which(a)]
        self._waiting = [a for a in self._waiting if not which(a)]
        if dropped:
            self._send({"cmd": "forget", "keys": [a.key for a in dropped]})
        labels = [a.label() for a in dropped]
        running = self._running
        if running is not None and which(running) and not running.ended:
            running.dropped += labels
            if not running.revoked:
                running.revoked = True
                running.message = running.message or why
            self._revoke(running.token)
            running.wake.set()
        self._cond.notify_all()
        return labels

    def stop(self, spare: Optional[str] = None) -> None:
        """Revoke every call and action: their motions end, nothing waits any more. The code running under
        ``spare`` (the one that asked for the stop) is revoked but neither interrupted nor marked stopped: it ends
        on the stop's own answer (a hold that could not be confirmed included), which its result or report
        carries."""
        with self._cond:
            if spare is not None:
                self._spared.add(spare)
            running = self._running
            spared_action = running is not None and running.token == spare
            labels = self._drop(lambda a: not (spared_action and a is running), "stopped on request")
            if spared_action:
                running.dropped += labels
            for token in list(self._tokens):
                self._revoke(token, interrupt=token != spare)

    def interrupt(self) -> None:
        """Before a motion not sent by a queued action: stop the running action and drop what waits."""
        with self._cond:
            if self._running is not None or self._waiting:
                self._drop(lambda a: True, "stopped: a motion was sent from outside the queue")

    def cancel_owner(self, owner: str) -> list[str]:
        """Revoke the running calls of ``owner`` and the actions they queued, waiting or running."""
        with self._cond:
            stopped = [a.label() for a in self._waiting if a.owner == owner]
            if self._running is not None and self._running.owner == owner and not self._running.ended:
                stopped.insert(0, self._running.label())
            self._drop(lambda a: a.owner == owner, "stopped: the run that queued it ended")
            for token, cell_owner in list(self._cells.items()):
                if cell_owner == owner:
                    self._revoke(token)
            return stopped

    def _cancel_cell_work(self, token: str, why: str) -> None:
        """Revoke a call and every action of its submission not yet acknowledged, those its actions queued
        included (the lock is held)."""
        self._drop(lambda a: a.cell == token and not a.confirmed, why)
        self._revoke(token)

    def close(self) -> Optional[str]:
        """Revoke everything, end the child process and the queue's worker, and confirm both ended. None when
        they have, else what has not."""
        with self._cond:
            self._closed = True
            proc = self._proc
            for a in self._waiting + ([self._running] if self._running else []):
                a.wake.set()
            self._cond.notify_all()
        self.stop()
        problems = []
        if proc is not None and not self._kill(proc):
            problems.append(f"the kernel process {proc.pid} did not end")
        self._dispatcher.join(GRACE_S + 5.0)
        if self._dispatcher.is_alive():
            problems.append("the kernel's queue worker did not end")
        return "; ".join(problems) or None

    # ── requests from the child ──
    def request(self, op: str, request: dict, ticket) -> dict:
        """``kqueue`` or ``kflush`` from code running under ``request["token"]``, admitted with ``ticket``."""
        token = request.get("token")
        with self.server._gate, self._cond:
            if self._closed or not ticket.valid() or not isinstance(token, str) or token not in self._tokens:
                raise SessionError("stopped: this code may no longer queue actions (it was stopped or has ended)")
            if op == "kflush":
                return {"dropped": self._drop(lambda a: a is not self._running, "")}
            keys, names, timeout = request.get("keys"), request.get("names"), request.get("timeout", ACTION_TIMEOUT_S)
            replace_from = request.get("replace_from")
            if not (isinstance(keys, list) and isinstance(names, list) and 0 < len(keys) == len(names) <= 10
                    and all(isinstance(k, str) and len(k) <= 64 for k in keys)
                    and all(isinstance(n, str) for n in names)):
                raise SessionError("kqueue: keys and names must be two lists of 1 to 10 strings")
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or \
                    not 0 < timeout <= MAX_ACTION_TIMEOUT_S:
                raise SessionError(f"timeout must be above 0 and at most {MAX_ACTION_TIMEOUT_S:g} s")
            waiting = list(self._waiting)
            if replace_from is not None:
                at = next((i for i, a in enumerate(waiting) if replace_from in (a.id, a.name)), None)
                if at is None:
                    raise SessionError(f"{replace_from!r} is not waiting (it has started, ended or never existed); "
                                       "nothing changed")
                waiting = waiting[:at]
            room = MAX_WAITING + (0 if self._running is not None else 1)
            if len(waiting) + len(keys) > room:
                raise SessionError(f"at most {MAX_WAITING} actions wait behind the running one; nothing changed "
                                   f"({len(waiting)} already wait)")
            replaced = self._waiting[len(waiting):]
            if replaced:
                self._send({"cmd": "forget", "keys": [a.key for a in replaced]})
            if token in self._cells:
                cell, owner, confirmed = token, self._cells[token], False
            else:                                 # queued by a running action: its submission's identity
                parent = self._running
                if parent is None or parent.token != token:
                    raise SessionError("stopped: this code may no longer queue actions")
                cell, owner, confirmed = parent.cell, parent.owner, parent.confirmed
            new = [_Queued(next(self._ids), k, n[:80] or "action", float(timeout), cell=cell, owner=owner,
                           confirmed=confirmed) for k, n in zip(keys, names)]
            self._waiting = waiting + new
            self._cond.notify_all()
            return {"ids": [a.id for a in new]}

    def queue_status(self) -> dict:
        """``{"running": {"id", "name", "seconds"} or None, "waiting": [{"id", "name"}], "reports_pending"}``."""
        with self._cond:
            running = self._running
            return {"running": None if running is None else {
                        "id": running.id, "name": running.name,
                        "seconds": round(time.monotonic() - running.started, 1)},
                    "waiting": [{"id": a.id, "name": a.name} for a in self._waiting],
                    "reports_pending": len(self._reports)}

    def issue(self, path: str, jpeg: Optional[bytes]) -> None:
        """Record an image file the session made and the digest of the bytes it wrote, so a reply may name it."""
        if jpeg is None:
            return
        with self._cond:
            self._issued[os.path.realpath(path)] = hashlib.sha256(jpeg).hexdigest()
            while len(self._issued) > 10 * MAX_REPORTS:
                self._issued.pop(next(iter(self._issued)))

    # ── reports ──
    def ack(self, upto: int) -> None:
        """Forget the reports numbered up to ``upto``: their client has shown them."""
        with self._cond:
            self._reports = [r for r in self._reports if r["seq"] > upto]

    def _keep(self, report: dict) -> None:
        """Add a report (the lock is held), keeping at most MAX_REPORTS."""
        self._reports.append(report)
        if len(self._reports) > MAX_REPORTS:
            self._lost_reports += len(self._reports) - MAX_REPORTS
            del self._reports[:-MAX_REPORTS]
        self._cond.notify_all()

    # ── calls ──
    def execute(self, code: str, wait_s: Optional[float], timeout_s: float, ticket, ack: Optional[int] = None,
                owner: Optional[str] = None) -> tuple:
        """Run ``code`` (if any) in the child for at most ``timeout_s``, then with ``wait_s`` wait up to that long
        for a report or an empty queue. ``ticket`` (the request's admission) cancelled by a stop or a hang-up
        revokes the call. Reports up to ``ack`` are forgotten first; the reply carries every report not yet
        acknowledged, each with its ``seq``. Returns the reply, ``{"output", "error", "stopped", "motion", "arms",
        "faults", "queue", "busy", "reports", "images", "image_dir"}``, and a function to call with whether it
        reached the client: actions the call queued persist once it has; otherwise they are stopped."""
        if len(code) > MAX_CODE:
            raise SessionError(f"the code is longer than {MAX_CODE} characters")
        if ack is not None:
            self.ack(ack)
        result = {"output": "", "error": False, "stopped": False, "motion": None, "images": []}
        token = None
        if code.strip():
            started = time.monotonic()
            if not self._calls.acquire(timeout=CALL_WAIT_S):
                raise SessionError("another py call is running in the kernel; wait for it, or stop it with "
                                   "`llm-harness op stop`")
            try:
                result, token = self._cell(code, timeout_s, ticket, owner)
            finally:
                self._calls.release()
            self._log("py", seconds=round(time.monotonic() - started, 3), chars=len(code), error=result["error"])
        if wait_s and not ticket.cancel.is_set():
            self.wait(wait_s, ticket.cancel)
        with self._cond:
            reports = list(self._reports)
            lost, self._lost_reports = self._lost_reports, 0
            busy = self._running is not None or bool(self._waiting)
            queue_text = self._queue_text()
        arms, faults = self._arms()
        image_dir = str(self.harness.image_dir) if self.harness.image_dir is not None else None
        with self._cond:
            issued = dict(self._issued)
        images = [p for p in list(result.get("images", [])) + [r["image"] for r in reports if r.get("image")]
                  if _inside(p, image_dir) and os.path.realpath(p) in issued]
        digests = {p: issued[os.path.realpath(p)] for p in images}
        reply = {**result, "arms": arms, "faults": faults, "queue": queue_text, "busy": busy, "reports": reports,
                 "images": images, "image_dir": image_dir, "image_digests": digests}
        if lost:
            reply["reports_lost"] = lost

        def delivered(sent: bool) -> None:
            if token is None:
                return
            with self._cond:
                if sent:
                    for a in self._waiting + ([self._running] if self._running else []):
                        if a.cell == token:
                            a.confirmed = True
                else:
                    self._cancel_cell_work(token, "stopped: the call that queued it did not reach its client")
        if ticket.cancel.is_set() and token is not None:
            with self._cond:
                self._cancel_cell_work(token, "stopped: the call that queued it was stopped or its client left")
        return reply, delivered

    def wait(self, seconds: float, cancel: Optional[threading.Event] = None) -> None:
        """Until a report is pending or nothing runs or waits, at most ``seconds``."""
        deadline = time.monotonic() + seconds
        with self._cond:
            while not (self._reports or (self._running is None and not self._waiting) or self._closed):
                left = deadline - time.monotonic()
                if left <= 0 or (cancel is not None and cancel.is_set()):
                    return
                self._cond.wait(min(left, 0.2))

    def _cell(self, code: str, timeout_s: float, ticket, owner: Optional[str]) -> tuple[dict, Optional[str]]:
        self._ensure_child()
        cell = next(self._cell_ids)
        done = threading.Event()
        with self.server._gate, self._cond:
            if self._closed or not ticket.valid():
                return {"output": "stopped: a stop came before this code started", "error": True, "stopped": True,
                        "motion": None, "images": []}, None
            token = self._new_token("cell", owner)
            self._cell_events[cell] = done
            self._cell_tokens[cell] = token
            sent = self._send({"cmd": "cell", "id": cell, "token": token, "code": code})
        try:
            if not sent:
                raise SessionError("the kernel process is not running")
            deadline = time.monotonic() + timeout_s
            given_up = None
            while not done.wait(0.05):
                if not self._alive:
                    break
                if ticket.cancel.is_set() and given_up is None and token not in self._spared:
                    with self._cond:
                        self._cancel_cell_work(token, "stopped: the call that queued it was stopped or its client "
                                                      "left")
                    given_up = time.monotonic() + GRACE_S
                if given_up is not None and time.monotonic() > given_up:
                    break
                if time.monotonic() > deadline:
                    with self._cond:
                        self._cancel_cell_work(token, "stopped: the call that queued it ran out of time")
                        proc = self._proc
                    if proc is not None:
                        self._kill(proc)
                    return {"output": f"the code did not finish within {timeout_s:g} s; the kernel was restarted "
                                      "(its variables are gone)", "error": True, "stopped": True, "motion": None,
                            "images": []}, token
            with self._cond:
                found = self._cell_results.pop(cell, None)
            if found is None:
                return {"output": "the kernel process ended before the code finished; it restarts with the next "
                                  "call (its variables are gone)" if not self._alive else
                        "the code was stopped", "error": True, "stopped": True, "motion": None, "images": []}, token
            motion = found.get("motion")
            return {"output": str(found.get("output", "")), "error": bool(found.get("error")),
                    "stopped": bool(found.get("stopped")),
                    "motion": motion if motion in ("done", "rejected", "stopped", "fault") else None,
                    "images": [str(p) for p in found.get("images", [])][:20]}, token
        finally:
            with self._cond:
                self._revoke(token)
                self._cell_events.pop(cell, None)
                self._cell_tokens.pop(cell, None)
                self._cell_results.pop(cell, None)

    # ── the child process ──
    def _ensure_child(self) -> None:
        with self._start_lock:
            if self._alive:
                return
            with self._cond:
                if self._closed:
                    raise SessionError("the session is closing")
            self._ready.clear()
            proc = subprocess.Popen([sys.executable, "-m", "makermods_llm_harness.kernel_child",
                                     kernel_address(self.server.address)],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=dict(os.environ))
            outbox: queue.Queue = queue.Queue()
            outbox.put(json.dumps({"key": self.server.kernel_key.hex()}) + "\n")
            with self._cond:
                if self._closed:
                    proc.kill()
                    proc.wait()
                    raise SessionError("the session is closing")
                self._proc, self._outbox, self._alive = proc, outbox, True
            threading.Thread(target=self._writer, args=(proc, outbox), name="kernel-writer", daemon=True).start()
            threading.Thread(target=self._reader, args=(proc,), name="kernel-reader", daemon=True).start()
            if not self._ready.wait(START_S) or not self._alive:
                self._kill(proc)
                raise SessionError(f"the kernel process did not start within {START_S:g} s")

    def _send(self, message: dict) -> bool:
        """Queue a message for the child; never blocks. False when no child runs."""
        outbox = self._outbox
        if outbox is None or not self._alive:
            return False
        outbox.put(json.dumps(message) + "\n")
        return True

    @staticmethod
    def _writer(proc: subprocess.Popen, outbox: queue.Queue) -> None:
        while True:
            line = outbox.get()
            if line is None:
                return
            try:
                proc.stdin.write(line.encode())
                proc.stdin.flush()
            except (OSError, ValueError):
                return

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            while True:
                line = proc.stdout.readline(MAX_LINE)
                if not line:
                    break
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._on_message(message)
        except (OSError, ValueError):
            pass
        finally:
            self._child_ended(proc)

    def _on_message(self, message: dict) -> None:
        """A message from the child. An ending call or action loses its authority here, as the message comes."""
        event = message.get("event")
        if event == "ready":
            self._ready.set()
        elif event == "cell_done":
            with self._cond:
                cell = message.get("id")
                self._revoke(self._cell_tokens.get(cell))
                done = self._cell_events.get(cell)
                if done is not None:
                    self._cell_results[cell] = message
                    done.set()
        elif event == "action_done":
            with self._cond:
                running = self._running
                if running is not None and running.id == message.get("id") and not running.done.is_set() \
                        and not running.ended:
                    self._revoke(running.token)
                    status = message.get("status")
                    running.status = status if status in ("done", "error", "stopped") else "error"
                    running.message = running.message or str(message.get("message", ""))[:4000]
                    running.output = str(message.get("output", ""))
                    running.done.set()
                    running.wake.set()

    def _child_ended(self, proc: subprocess.Popen) -> None:
        with self._cond:
            if self._proc is not proc:
                return
            self._alive = False
            if self._outbox is not None:
                self._outbox.put(None)
            for done in self._cell_events.values():
                done.set()
            running = self._running
            if running is not None and not running.done.is_set():
                running.message = running.message or "the kernel process ended"
                running.status = "error"
                running.done.set()
                running.wake.set()
            dropped, self._waiting = self._waiting, []
            if running is not None:
                running.dropped += [a.label() for a in dropped]
            for token in list(self._tokens):
                for cancel in self._tokens.pop(token):
                    cancel.set()
            self._cells.clear()
            self._proc, self._outbox = None, None
            self._cond.notify_all()
        self._ready.set()

    def _kill(self, proc: subprocess.Popen) -> bool:
        """End the child: terminate, then kill after KILL_S; never waits on its pipes. Whether it ended."""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(KILL_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(5 * KILL_S)
                except subprocess.TimeoutExpired:
                    return False
        self._child_ended(proc)
        return True

    # ── the queue's worker ──
    def _dispatch(self) -> None:
        while True:
            with self._cond:
                while not self._closed and not (self._waiting and self._alive):
                    self._cond.wait()
                if self._closed:
                    return
                a = self._waiting.pop(0)
                a.token = self._new_token("action")
                a.status, a.started = "running", time.monotonic()
                self._running = a
                sent = self._send({"cmd": "run", "id": a.id, "key": a.key, "token": a.token})
            if not sent:
                a.status, a.message = "error", "the kernel process is not running"
            else:
                deadline = a.started + a.timeout
                while not a.wake.is_set() and not self._closed:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        with self._cond:
                            a.message = a.message or f"timed out after {a.timeout:g} s"
                            a.revoked = a.timed_out = True
                            self._revoke(a.token)
                        break
                    a.wake.wait(min(left, 0.5))
                if (a.revoked or self._closed) and not a.done.is_set():
                    a.done.wait(0 if self._closed else GRACE_S)
            with self._cond:
                self._revoke(a.token)
                if sent and not a.done.is_set():
                    a.message = (a.message + "; " if a.message else "") + \
                        "its code did not end and was left behind (it can no longer move the arm)"
                    a.status = "stopped" if (a.revoked or self._closed) and not a.timed_out else "error"
                elif a.revoked:
                    a.status = "error" if a.timed_out else "stopped"
                a.ended = time.monotonic()
                closed = self._closed
            if a.status != "done" and not closed:
                problem = self.harness.hold()
                if problem:
                    a.message = (a.message + "; " if a.message else "") + f"the hold is not confirmed: {problem}"
            report = self._report(a)
            with self._cond:
                if a.status != "done":
                    a.dropped += self._drop(lambda w: w is not a, "")
                report["dropped"] = list(a.dropped)
                report["started"] = self._waiting[0].label() if self._waiting else None
                report["waiting"] = [w.label() for w in self._waiting[1:]]
                self._keep(report)
                self._running = None
                self._cond.notify_all()
                if self._closed:
                    return

    def _report(self, a: _Queued) -> dict:
        """The action's report, taken before anything else starts."""
        try:
            look = self.harness.look()
            image, camera_errors = look["path"], look["errors"]
            if image:
                self.issue(image, look["jpeg"])
        except Exception as e:
            image, camera_errors = None, {"grid": f"{type(e).__name__}: {e}"}
        arms, _ = self._arms()
        report = {"seq": next(self._seqs), "id": a.id, "name": a.name, "status": a.status,
                  "seconds": round(a.ended - a.started, 2), "message": a.message, "output": a.output, "arms": arms,
                  "image": image, "image_time": round(time.time(), 3), "dropped": [], "started": None, "waiting": []}
        if camera_errors:
            report["camera_errors"] = camera_errors
        self._log("kernel_action", id=a.id, name=a.name, status=a.status, seconds=report["seconds"], message=a.message)
        return report

    def _log(self, event: str, **fields) -> None:
        log = getattr(self.harness, "log", None)
        if log is not None:
            try:
                log.write(event, **fields)
            except (OSError, ValueError):
                pass

    # ── text ──
    def _queue_text(self) -> str:
        running = self._running
        parts = []
        if running is not None:
            parts.append(f"running {running.label()} for {time.monotonic() - running.started:.1f} s")
        if self._waiting:
            parts.append("waiting " + ", ".join(a.label() for a in self._waiting))
        return "; ".join(parts) or "empty"

    def _arms(self) -> tuple[dict, dict]:
        try:
            obs = self.harness.state()
        except Exception as e:
            return {"?": f"state not read: {type(e).__name__}: {e}"}, {}
        return ({name: arm_line(arm) for name, arm in obs.arms.items()},
                {name: arm.fault for name, arm in obs.arms.items() if arm.fault})


def _inside(path, folder: Optional[str]) -> bool:
    """Whether ``path`` names a file inside ``folder``."""
    if not path or folder is None:
        return False
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(folder)]) == os.path.realpath(folder)
    except ValueError:
        return False


class RemoteKernel:
    """The kernel of a running session, through a session ``Client``; its calls carry ``owner``, so ``cancel()``
    stops only the work they queued."""

    def __init__(self, client):
        self.client = client
        self.owner = secrets.token_hex(8)

    def execute(self, code: str = "", wait_s: Optional[float] = None,
                cancel: Optional[threading.Event] = None) -> dict:
        return self.client.py(code, wait_s, cancel=cancel, owner=self.owner)

    def cancel(self) -> list[str]:
        """Stop the work this kernel's calls queued; SessionError when the session cannot."""
        return self.client.cancel(self.owner)


def kernel_for(target) -> Optional[RemoteKernel]:
    """The kernel of ``target`` when it is a session Client whose session offers one, else None."""
    if not hasattr(target, "py"):
        return None
    try:
        return RemoteKernel(target) if target.status().get("kernel") else None
    except Exception:
        return None


def reply_images(reply: dict, count: int) -> list[bytes]:
    """The JPEG bytes of the newest ``count`` images a kernel reply names: only files inside its ``image_dir``, at
    most MAX_IMAGE_BYTES each, whose bytes still match the digest the session recorded when it wrote them."""
    out = []
    digests = reply.get("image_digests") or {}
    for path in [p for p in reply.get("images", []) if _inside(p, reply.get("image_dir")) and p in digests][-count:]:
        try:
            if os.path.getsize(path) <= MAX_IMAGE_BYTES:
                with open(path, "rb") as f:
                    data = f.read(MAX_IMAGE_BYTES)
                if hashlib.sha256(data).hexdigest() == digests[path]:
                    out.append(data)
        except OSError:
            continue
    return out


def arm_line(arm) -> str:
    """One line for an ``ArmObservation``."""
    x, y, z = arm.tool.xyz
    clearance = "unknown" if arm.clearance_m is None else f"{arm.clearance_m:.3f} m"
    line = (f"tool [{x:.3f}, {y:.3f}, {z:.3f}] pitch {arm.tool.pitch_deg:.0f} yaw {arm.tool.yaw_deg:.0f} roll "
            f"{arm.tool.roll_deg:.0f}; gripper {arm.gripper:.2f}; clearance {clearance}")
    if arm.enabled is False:
        line += "; motors off"
    if arm.fault:
        line += f"; FAULT: {arm.fault}"
    return line


def format_reply(reply: dict) -> str:
    """A kernel reply as text: what the code printed, ``[state]`` per arm, ``[queue]``, then each report."""
    lines = [reply["output"].rstrip("\n")] if reply.get("output") else []
    lines += [f"[state] {name}: {text}" for name, text in reply.get("arms", {}).items()]
    lines.append(f"[queue] {reply.get('queue', 'empty')}")
    if reply.get("reports_lost"):
        lines.append(f"[reports] {reply['reports_lost']} older reports were dropped undelivered")
    for report in reply.get("reports", []):
        lines.append(report_text(report))
    return "\n".join(lines)


def report_text(report: dict) -> str:
    lines = [f"[action #{report['id']} {report['name']}] {report['status']} in {report['seconds']:.1f} s"
             + (f": {report['message']}" if report.get("message") else "")]
    lines += [f"  | {line}" for line in report.get("output", "").splitlines()]
    after = []
    if report.get("started"):
        after.append(f"started {report['started']}")
    if report.get("waiting"):
        after.append("waiting " + ", ".join(report["waiting"]))
    if report.get("dropped"):
        after.append("dropped " + ", ".join(report["dropped"]))
    lines.append("  then: " + ("; ".join(after) or "nothing queued behind it"))
    lines += [f"  {name}: {text}" for name, text in report.get("arms", {}).items()]
    if report.get("image"):
        lines.append(f"  image: {report['image']}")
    for camera, error in report.get("camera_errors", {}).items():
        lines.append(f"  camera {camera}: {error}")
    return "\n".join(lines)
