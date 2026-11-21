"""A model loop: a model API drives the robot through the action tools until the task is done.

    from makermods_llm_harness import Harness, agent
    with Harness.open("my-robot") as h:
        result = agent.run(h, "pick up the red cube and lift it 10 cm")

``target`` is a ``Harness`` or a session ``Client`` (anything with ``observe``, ``step`` and ``describe``).

The loop talks to a provider: any object with ``complete(system, messages, tools) -> Reply``.

- ``system`` is the prompt text and ``tools`` is ``actions.tools()`` (name, description, input_schema).
- ``messages`` alternate between ``{"role": "user", "content": [part, ...]}``, with parts
  ``{"type": "text", "text"}``, ``{"type": "image", "camera", "jpeg": bytes}`` and
  ``{"type": "tool_result", "id", "content": [text and image parts]}``, and
  ``{"role": "assistant", "text": str, "tool_calls": [ToolCall]}``.
- ``Reply`` is the model's text and its tool calls; a reply without tool calls ends the run.

``AnthropicProvider`` is the default; another API needs only that one method.
"""

from __future__ import annotations

import base64
import importlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .actions import Observation, tools

DEFAULT_MODEL = "claude-opus-5-5"
KEEP_IMAGES = 3          # the most recent observations whose images stay in the conversation
MAX_TOKENS = 16000
GRACE_S = 10.0           # how long an ended action may take to hold before the loop gives up on it

ROLE = ("You control a robot through tools. Each tool call runs one action and returns the robot's state as text "
        "and a fresh image from every camera.")
RULES = """How to work:
- Look before you move and after every move: check the images and the reported tool position.
- Near the table and near objects, move in small steps and check the clearance each time.
- "rejected" means nothing moved: read the message, change the request and try again.
- "stopped" means the action ended early: look before going on.
- "fault" means the arm needs the operator: call no more tools and report what happened.
- When the task is done, or cannot be done, reply with a short plain-text summary and no tool call."""
FAULT_NOTE = "The arm reported a fault. Call no more tools: tell the operator what happened and what you saw."


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    refused: bool = False     # the model declined the request


@dataclass
class RunResult:
    """``status``: "done", "step_limit", "timeout", "stopped" (the operator cancelled), "fault" or "refused".
    ``message`` is the model's last text (its summary or report) or why the loop stopped."""
    status: str
    steps: int
    message: str


def _parts(obs: Observation, text: str) -> list[dict]:
    missing = obs.missing_images()
    if missing:
        text += "\n(no image delivered for " + ", ".join(missing) + ")"
    return [{"type": "text", "text": text}] + [{"type": "image", "camera": name, "jpeg": jpeg}
                                               for name, jpeg in obs.jpegs().items()]


def _drop_old_images(holders: list[list[dict]]) -> None:
    """Replace the images of all but the last KEEP_IMAGES observations with a short note."""
    for parts in holders[:-KEEP_IMAGES]:
        for i, part in enumerate(parts):
            if part["type"] == "image":
                parts[i] = {"type": "text", "text": f"(older {part['camera']} image left out)"}


def _line(call: ToolCall, obs: Observation) -> str:
    line = f"{call.name} {json.dumps(call.input)} -> {obs.result.status}"
    return f"{line}: {obs.result.message[:100]}" if obs.result.message else line


def _in_thread(fn, cancel: threading.Event, deadline: float, ended: Optional[threading.Event] = None):
    """``fn()`` on a worker thread, waited for until it returns, ``cancel`` is set or ``deadline`` (monotonic)
    passes; then ``ended`` is set and the worker is waited for no longer than GRACE_S more. Returns (done,
    value)."""
    box: dict = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as e:      # handed to the caller
            box["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive() and not cancel.is_set() and time.monotonic() < deadline:
        worker.join(0.05)
    if worker.is_alive() and ended is not None:
        ended.set()
        worker.join(GRACE_S)
    if "error" in box:
        raise box["error"]
    return "value" in box, box.get("value")


def run(target, task: str, *, model: str = DEFAULT_MODEL, max_steps: int = 40, timeout_s: float = 600.0,
        provider=None, say: Callable[[str], None] = print, cancel: Optional[threading.Event] = None) -> RunResult:
    """Let the model work on ``task`` until it answers without a tool call, ``max_steps`` actions have run,
    ``timeout_s`` has passed, ``cancel`` is set, or an action ends in a fault (the model then reports, without
    more actions). The model's replies and the actions are waited for no longer than the time left; an action
    still running then is ended (the arm holds). After an action that stops early, the rest of that reply's
    tool calls are not run."""
    provider = provider or AnthropicProvider(model)
    cancel = cancel or threading.Event()
    deadline = time.monotonic() + timeout_s
    system = f"{ROLE}\n\n{target.describe()}\n\n{RULES}"
    defs = tools()
    obs = target.observe()
    first = _parts(obs, f"Task: {task}\n\nThe robot now:\n{obs.text()}")
    messages: list[dict] = [{"role": "user", "content": first}]
    holders = [first]
    steps, fault = 0, None

    def ended() -> Optional[RunResult]:
        if cancel.is_set():
            return RunResult("stopped", steps, "stopped by the operator")
        if time.monotonic() >= deadline:
            return RunResult("timeout", steps, f"the run took longer than {timeout_s:g} s")
        return None

    while True:
        if ended():
            return ended()
        done, reply = _in_thread(lambda: provider.complete(system, messages, defs), cancel, deadline)
        if not done:
            return ended()
        messages.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})
        if reply.refused:
            return RunResult("refused", steps, reply.text or "the model declined the task")
        if fault is not None:
            return RunResult("fault", steps, reply.text or fault)
        if not reply.tool_calls:
            return RunResult("done", steps, reply.text)
        results: list[dict] = []
        skip = None
        for call in reply.tool_calls:
            if skip is None and steps >= max_steps:
                return RunResult("step_limit", steps, f"stopped after {steps} actions without finishing")
            if skip is None and ended():
                return ended()
            if skip is not None:
                results.append({"type": "tool_result", "id": call.id, "content": [{"type": "text", "text": skip}]})
                continue
            stop = threading.Event()
            done, obs = _in_thread(lambda: target.step({**call.input, "action": call.name}, cancel=stop),
                                   cancel, deadline, ended=stop)
            if not done:
                return ended() or RunResult("stopped", steps, "the action did not end")
            steps += 1
            say(f"[{steps}] {_line(call, obs)}")
            parts = _parts(obs, obs.text())
            results.append({"type": "tool_result", "id": call.id, "content": parts})
            holders.append(parts)
            if obs.result.status == "fault":
                fault = obs.result.message
                skip = "not run: the arm reported a fault"
            elif obs.result.status == "stopped":
                if ended():
                    return ended()
                skip = "not run: the action before it stopped early; look before going on"
        if fault is not None:
            results.append({"type": "text", "text": FAULT_NOTE})
        _drop_old_images(holders)
        messages.append({"role": "user", "content": results})


class AnthropicProvider:
    """The Anthropic Messages API. The API key comes from the environment (``ANTHROPIC_API_KEY``) unless
    ``client`` (an ``anthropic.Anthropic``) is given."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Optional[Any] = None):
        if client is None:
            try:
                anthropic = importlib.import_module("anthropic")
            except ImportError as e:
                raise ImportError("the model loop needs the anthropic package: "
                                  "pip install 'makermods-llm-harness[agent]'") from e
            client = anthropic.Anthropic()
        self.model = model
        self.client = client

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=system, tools=tools,
                messages=[_to_anthropic(m) for m in messages], cache_control={"type": "ephemeral"})
        except TypeError as e:
            if "authentication" not in str(e):
                raise
            raise RuntimeError("no Anthropic API key: set ANTHROPIC_API_KEY") from e
        text = "\n".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return Reply(text, calls, refused=response.stop_reason == "refusal")


def _to_anthropic(message: dict) -> dict:
    """One message in the Messages API shape. Thinking blocks are not sent back: dropping old images edits
    earlier turns, which would invalidate them."""
    if message["role"] == "assistant":
        content = [{"type": "text", "text": message["text"]}] if message["text"] else []
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in message["tool_calls"]]
        return {"role": "assistant", "content": content}
    return {"role": "user", "content": [_part(p) for p in message["content"]]}


def _part(part: dict) -> dict:
    if part["type"] == "image":
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(part["jpeg"]).decode("ascii")}}
    if part["type"] == "tool_result":
        return {"type": "tool_result", "tool_use_id": part["id"], "content": [_part(p) for p in part["content"]]}
    return {"type": "text", "text": part["text"]}
