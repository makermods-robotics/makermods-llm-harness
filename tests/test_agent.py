"""The model loop, driven by scripted providers on the kinematic simulator. No network: the Anthropic
provider is tested against the real SDK with a mock HTTP transport."""

import copy
from pathlib import Path
import json
import threading
import time

import numpy as np
import pytest

from makermods_llm_harness import agent, tools
from makermods_llm_harness.agent import AnthropicProvider, Reply, ToolCall
from makermods_llm_harness.kernel import TOOL

from conftest import APPROACH, GRASP_PITCH, GRASP_Z, sim_backend


def act(*actions):
    """A reply that calls one tool per action dict."""
    return Reply(tool_calls=[ToolCall(f"call-{i}", a["action"], {k: v for k, v in a.items() if k != "action"})
                             for i, a in enumerate(actions)])


class Scripted:
    """A provider that plays back replies (or functions of the messages) and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, system, messages, tools):
        self.requests.append((system, copy.deepcopy(messages), tools))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def tool_results(messages):
    return [p for m in messages if m["role"] == "user" for p in m["content"] if p["type"] == "tool_result"]


def images(parts):
    return [p for p in parts if p["type"] == "image"]


def test_scripted_pick_finishes_with_the_cube_lifted(sim_harness):
    h = sim_harness()
    moves = [{"action": "move_tool", "xyz": list(xyz), "pitch_deg": p} for xyz, p in APPROACH["metal"]]
    provider = Scripted(
        act({"action": "observe"}), act({"action": "home"}), act(*moves), act({"action": "gripper", "opening": 1}),
        act({"action": "move_tool", "xyz": [0.28, 0, GRASP_Z["metal"]], "pitch_deg": GRASP_PITCH["metal"]}),
        act({"action": "gripper", "opening": "close"}), act({"action": "nudge", "dxyz": [0, 0, 0.08]}),
        Reply("The red cube is in the gripper, 8 cm above the table."))
    lines = []
    result = agent.run(h, "pick up the red cube", provider=provider, say=lines.append)
    assert result == agent.RunResult("done", 12, "The red cube is in the gripper, 8 cm above the table.")
    cube = h.backend.objects[0]
    assert cube["held_by"] == "arm" and cube["xyz"][2] > 0.09
    assert len(lines) == 12 and lines[1] == "[2] home {} -> done"
    first = provider.requests[0][1][0]["content"]
    assert first[0]["text"].startswith("Task: pick up the red cube\n\nThe robot now:\n") and len(images(first)) == 2
    system, messages, defs = provider.requests[-1]
    assert "Robot " in system and "rejected" in system and defs == tools()
    last = tool_results(messages)[-1]
    assert last["id"] == "call-0" and "tool xyz" in last["content"][0]["text"]
    assert [p["camera"] for p in images(last["content"])] == ["top", "side"]
    assert all(p["jpeg"][:2] == b"\xff\xd8" for p in images(last["content"]))


def test_a_rejection_is_fed_back_and_the_run_goes_on(sim_harness):
    h = sim_harness()
    provider = Scripted(act({"action": "home"}), act({"action": "nudge", "dxyz": [0, 0, -0.6]}),
                        act({"action": "nudge", "dxyz": [0, 0, -0.05]}), Reply("Lowered the gripper by 5 cm."))
    result = agent.run(h, "lower the gripper", provider=provider, say=lambda line: None)
    assert result.status == "done" and result.steps == 3
    rejected = tool_results(provider.requests[2][1])[-1]["content"][0]["text"]
    assert rejected.startswith("rejected: ")
    assert h.observe().arms["arm"].tool.xyz[2] == pytest.approx(0.531, abs=2e-3)


def test_step_limit(sim_harness):
    provider = Scripted(*[act({"action": "observe"}, {"action": "observe"})] * 5)
    result = agent.run(sim_harness(), "look around", provider=provider, max_steps=3, say=lambda line: None)
    assert result.status == "step_limit" and result.steps == 3 and len(provider.requests) == 2


def test_timeout(sim_harness):
    def slow(messages):
        time.sleep(0.8)
        return act({"action": "observe"})

    result = agent.run(sim_harness(), "look", provider=Scripted(*[slow] * 10), timeout_s=2.0, say=lambda line: None)
    assert result.status == "timeout" and 1 <= result.steps <= 3     # each model turn takes at least 0.8 s


def test_a_fault_stops_the_actions_and_the_model_reports(sim_harness):
    h = sim_harness()

    def fault_then_move(messages):
        h.backend.set_fault("arm", "motor 3 lost power")
        return act({"action": "home"}, {"action": "observe"})

    provider = Scripted(act({"action": "home"}), fault_then_move, Reply("Motor 3 lost power; the operator must check it."))
    result = agent.run(h, "wave", provider=provider, say=lambda line: None)
    assert result == agent.RunResult("fault", 2, "Motor 3 lost power; the operator must check it.")
    last_user = provider.requests[-1][1][-1]["content"]
    assert last_user[0]["content"][0]["text"].startswith("fault: arm: motor 3 lost power")
    assert last_user[1]["content"][0]["text"] == "not run: the arm reported a fault"
    assert last_user[2] == {"type": "text", "text": agent.FAULT_NOTE}


def test_only_the_latest_observations_keep_their_images(sim_harness):
    provider = Scripted(*[act({"action": "observe"})] * 6, Reply("Seen enough."))
    agent.run(sim_harness(), "look", provider=provider, say=lambda line: None)
    messages = provider.requests[-1][1]
    results = tool_results(messages)
    assert [len(images(r["content"])) for r in results] == [0, 0, 0, 2, 2, 2]
    assert not images(messages[0]["content"])
    assert {"type": "text", "text": "(older top image left out)"} in results[0]["content"]


def test_no_action_runs_past_the_step_budget(sim_harness):
    h = sim_harness()
    result = agent.run(h, "home", provider=Scripted(act({"action": "home"})), max_steps=0, say=lambda line: None)
    assert result.status == "step_limit" and result.steps == 0 and not h.observe().arms["arm"].enabled


def test_operator_cancel_ends_the_run_during_inference(sim_harness):
    h = sim_harness()
    cancel = threading.Event()

    def thinking(messages):
        time.sleep(3)
        return act({"action": "home"})

    threading.Timer(0.2, cancel.set).start()
    start = time.monotonic()
    result = agent.run(h, "home", provider=Scripted(thinking), cancel=cancel, say=lambda line: None)
    assert result.status == "stopped" and result.steps == 0 and time.monotonic() - start < 2
    time.sleep(3.2)                              # the abandoned reply arrives and dispatches nothing
    assert not h.observe().arms["arm"].enabled


def test_the_deadline_ends_an_action_in_progress(sim_harness):
    h = sim_harness(backend=sim_backend(realtime=True))
    start = time.monotonic()
    result = agent.run(h, "home", provider=Scripted(act({"action": "home"})), timeout_s=0.5, say=lambda line: None)
    assert result.status == "timeout" and time.monotonic() - start < 3
    state = h.backend.read()["arm"]
    assert np.array_equal(state.q_target, state.q)                 # held where the deadline found it


def test_a_stopped_action_skips_the_rest_of_its_batch(sim_harness):
    h = sim_harness()
    h.step({"action": "home"})
    h.backend.block("arm")
    provider = Scripted(act({"action": "move_joints", "joints_deg": {"wrist_roll": 20}, "relative": True},
                            {"action": "nudge", "dxyz": [0, 0, -0.05]}), Reply("Blocked."))
    result = agent.run(h, "lower", provider=provider, say=lambda line: None)
    assert result.status == "done" and result.steps == 1
    skipped = tool_results(provider.requests[-1][1])[-1]["content"][0]["text"]
    assert skipped.startswith("not run: the action before it stopped early")


# ── the Anthropic provider ──
@pytest.fixture
def mock_api(monkeypatch, tmp_path):
    """An ``anthropic.Anthropic`` client whose HTTP transport records requests and answers from ``replies``."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    sent, replies = [], []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=replies.pop(0))

    def make(**kwargs):
        return anthropic.Anthropic(max_retries=0, http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(handler)), **kwargs)

    return make, sent, replies


def message(*content, stop_reason="tool_use"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": list(content),
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_anthropic_request_and_reply(mock_api):
    make, sent, replies = mock_api
    replies.append(message({"type": "text", "text": "Homing first."},
                           {"type": "tool_use", "id": "toolu_1", "name": "home", "input": {"arm": "left"}}))
    provider = AnthropicProvider(client=make(api_key="test-key"))
    history = [{"role": "user", "content": [{"type": "text", "text": "Task: wave"},
                                            {"type": "image", "camera": "top", "jpeg": b"\xff\xd8jpeg"}]},
               {"role": "assistant", "text": "", "tool_calls": [ToolCall("toolu_0", "observe", {})]},
               {"role": "user", "content": [{"type": "tool_result", "id": "toolu_0", "content": [
                   {"type": "text", "text": "arm: tool xyz"}, {"type": "image", "camera": "top", "jpeg": b"\xff\xd8"}]}]}]
    reply = provider.complete("system text", history, tools())
    assert reply == Reply("Homing first.", [ToolCall("toolu_1", "home", {"arm": "left"})])
    body = sent[0]
    assert body["model"] == agent.DEFAULT_MODEL and body["system"] == "system text"
    assert body["tools"] == tools() and body["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9hqcGVn"}}
    assert body["messages"][1] == {"role": "assistant",
                                   "content": [{"type": "tool_use", "id": "toolu_0", "name": "observe", "input": {}}]}
    result = body["messages"][2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_0"
    assert [p["type"] for p in result["content"]] == ["text", "image"]


def test_anthropic_refusal_and_missing_key(mock_api, monkeypatch, tmp_path):
    make, sent, replies = mock_api
    replies.append(message({"type": "text", "text": "I can't help with that."}, stop_reason="refusal"))
    reply = AnthropicProvider(client=make(api_key="test-key")).complete("s", [{"role": "user", "content": []}], [])
    assert reply.refused and reply.text == "I can't help with that."

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                 "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_CONFIG_DIR", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(client=make()).complete("s", [{"role": "user", "content": []}], [])
    assert len(sent) == 1


def test_the_model_queues_actions_in_the_kernel_and_gets_their_reports(served):
    server, client = served()
    h = server.harness
    code = "def home(r):\n    r.home()\ndef lower(r):\n    r.nudge([0, 0, -0.05])\nkernel.queue([home, lower])"
    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python", {"code": code})]),
                        Reply("Queued; waiting."), Reply("Waiting for the last one."), Reply("Done."))

    lines = []
    result = agent.run(client, "home then lower", provider=provider, say=lines.append)
    assert result.status == "done" and result.steps == 1 and lines[0].startswith("[1] run_python -> ok")
    assert "kernel.queue" in provider.requests[0][0] and provider.requests[0][2] == tools() + [TOOL]
    delivered = "\n".join(p["text"] for _, messages, _ in provider.requests for m in messages if m["role"] == "user"
                          for p in m["content"] if p["type"] == "text")
    assert "[action #1 home] done" in delivered and "[action #2 lower] done" in delivered
    assert h.state().arms["arm"].tool.xyz[2] == pytest.approx(0.531, abs=2e-3)


def test_a_run_that_ends_abnormally_stops_what_it_queued(served, tmp_path):
    server, client = served()
    gate = tmp_path / "never"
    code = (f"import os, time\ndef waits(r):\n    r.home()\n    while not os.path.exists({str(gate)!r}):\n"
            "        time.sleep(0.01)\nkernel.queue(waits)")

    def fails(messages):
        deadline = time.monotonic() + 60
        while client.status()["queue"]["running"] is None:      # the action runs when the provider fails
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raise RuntimeError("provider down")

    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python", {"code": code})]), fails)
    with pytest.raises(RuntimeError, match="provider down"):
        agent.run(client, "home", provider=provider, say=lambda line: None)
    status = client.status()["queue"]
    reports = client.py("", wait_s=30)["reports"]
    assert status["waiting"] == [] and [(r["name"], r["status"]) for r in reports] == [("waits", "stopped")]


def test_a_stopped_kernel_call_skips_the_rest_of_its_batch(served):
    server, client = served()
    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python", {"code": "kernel.stop()"}),
                                          ToolCall("call-2", "home", {})]), Reply("Stopped."))
    result = agent.run(client, "home", provider=provider, say=lambda line: None)
    assert result.status == "done" and result.steps == 1
    skipped = tool_results(provider.requests[-1][1])[-1]["content"][0]["text"]
    assert skipped.startswith("not run: ") and not server.harness.state().arms["arm"].enabled


def test_a_motion_that_did_not_end_done_in_a_call_skips_the_rest_of_its_batch(served):
    server, client = served()
    client.step({"action": "home"})
    server.harness.backend.block("arm")
    code = "r.move_joints({'wrist_roll': 20}, relative=True)"
    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python", {"code": code}),
                                          ToolCall("call-2", "gripper", {"opening": 1})]), Reply("Blocked."))
    result = agent.run(client, "turn", provider=provider, say=lambda line: None)
    assert result.steps == 1
    assert tool_results(provider.requests[-1][1])[-1]["content"][0]["text"].startswith("not run: ")
    assert server.harness.state().arms["arm"].gripper < 0.05


def test_cleanup_stops_only_the_runs_own_work_and_reports_a_failure(served, tmp_path, monkeypatch):
    server, client = served()
    gate = tmp_path / "never"
    others = (f"import os, time\ndef theirs(r):\n    while not os.path.exists({str(gate)!r}):\n"
              "        time.sleep(0.01)\nkernel.queue(theirs)")
    client.py(others)                                         # queued by someone else
    provider = Scripted(Reply("Nothing to do."))
    monkeypatch.setattr(agent, "_loop", lambda *a, **k: agent.RunResult("step_limit", 0, "limit"))
    result = agent.run(client, "x", provider=provider, say=lambda line: None)
    assert result.status == "step_limit" and client.status()["queue"]["running"]["name"] == "theirs"

    def failing(owner):
        raise RuntimeError("session gone")

    monkeypatch.setattr(client, "cancel", failing)
    result = agent.run(client, "x", provider=provider, say=lambda line: None)
    assert "could not be stopped" in result.message and "session gone" in result.message
    gate.touch()


def test_reply_images_reads_only_files_in_the_run_image_dir_within_the_size_limit(tmp_path, monkeypatch):
    from makermods_llm_harness import kernel
    inside, outside = tmp_path / "images", tmp_path / "secret"
    inside.mkdir()
    outside.mkdir()
    (inside / "grid-0001.jpg").write_bytes(b"\xff\xd8ok")
    (outside / "grid-0002.jpg").write_bytes(b"\xff\xd8no")
    (inside / "grid-0003.jpg").write_bytes(b"\xff\xd8" + b"x" * 100)
    import hashlib
    paths = [str(inside / "grid-0001.jpg"), str(outside / "grid-0002.jpg"), str(inside / "../secret/grid-0002.jpg")]
    digests = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    reply = {"image_dir": str(inside), "images": paths, "image_digests": digests}
    assert kernel.reply_images(reply, 3) == [b"\xff\xd8ok"]
    big = str(inside / "grid-0003.jpg")
    monkeypatch.setattr(kernel, "MAX_IMAGE_BYTES", 50)
    assert kernel.reply_images({"image_dir": str(inside), "images": [big],
                                "image_digests": {big: hashlib.sha256(Path(big).read_bytes()).hexdigest()}}, 1) == []


def test_a_motion_fault_in_python_ends_the_run_as_a_fault(served, monkeypatch):
    server, client = served()
    client.step({"action": "home"})
    original = server.harness.confirm_hold
    monkeypatch.setattr(server.harness, "confirm_hold", lambda names=None: "arm: the hold is not confirmed: test")
    server.harness.backend.block("arm")
    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python",
                                                   {"code": "r.move_joints({'wrist_roll': 20}, relative=True)"})]),
                        Reply(tool_calls=[ToolCall("call-2", "gripper", {"opening": 1})]),
                        Reply("Reported."))
    result = agent.run(client, "turn", provider=provider, say=lambda line: None)
    monkeypatch.setattr(server.harness, "confirm_hold", original)
    assert result.status == "fault" and result.steps == 1
    assert server.harness.state().arms["arm"].gripper < 0.05


def test_a_fault_seen_only_in_the_final_poll_ends_the_run_as_a_fault(served):
    server, client = served()
    client.step({"action": "home"})
    server.harness.backend.set_fault("arm", "motor 3 overcurrent")
    provider = Scripted(Reply(tool_calls=[ToolCall("py-1", "run_python", {"code": "x = 1"})]),
                        Reply("All done."), Reply("The arm reports a fault."))
    result = agent.run(client, "nothing", provider=provider, say=lambda line: None)
    server.harness.backend.set_fault("arm", None)
    assert result.status == "fault"
