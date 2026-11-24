import json

import pytest

from makermods_llm_harness.log import EpisodeBusy, EpisodeLog, latest_run, new_run_dir, read_episode, summary


def _observation(step, status="done", message=""):
    return {"step": step, "time": 1.0, "result": {"status": status, "message": message, "action": None},
            "arms": {"arm": {"joints_deg": {"shoulder_pan": 1.0}, "gripper": 0.5,
                             "tool": {"xyz": [0.3, 0.0, 0.1], "pitch_deg": -90, "yaw_deg": 0, "roll_deg": 0},
                             "clearance_m": 0.08, "enabled": True, "fault": None, "temperature_c": 30.0}},
            "cameras": {"top": {"stamp": 1.0, "width": 320, "height": 240,
                                "path": f"/run/images/step-{step:04d}-top.jpg"}}}


def test_new_run_dirs_are_unique_and_latest_is_found(harness_home):
    a, b = new_run_dir("bench"), new_run_dir("bench")
    assert a != b and a.parent == harness_home / "runs" and a.name.endswith("bench")
    assert latest_run() is None
    EpisodeLog(a).close()
    assert latest_run() == a


def test_events_are_written_one_per_line(tmp_path):
    log = EpisodeLog(tmp_path / "run")
    assert log.image_dir.is_dir()
    log.start("bench", ["arm"], ["top"], {"arm": 0.0}, {"max_speed_deg_s": 20})
    log.note("pick up the red cube")
    log.step({"action": "home"}, _observation(1))
    log.close()
    log.close()
    events = read_episode(tmp_path / "run")
    assert [e["event"] for e in events] == ["start", "note", "step", "end"]
    assert events[2]["step"] == 1 and events[2]["action"] == {"action": "home"}
    with pytest.raises(ValueError, match="closed"):
        log.note("late")


def test_resume_continues_the_step_count(tmp_path):
    log = EpisodeLog(tmp_path / "run")
    log.step({"action": "home"}, _observation(1))
    log.step({"action": "observe"}, _observation(2))
    log.close()
    resumed = EpisodeLog.resume(tmp_path / "run")
    assert resumed.steps == 2
    resumed.step({"action": "rest"}, _observation(3))
    resumed.close()
    assert [e["step"] for e in read_episode(tmp_path / "run") if e["event"] == "step"] == [1, 2, 3]
    with pytest.raises(FileNotFoundError):
        EpisodeLog.resume(tmp_path / "nothing")


def test_read_episode_skips_a_torn_last_line(tmp_path):
    log = EpisodeLog(tmp_path)
    log.note("x")
    log.close()
    with open(tmp_path / "episode.jsonl", "a") as f:
        f.write('{"event": "st')
    assert [e["event"] for e in read_episode(tmp_path / "episode.jsonl")] == ["note", "end"]


def test_summary_is_short_and_has_what_a_fresh_agent_needs(tmp_path):
    log = EpisodeLog(tmp_path)
    log.start("bench", ["arm"], ["top"], {"arm": 0.0}, {})
    log.note("pick up the red cube")
    for i in range(1, 9):
        log.step({"action": "nudge", "dxyz": [0, 0, 0.01 * i]}, _observation(i))
    log.step({"action": "move_tool", "xyz": [9, 9, 9]}, _observation(9, "rejected", "that pose is out of reach"))
    text = summary(tmp_path, last=3)
    lines = text.splitlines()
    assert "robot bench" in lines[0] and "9 steps" in lines[0]
    assert "Note: pick up the red cube" in text
    assert "rejected (that pose is out of reach)" in text
    assert '"dxyz": [0, 0, 0.07]' in text and '"dxyz": [0, 0, 0.06]' not in text
    assert "step-0009-top.jpg" in text and "clearance above table 0.080 m" in text
    assert len(lines) < 15


def test_a_torn_last_line_is_cut_before_appending(tmp_path):
    log = EpisodeLog(tmp_path / "run")
    log.step({"action": "home"}, {"result": {"status": "done"}})
    log.close()
    with (tmp_path / "run" / "episode.jsonl").open("a") as f:
        f.write('{"event": "step", "step": 2, "act')            # a crash mid-write
    resumed = EpisodeLog.resume(tmp_path / "run")
    assert resumed.steps == 1
    resumed.note("go on")
    resumed.close()
    events = read_episode(tmp_path / "run")
    assert [e["event"] for e in events] == ["step", "end", "note", "end"]


def test_one_writer_at_a_time_and_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = EpisodeLog("run")
    assert log.run_dir.is_absolute() and log.image_dir == tmp_path / "run" / "images"
    with pytest.raises(EpisodeBusy, match="another session"):
        EpisodeLog.resume("run")
    log.close()
    EpisodeLog.resume("run").close()


def test_profile_splits_an_episode_into_steps_and_idle_gaps(tmp_path, capsys):
    from makermods_llm_harness import cli
    from makermods_llm_harness.log import profile

    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"event": "start", "time": 100.0, "robot": "bench", "arms": ["arm"]},
        {"event": "action", "time": 102.0, "step": 1, "action": {"action": "home"}},
        {"event": "step", "time": 106.0, "step": 1, "action": {"action": "home"},
         "observation": {"result": {"status": "done"}}},
        {"event": "py", "time": 107.5, "seconds": 0.5, "chars": 40, "error": False},
        {"event": "action", "time": 110.0, "step": 2, "action": {"action": "move_tool"}},
        {"event": "step", "time": 110.1, "step": 2, "action": {"action": "move_tool"},
         "observation": {"result": {"status": "rejected"}}},
        {"event": "step", "time": 111.0, "step": 3, "action": None, "observation": {"result": {"status": "rejected"}}},
        {"event": "action", "time": 111.0, "step": 4, "action": {"action": "nudge"}},
        {"event": "step", "time": 114.0, "step": 4, "action": {"action": "nudge"},
         "observation": {"result": {"status": "done"}}},
        {"event": "kernel_action", "time": 114.1, "id": 1, "name": "grasp", "status": "done", "seconds": 3.0},
        {"event": "end", "time": 120.0},
    ]
    (run / "episode.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    p = profile(run)
    assert p["wall_s"] == 20.0 and p["in_steps_s"] == pytest.approx(7.1)
    assert p["motion_s"] == pytest.approx(7.0) and p["refused_s"] == pytest.approx(0.1)
    assert p["between_s"] == pytest.approx(2.0 + 4.0 + 0.9 + 6.0) and p["after_last_s"] == pytest.approx(6.0)
    assert p["in_steps_s"] + p["between_s"] + p["between_sessions_s"] + p["shutdown_s"] == pytest.approx(p["wall_s"])
    assert [(s["step"], s["seconds"], s["gap_s"]) for s in p["steps"]] == [
        (1, 4.0, 2.0), (2, 0.1, 4.0), (3, 0.0, 0.9), (4, 3.0, 0.0)]
    assert p["statuses"] == {"done": 2, "rejected": 2} and p["by_action"]["home"] == {"count": 1, "seconds": 4.0}
    assert p["py_calls"] == 1 and p["kernel_actions"] == [{"id": 1, "name": "grasp", "status": "done", "seconds": 3.0}]
    assert cli.main(["log", "profile", str(run)]) == 0
    out = capsys.readouterr().out
    assert "4 steps over 20.0 s" in out and "longest gaps: 4.0 s before step 2" in out and "#1 grasp" in out
    assert cli.main(["log", "profile", str(run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["wall_s"] == 20.0


def test_profile_counts_the_time_between_resumed_sessions_apart(tmp_path):
    from makermods_llm_harness.log import profile

    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"event": "start", "time": 0.0},
        {"event": "action", "time": 1.0, "step": 1, "action": {"action": "home"}},
        {"event": "step", "time": 3.0, "step": 1, "action": {"action": "home"},
         "observation": {"result": {"status": "done"}}},
        {"event": "end", "time": 4.0},
        {"event": "start", "time": 1004.0},
        {"event": "action", "time": 1005.0, "step": 2, "action": {"action": "nudge"}},
        {"event": "step", "time": 1006.0, "step": 2, "action": {"action": "nudge"},
         "observation": {"result": {"status": "done"}}},
    ]
    (run / "episode.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    p = profile(run)
    assert p["sessions"] == 2 and p["between_sessions_s"] == pytest.approx(1000.0)
    assert p["in_steps_s"] == pytest.approx(3.0) and p["between_s"] == pytest.approx(1 + 1 + 1)
    assert p["in_steps_s"] + p["between_s"] + p["between_sessions_s"] + p["shutdown_s"] == pytest.approx(p["wall_s"])


def test_profile_counts_the_shutdown_rest_apart_from_idle_time(tmp_path):
    from makermods_llm_harness.log import profile

    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"event": "start", "time": 0.0},
        {"event": "action", "time": 1.0, "step": 1, "action": {"action": "home"}},
        {"event": "step", "time": 2.0, "step": 1, "action": {"action": "home"},
         "observation": {"result": {"status": "done"}}},
        {"event": "closing", "time": 5.0},
        {"event": "end", "time": 9.0},
    ]
    (run / "episode.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    p = profile(run)
    assert p["shutdown_s"] == pytest.approx(4.0) and p["after_last_s"] == pytest.approx(3.0)
    assert p["in_steps_s"] + p["between_s"] + p["between_sessions_s"] + p["shutdown_s"] == pytest.approx(p["wall_s"])
