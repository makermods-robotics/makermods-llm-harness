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
