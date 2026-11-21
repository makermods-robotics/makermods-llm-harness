import json
import math

import pytest

from makermods_llm_harness.actions import (ACTIONS, SCHEMA, ActionError, ArmObservation, CameraImage, Gripper, Home,
                                           MoveJoints, MoveTool, Nudge, Observation, Observe, Rest, Result, ToolPose,
                                           Wait, action_to_dict, parse_action, tools)


@pytest.mark.parametrize("raw, expected", [
    ({"action": "observe"}, Observe()),
    ({"action": "move_tool", "xyz": [0.3, 0, 0.1]}, MoveTool((0.3, 0.0, 0.1))),
    ({"action": "move_tool", "xyz": [0.3, 0, 0.1], "pitch_deg": -90, "yaw_deg": 10, "roll_deg": 5, "arm": "left"},
     MoveTool((0.3, 0.0, 0.1), -90.0, 10.0, 5.0, "left")),
    ({"action": "nudge", "dxyz": [0, 0, 0.02]}, Nudge((0.0, 0.0, 0.02))),
    ({"action": "nudge", "dxyz": [0, 0, 0.02], "frame": "tool"}, Nudge((0.0, 0.0, 0.02), "tool")),
    ({"action": "gripper", "opening": "open"}, Gripper(1.0)),
    ({"action": "gripper", "opening": "close"}, Gripper(0.0)),
    ({"action": "gripper", "opening": 0.4, "arm": "right"}, Gripper(0.4, "right")),
    ({"action": "move_joints", "joints_deg": {"elbow_flex": 30}}, MoveJoints({"elbow_flex": 30.0})),
    ({"action": "move_joints", "joints_deg": {"elbow_flex": 5}, "relative": True},
     MoveJoints({"elbow_flex": 5.0}, True)),
    ({"action": "home"}, Home()),
    ({"action": "rest"}, Rest()),
    ({"action": "wait", "seconds": 2}, Wait(2.0)),
    ({"action": "wait"}, Wait(1.0)),
])
def test_parse_accepts(raw, expected):
    assert parse_action(raw) == expected
    assert parse_action(json.dumps(raw)) == expected
    assert parse_action(action_to_dict(expected)) == expected


@pytest.mark.parametrize("raw, words", [
    ({"action": "fly"}, "unknown action"),
    ({"xyz": [0, 0, 0]}, "unknown action"),
    ({"action": "move_tool"}, "missing xyz"),
    ({"action": "move_tool", "xyz": [0, 0]}, "three numbers"),
    ({"action": "move_tool", "xyz": [0, 0, float("nan")]}, "finite"),
    ({"action": "move_tool", "xyz": [0, 0, True]}, "finite"),
    ({"action": "move_tool", "xyz": [0, 0, 0.1], "pitch_deg": -100}, "pitch_deg"),
    ({"action": "move_tool", "xyz": [0, 0, 0.1], "speed": 3}, "unknown fields"),
    ({"action": "nudge", "dxyz": [0, 0, 1], "frame": "world"}, "frame"),
    ({"action": "gripper", "opening": 1.5}, "between 0"),
    ({"action": "gripper", "opening": "wide"}, "between 0"),
    ({"action": "move_joints", "joints_deg": {}}, "joints_deg"),
    ({"action": "move_joints", "joints_deg": {"elbow_flex": 1}, "relative": "yes"}, "relative"),
    ({"action": "wait", "seconds": 100}, "seconds"),
    ({"action": "home", "arm": 3}, "arm must be a name"),
    ([1, 2], "JSON object"),
])
def test_parse_rejects_readably(raw, words):
    with pytest.raises(ActionError, match=words):
        parse_action(raw)


def test_parse_rejects_bad_json():
    with pytest.raises(ActionError, match="not valid JSON"):
        parse_action("{action: home")


def test_tools_and_schema_cover_every_action():
    names = [t["name"] for t in tools()]
    assert names == list(ACTIONS)
    assert all(t["description"] and t["input_schema"]["type"] == "object" for t in tools())
    assert len(SCHEMA["oneOf"]) == len(ACTIONS)


def _observation() -> Observation:
    arm = ArmObservation({"shoulder_pan": 1.5}, 0.5, ToolPose((0.3, 0.0, 0.1), -90.0, 0.0, 0.0), None,
                         enabled=False, fault="overheated", temperature_c=71.0)
    cams = {"top": CameraImage(1.0, 320, 240, "/x/top.jpg", b"\xff\xd8jpeg"), "side": CameraImage(error="no image")}
    return Observation(3, 12.5, {"arm": arm}, cams, Result("stopped", "blocked", {"action": "home"}))


def test_observation_round_trips_with_and_without_images():
    obs = _observation()
    back = Observation.from_dict(json.loads(obs.to_json(images=True)))
    assert back == obs
    plain = obs.to_dict()
    assert "jpeg_base64" not in plain["cameras"]["top"] and plain["cameras"]["top"]["path"] == "/x/top.jpg"
    assert Observation.from_dict(plain).cameras["top"].jpeg is None


def test_observation_text_names_result_state_and_cameras():
    text = _observation().text()
    assert text.startswith("stopped: blocked")
    assert "motors off" in text and "clearance above table unknown" in text and "FAULT: overheated" in text
    assert "camera side: no image" in text
    assert math.isclose(_observation().arms["arm"].tool.pitch_deg, -90.0)


@pytest.mark.parametrize("raw, words", [
    (Gripper(2.0), "opening must be between 0"),
    (MoveTool((0.3, 0, float("nan"))), "finite number"),
    ({"action": "nudge", "dxyz": [0, 0, 0.01], "frame": False}, "frame must be one of"),
    ({"action": ["nudge"], "dxyz": [0, 0, 0.01]}, "unknown action"),
    ({"action": "move_tool", "xyz": [10 ** 400, 0, 0]}, "finite number"),
    ({"action": "move_tool", "xyz": [0.3, 0, 0.1], "arm": ""}, "arm must be a name"),
    ({"action": "move_joints", "joints_deg": {"wrist_roll": 10 ** 5000}}, "too large to print"),
    ({"action": "move_joints", "joints_deg": {"wrist_roll": True}}, "finite number"),
])
def test_every_input_goes_through_the_same_checks(raw, words):
    with pytest.raises(ActionError, match=words):
        parse_action(raw)


def test_omitted_or_null_fields_take_their_defaults():
    assert parse_action({"action": "nudge", "dxyz": [0, 0, 0.01], "frame": None}).frame == "base"
    assert parse_action(Gripper(0.5)) == Gripper(0.5)
