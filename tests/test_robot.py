import math

import pytest
import yaml

from makermods_llm_harness import robot as robots
from makermods_llm_harness.backends.sim import SimBackend
from makermods_llm_harness.robot import Robot, RobotError


def test_home_follows_the_environment(harness_home):
    assert robots.home() == harness_home
    assert robots.robot_names() == []


def test_save_load_round_trip_and_only_robot():
    robot = robots.simulated(arms=("left", "right"), name="bench")
    robot.safety = {"max_speed_deg_s": 15}
    path = robots.save(robot)
    assert path == robots.robots_dir() / "bench.yaml"
    loaded = robots.load()
    assert loaded.to_dict() == robot.to_dict()
    assert loaded.path == path and loaded.name == "bench"
    assert robots.load(path).to_dict() == robot.to_dict()


def test_load_errors_say_what_to_do():
    with pytest.raises(RobotError, match="llm-harness robot init"):
        robots.load()
    robots.save(robots.simulated(name="a"))
    robots.save(robots.simulated(name="b"))
    with pytest.raises(RobotError, match="several robots"):
        robots.load()
    with pytest.raises(RobotError, match="robots: a, b"):
        robots.load("c")


@pytest.mark.parametrize("raw, words", [
    ({"backend": "gazebo", "arms": {"a": {}}}, "backend must be"),
    ({"backend": "sim", "arm": "steel", "arms": {"a": {}}}, "arm must be"),
    ({"backend": "sim", "arms": {}}, "one or two arms"),
    ({"backend": "sim", "arms": {"a": {}, "b": {}, "c": {}}}, "one or two arms"),
    ({"backend": "sim", "arms": {"all": {}}}, "one or two arms"),
    ({"backend": "sim", "arms": {"a": {"port": "x"}}}, "only 'profile'"),
    ({"backend": "sim", "arms": {"a": {}}, "bases": {"b": {}}}, "no arm named"),
    ({"backend": "sim", "arms": {"a": {}}, "bases": {"a": {"xyz": [0, 0]}}}, "bases.a"),
    ({"backend": "real", "arms": {"a": {"backend": "carrier-pigeon"}}}, "arms.a"),
    ({"backend": "real", "arms": {"a": {"backend": "gs_usb"}}, "cameras": ["top"]}, "camera id"),
    ({"backend": "real", "arms": {"a": {"backend": "gs_usb"}}, "scene": {"table_z": 0}}, "calibrate"),
    ({"backend": "sim", "arms": {"a": {}}, "scene": {"objects": [{"name": "c"}]}}, "scene.objects"),
    ({"backend": "sim", "arms": {"a": {}}, "safety": {"max_speed": 3}}, "unknown settings"),
    ({"backend": "sim", "arms": {"a": {}}, "safety": {"max_speed_deg_s": -1}}, "max_speed_deg_s"),
    ({"backend": "sim", "arms": {"a": {}}, "colour": "red"}, "unknown keys"),
])
def test_validation_rejects(raw, words):
    with pytest.raises(RobotError, match=words):
        Robot.from_dict("r", raw)


def test_real_robot_accepts_rig_entries():
    robot = Robot.from_dict("r", {"backend": "real", "arm": "maker", "arms": {"left": {"backend": "gs_usb",
                                  "port": "gs_usb:abc"}}, "cameras": {"top": "usb:1-2"}})
    assert robot.arms["left"]["port"] == "gs_usb:abc"


def test_from_rig_reads_a_rig_file_without_devices(tmp_path):
    rig = tmp_path / "rig.yaml"
    rig.write_text(yaml.safe_dump({
        "arms": {"left": {"backend": "gs_usb", "port": "gs_usb:A", "profile": "mine.yaml"},
                 "right": {"backend": "gs_usb", "port": "gs_usb:B"}},
        "cameras": {"top": "usb:1", "left_wrist": "usb:2"}}))
    robot = robots.from_rig("bench", "metal", rig)
    assert robot.backend == "real" and list(robot.arms) == ["left", "right"]
    assert robot.arms["left"]["profile"] == str(tmp_path / "mine.yaml")
    assert robot.cameras == {"top": "usb:1", "left_wrist": "usb:2"}
    only = robots.from_rig("bench", "metal", rig, arms=["right"], cameras=["top"])
    assert list(only.arms) == ["right"] and only.cameras == {"top": "usb:1"}
    with pytest.raises(RobotError, match="no arm named middle"):
        robots.from_rig("bench", "metal", rig, arms=["middle"])
    with pytest.raises(RobotError, match="no camera named front"):
        robots.from_rig("bench", "metal", rig, cameras=["front"])


def test_simulated_robot_places_two_arms_and_cubes():
    robot = robots.simulated(arm="maker", arms=("left", "right"))
    infos = robot.arm_infos()
    assert [(i.name, i.type, i.base_xyz) for i in infos] == [("left", "maker", (0.0, 0.25, 0.0)),
                                                              ("right", "maker", (0.0, -0.25, 0.0))]
    assert [o["xyz"][1] for o in robot.scene["objects"]] == [0.25, -0.25]


def test_bases_yaw_is_given_in_degrees():
    robot = Robot.from_dict("r", {"backend": "sim", "arms": {"a": {}},
                                  "bases": {"a": {"xyz": [1, 2, 0], "yaw_deg": 90}}})
    info = robot.arm_infos()[0]
    assert info.base_xyz == (1.0, 2.0, 0.0) and math.isclose(info.base_yaw, math.pi / 2)


def test_open_backend_builds_the_simulator_with_its_scene():
    robot = robots.simulated()
    robot.cameras = {"top": ""}
    robot.scene["table_z"] = 0.05
    backend = robot.open_backend()
    assert isinstance(backend, SimBackend)
    assert backend.cameras == ["top"] and backend.table_z == 0.05
    assert [o["name"] for o in backend.objects] == ["red_cube"]


def test_sdk_imports_the_arm_package():
    assert robots.sdk("metal").__name__ == "metal_arm" and robots.sdk("maker").kinematics
    with pytest.raises(RobotError):
        robots.sdk("wood")


@pytest.mark.parametrize("arms", [
    {"left": {"backend": "gs_usb", "port": "gs_usb:A"}, "right": {"backend": "gs_usb", "port": "gs_usb:A"}},
    {"left": {"backend": "gs_usb"}, "right": {"backend": "gs_usb"}},
    {"left": {"backend": "socketcan"}, "right": {"backend": "socketcan", "channel": "can0"}},
])
def test_two_arm_names_never_reach_the_same_motors(arms):
    with pytest.raises(RobotError, match="same bus"):
        Robot.from_dict("bench", {"backend": "real", "arm": "metal", "arms": arms, "cameras": {}})


def test_rest_q_is_six_degrees_per_known_arm():
    raw = {"backend": "real", "arm": "metal", "arms": {"left": {"backend": "gs_usb", "port": "gs_usb:A"}},
           "cameras": {}, "rest_q": {"left": [0, -170, 170, 0, 0, 0]}}
    robot = Robot.from_dict("bench", raw)
    assert robot.to_dict()["rest_q"] == {"left": [0, -170, 170, 0, 0, 0]}
    for bad in ({"right": [0] * 6}, {"left": [0] * 5}, {"left": [0, 0, 0, 0, 0, float("nan")]}):
        with pytest.raises(RobotError, match="rest_q"):
            Robot.from_dict("bench", {**raw, "rest_q": bad})
