"""The example scripts run as documented."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimal_loop(capsys):
    assert load("minimal_loop").main() == 0
    assert "gripper 1.00" in capsys.readouterr().out


def test_pick_and_lift_on_metal(capsys, tmp_path):
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("needs mujoco")
    assert load("pick_and_lift").main(["--arm", "metal", "--save", str(tmp_path)]) == 0
    assert "red_cube lifted" in capsys.readouterr().out
    assert (tmp_path / "step-0001-top.jpg").is_file()
