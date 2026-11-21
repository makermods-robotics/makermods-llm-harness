"""Shared fixtures: every test gets its own empty harness home; ``sim_harness`` builds harnesses on the
kinematic simulator and closes them afterwards."""

from __future__ import annotations

import pytest

from makermods_llm_harness import Harness
from makermods_llm_harness import robot as robots
from makermods_llm_harness.backends.sim import SimBackend


@pytest.fixture(autouse=True)
def harness_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("LLM_HARNESS_HOME", str(home))
    return home


def sim_backend(arm="metal", arms=("arm",), cameras=("top", "side"), bases=None, **kwargs) -> SimBackend:
    robot = robots.simulated(arm=arm, arms=arms)
    robot.bases.update(bases or {})
    return SimBackend(robot.arm_infos(), cameras=cameras, table_z=robot.scene["table_z"],
                      objects=robot.scene["objects"], **kwargs)


@pytest.fixture
def sim_harness():
    """``make(arm="metal", arms=("arm",), cameras=..., bases=..., backend=None, **Harness.open kwargs)``."""
    opened = []

    def make(arm="metal", arms=("arm",), cameras=("top", "side"), bases=None, backend=None, **kwargs) -> Harness:
        backend = backend or sim_backend(arm, arms, cameras, bases)
        harness = Harness.open(backend=backend, **kwargs)
        opened.append(harness)
        return harness

    yield make
    for harness in opened:
        harness.close()


# Stepped top-down approaches to the 4 cm cube at x = 0.28 (in front of an arm based at the origin): each action
# stays under the 35 deg per-action cap. (xyz, pitch_deg)
APPROACH = {
    "metal": [((0.31, 0, 0.54), -5), ((0.30, 0, 0.47), -25), ((0.29, 0, 0.39), -45), ((0.285, 0, 0.30), -65),
              ((0.28, 0, 0.21), -85), ((0.28, 0, 0.12), -90)],
    "maker": [((0.28, 0, 0.11), -45)],
}
GRASP_PITCH = {"metal": -90, "maker": -45}
# Tool point for the grasp: the fingertips reach about halfway down the cube (Maker's 34 mm fingers, tilted 45 deg,
# reach 24 mm forward of its tool point).
GRASP_X = {"metal": 0.28, "maker": 0.26}
GRASP_Z = {"metal": 0.035, "maker": 0.035}
