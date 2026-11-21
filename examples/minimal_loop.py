"""The smallest loop: open the built-in kinematic simulator, look, act, and read what came back."""

from makermods_llm_harness import Harness, robot


def main() -> int:
    with Harness.open(robot.simulated()) as h:
        print(h.observe().text())
        for action in ({"action": "home"},
                       {"action": "move_tool", "xyz": [0.31, 0.03, 0.54], "pitch_deg": -5},
                       {"action": "gripper", "opening": "open"}):
            obs = h.step(action)
            print(f"\n{action}\n{obs.text()}")
            if not obs.result.ok:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
