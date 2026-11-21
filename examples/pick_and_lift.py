"""Pick up the cube and lift it in MuJoCo with a scripted policy: the actions an LLM would send, split into
steps the default safety envelope accepts (no joint turns more than 35 deg in one action, slow near the table).

    python examples/pick_and_lift.py --arm maker --save frames/
"""

import argparse
import json
from typing import Optional

from makermods_llm_harness import Harness, robot

# The way down to the pre-grasp pose above the cube: (x offset from the cube, tool z, pitch_deg) per action.
# Metal's home points forward, so it tilts down in six steps; Maker's home is already low and tilted.
APPROACH = {
    "metal": [(0.03, 0.54, -5), (0.02, 0.47, -25), (0.01, 0.39, -45), (0.005, 0.30, -65), (0.0, 0.21, -85),
              (0.0, 0.125, -90)],
    "maker": [(0.02, 0.125, -50), (0.0, 0.125, -60)],   # Maker cannot point straight down this close to its base
}
# Tool point height for the grasp: the fingertips reach about halfway down the 4 cm cube.
GRASP_Z = {"metal": 0.035, "maker": 0.035}
LIFT_M = {"metal": 0.10, "maker": 0.06}   # Maker at pitch -60 reaches about 7 cm higher from the grasp


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("metal", "maker"), default="metal")
    parser.add_argument("--save", metavar="DIR", help="write every camera image here as JPEG")
    args = parser.parse_args(argv)

    sim = robot.simulated("mujoco", args.arm)
    cube = sim.scene["objects"][0]
    x, y, z0 = cube["xyz"]
    pitch = APPROACH[args.arm][-1][2]
    actions = [{"action": "home"}]
    actions += [{"action": "move_tool", "xyz": [round(x + dx, 3), y, z], "pitch_deg": p} for dx, z, p in APPROACH[args.arm]]
    actions += [{"action": "gripper", "opening": "open"},
                {"action": "move_tool", "xyz": [x, y, GRASP_Z[args.arm]], "pitch_deg": pitch},
                {"action": "gripper", "opening": "close"},
                {"action": "wait", "seconds": 1.0},
                {"action": "nudge", "dxyz": [0, 0, LIFT_M[args.arm]]}]

    with Harness.open(sim, image_dir=args.save) as h:
        for action in actions:
            obs = h.step(action)
            arm = obs.arms["arm"]
            print(f"{json.dumps(action)}: {obs.result.status} {obs.result.message}".rstrip()
                  + f" | tool z {arm.tool.xyz[2]:.3f} m, gripper {arm.gripper:.2f}")
            if not obs.result.ok:
                return 1
        lifted = h.backend.object_xyz(cube["name"])[2] - z0
    print(f"{cube['name']} lifted {lifted * 100:.1f} cm")
    return 0 if lifted > 0.8 * LIFT_M[args.arm] else 1


if __name__ == "__main__":
    raise SystemExit(main())
