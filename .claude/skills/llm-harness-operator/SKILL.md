---
name: llm-harness-operator
description: Operate a Metal or Maker robot arm (real or simulated) through the `llm-harness` CLI, one action at a time, looking at the camera images between actions. Use when asked to move, pick, place, look at or rest a robot arm through llm-harness, or to take over a running llm-harness session.
---

# Operating a robot through llm-harness

You are the policy. A session (`llm-harness serve` or `llm-harness sim`) holds the robot open; each
`llm-harness op ...` sends one action and prints the result, the arm state and one JPEG path per camera.
Details: [docs/OPERATING.md](../../../docs/OPERATING.md).

## Start

```bash
llm-harness op status        # is a session running? which robot, arms, cameras
llm-harness op describe      # arms, joint ranges, home pose, table height, envelope numbers
llm-harness op observe       # then Read every JPEG path it prints
```

If no session is running, ask the operator to start one. Never start `serve` on a real robot yourself, and never run
`calibrate` (table height and rest pose): both are the operator's. To take over a task in progress, read `llm-harness log resume` first.

## The loop

Observe, decide one action, send it, read the result line and the numbers, Read the fresh images, repeat. Say what you
expect to see before each motion and check it after.

| Op | Use |
|---|---|
| `op move_tool X Y Z [--pitch D] [--yaw D] [--roll D]` | tool point to an absolute world position, m; pitch -90 points straight down |
| `op nudge DX DY DZ [--frame base\|tool]` | small relative move, keeps the orientation; added to the last commanded pose while the arm holds near it, so small nudges add up |
| `op gripper open\|close\|0..1` | the reply's `gripper` is the opening reached |
| `op move_joints NAME=DEG ... [--relative]` | joints by name, e.g. `wrist_roll=30 --relative` |
| `op home` / `op wait S` / `op observe` | ready pose / let time pass / look |
| `llm-harness rest` | every arm back to rest, motors off: how every task ends |

With two arms add `--arm left` or `--arm right` to every motion. `llm-harness ops "A" "B"` runs several ops and
stops at the first one not done; never chain `op` calls with `;`.

## Results

| Status | Do |
|---|---|
| `done` | look, decide the next action. A tool move is done only within 10 mm and 3 deg of its target. |
| `rejected` | nothing moved. Read the message, change the request; the same request gets the same answer. |
| `stopped` | the arm holds where it is (stop, blocked, clearance, keep-out zone or heat, did not arrive, ended off its target, rest not released). Look at every camera before moving. |
| `fault` | stop sending actions. Tell the operator what you did and saw. |

## Rules

- One action per call, and look between motions. Plan in the reported `tool xyz` and `clearance`, not in pixels.
- No joint may move more than 35 deg in one action: split big moves, and change the pitch by 20-30 deg at a time.
  A Metal arm needs several `move_tool` steps from `home` to point straight down. The Maker arm points straight down
  only a few centimetres above the table: near the table use pitch -45 to -60.
- By default the hard floor lets the elbow and wrist joints, the tool point, the fingertips and the tips of the open
  jaws come down to the table surface, not below it: the arm may touch the table. The lowest tool z is about 0.025 m
  pointing straight down on Metal and about 0.03 m on Maker (pitch -45 to -60), so small objects (a 4 cm cube) can be
  grasped. A robot file may set a stricter floor under `safety:`; `op describe` states the one in force. Within 76 mm
  of the table motions run slowly: descend in 2-5 cm steps and look each time.
- A joint more than 6 deg off its path, or a measured point more than 10 mm under the table, stops the motion. The motors turn on only at the rest pose: if the
  first motion is refused for that, ask the operator.
- Open the gripper fully before descending onto an object. After closing, `gripper` well above 0 means it holds
  something; near 0 means it missed. Lift straight up before moving sideways.
- Pointing straight down, turn the gripper with `--yaw` (about 30 deg per action); there the observation reads roll 0
  and yaw carries the whole turn. A camera line marked `stale` may show the arm before the action ended: observe
  again.
- Keep-out zones (`op describe` lists them with their corners; each observation shows the nearest one and its
  distance) are boxes, such as a camera mount, that no part of the arm may come within 30 mm of. A move into one is
  rejected: go around, moving away from the zone first. An arm inside one may only move out.
- Nothing else checks the arms against each other or against objects. Keep the two arms apart.
- `llm-harness op stop` ends the action in progress and any waiting behind it. It is not an emergency stop: the
  operator has the hardware one.
- If the images and the numbers disagree, or anything is unexpected, stop and ask the operator.

## Finish

```bash
llm-harness rest
```

Then report what was done, what you saw, and anything the operator should check (an arm `rest` did not release is
still holding). Leave `op quit` to the operator unless asked.
