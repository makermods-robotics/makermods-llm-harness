# makermods-llm-harness

Let an LLM drive one or two MakerMods arms (Metal or Maker), on real hardware or in a simulator. The model asks for
one action at a time, usually a tool target; the harness checks the motion against a safety envelope above the table,
solves the inverse kinematics, moves the arm and answers with camera images and the arm's state.

Built on the arm SDKs, [`metal-arm`](https://github.com/makermods-robotics/metal-arm-sdk) and
[`maker-arm`](https://github.com/makermods-robotics/maker-arm-sdk), which have the same API.

## Install

Python 3.10 or newer. The two SDKs are installed from GitHub with the package.

```bash
pip install "makermods-llm-harness @ git+https://github.com/makermods-robotics/makermods-llm-harness"
# or: uv pip install "makermods-llm-harness @ git+https://github.com/makermods-robotics/makermods-llm-harness"
```

| Extra | Adds | Needed for |
|---|---|---|
| `mujoco` | MuJoCo | `llm-harness sim --mujoco`, `examples/pick_and_lift.py` |
| `mcp` | the MCP SDK | `llm-harness mcp` |
| `agent` | the Anthropic SDK | `llm-harness run` |

```bash
pip install "makermods-llm-harness[mujoco,mcp,agent] @ git+https://github.com/makermods-robotics/makermods-llm-harness"
```

Real arms need the SDKs' USB setup (libusb, and on Linux a udev rule for gs_usb adapters): see the SDK README.

## Quick start in the simulator

Terminal 1 holds a simulated Metal arm open as a session:

```bash
llm-harness sim                  # --arm maker, --arms left right, --mujoco
```

Terminal 2 sends one action at a time and prints what came back:

```bash
llm-harness op observe
llm-harness op home
llm-harness op move_tool 0.31 0.03 0.54 --pitch -5
llm-harness op gripper open
llm-harness op move_tool 0.28 0 0.12 --pitch -90
llm-harness op nudge 0 0 -0.3
llm-harness rest
llm-harness op quit
```

Each reply is the result (when it is not plain "done"), the state of each arm and one JPEG path per camera:

```
rejected: arm: wrist_flex would move 70 deg in one action, over the 35 deg per-action limit. Split the motion into smaller steps and look at the cameras in between.
arm: tool xyz [0.280, -0.000, 0.120] m, pitch -90, yaw -0, roll 0 deg; gripper 1.00; clearance above table 0.095 m
  joints (deg): shoulder_pan 0.1, shoulder_lift -100.3, elbow_flex 74.0, wrist_flex -62.7, wrist_yaw 0.0, wrist_roll 0.1
camera top: /home/you/.makermods-llm-harness/runs/20261120-182307-sim/images/step-0006-top.jpg
camera side: /home/you/.makermods-llm-harness/runs/20261120-182307-sim/images/step-0006-side.jpg
```

Open the JPEGs: the kinematic simulator draws a top view and a side view of the table, the 4 cm cube and the arm. Every
step is also in the run's `episode.jsonl` (`llm-harness log show`).

Then the scripts:

```bash
python examples/minimal_loop.py                      # Harness.open on the kinematic simulator, three actions
llm-harness sim --mujoco                             # physics and rendered cameras: top, arm_wrist
python examples/pick_and_lift.py                     # MuJoCo: approach, grasp and lift the cube (Metal)
python examples/pick_and_lift.py --arm maker --save frames/
```

## Quick start on a real robot

1. Set up the arms and cameras with the SDK first, as its README describes (`metal-arm rig init --detect`,
   `metal-arm identify --assign --top camera1`, `metal-arm scan`, `metal-arm hold`; `maker-arm ...` for Maker). The
   rig file it writes names the arms (`left`, `right`) and the cameras.
2. Turn the rig into a robot, look at it without moving anything, and calibrate it with the motors off:

```bash
llm-harness robot init bench --arm metal --from-rig --arms left --cameras top left_wrist
llm-harness robot check bench        # read-only: joints, torque, temperatures, one image per camera, calibration
llm-harness calibrate --robot bench  # per arm: touch the table with the closed fingertips, Enter; then fold the
                                     # arm onto its rest support and answer y: its rest pose is recorded
```

3. Serve it and make the first moves with a hand on the emergency stop:

```bash
llm-harness serve --robot bench      # terminal 1; Ctrl-C or `op quit` rests the arms and ends it
llm-harness op home                  # terminal 2
llm-harness op nudge 0 0 -0.05
llm-harness rest
llm-harness op quit
```

Robot files live in `~/.makermods-llm-harness/robots/<name>.yaml` (`$LLM_HARNESS_HOME` moves the whole home: robots,
calibration, runs and the session socket). The operator's side is in [docs/OPERATING.md](docs/OPERATING.md#operator).

## Connecting an LLM

| Way | Command | Who decides |
|---|---|---|
| A coding agent runs the CLI | `llm-harness op ...` against `llm-harness serve` | the agent (Claude Code, Codex, ...); it reads the JPEG paths |
| MCP tools | `llm-harness mcp --connect` | the MCP client's model; images come back inline |
| Built-in model loop | `llm-harness run "pick up the red cube"` | a model over the API |

**Coding agent.** Point it at [docs/OPERATING.md](docs/OPERATING.md). Claude Code picks up the skill in
[.claude/skills/llm-harness-operator](.claude/skills/llm-harness-operator/SKILL.md) when it works in this repository.

**MCP.** `llm-harness mcp` serves one tool per action on stdio, with the robot description as its instructions.
`--connect` drives a session you started with `serve` or `sim` (so `op stop` and Ctrl-C stay with you); `--robot NAME`,
`--sim` or `--mujoco` open the robot inside the MCP server instead. A tool call the client cancels, or that is still
running when the client disconnects, ends its motion and the arm holds. Registration (not yet tried against these
clients; use the full path to `llm-harness` if the client does not see your virtual environment):

```bash
claude mcp add llm-harness -- llm-harness mcp --connect          # Claude Code
codex mcp add llm-harness -- llm-harness mcp --connect           # Codex
```

```json
{"mcpServers": {"llm-harness": {"command": "llm-harness", "args": ["mcp", "--connect"]}}}
```

The JSON goes in Claude Desktop's `claude_desktop_config.json`.

**Model loop.** Needs the `agent` extra and `ANTHROPIC_API_KEY`. The default model is `claude-opus-5-5`
(`agent.DEFAULT_MODEL`); `--model` picks another. It stops when the model answers without a tool call, after
`--max-steps` (40) actions, after `--timeout` (600 s: a model reply or an action still running then is not waited
for, and the action ends with the arm holding), on Ctrl-C or SIGTERM, or on a fault. After an action that stops
early, the rest of that reply's tool calls are not run. Exit status 0 only when the model finished with a plain
answer and every arm is confirmed off at the end.

```bash
llm-harness run "pick up the red cube and lift it 10 cm" --sim
llm-harness run "put the marker in the cup" --connect --model claude-opus-5-5 --max-steps 60
```

## Actions and observations

Metres, degrees, seconds; positions in the world frame (z up). `arm` may be left out when the robot has one arm. The
CLI op, the JSON `"action"` and the MCP tool have the same name.

| Action | Fields (default) | What it does |
|---|---|---|
| `observe` | | Fresh images and arm state. Moves nothing. |
| `move_tool` | `xyz` [x, y, z]; `pitch_deg` (keep), `yaw_deg` (heading from the arm base to xyz), `roll_deg` (keep), `arm` | Straight line of the tool point to xyz. Pitch -90 points straight down, 0 horizontal. |
| `nudge` | `dxyz` [dx, dy, dz]; `frame` `base` or `tool` (`base`), `arm` | Move the tool point by an offset, keeping its orientation. The offset is added to the last commanded tool pose while the arm holds still within 10 mm and 3 deg of it (so small nudges add up despite a standing error), else to the measured pose. |
| `gripper` | `opening` 0 closed .. 1 open, or `"open"` / `"close"`; `arm` | Reports the opening reached; staying above the request means the jaws hold something. |
| `move_joints` | `joints_deg` {name: deg}; `relative` (false), `arm` | Named joints to angles; the others stay. |
| `home` | `arm` | The ready pose: gripper forward, above the table. |
| `rest` | `arm` (every arm) | Back to the rest pose along a checked path (by way of `home` when the straight path is refused), then motors off. |
| `wait` | `seconds` 0..30 (1) | Let time pass, then observe. |

```bash
llm-harness op '{"action": "move_tool", "xyz": [0.31, 0.03, 0.54], "pitch_deg": -5}'
```

| Observation field | Meaning |
|---|---|
| `step`, `time` | actions so far in this run; Unix time |
| `result.status`, `result.message`, `result.action` | how the action ended, why, and the action as parsed |
| `result.error` | after a tool move: `position_m` and `angle_deg` between the measured and the requested tool pose |
| `arms.<arm>.tool` | `xyz` (m) and `pitch_deg`, `yaw_deg`, `roll_deg`: the same angles `move_tool` takes |
| `arms.<arm>.clearance_m` | lowest monitored point (elbow, wrist, tool point, fingertips, open-jaw tips) above the table; null without a table height |
| `arms.<arm>.nearest_keep_out` | `name` and `distance_m` of the keep-out zone closest to any point along the arm (negative inside); null without zones |
| `arms.<arm>.gripper` | measured opening, 0 closed .. 1 open |
| `arms.<arm>.joints_deg`, `enabled`, `fault`, `temperature_c` | joints (no gripper), motors on (null when not known), fault text or null, hottest motor |
| `cameras.<cam>.path`, `width`, `height`, `stamp`, `error` | the JPEG on disk (an absolute path), its size, when it was taken, or why there is none |
| `cameras.<cam>.age_s`, `stale` | how old the frame was; `stale` true when no frame taken after the action ended came within 0.5 s |

| Status | Meaning | Next |
|---|---|---|
| `done` | the action finished (a tool move: within 10 mm and 3 deg of the target) | look, decide the next one |
| `rejected` | refused before anything moved | read the message, change the request |
| `stopped` | ended early (`op stop`, the arm was blocked, rest not confirmed), ended further from a tool target than that, or never started (`op stop` while it waited); the arm holds where it is | look before going on |
| `fault` | the arm reports a fault | stop; the operator has to look at the arm |

`llm-harness op` exits 0 on `done`, 1 on any other status, 2 on usage errors or when no session is running.
`llm-harness op quit` waits until the session has rested its arms, prints each arm's outcome and exits 1 when an arm
is not confirmed off.

## Safety model

Every motion is planned in full from the measured joints and checked before the first target is sent, then watched
tick by tick while it runs. The numbers are the `SafetyConfig` defaults; a robot file can change them under `safety:`.

| Rule | Default |
|---|---|
| Table height: measured per arm by touching it (`llm-harness calibrate`), and used only with the bus, profile, base pose and finger length it was measured with; without it every action except `observe` and `wait` is refused | |
| Rest pose: recorded per arm with its motors off (`llm-harness calibrate`); the motors turn on only within this of it on every joint, and an arm is released only there | 5 deg (`REST_TOLERANCE_DEG`) |
| Hard floor: the elbow and wrist joints, the tool point, the fingertips and the tips of the fully open jaws stay this far above the table at every waypoint, with every joint off its plan by the floor tolerance (a worst case through the kinematics) and the whole arm lower by the sag allowance. By default they may come down to the table surface: the arm may touch the table | 0 mm, 0 deg, 0 mm (`floor_margin_m`, `floor_tolerance_deg`, `sag_allowance_m`) |
| Recovery: an arm already under the floor may only gain height, and once above it the floor holds again | |
| Keep-out zones: boxes in the world frame (robot file `keep_out:`). No point along an arm, sampled every 10 mm on the links from its base to the tool point, the fingers and the open jaws, comes within the margin of a box at any waypoint. An arm already inside may only move further out; the way down to a rest pose may end inside when the rest pose lies there. A measured point entering a box holds the arm | 30 mm (`keep_out_margin_m`) |
| Tool targets: a tool move is refused when the closest pose its path can reach is further than this from the request, and is `done` only when the measured tool pose ends within the arrival tolerance | 5 mm, 2 deg (`POSE_TOL_M`, `POSE_TOL_DEG`); 10 mm, 3 deg (`ARRIVAL_TOL_M`, `ARRIVAL_TOL_DEG`) |
| Rest: the way down to the recorded rest pose may end under the floor when the rest pose lies there, if no point on it is lower than the rest pose and it runs at the slow speed | |
| Watch: a joint farther than the tracking tolerance from its recent waypoints, a measured monitored point more than the contact allowance under the floor, a fault on any arm or a motor over the temperature limit holds the arm where it is | 6 deg, 10 mm (`tracking_tolerance_deg`, `contact_allowance_m`) |
| Slow zone: the whole motion runs slower when it starts or passes this close to the table | 76.2 mm, 5 deg/s (`slow_zone_m`, `slow_speed_deg_s`) |
| Joint speed elsewhere, sent with every target as the arm's speed cap | 20 deg/s (`max_speed_deg_s`) |
| Per-action excursion cap: no joint moves farther than this from where the action started (`home` and `rest` exempt) | 35 deg (`max_excursion_deg`) |
| Temperature: no motion while a motor is hotter (`rest` is allowed, to let it cool) | 70 C (`max_temp_c`) |
| Joint targets streamed while moving | 25 per second (`rate_hz`) |

The gripper's geometry comes from a table per arm type (`arm_model.GRIPPERS`): how far the fingertips reach past the
tool point and how far the open jaws reach to each side, plus a margin, all overridable under `safety:` as
`finger_length_m`, `jaw_half_width_m` and `finger_margin_m`.

| Arm | Fingertips past the tool point | Open jaws from the tool axis | Margin | Source |
|---|---|---|---|---|
| Maker | 34.2 mm | 72.5 mm | 3 mm | the SDK URDF's gripper collision meshes and jaw travel |
| Metal | 15 mm | 50 mm | 10 mm | the MuJoCo model's fingers (the SDK URDF has no gripper geometry) |

**Measure these on the gripper you have installed** and set them in the robot file before you calibrate: the
fingertip length decides where the table is, and too small a jaw width lets the jaws reach under the floor.

With these defaults a Metal tool point pointing straight down comes down to about 25 mm above the table (its
fingertips 10 mm, the Metal finger margin), and a Maker one at pitch -45 to -70 to about 26-35 mm (its fingertips
2-3 mm): the arms grasp small objects such as the 4 cm cube, and may touch the table. For a stricter envelope raise
`floor_margin_m`, `floor_tolerance_deg` and `sag_allowance_m` under `safety:` (for example 0.01, 2 and 0.005).

What it does not do:

- No collision checking between the two arms, or between an arm and anything on the table that is not a keep-out
  zone.
- Opening and closing the gripper is not checked against the table. A held object is not checked, and the floor
  watches only the monitored points (the keep-out zones watch points along the whole arm).
- `op stop` ends the action (and any waiting behind it) and holds the pose; it is not an emergency stop.

The motors of an arm come on with its first accepted motion or gripper action, and only at its rest pose; a refused
one leaves them off. A joint that rests past a limit is then brought inside it slowly along a checked path. `rest`
and every way out (`op quit`, Ctrl-C or SIGTERM on `serve`, `mcp` or `run` on a robot they opened,
`Harness.close()`) move an arm that is on to its rest pose only along a checked path, and release it only when it
arrived there. An arm with a fault, one whose way to rest the envelope refuses, or one that does not arrive is left
holding its pose with the motors on. Each of those commands then prints every arm's torque outcome and exits 1 when
any arm is not confirmed off, with the command that brings that arm down (`metal-arm rest --arm NAME --pose ...`;
`release` would drop it where it is). `serve` on a terminal first asks: Enter leaves the arm holding, `RELEASE` turns
its torque off there. `serve --attach` takes over an arm an earlier session left holding, so `rest` can bring it down.

Underneath, the SDK runs each arm's control loop: it keeps targets inside the joint limits, limits speed, holds the
pose on a fault, and confirms torque off motor by motor. The arms have no brakes: they drop when torque goes off away
from rest. Keep a hardware emergency stop on motor power. See the SDK's safety notes.

## Two arms

```yaml
# ~/.makermods-llm-harness/robots/bench.yaml
backend: real
arm: metal
arms:
  left:  {backend: gs_usb, port: "gs_usb:<serial>"}
  right: {backend: gs_usb, port: "gs_usb:<serial>"}
bases:
  left:  {xyz: [0.0, 0.25, 0.0], yaw_deg: 0}
  right: {xyz: [0.0, -0.25, 0.0], yaw_deg: 0}
cameras: {top: "usb:<path>", left_wrist: "usb:<path>", right_wrist: "usb:<path>"}
keep_out:
  - {name: camera_mount, min: [-0.10, -0.05, 0.0], max: [0.05, 0.05, 1.20]}
```

`bases:` places each arm base in one shared world frame (default: both at the origin, so measure and fill it in).
Targets and observed tool poses are in that frame. Every motion action names its arm (`--arm left`); `rest` without
an arm rests both. One action moves one arm. Each arm has its own table height and rest pose: `llm-harness calibrate`
measures them in turn and writes the rest poses to the robot file under `rest_q`. Two arms must be on two buses,
each named by its own `port` or `channel`.

`keep_out:` lists boxes no part of either arm may enter, such as a camera mount between the arms: each has a unique
`name` and its lowest (`min`) and highest (`max`) corner in the world frame, in metres. Simulator robots take the
same key (MuJoCo draws the boxes). Edit them with the CLI; [docs/OPERATING.md](docs/OPERATING.md#keep-out-zones)
says how to measure one:

```bash
llm-harness robot keep-out add camera_mount --min -0.10 -0.05 0 --max 0.05 0.05 1.2 --robot bench
llm-harness robot keep-out list --robot bench
llm-harness robot keep-out remove camera_mount --robot bench
```

## Your own simulator

A backend is a subclass of `Backend` with `arms`, `read()` and `step(targets, dt)`, plus optional `cameras`,
`table_z`, `enable`, `images`, `rest` and `close`: joint targets in, joint state and images out. The harness
does the rest (actions, envelope, IK, logging); `Harness.open(backend=MySim())` drives it.
[docs/backends.md](docs/backends.md) has the protocol, a complete 50-line example and notes for Isaac Sim and Newton.

## Python

```python
from makermods_llm_harness import Harness, robot

with Harness.open(robot.simulated()) as h:          # or Harness.open("bench") for a robot file
    print(h.describe())                             # robot, frames and envelope in words: prompt material
    obs = h.step({"action": "home"})
    obs = h.step({"action": "move_tool", "xyz": [0.31, 0.03, 0.54], "pitch_deg": -5})
    print(obs.result.status, obs.arms["arm"].tool.xyz)
    print(h.observe().text())
```

`step` never raises for a bad, refused or failed action: the outcome is in `obs.result`. `h.stop()` (from another
thread) ends the action in progress and the ones waiting behind it; `h.step(action, cancel=event)` ends that one
action when `event` is set. `h.close()`, which leaving the `with` block calls, rests every arm that is on and
returns each arm's torque outcome (`"released"`, `"always_on"` for a simulator, anything else with the reason when
the arm may still be on); after it `step` and `observe` raise `HarnessClosed`. `session.Client()` has the same
`observe`, `step`, `rest`, `stop` and `describe`
for a running `serve`; `agent.run(target, task)` is the model loop; `actions.tools()` and `actions.SCHEMA` give the
actions as tool definitions and as one JSON Schema.

## Development

```bash
uv venv --python 3.12
uv pip install -e ".[dev,mujoco,mcp,agent]"
uv pip install -e ../metal-arm-sdk -e ../maker-arm-sdk      # optional: local SDK checkouts
.venv/bin/python -m pytest -q
```

The tests run only against the simulators, injected fakes and simulated SDK motors; none opens a device. Tests for an
optional extra skip when it is not installed. Not yet verified with this version: real hardware, live MCP clients, the
live model API, and sessions over Windows named pipes.

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
