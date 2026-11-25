# Reference

Commands, actions, the safety model and the Python API in full. Start with the [README](../README.md).

## Connecting an LLM

| Way | Command | Who decides |
|---|---|---|
| A coding agent runs the CLI | `llm-harness op ...` against `llm-harness serve` | the agent (Claude Code, Codex, ...); it reads the JPEG paths |
| A coding agent runs Python | `llm-harness py` against `llm-harness serve` | the agent, queueing actions ahead ([docs/AGENT_GUIDE.md](AGENT_GUIDE.md)) |
| MCP tools | `llm-harness mcp --connect` | the MCP client's model; images come back inline |
| Built-in model loop | `llm-harness run "pick up the red cube"` | a model over the API |

**Coding agent.** Point it at [docs/OPERATING.md](OPERATING.md). Claude Code picks up the skill in
[.claude/skills/llm-harness-operator](../.claude/skills/llm-harness-operator/SKILL.md) when it works in this repository.

**MCP.** `llm-harness mcp` serves one tool per action on stdio, plus `run_python` (code in the session's kernel),
with the robot description and the kernel guide as its instructions.
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
early, the rest of that reply's tool calls are not run. The model also has `run_python`; when it ends its turn while
queued actions run, the loop waits for their reports and hands them to it; a run that ends any other way stops what
it queued. With `--sim`, `--mujoco` or `--robot` the exit status is 0 only when the model finished with a plain
answer and every arm is confirmed off at the end; with `--connect` the session keeps the arms, and the exit status
is 0 when the model finished with a plain answer.

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
| `move_tool` | `xyz` [x, y, z]; `pitch_deg` (keep), `yaw_deg` (heading from the arm base to xyz), `roll_deg` (keep), `line` (true), `until`, `arm` | Straight line of the tool point to xyz (`line` false: the joints move straight to the pose). Pitch -90 points straight down, 0 horizontal. |
| `nudge` | `dxyz` [dx, dy, dz]; `frame` `base` or `tool` (`base`), `until`, `arm` | Move the tool point by an offset, keeping its orientation. The offset is added to the last commanded tool pose while the arm holds still within 10 mm and 3 deg of it (so small nudges add up despite a standing error), else to the measured pose. |
| `gripper` | `opening` 0 closed .. 1 open, or `"open"` / `"close"`; `arm` | The result's `grip` says where the jaws stopped and whether they hold something. |
| `move_joints` | `joints_deg` {name: deg}; `relative` (false), `arm` | Named joints to angles; the others stay. |
| `home` | `arm` | The ready pose: gripper forward, above the table. |
| `rest` | `arm` (every arm) | Back to the rest pose along a checked path (by way of `home` when the straight path is refused), then motors off. |
| `wait` | `seconds` 0..30 (1) | Let time pass, then observe. |

`until: "contact"` (on `move_tool` and `nudge`) ends the move early, `done` and holding, once the arm has followed its
command (a joint moved at least 1 deg and half as far as its command) and then, while the command was still moving,
its tool point advanced along the approach less than a quarter as far as the command had one tracking lag earlier,
for more than 0.3 s, and fell more than `contact_tolerance_m` beyond both the offset it stood at before the move and
the largest shortfall it had while advancing freely (load changes, lag), from every tool point commanded within the
lag; `result.contact` says where. A move that reaches the end of its command more than `contact_tolerance_m` short of
the target along the approach, and then stands still for 0.3 s, also ends on contact there ("stopped on contact at the
end of the move"); short and still moving, it is `stopped`. Lag, gear take-up, a standing error within the tolerance,
or a miss in orientation only, are never contact: such a move ends `done` if the measured tool pose arrived, else
`stopped`. A move whose target is under the floor is refused with how far down the lowest point may come from here. `speed_deg_s` (a joint speed cap, at least 0.5 deg/s, sent to the arm with every target) or
`duration_s` (at most 60 s; a shorter move is stretched to it) slows `move_tool`, `nudge`, `move_joints` and `home`;
nothing goes faster than the envelope, and no plan lasts more than 120 s. The motion ops (`move_tool`, `nudge`,
`gripper`, `move_joints`, `home`, `rest`) take `--dry-run`: checked as they would be, nothing moves; so do the
session's `check` request and `Harness.check(action)`. What only feedback during the motion can show (blocking,
contact, arrival) is not known before.

```bash
llm-harness op '{"action": "move_tool", "xyz": [0.31, 0.03, 0.54], "pitch_deg": -5}'
llm-harness op nudge 0 0 -0.08 --until contact
```

| Observation field | Meaning |
|---|---|
| `step`, `time` | actions so far in this run; Unix time |
| `result.status`, `result.message`, `result.action` | how the action ended, why, and the action as parsed |
| `result.error` | after a tool move: `position_m` and `angle_deg` between the measured and the requested tool pose, and `short_m`, how far the tool point ended short of the target along the approach (negative past it) |
| `result.contact` | after a move with `until: "contact"` that stopped on contact: `xyz` of the tool point and `short_m` from the target |
| `result.grip` | after `gripper`: `opening` (measured), `target`, `settled`, `holding`. Done, not holding, within 0.02 of the request, or settled (less than 0.005 over 0.5 s) within 0.04 of it (0.15 when opening: jaws stop at their mechanical end). Holding when, closing, the jaws stalled (less than 0.003 over 1 s) more than 0.06 short of the request after closing at least 0.05, or were already closed on something. Anything else by the 2.5 s limit (still moving or creeping, stalled without that progress, an opening that does not open): `stopped` (undetermined or blocked), the gripper held where it is |
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
| Table height: measured per arm by touching it (`llm-harness calibrate`), and used only with the bus, profile, base pose and finger length it was measured with; measured at several reaches (`calibrate --powered`), the height under each monitored point follows its distance from the arm base, linear between the measured reaches and flat beyond them; without it every action except `observe` and `wait` is refused | |
| Rest pose: recorded per arm with its motors off (`llm-harness calibrate`); the motors turn on only within this of it on every joint, and an arm is released only there | 5 deg (`REST_TOLERANCE_DEG`) |
| Hard floor: the elbow and wrist joints, the tool point, the fingertips and the tips of the fully open jaws stay this far above the table at every waypoint, with every joint off its plan by the floor tolerance (a worst case through the kinematics) and the whole arm lower by the sag allowance. By default they may come down to the table surface: the arm may touch the table | 0 mm, 0 deg, 0 mm (`floor_margin_m`, `floor_tolerance_deg`, `sag_allowance_m`) |
| Recovery: an arm already under the floor may only gain height, and once above it the floor holds again | |
| Keep-out zones: boxes in the world frame (robot file `keep_out:`). No point along an arm, sampled every 10 mm on the links from its base to the tool point, the fingers and the open jaws, comes within the margin of a box at any waypoint. An arm already inside may only move further out; the way down to a rest pose may end inside when the rest pose lies there. A measured point entering a box holds the arm | 30 mm (`keep_out_margin_m`) |
| Tool targets: a tool move is refused when the closest pose its path can reach is further than this from the request, and is `done` when the measured tool pose ends within the arrival tolerance, or within the sag tolerance when the standing error the joints held before the move explains the miss (the message then says how far off it ended), unless it ended more than the contact tolerance short of the target along the approach: that is `stopped`, with the shortfall | 5 mm, 2 deg (`POSE_TOL_M`, `POSE_TOL_DEG`); 10 mm, 3 deg (`ARRIVAL_TOL_M`, `ARRIVAL_TOL_DEG`); 30 mm, 6 deg (`SAG_ARRIVAL_TOL_M`, `SAG_ARRIVAL_TOL_DEG`) |
| Rest: the way down to the recorded rest pose may end under the floor when the rest pose lies there, if no point on it is lower than the rest pose and it runs at the slow speed | |
| Watch: a joint farther than the tracking tolerance from every waypoint sent within the tracking lag (elapsed time, every tick at least 1/`rate_hz`, at most 0.5 s: the allowance along the path grows with the commanded speed), a measured monitored point more than the contact allowance under the floor, a fault on any arm or a motor over the temperature limit holds the arm where it is | 6 deg, 0.2 s, 10 mm (`tracking_tolerance_deg`, `tracking_lag_s`, `contact_allowance_m`) |
| Settling: after the last waypoint the goal is sent again until the joints are within 1 deg of it, for at most | 1 s (`settle_s`) |
| End of a motion (`harness.judge_motion`, a table of evidence to outcome), per joint: a joint the caller named is judged from where it was measured, any other from the command it held. A travel of 0.5 deg or more must end within 2 deg, its standing error before plus 0.5 deg, or a quarter of the travel (whichever is most, at most the tracking tolerance), and from 4 deg of travel must also be covered at least half way, else `stopped` (ended short). Under 4 deg a real arm's friction hides progress, so a block that small can end `done`: check the cameras. A smaller named request must end within its standing error plus 0.5 deg having moved at least half of it the right way, else `stopped` (undetermined); a joint not named may end within that allowance. Joints `move_joints` does not name keep their command and may only drift 2 deg; the speed near the table is the lower of the command's and the measured pose's. The settling ends early only once the motion is judged arrived. A tool move must end with the measured tool pose within 10 mm and 3 deg of its target, from fresh valid feedback: a miss is `stopped`, unless the standing error held before the move explains it and it is within 30 mm and 6 deg, which is `done` with the miss in the message | 0.5, 2, 0.5 deg, 25% |
| Holds: every way a motion ends other than a verified arrival (stopped, blocked, contact, arrival miss, jaws not settled, a failed action, a stop, shutdown) replaces the joint and gripper targets of every arm that may be on with its measured pose (the backend returns that setpoint) and confirms it from feedback read after the control loop has had time to send it (two control periods on real arms): no fault, torque known, every joint's sample at most 0.25 s old, every joint inside its range (one past it cannot be held there), the setpoint where the arm was measured, and the commands the backend reports equal to it within 0.2 deg (gripper 0.01); what cannot be confirmed makes the result a `fault` that says so. `op stop` answers only after that, or with the failure; once the session shuts down it changes nothing | 0.2 deg, 0.01 opening, 0.25 s |
| Release at rest: torque goes off only after every joint (gripper included) has reported a sample newer than the last rest waypoint, within 5 deg of the rest pose with no fault, and only if no stop came first | 1 s wait |
| Contact: a move with `until: "contact"` ends once the arm has stood still with the measured tool point this far from every tool point commanded within the tracking lag | 10 mm (`contact_tolerance_m`) |
| Slow zone: the whole motion runs slower when it starts or passes this close to the table | 76.2 mm, 5 deg/s (`slow_zone_m`, `slow_speed_deg_s`) |
| Joint speed elsewhere, sent with every target as the arm's speed cap | 20 deg/s (`max_speed_deg_s`) |
| Per-action excursion cap: no joint moves farther than this from where the action started (`home` and `rest` exempt) | 35 deg (`max_excursion_deg`) |
| Temperature: no motion while a motor is hotter (`rest` is allowed, to let it cool) | 70 C (`max_temp_c`) |
| Joint targets streamed while moving | 25 per second (`rate_hz`) |

The settings that cost time are `max_speed_deg_s`, `slow_speed_deg_s`, `slow_zone_m`, `max_excursion_deg`, `settle_s`,
`tracking_tolerance_deg` and `tracking_lag_s`. Faster motion needs a tracking lag that covers how late the arm follows
its command (the real arms trail it by roughly 70-170 ms):

```yaml
safety: {max_speed_deg_s: 60, slow_speed_deg_s: 15, max_excursion_deg: 60, settle_s: 0.5, tracking_lag_s: 0.2}
```

Every setting has an accepted range (`safety.LIMITS`); a robot file outside it is refused with the range in the
message.

`llm-harness log profile` shows where an episode's time went (in steps, moving or refused; outside steps, the arm
idle; between resumed sessions; per action), so a change can be measured.

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
its torque off there (reported as off but NOT at its rest pose). `serve --attach` takes over an arm an earlier session left holding, so `rest` can bring it down.

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
same key (MuJoCo draws the boxes). Edit them with the CLI; [docs/OPERATING.md](OPERATING.md#keep-out-zones)
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
[docs/backends.md](backends.md) has the protocol, a complete 50-line example and notes for Isaac Sim and Newton.

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

`h.state()`, `h.look()` (one camera grid), `h.check(action)` and `h.reachable(poses)` move nothing and do not wait for
the action in progress. `program.Program(h)` (or `Program(client)` for a session) is the object the kernel's `r` is:
`state`, `look`, `reachable`, `move`, `nudge`, `move_joints`, `gripper`, `home`, `rest`, `wait`, each motion without
camera images, raising `MotionFailed` when it does not end done:

```python
from makermods_llm_harness import Harness, robot
from makermods_llm_harness.program import Program

with Harness.open(robot.simulated()) as h:
    r = Program(h)
    r.home()
    print(r.reachable([{"xyz": [0.31, 0.03, 0.54], "pitch": -5}]))
    print(r.move([0.31, 0.03, 0.54], pitch=-5).error)
```

A session's Python kernel (`kernel.py`) runs model code in a separate process the session starts and can kill. That
process holds only a client of the session's kernel channel, whose key the session hands it: there every request
that moves or queues needs a live execution token, which the session revokes on a stop, a hang-up, the end of the
call or action, a timeout or shutdown. The action queue, its reports and the authority to move stay in the session.
`llm-harness py`, the MCP `run_python` tool and the model loop use it; `mcp` and `run` without `--connect` serve
their robot through a session of their own for that, and a `quit` there rests the arms through its owner. Model code
runs as the user who started the session, with all of that user's rights: it is not sandboxed.

Every way a session the CLI owns ends (`op quit`, Ctrl-C, SIGTERM, an error in its server or MCP loop, `mcp` or
`run` on their own robot) goes through one routine (`cli.close_session`): the session's part (shutdown phase, kernel
ended) first, then, whatever happened there, a confirmed hold, every arm rested and released, the episode log ended
(`closing`, `end`), each arm's outcome reported, and last the address lock given up.

Every connection to a session first answers a challenge with its channel's key; the operator key is in an
owner-only file next to the socket, published once the server owns the address (a lock file there keeps a second
server out). Requests that can move or queue are admitted under one gate with stop and shutdown: a stop cancels
everything admitted before it, and once shutdown has begun only its own rest moves the arms. On Windows, where the
key file cannot be shown to be private, the kernel is not available.

