# Backends

A backend is the robot as the harness sees it: joint targets in, joint state and camera images out. The harness does
everything above that: parsing actions, the safety envelope, inverse kinematics and path planning, logging, the
session server and the MCP tools. So driving another simulator from an LLM takes one subclass of
`makermods_llm_harness.Backend`.

## The protocol

Units are radians, metres and seconds. `q` is the six arm joints in the SDK profile's order (`shoulder_pan,
shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll`), in the SDK's joint coordinates (what a real arm
reports), without the gripper. The gripper is one number: its opening, 0 closed to 1 open.

| Member | Required | Returns / does |
|---|---|---|
| `arms` (property) | yes | `list[ArmInfo]`, in a fixed order: each arm's `name`, `type` (`"metal"` or `"maker"`), optional `profile`, and base pose in the world frame (`base_xyz` in m, `base_yaw` in rad about z) |
| `read()` | yes | `dict[name, ArmState]` for every arm |
| `step(targets, dt)` | yes | apply `targets` (`dict[name, ArmTarget]`, only the arms named; the others keep their last target), then let `dt` seconds pass: a simulator advances its clock, hardware sleeps |
| `cameras` (property) | no (`[]`) | camera names |
| `table_z` (property) | no (`None`) | world-frame height of the table top when the backend knows it; `None` means the harness needs a stored calibration |
| `enable(arm, cancel=None)` | no | make the arm follow targets, holding its current pose (real arms: torque on); give up when the `threading.Event` `cancel` is set. Called only at the arm's rest pose, once a motion has been planned and checked, before its first target. |
| `hold(arm)` | no (one `step` to the measured pose) | freeze the arm at its measured joints and gripper opening, replacing every outstanding target |
| `images()` | no (`{}`) | `dict[camera, Frame]`: the newest image of each camera; leave out a camera that has none |
| `rest(arm)` | no (`"always_on"`) | release the arm where it stands; the harness has already brought it to `rest_q`. Returns `"released"` (torque off confirmed), `"always_on"` (a simulator without torque off), `"holding"` (not released) or an error text. It must not move the arm. |
| `close()` | no | free the devices without moving anything. The harness rests every arm first; an arm still on is left holding its pose where the hardware can do that. May return `dict[name, outcome]` with the words of `rest`, `"unknown"` or `"energized"` (on, but not from this session). |

`ArmState` fields:

| Field | Meaning |
|---|---|
| `q` | measured joints, (6,). Every plan starts here. |
| `gripper` | measured opening, 0..1 |
| `q_target`, `gripper_target` | the command the arm is following (equal to the measurement before the first command) |
| `enabled` | `True` while the motors hold torque, `False` when torque off is confirmed, `None` when it is not known. An arm that is not `True` is enabled (`enable(arm)`) before its first motion. |
| `fault` | `None` while the arm can move; any text stops the action with a `fault` result, and no arm moves while one reports a fault |
| `temperature` | hottest motor in °C, or `None`; above `max_temp_c` motions are refused, and one in progress holds |
| `rest_q` | the folded rest pose. An arm that is off is turned on only within `REST_TOLERANCE_DEG` (5 deg) of it on every joint, and `rest` releases it only there. `None`: the arm is never turned on or released. |

`ArmTarget(q, gripper, speed)` is one command; `speed` (rad/s) caps every joint on the way to `q`, and is `None` only
for the gripper action, which repeats the arm's current command. `Frame(image, stamp, captured)` is an RGB `uint8`
array of shape (height, width, 3), its time in seconds (wall clock or simulated) and its `time.monotonic()` capture
time (`None`: made when `images()` was called). The harness encodes each frame to JPEG once per observation.

## What the harness sends

- Only waypoints of a plan that passed the envelope, planned from the measured joints: inside the joint limits, no
  joint more than `max_excursion_deg` from the start of the action (except `home` and `rest`), every monitored point
  above the floor even with every joint `floor_tolerance_deg` off and `sag_allowance_m` lower, and no joint step
  larger than the speed limit allows in one tick. Each target carries that speed limit.
- One `step(targets, dt)` per waypoint with `dt = 1 / rate_hz` (25 Hz by default), for one arm at a time, with a
  `read()` after each: a fault on any arm, a hot motor, a joint more than `tracking_tolerance_deg` from every
  waypoint sent within the last `tracking_lag_s`, or a measured clearance more than `contact_allowance_m` under the
  floor ends the motion with `hold(arm)`. A move with `until: "contact"` also ends with `hold(arm)` once the arm has
  followed its command and then its tool point stood still for 0.3 s while the command moved on, `contact_tolerance_m`
  from every tool point sent within the lag. Elapsed time counts each tick from the end of the one before, and at
  least `dt`. Each target goes out only after a stop check made under the lock that serializes backend calls.
- `hold(arm)` must replace the arm's joint and gripper targets with its measured pose before it returns, or raise,
  and returns the setpoint it used, `(q, opening)`. The harness reads every arm `hold_settle_s` later (0 by default;
  two control periods on real arms): a fault, missing feedback, unknown torque, a joint sample older than
  0.25 s (`feedback_ages`), or a commanded target (`q_target`, `gripper_target`: what the motors are commanded,
  not a local copy) more than 0.2 deg (0.01 opening) from that setpoint, a setpoint away from where the arm was
  measured, or a joint measured past its range makes the outcome a `fault`. Every stop, and every motion that ends other than by arriving, holds every
  arm that reports `enabled` this way.
- `feedback_ages(arm)` (optional): seconds since each joint, gripper last, reported a sample. The harness releases
  an arm at rest only after every joint has reported one newer than the last rest waypoint (up to 1 s); the default
  `None` says every `read()` is fresh. The real backend takes it from the SDK's `status()`.
- `enable(arm, cancel, speed)`: `speed` (rad/s) is passed only when the motion asked for a lower speed; it caps any
  motion of the backend's own while enabling.
- After the last waypoint, the goal again for up to `settle_s` (1 s) until the measured joints are within 1 deg of
  it; a joint still further than 2 deg, or a quarter of its travel (at most `tracking_tolerance_deg`), away holds
  the arm and the action is `stopped`.
- Turning an arm on: at its rest pose only, `enable(arm, cancel)`; a joint that rests past a limit is then brought
  inside it along a checked path at the slow speed.
- For `rest`: a checked joint path straight to `rest_q` (clipped into the limits), or, when the envelope refuses
  that, one to the home pose and one from where that ends to `rest_q`; then, when the measured joints are within
  5 deg of `rest_q`, `rest(arm)`. An arm with a fault, or one that does not arrive, is not moved further and
  `rest(arm)` is not called.
- To hold (after `op stop`, a cancelled request or any watch above): `hold(arm)`.
- For the gripper action: the arm's `q_target` unchanged (`speed` `None`) and the requested opening, tick by tick
  for up to 2 s, until the measured opening reaches the request or stops changing.
- For `wait`: `step({}, dt)` with no targets, so time passes.
- After an action (unless it was sent without images, as the kernel's motions are), `images()` until every frame was
  captured after the action ended, for up to 0.5 s; an older frame is marked `stale` in the observation. `look()`
  does the same for one camera grid.

Calls may come from different threads but never at the same time: the harness runs every backend call under one
lock, so `read()` and `images()` for `state()`, `look()` and reach checks fall between the ticks of a motion. A
backend whose rendering belongs to one thread (an EGL context) renders on a thread of its own, as the MuJoCo backend
does.
`Harness.stop()` (or a request's own cancel event) only sets a flag that the motion loop checks between waypoints.

Call order over a session: `arms`, `table_z` and `cameras` when the harness opens; then per action `read()`, maybe
`enable(arm)`, the `step` calls with `read()` in between; `rest(arm)` at the end of a `rest` action.
`Harness.close()` runs the same rest for every arm that is on, then calls `close()` once and returns each arm's
outcome.

The envelope computes the monitored points with the SDK's kinematics for each `ArmInfo` (type, profile, base pose),
not with the simulator's geometry. A simulated arm must therefore match its SDK profile, and sit where its `ArmInfo`
says. The harness takes base poses from `backend.arms`, so a backend that moves an arm base (for example to keep it
above the table) reports the moved pose there.

## A complete example

One Metal arm whose joints jump to each target, a table at z = 0 and one grey camera. It runs as shown:

```python
import time

import numpy as np

from makermods_llm_harness import ArmInfo, ArmState, Backend, Frame, Harness


class InstantSim(Backend):
    """One Metal arm whose joints jump to each target, over a table at z = 0, with one grey camera."""

    def __init__(self):
        self.q = np.zeros(6)               # the rest pose: arm stretched forward, inside the joint limits
        self.q_target = self.q.copy()
        self.gripper = 0.0
        self.enabled = False

    @property
    def arms(self):
        return [ArmInfo("arm", "metal")]

    @property
    def cameras(self):
        return ["top"]

    @property
    def table_z(self):
        return 0.0

    def enable(self, arm, cancel=None):
        self.enabled, self.q_target = True, self.q.copy()

    def read(self):
        return {"arm": ArmState(q=self.q.copy(), gripper=self.gripper, q_target=self.q_target.copy(),
                                gripper_target=self.gripper, enabled=self.enabled, rest_q=np.zeros(6))}

    def step(self, targets, dt):
        target = targets.get("arm")
        if target is not None and self.enabled:
            self.q = self.q_target = np.asarray(target.q, dtype=float).copy()
            self.gripper = float(target.gripper)

    def images(self):
        return {"top": Frame(np.full((240, 320, 3), 128, np.uint8), time.time())}

    def rest(self, arm):
        self.enabled = False               # the harness has brought it to rest_q; release it there
        return "released"


with Harness.open(backend=InstantSim()) as h:
    for action in ({"action": "home"},
                   {"action": "move_tool", "xyz": [0.31, 0.0, 0.50], "pitch_deg": -10},
                   {"action": "gripper", "opening": "open"},
                   {"action": "rest"}):
        obs = h.step(action)
        print(action["action"], "->", obs.result.status, obs.arms["arm"].tool.xyz)
```

```
home -> done (0.3128, -0.0003, 0.581)
move_tool -> done (0.31, -0.0, 0.4999)
gripper -> done (0.31, -0.0, 0.4999)
rest -> done (0.1743, -0.0003, 0.189)
```

To drive it from the CLI or an MCP client, serve the harness yourself; `llm-harness op ...` then works against it
until `llm-harness op quit`:

```python
from makermods_llm_harness import EpisodeLog, Harness, log
from makermods_llm_harness.session import Server

with Harness.open(backend=InstantSim(), log=EpisodeLog(log.new_run_dir("instant"))) as h:
    Server(h).serve_forever()           # or: from makermods_llm_harness import mcp_server; mcp_server.run(h)
```

The episode log gives the session its run directory and image files; without one, observations carry the JPEG bytes
but no paths.

## The built-in backends

| Backend | Module | `step` | `images` | `table_z` | `rest` |
|---|---|---|---|---|---|
| Kinematic simulator | `backends/sim.py` (`SimBackend`) | joints follow at up to 180 deg/s, the gripper at 2 openings/s; no sleeping | cv2 drawings: `top`, `side`, `<arm>_wrist` | the scene's | `"released"` (`"holding"` when blocked or faulted) |
| MuJoCo | `backends/mujoco.py` (`MujocoBackend`, extra `mujoco`) | position servos on the URDF's links, `dt / 0.002` physics steps | offscreen rendering: `top`, `<arm>_wrist` | the scene's | `"always_on"` |
| Real arms | `backends/real.py` (`RealBackend`) | `Arm.move_to(q + [gripper_target(opening)], speed=target.speed, wait=False)`, then sleep to a monotonic deadline | `Camera.newest()`, BGR to RGB, with its capture time | `None`: calibrated | `Arm.disable()` where it stands: `"released"` when every motor confirmed it |

The simulators also have test hooks: `SimBackend.set_fault`, `set_temperature`, `block` and `objects`;
`MujocoBackend.object_xyz` and `tool_pose`. A robot file chooses between them with `backend: sim | mujoco | real`.

## Table height and custom arms

A simulator knows its table: return its world-frame height from `table_z` and every arm uses it. `Harness.open(...,
table_z=0.0)` (one height) or `table_z={"left": 0.0, "right": 0.01}` overrides it, and the backend's value overrides a
stored calibration.

`ArmInfo.type` selects the SDK (`metal_arm` or `maker_arm`) and with it the shipped profile: joint names and limits,
URDF, tool point, rest pose. `ArmInfo.profile` names another profile file in the same format (see the SDK's
`profiles/` directory), for an arm with other limits or another tool. In a simulator robot file it is the one field an
arm entry takes:

```yaml
backend: mujoco
arm: metal
arms:
  arm: {profile: /path/to/my_profile.yaml}
```

## Isaac Sim and Newton

Not tried here: nobody has run either with this harness. These notes say which calls of theirs map to which member;
check the names against the version you use.

- **Joint order and coordinates.** Import the URDF the profile points to. The articulation's joints come in its own
  order; map them by name to `ArmModel(info).chain.joint_names` (the URDF names of the six arm joints). The URDF angle
  of joint i is `q[i] * sign + offset` with `sign`, `offset` = `config.joints[i].urdf_sign`, `.urdf_offset` of the
  same `ArmModel(info).config`; `MujocoBackend` does exactly this (`_Arm.theta`). Convert back the same way in
  `read()`.
- **Base pose.** The URDF root sits in the world as: translate by `ArmInfo.base_xyz`, turn `base_yaw` about z, then
  turn by the profile's `urdf_base_rpy` (`config.urdf_base_rpy`, radians; fixed-axis roll, pitch, yaw about x, y,
  z). Metal's is [0, 0, 0]. The Maker URDF is drawn y-up, so its profile has [90, 0, 180] deg: leaving that rotation
  out lays the arm on its side. `MujocoBackend` nests exactly these two frames (`_Arm.xml`).
- **Check the import.** Before anything else, put the arm at its home pose (`ArmModel(info).home_q`) and compare the
  simulator's tool pose (the link the profile's `tcp_xyz`, `tcp_rpy` hang from, offset by them) with
  `ArmModel(info).tool_pose(home_q)`: they must agree within a millimetre and a degree. Do the same at two more
  poses with every joint moved. A mismatch means a sign, offset, joint order or base rotation is wrong, and the
  envelope would then check a different arm from the one that moves.
- **`step(targets, dt)`.** Write the joint position targets (and the finger joints for the opening), then advance the
  physics by `dt` in as many substeps as the solver needs. In Isaac Sim that is the articulation's position-target
  setter followed by stepping the world without rendering; in Newton, writing the control's joint targets and calling
  the solver's step for each substep.
- **`read()`.** The articulation's joint positions, converted to SDK coordinates; the finger gap divided by the gap at
  opening 1; `q_target` the last targets; `enabled=True` unless you model torque; `rest_q` the pose the arm starts
  in.
- **`images()`.** Render headless: Isaac Sim's camera sensor RGBA output without its alpha channel, Newton's offscreen
  or tiled camera output, as `uint8` RGB. Render only here, not in `step`, so motions stay fast.
- **`table_z`.** The height of the table top you placed.
- **`rest(arm)`.** Do not move: the harness has driven the arm to `rest_q` along a checked path. Return
  `"always_on"` when the simulation keeps the servos on, or turn torque off and return `"released"`.
