# Operating guide

For whoever is the policy: a coding agent that runs `llm-harness op ...` and reads the JPEG files, or an MCP client
whose model calls the same actions as tools. The [operator section](#operator) at the end is for the person who
starts the robot and keeps a hand on the emergency stop.

The CLI forms are shown. Over MCP the tool names and fields are the same (`move_tool` with `xyz`, `pitch_deg`, ...)
and the images come back inline instead of as paths. For faster episodes, drive the session from its Python kernel
instead (`llm-harness py`, the MCP tool `run_python`): motions are queued as actions that run while the next ones are
prepared. [AGENT_GUIDE.md](AGENT_GUIDE.md) is the guide for that.

## The loop

1. Once: `llm-harness op describe`. It names the arms, cameras, joint ranges, where `home` puts the tool, the table
   height and the envelope numbers.
2. `llm-harness op observe`, then open every image path it prints.
3. Decide one action. Say what you expect to see after it.
4. Send it, read the result line, the numbers and the fresh images. Back to 3.

One action per call keeps you looking between motions. `llm-harness ops` sends a short list and stops at the first one
that is not `done`, so a refused descent is never followed by a close on air:

```bash
llm-harness ops "nudge 0 0 -0.02" "gripper close"
```

Do not chain `op` calls with `;` in the shell for the same reason.

Two flags save steps: `--dry-run` checks a motion exactly as it would be checked and moves nothing, and
`--until contact` on `move_tool` or `nudge` ends the move where the arm stops on something, instead of descending a
centimetre at a time:

```bash
llm-harness op move_tool 0.28 0 0.12 --pitch -90 --dry-run
llm-harness op nudge 0 0 -0.08 --until contact
```

## Reading an observation

```
arm: tool xyz [0.280, -0.000, 0.120] m, pitch -90, yaw -0, roll 0 deg; gripper 1.00; clearance above table 0.095 m
  joints (deg): shoulder_pan 0.1, shoulder_lift -100.3, elbow_flex 74.0, wrist_flex -62.7, wrist_yaw 0.0, wrist_roll 0.1
camera top: /home/you/.makermods-llm-harness/runs/20261120-182307-sim/images/step-0005-top.jpg
camera side: /home/you/.makermods-llm-harness/runs/20261120-182307-sim/images/step-0005-side.jpg
```

| Part | How to use it |
|---|---|
| first line, when present | `rejected: ...`, `stopped: ...` or `fault: ...` and why. No such line means `done`. |
| `tool xyz` | the tool point (between the gripper fingers) in the world frame, metres. Plan in these numbers. |
| `pitch, yaw, roll` | the gripper's orientation, in the angles `move_tool` takes (next section) |
| `gripper` | measured opening, 0 closed .. 1 open. After a close, a value well above 0 means the jaws stopped on something. |
| `clearance above table` | height of the lowest monitored point (elbow and wrist joints, tool point, fingertips, tips of the open jaws) above the table, as measured. Motions keep it above the hard floor (0 mm by default: the table surface; next sections). |
| `keep-out zone NAME D m away` | the keep-out zone closest to any part of the arm and how far (only when the robot has zones; `op describe` lists them with their corners) |
| `(motors off)` after the arm name | the arm is released at rest; the next motion turns it on |
| `(torque state unknown)` | the motors do not report whether torque is on (older firmware); the next motion turns them on |
| `joints (deg)` | the six arm joints; the names are the ones `move_joints` takes |
| `camera NAME: PATH` | a JPEG written for this step; open it. An error text instead means that camera had no recent image. `(stale: N s old ...)` means no frame taken after the action ended arrived in time: the picture may show the arm before it stopped. |

`--json` prints the same as one JSON object (see the README for the fields). Images are in the run's `images/`
directory: `step-NNNN-<camera>.jpg` for an action (`NNNN` is the step number) and `look-NNNN-<camera>.jpg` for a look
that is not an action (a model loop's first look, a session client's `observe()`).

Cameras in the simulators:

| Camera | View |
|---|---|
| `top` | looking down: +x (away from the arm) is up the image, +y is to the left |
| `side` (kinematic simulator) | looking along +y: +x to the right, z up; best for heights |
| `<arm>_wrist` (`arm_wrist` in `sim --mujoco`) | from the gripper along its pointing direction; the jaws are at the bottom of the image |

On a real robot the cameras are whatever the robot file names; `op describe` lists them. Read heights from the
numbers rather than from the images.

## Frames and tool angles

The world frame has z up. With one arm at the default base it is the arm's base frame: x forward (away from the
base), y left, z up, origin at the base. With two arms each base sits where the robot file's `bases:` puts it.

The pointing direction of the gripper is `(cos p cos y, cos p sin y, sin p)` for pitch p and yaw y:

| Angle | Meaning |
|---|---|
| `pitch_deg` | tilt of the pointing direction: -90 straight down, 0 horizontal, +90 straight up |
| `yaw_deg` | heading of the pointing direction about the vertical: 0 = +x, 90 = +y |
| `roll_deg` | turn of the gripper about its pointing direction |

With roll 0 the tool x axis is horizontal; in both simulators the jaws close along it (on a real arm, confirm in the
wrist image). `move_tool` keeps the current pitch and roll when you leave them out, and points the yaw from the arm's
base towards the target; leaving out both pitch and roll keeps the gripper's exact orientation, turned about the
vertical by the change of heading.

Example 1, straight down. `op move_tool 0.28 0 0.12 --pitch -90` points the gripper down over x = 0.28 with yaw 0:
the tool x axis, and so the jaws, run along world y (left-right). To close them along x instead, turn the gripper 90
deg about the vertical, at most about 30 deg per action:

```bash
llm-harness op move_tool 0.28 0 0.12 --pitch -90 --yaw 30
llm-harness op move_tool 0.28 0 0.12 --pitch -90 --yaw 60
llm-harness op move_tool 0.28 0 0.12 --pitch -90 --yaw 90
```

Within 0.5 deg of straight down (or up) yaw and roll would turn the gripper about the same axis, so the observation
reads roll 0 there and yaw carries the whole turn; the last command above reads `pitch -90, yaw 90, roll 0`.

Example 2, tilted. From the Maker arm's `home`, `op move_tool 0.30 0.10 0.15 --pitch -45` puts the tool point at
(0.30, 0.10, 0.15) pointing 45 deg below horizontal; the yaw defaults to the heading from the base to the target,
atan2(0.10, 0.30) = 18 deg, so the gripper points forward-left and down, away from the base. The observation reads
`pitch -45, yaw 18, roll 0`.

`nudge` offsets are in the arm's base axes (`--frame base`: x forward, y left, z up) or in the gripper's own axes
(`--frame tool`: z is the pointing direction). Pointing down, `op nudge 0 0 0.02 --frame tool` moves 2 cm down.
A nudge is added to the pose the last motion was sent to while the arm holds still within 10 mm and 3 deg of it, so
a real arm's standing error of a few millimetres does not eat into small nudges; after a stop or a fault, or when
the arm stands further off, it is added to the measured pose.

## Approaching and grasping from above

1. Look at the scene from high up and find the object in the images. Get its x, y from where it sits relative to the
   gripper and the table, then check by hovering above it.
2. Turn the gripper down in steps (below), ending about 10 cm above the object.
3. Open the gripper fully before you descend: jaws only slightly wider than the object land on its edges.
4. Line up over the object in the top and wrist images (`nudge` by 1-2 cm sideways).
5. Descend in small steps, 2-5 cm each, looking after each one. Within 76 mm of the table every motion runs at the
   slow speed, so short steps take a few seconds each. By default the envelope lets the tool point come down to about
   25 mm above the table pointing straight down on a Metal arm (fingertips 10 mm) and 26-35 mm on a Maker arm at
   pitch -45 to -70 (fingertips 2-3 mm): the arm may touch the table.
6. Close. Read `gripper`: well above 0 means the jaws stopped on the object; near 0 means they closed on air. In the
   simulators a closed grip on the 4 cm cube reads about 0.4-0.5.
7. Lift straight up (`op nudge 0 0 0.06`) before moving sideways, and look: the object should rise with the jaws.

A Metal arm from `home` (tool at about [0.31, 0, 0.58], pointing forward and slightly up) needs several `move_tool`
steps to point straight down; each changes the pitch by about 20 deg. This sequence picks the cube at x = 0.28 in the
simulators:

```bash
llm-harness op home
llm-harness op move_tool 0.31 0 0.54 --pitch -5
llm-harness op move_tool 0.30 0 0.47 --pitch -25
llm-harness op move_tool 0.29 0 0.39 --pitch -45
llm-harness op move_tool 0.285 0 0.30 --pitch -65
llm-harness op move_tool 0.28 0 0.21 --pitch -85
llm-harness op move_tool 0.28 0 0.125 --pitch -90
llm-harness op gripper open
llm-harness op move_tool 0.28 0 0.035 --pitch -90
llm-harness op gripper close
llm-harness op nudge 0 0 0.10
```

The Maker arm cannot point straight down this close to its base. Near the table use a pitch of -45 to -60. Its
`home` already sits low (tool at about [0.32, 0, 0.12], pitch -38):

```bash
llm-harness op home
llm-harness op move_tool 0.30 0 0.125 --pitch -50
llm-harness op move_tool 0.28 0 0.125 --pitch -60
llm-harness op gripper open
llm-harness op move_tool 0.28 0 0.035 --pitch -60
llm-harness op gripper close
llm-harness op nudge 0 0 0.06
```

Set heights with `move_tool` to an absolute z when you know where you want to be; a chain of relative nudges from a
misremembered height ends somewhere else.

## Results and what to do next

| Status | What happened | What to do |
|---|---|---|
| `done` | the action finished; the numbers are measured. A tool move is done only within 10 mm and 3 deg of its target; `--json` gives how far off it ended under `result.error`. | look, then decide |
| `rejected` | refused before anything moved | read the message: it names the rule and usually the fix. Change the request; repeating it gets the same answer. |
| `stopped` | ended early: `op stop`, a joint more than 6 deg off its path (blocked, maybe by a collision), a measured point more than 10 mm under the floor or inside a keep-out zone, a motor over the temperature limit, a motion that did not arrive, a tool move that `ended X mm and Y deg from the target` (up to 30 mm that the arm's sag explains is `done`, with the miss in the message) or `ended N mm short of the target along the approach` (more than 10 mm: aim that much further, or use `--until contact`), or a rest that ended without release. The arm holds where it is. | look at every camera before the next move. If it was blocked, find out by what. |
| `fault` | an arm reports a fault | stop sending actions and tell the operator what you did and saw |

Common rejections:

| Message says | Do |
|---|---|
| `would move N deg in one action, over the 35 deg per-action limit` | split the motion: smaller position steps, or change the pitch by 20-30 deg at a time |
| `that pose is out of reach (the closest reachable pose is ...)` | choose a closer point or another pitch |
| `the straight path to that pose ends ... off it` | go through an intermediate point |
| `... under the 0 mm hard floor. From here the lowest point may come down at most N mm (its clearance) ...` | aim at most N mm lower than now (with `--until contact` too) |
| `would bring the ... into keep-out zone NAME ... Go around it` | the path crosses a zone (`op describe` gives its corners): move away from it first, then approach the target from another side |
| `inside keep-out zone NAME ... only motions that keep moving out of the zone` | the arm is already in the zone: move straight away from it first |
| `under the 0 mm hard floor once tracking error and sag are allowed for ... only motions that keep gaining height` | the arm is already too low: lift first, `op nudge 0 0 0.02` |
| `the motors turn on only at the rest pose` | the arm was moved while off: the operator folds it back to rest |
| `no rest pose is recorded` | the operator runs `llm-harness calibrate --rest` |
| `name the arm: left or right` | add `--arm left` or `--arm right` |
| `a motor is at N C, over the 70 C limit` | `op wait 30` a few times, then try again |
| `the table height is not known` | the operator has to run `llm-harness calibrate` |
| `is outside its range [lo, hi] deg` | stay inside the range `op describe` lists |

## Working within the envelope

- One action may move no joint more than 35 deg from where it started (`home` and `rest` are exempt). Large tool moves
  and large changes of pitch or yaw touch several joints at once: changing the pitch by more than about 20-30 deg in
  one action tends to exceed the cap. Split them; `r.reachable(poses, chain=True)` in the kernel checks a whole
  sequence in one call.
- Turning the gripper far about its axis: `op move_joints wrist_roll=30 --relative`, repeated.
- The hard floor (0 mm by default: the arm may touch the table) applies at every waypoint, not only at the end, to
  the elbow and wrist joints, the tool point, the fingertips and the tips of the fully open jaws (`op describe` gives
  both distances), whatever the gripper's actual opening. Pointing straight down the fingertips are lowest: a Metal
  tool point comes down to about 25 mm above the table, a Maker one (pitch -45 to -70) to about 26-35 mm. A held
  object and the links between those points are not checked. For a stricter envelope set `floor_margin_m`,
  `floor_tolerance_deg` (every joint that far off its plan) and `sag_allowance_m` under `safety:`.
- While moving, a joint more than 6 deg off its path, or a measured point more than 10 mm under the floor (pressing
  into the table), stops the motion and the arm holds.
- Keep-out zones (`op describe` lists each with its corners) are boxes no part of the arm may come within 30 mm of
  at any point of a motion, from the base along the links to the fingers and open jaws. The observation's
  `keep-out zone NAME D m away` shows how close the arm is. A refused path has to go around: move away from the
  zone, then approach from another side. An arm already inside may only move out; entering one while moving holds
  the arm.
- Joints move at up to 20 deg/s, and at 5 deg/s when a motion starts or passes within 76 mm of the table (the robot
  file may set other speeds; `op describe` states the ones in force).
- An arm already under the floor accepts only motions that keep gaining height.
- `rest` moves to the rest pose along a checked path, by way of `home` when the straight path would cross the floor.
  When both are refused nothing moves: lift, then rest again. An arm with a fault is not moved.
  The final approach to the recorded rest pose may go under the floor when the rest pose itself lies there, as long
  as nothing on the way is lower than the rest pose and it runs at the slow speed.

## Two arms

Name the arm in every motion (`--arm left`). `observe`, `wait` and `rest` without `--arm` cover both arms; `rest`
then rests each one. One action moves one arm; the other holds. The arms are not checked against each other: keep
them apart, and move one away before the other works near it. Both arms share the world frame, so a target means the
same point for either arm.

```bash
llm-harness op home --arm right
llm-harness op move_tool 0.30 -0.25 0.50 --arm right
llm-harness op rest
```

## Session control

| Command | Effect |
|---|---|
| `llm-harness op status` | robot, arms, cameras, step count, run directory, whether an action or a queued kernel action is running, the kernel's queue; touches no device |
| `llm-harness op describe` | the robot description |
| `llm-harness op stop` | ends the action in progress and any sent while it ran, and the kernel's running code and queue; answers once every arm that is on holds where it is, confirmed, or with what could not be confirmed. Ctrl-C on a waiting `op` sends it too. |
| `llm-harness rest` | every arm back to rest, motors off (`op rest --arm left` for one) |
| `llm-harness op quit` | ends the session: the server rests the arms, then the command prints each arm's outcome (exit 1 when one is not confirmed off) |
| `llm-harness py -c CODE`, `llm-harness py < file.py` | runs code in the session's Python kernel ([AGENT_GUIDE.md](AGENT_GUIDE.md)); `--timeout S` (300) bounds how long it may run, after which the kernel restarts empty |
| `llm-harness py --wait [S]` | waits up to S seconds (60) for the next report of a queued action, or until nothing runs or waits |

Finish a task with `llm-harness rest`, then report.

## The episode log and taking over

Every action is a line in `<home>/runs/<stamp>-<robot>/episode.jsonl` with its result and observation, and the images
are next to it. To take over a task someone else started:

```bash
llm-harness log resume       # robot, task notes, step count, last five actions, last observation and images
llm-harness log show         # every step: action -> result
llm-harness log profile      # where the time went: in steps, motion, idle gaps, per action, kernel actions
```

Both read the latest run, or the run directory given as an argument. The operator can continue the same episode in
a new session with `llm-harness serve --resume`; step numbers carry on.

## When something is wrong

- The images do not match the numbers: trust neither; `op observe` again, then ask the operator.
- `no session is running`: the operator has to start `llm-harness serve` (or `sim`).
- A camera line shows an error: carry on with the others only if they show what you need; tell the operator.
- `stopped` with "blocked": something is in the way, or the arm is touching it. Look; back off with a small nudge
  away from it.
- A grasp reads near 0: the jaws missed. Open, lift a little, re-align, try again.
- The same rejection twice: you are repeating the request. Change it.
- `fault`: stop and report. Do not try to clear it with more actions.

## Operator

### Starting a session on a real robot

```bash
llm-harness robot check bench        # read-only: joints, torque, temperatures, faults, rest pose, cameras, calibration
llm-harness calibrate --robot bench  # the table under each arm, then its folded rest pose
llm-harness serve --robot bench
```

`robot check` prints per arm whether its motors are on, off or not known, how far it stands from its recorded rest
pose, and the table height; it exits 1 and lists the problems when an arm reports a fault (torque left on by another
program counts) or a camera gives no image. The images are in `<home>/checks/`. `serve --read-only` lets an agent look
(`observe`, `wait`) without moving anything. `serve --address PATH` and `op --address PATH` run more than one session
side by side; the socket's directory must belong to you and be closed to others (`chmod 700`).

The arms are connected with their torque as it is. An arm's motors come on, holding its pose, with its first motion
or gripper action that the envelope accepts, and only within 5 deg (every joint) of its recorded rest pose; a refused
request leaves them off, and before the table is calibrated every action except `observe` and `wait` is refused.
Keep the workspace clear and a hand on the hardware emergency stop while an agent drives.

The session's Python kernel (`llm-harness py`, MCP `run_python`) runs model code as the user who started `serve`,
with all of that user's rights: it is not sandboxed. It runs in a separate process the session starts and can kill,
and reaches the robot only through the session's kernel channel, where everything that moves or queues needs an
authority the session revokes on `op stop`, a hang-up, a timeout or shutdown. `op stop` answers once every arm that
is on holds where it is, confirmed from fresh feedback, or answers what could not be confirmed. `op quit`, Ctrl-C or
SIGTERM on `serve`, or an error in it, revoke everything, hold and confirm, end the kernel process (killed after 1 s)
and only then rest and release the arms; nothing else can move them from then on, and the address stays taken until
each arm's outcome is printed. `mcp` and `run` without `--connect` print their session's address (`session at ...`) on
stderr, so `op stop --address` and `op quit --address` reach them; a quit there rests the arms and ends the process. Every connection must answer a challenge with the session key
(`<socket>.key`, readable only by you). The kernel is not available on Windows.

### Speed settings

Every setting that costs time lives in the robot file's `safety:` block (defaults in the README's safety section):
`max_speed_deg_s`, `slow_speed_deg_s`, `slow_zone_m`, `max_excursion_deg`, `settle_s`, `tracking_tolerance_deg` and
`tracking_lag_s`, plus `contact_tolerance_m` for `until contact`. When you raise the speeds, keep `tracking_lag_s` at
least as long as the arm trails its command, or fast motions stop as blocked. Compare episodes with
`llm-harness log profile`.

### Keep-out zones

A keep-out zone is a box in the world frame that neither arm may enter, for things the arms must not hit: a camera
mount between the arms, a wall, a fixture. The envelope keeps every point along each arm 30 mm
(`keep_out_margin_m`) outside every box, so measure the object itself and let the margin do its work.

Measure its corners in the world frame (with one arm at the default base, that arm's base frame: x forward, y left,
z up, origin at the base):

- With the arm: with the session stopped and the motors off, move the arm by hand until the closed fingertips touch
  each extreme of the object (its lowest x, highest x, and so on) and run `llm-harness robot check bench`; it prints
  the tool xyz. The fingertips reach past the tool point by the finger length (`op describe`), so add it along the
  pointing direction.
- With a tape measure: measure from the arm base's origin along its x and y axes; with two arms, add that arm's
  `bases:` offset.

Take the box a little larger than the object, up to its full height (a mount: from the table to above the highest
reach of the arm), and add it:

```bash
llm-harness robot keep-out add camera_mount --min -0.10 -0.42 0 --max 0.20 -0.21 1.2 --robot bench
```

`robot check` then reports an arm or a recorded rest pose that lies inside a zone. An arm inside a zone may only move
out of it, so a wrong box never traps an arm; `rest` may still end at a rest pose inside one.

### Gripper geometry

The envelope watches the fingertips and the tips of the open jaws at distances taken from a table per arm type
(the README's safety section). They are estimates, not your tool: measure on the installed gripper how far the
closed fingertips reach past the tool point along the pointing direction, and how far the fully open jaws reach to
each side of it along their closing direction, and set them in the robot file before you calibrate:

```yaml
safety: {finger_length_m: 0.036, jaw_half_width_m: 0.074}
```

### Calibration

The envelope needs the table height under each arm, in the world frame, and each arm's folded rest pose.
`llm-harness calibrate` opens the robot read-only, and every reading needs the motors to confirm torque off (an arm
whose torque is on or not known is refused: bring it down with the SDK first, `metal-arm rest --arm NAME`). For each
arm in turn:

1. Close the jaws, point the gripper down and move the arm by hand until the fingertips rest on the table. The arm
   has no brakes: support it.
2. Press Enter. Two readings 0.3 s apart must agree within 3 mm (three attempts); each needs fresh feedback from
   every motor, the gripper pitched below -30 deg and the jaws closed (opening at most 0.1), or nothing is stored.
3. The fingertip height is stored as that arm's table height in `<home>/calibration/<robot>.json`, together with the
   arm's bus, profile, base pose and finger length; a change to any of them makes the arm uncalibrated again.
4. Fold the arm back onto its rest support, as it should stand with the motors off, and answer `y`. Two readings
   0.3 s apart must agree within 0.5 deg; a joint resting more than 5 deg past its limit is refused. The pose is
   written to the robot file under `rest_q` (degrees).

| Option | Use |
|---|---|
| `--arm left` | one arm only |
| `--rest` | record only the rest pose (step 4) |
| `--yes` | record the rest pose without asking whether the arm is folded and supported (for unattended use: it must already be) |
| `--motors-off` | state that the motors are off, for an arm whose motors do not report their torque (older RobStride firmware). Check by hand that the arm moves freely first; motors that report torque on are still refused. |
| `--watch` | hands-free table touch: the reading is taken once the gripper has moved by more than 3 cm and then held still (within 3 mm) for 5 s; gives up after 8 minutes |
| `--table-z Z` | store a known table height without opening the robot: the world-frame z (metres) where the fingertips touch the table. A value more than 0.5 m from the arm base's height is refused. |
| `--powered` | the arm finds the table itself, from and back to its recorded rest pose: at three points straight ahead (0.25, 0.32, 0.39 m from the base; `--points X,Y ...` for others, clear of objects) the jaws close, the fingertips go above the point pointing down and then 2 mm at a time down to the table at 3 deg/s. Contact is a step after which the fingertips stood still more than 4 mm above their command (beyond the offset they held while descending freely); arm lag closes once a step ends, so it is not taken for contact. The fingertips are never sent more than 4 cm under `--expect Z` (default: the stored height, else the base's), and a table already above the start is refused. Stored as a table height per reach. Needs the rest pose recorded; refused with a motor at 50 C or more |

On an arm whose model sees a flat table at a height that changes with the reach (a Maker arm: about 1 cm between
0.26 and 0.40 m), measure it with `--powered`: the floor then follows the reach of each monitored point.

When the recorded rest pose lies under the floor for the measured table (within the hard floor once tracking error
and sag are allowed for), `calibrate` and `robot check` say so. That is allowed: the arm starts in recovery mode, where
it may only rise until it clears the floor, and `rest` may always end at the recorded rest pose, as long as no point of
the way down is lower than the rest pose itself and it runs at the slow speed.

Calibrate again when the table, an arm base, the gripper or the robot file's `bases:` change, and record the rest
pose again when the rest support moves.

### What happens on exit

`op quit`, Ctrl-C or SIGTERM on `serve` stop taking requests, end the action in progress and any waiting, then rest
every arm whose motors are on exactly as the `rest` action does: a checked path to the rest pose (by way of `home`
when needed), and torque off once the arm stands within 5 deg of it, confirmed motor by motor. Further Ctrl-C or
SIGTERM are ignored until that is over. `mcp` and `run` do the same when they end on a robot they opened, including
on SIGTERM. Each prints `Every arm is at rest with its motors off.` and exits 0, or prints every arm's outcome and
exits 1 when one is not confirmed off. Such an arm (a fault, a rest the envelope refused, a return that did not
arrive, torque left on by another program) is not moved: it holds its pose with the motors on. On a terminal, `serve`
keeps holding it and waits: Enter leaves it holding and ends the session, `RELEASE` (once it is supported) turns its
torque off there. The printed command, `metal-arm rest --arm NAME --pose <its rest pose>` (or `maker-arm`), then brings
it down and turns torque off; `release` would turn torque off where it is and drop it. `llm-harness serve --attach`
instead takes over an arm left holding where it is, and `rest` brings it down along a checked path.
