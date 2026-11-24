# Agent guide: driving the arm from the Python kernel

For a model (or a person) driving a session with code. Every name and command here exists as written; the Python
blocks run in order on `llm-harness sim`.

## How an episode spends its time

- One call is one model round trip. The arm moves on the wall clock between your calls, so the fast episodes keep it
  busy: while one action runs, check the next one and queue it.
- Code runs in one Python kernel kept by the session, in a process of its own that reaches the robot only through the
  session. Variables, functions and imports stay between calls: define helpers once and call them by name. A call
  that runs longer than its limit (`llm-harness py --timeout`, 300 s) restarts the kernel empty.
- A motion is a function `fn(r)` that you queue. `kernel.queue(fn)` returns at once; a worker runs it. Each action
  that ends produces one report (how it ended, how long it took, what it printed, the queue, each arm's state and a
  camera grid), delivered with your next call.
- The cycle: look, measure, check reach of the whole plan in one call, queue the action, then check and queue the
  next while it runs. Read every report before deciding what comes next.

## Being fast

The episode is timed from your first call to the arm at rest. Most of that time is yours, not the arm's.

- **Queue a whole phase as one action**: approach, descend, close, check the hold, lift. Do not queue one move and
  wait for its report.
- **While an action runs, prepare the next one.** Check its poses with `r.reachable` and queue it before the first
  one ends. Wait (`--wait`) only when nothing is left to prepare.
- **Check the whole plan before the first move**, the place pose included, in one `r.reachable` call. Straight down
  (`pitch=-90`) is only reachable close in front of the base; further out or to the side use -60 to -80.
- **Leave `speed=` and `duration=` out.** The harness already limits the speed and slows the arm near the table;
  passing a lower one only costs time. Use them for a motion that must be gentle, nothing else.
- **A refusal is fixed in the same call**: try the alternatives with one `r.reachable` and queue the best. Never
  stop to think with the arm holding something in the air.
- **Look once per decision.** `r.home()` leaves the wrist camera looking down at the table in front of the arm;
  one `r.look()` there shows the scene.

## Running code

```bash
llm-harness py <<'PY'
print(r.state().text())
PY
llm-harness py -c 'print(kernel.status())'
llm-harness py --wait 30
```

- In the namespace: `r` (the robot), `np`, `math`, `kernel`, and `MotionFailed` and `Stopped` (the exceptions a
  motion raises).
- Each call prints what your code printed (and the value of a last bare expression), then `[state]` per arm,
  `[queue]`, and every report that arrived since the last call. Read them instead of asking again.
- `--wait S` (no code) waits up to S seconds for the next report, or until nothing runs or waits. Use it instead of
  sleeping or polling when there is nothing to prepare.
- Through MCP or `llm-harness run` the same kernel is the `run_python` tool: `{"code": "...", "wait_s": 30}`.
- `llm-harness op stop` halts the arm where it is and drops the queue at any moment.

## First call: look and measure

```python
print(r.state().text())
print(r.look())
```

`r.state()` gives the arms with numbers (`r.state().arms["arm"].tool.xyz`); `r.look()` saves one labelled grid (each
camera at its own size up to 640x480 with one or two cameras) of
every camera and returns its path. `llm-harness op describe` gives the robot, its frames and the envelope.

## Define helpers once

```python
def down(x, y, z):
    """A pose with the jaws pointing straight down."""
    return {"xyz": [x, y, z], "pitch": -90}

def go(r, poses, **options):
    """Move through poses in order; each is a dict with "xyz" and optional "pitch", "yaw", "roll"."""
    for p in poses:
        r.move(p["xyz"], pitch=p.get("pitch"), yaw=p.get("yaw"), roll=p.get("roll"), **options)
```

## Check reach in one call

```python
approach = [{"xyz": [0.31, 0, 0.54], "pitch": -5}, {"xyz": [0.30, 0, 0.47], "pitch": -25},
            {"xyz": [0.29, 0, 0.39], "pitch": -45}, {"xyz": [0.285, 0, 0.30], "pitch": -65},
            {"xyz": [0.28, 0, 0.21], "pitch": -85}, down(0.28, 0, 0.12)]
for pose, check in zip(approach, r.reachable(approach, chain=True, start="home")):
    print(pose["xyz"], check["ok"], check["why"] or f"{check['seconds']} s, {check['margin_deg']} deg spare")
```

- `r.reachable(poses)` checks each pose exactly as `r.move` would (IK, joint limits, table floor, keep-out zones,
  the per-action joint travel limit), moves nothing, and does not disturb a running action. Per pose: `ok`, `why`
  (the refusal a move would get), `margin_deg` and `margin_joint` (the goal's nearest joint limit), `joints_deg`,
  `seconds`.
- Poses are checked from where the arm is, or from where its running motion ends; `start="home"` or `start=pose`
  checks from there, and `chain=True` checks each pose from the one before it.
- `line=False` checks a joint-space move (the default `line=True` is a straight line of the tool point).
- Any motion takes `dry_run=True`: `r.move([0.28, 0, 0.12], pitch=-90, dry_run=True)` returns the Result it would
  get, without moving.
- One action turns no joint more than the per-action limit (`max_excursion_deg`, in `llm-harness op describe`):
  split long moves into several poses, as `approach` does.

## Write an action

```python
def to_cube(r):
    r.home()
    go(r, approach)

kernel.queue(to_cube)
```

While it runs, write the next ones:

```python
def grasp(r):
    r.gripper("open")
    r.move([0.28, 0, 0.035], pitch=-90, until="contact")
    if not r.gripper("close").grip["holding"]:
        raise RuntimeError("the jaws closed on nothing")

def lift(r):
    r.nudge([0, 0, 0.08])

kernel.queue([grasp, lift])
```

- A motion that does not end done raises `MotionFailed` (its `.result` says why), which ends the action with an
  error and drops what waits behind it. A check at the top of an action (`raise`) stops it before it moves.
- `until="contact"` ends a move or nudge early, holding, where the measured arm stops following its command; the
  result says so (`result.contact["xyz"]`, `result.contact["short_m"]`). Use it instead of stepping down a
  centimetre at a time.
- `r.gripper(...)` returns `grip` with `opening` (measured), `target` and `holding` (the jaws closed and then
  stalled on something). A close that stalls without closing far enough to tell, or an opening that does not open,
  raises `MotionFailed` (stopped): look before going on.
- `r.move_joints(...)` keeps the joints it does not name at their command. A request of under 0.5 deg that the
  joint does not visibly follow ends stopped as undetermined.
- `speed=` (joint deg/s) or `duration=` (seconds) slows a move; nothing goes faster than the envelope allows.
- Motions come only from a call or a queued action: a thread your code starts cannot move the arm. Inside an action,
  use the `r` it is given. Code inside an action runs later, on fresh readings: `r.state()` there
  reads the arm as the action runs.
- With two arms, every motion names its arm: `arm="left"`.

## Pipelining

- At most three actions wait behind the running one. `kernel.queue([f, g])` queues several in one call.
- `kernel.queue(new, replace_from=id)` replaces that waiting action and everything behind it (`queue` returns the
  ids). `kernel.flush()` drops what waits; `kernel.stop()` halts the arm, drops the queue and ends the code that
  called it (with `Stopped`). `kernel.status()` says
  what runs and what waits.
- An error or a stop drops everything queued and the arm holds. Read the report, then queue the fix and the rest of
  the plan again in one call.
- A motion sent from your own call (outside an action) while an action runs stops that action first and drops the
  queue: put motions in actions.
- What a picture shows is older than what happens now: plan from where the queued actions will leave the arm.

## When something fails

| Report says | What to do |
|---|---|
| `rejected: ... out of reach` | try other heights or pitches in one `r.reachable` call, queue the best |
| `rejected: ... over the 35 deg per-action limit (safety setting max_excursion_deg)` | split the move into intermediate poses |
| `rejected: ... hard floor` | aim higher; the fingertips may touch the table, never go below it |
| `rejected: ... keep-out zone` | go around the zone |
| `stopped: ... blocked` | the arm hit something and holds: look, then move away |
| `fault: ...` | stop sending motions and tell the operator |

## Finishing

```python
def put_back(r):
    r.nudge([0, 0, -0.08], until="contact")
    r.gripper("open")
    r.nudge([0, 0, 0.1])
    r.home()

kernel.queue(put_back)
```

When its report comes, look once (`r.look()`), then `r.rest()` in its own call or end the session.
`llm-harness log profile` shows where the episode's time went.
