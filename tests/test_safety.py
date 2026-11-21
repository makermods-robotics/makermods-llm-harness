import math

import numpy as np
import pytest

from makermods_llm_harness.arm_model import GRIPPERS, ArmModel, Gripper
from makermods_llm_harness.backends.base import ArmInfo
from makermods_llm_harness.safety import Envelope, MoveRejected, SafetyConfig

# A strict envelope (the defaults let the arm touch the table): the floor rules are tested with it.
STRICT = dict(floor_margin_m=0.010, floor_tolerance_deg=2.0, sag_allowance_m=0.005)
CFG = SafetyConfig(**STRICT)
RATE = CFG.rate_hz
MODEL = ArmModel(ArmInfo("arm", "metal"))
# Pointing straight down above the table: the fingertips are the lowest monitored point.
HOME = np.array(MODEL.chain.ik(MODEL.pose_from_angles([0.28, 0, 0.15], -math.pi / 2, 0, 0), MODEL.home_q,
                               limits=(MODEL.ik.lo, MODEL.ik.hi)).q)
HOME_LOW = float(MODEL.lowest_z(HOME))
# How far below its nominal height the envelope takes the lowest point near HOME: tracking tolerance and sag.
ALLOW = HOME_LOW - MODEL.lowest_z_within(HOME, math.radians(CFG.floor_tolerance_deg)) + CFG.sag_allowance_m


def envelope(clearance_at_home=0.30, **overrides) -> Envelope:
    """An envelope whose table sits ``clearance_at_home`` below the start pose's lowest point."""
    return Envelope(MODEL, SafetyConfig(**{**STRICT, **overrides}), HOME_LOW - clearance_at_home)


def joint_path(q0, q1, speed_deg=CFG.max_speed_deg_s):
    return MODEL.plan_joint_path(q0, q1, math.radians(speed_deg), RATE)


def tool_path(q0, dz, speed_deg):
    goal = MODEL.tool_pose(q0)
    goal[2, 3] += dz
    return MODEL.plan_tool_path(q0, goal, math.radians(speed_deg), RATE)


def moved(q, joint, deg):
    out = np.array(q, dtype=float)
    out[MODEL.joint_names.index(joint)] += math.radians(deg)
    return out


# ── config ──
def test_config_defaults_and_validation():
    default = SafetyConfig()
    assert default.max_speed_deg_s == 20 and default.slow_speed_deg_s == 5 and default.max_excursion_deg == 35
    assert default.floor_margin_m == 0 and default.floor_tolerance_deg == 0 and default.sag_allowance_m == 0
    assert default.tracking_tolerance_deg == 6 and default.contact_allowance_m == 0.010
    assert default.finger_length_m is None
    assert CFG.gripper("maker") == GRIPPERS["maker"]
    assert SafetyConfig(finger_length_m=0.05).gripper("metal").finger_length_m == 0.05
    assert SafetyConfig.from_dict({"rate_hz": 30}).rate_hz == 30
    with pytest.raises(ValueError, match="unknown settings"):
        SafetyConfig.from_dict({"speed": 1})
    for bad in ({"max_speed_deg_s": 0}, {"slow_speed_deg_s": 30}, {"floor_margin_m": float("nan")},
                {"rate_hz": "fast"}):
        with pytest.raises(ValueError):
            SafetyConfig(**bad)
    assert SafetyConfig(floor_margin_m=0).floor_margin_m == 0


# ── no table ──
def test_a_table_height_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite"):
        Envelope(MODEL, CFG, float("nan"))


def test_without_a_table_height_motion_is_refused_naming_calibrate():
    env = Envelope(MODEL, CFG, None)
    assert env.clearance(HOME) is None
    with pytest.raises(MoveRejected, match="llm-harness calibrate"):
        env.check(HOME, joint_path(HOME, moved(HOME, "wrist_roll", 5)))


# ── rule 1 ──
def test_accepts_a_safe_path():
    env = envelope()
    env.check(HOME, joint_path(HOME, moved(HOME, "wrist_flex", -20)))


@pytest.mark.parametrize("where", ["path", "start"])
def test_rejects_non_finite_values(where):
    env = envelope()
    path = joint_path(HOME, moved(HOME, "wrist_flex", -5))
    start = HOME.copy()
    if where == "path":
        path[3, 2] = np.nan
    else:
        start[0] = np.inf
    with pytest.raises(MoveRejected, match="not a finite number"):
        env.check(start, path)


def test_rejects_joint_limits_naming_the_joint():
    env = envelope()
    q = HOME.copy()
    q[4] = MODEL.hi[4] - math.radians(1)
    with pytest.raises(MoveRejected, match=r"wrist_yaw would go to .* outside its range \[-85, 85\]"):
        env.check(q, joint_path(q, moved(q, "wrist_yaw", 3)))
    env.check(q, joint_path(q, moved(q, "wrist_yaw", 1)))


def test_a_start_past_a_limit_may_stay_or_move_inward_but_not_further_out():
    env = envelope()
    q = HOME.copy()
    q[4] = MODEL.hi[4] + math.radians(0.01)
    env.check(q, joint_path(q, moved(q, "wrist_yaw", -3)))
    env.check(q, np.repeat(q[None], 3, axis=0))
    with pytest.raises(MoveRejected, match=r"wrist_yaw would go to 85.0 deg, outside its range"):
        env.check(q, joint_path(q, moved(q, "wrist_yaw", 1)))
    q[4] = MODEL.lo[4] - math.radians(0.01)
    env.check(q, joint_path(q, moved(q, "wrist_yaw", 2)))
    with pytest.raises(MoveRejected, match="wrist_yaw"):
        env.check(q, joint_path(q, moved(q, "wrist_yaw", -0.5)))


# ── rule 2 ──
def test_excursion_cap_rejects_over_and_accepts_under():
    env = envelope()
    with pytest.raises(MoveRejected, match="wrist_roll would move 40 deg .* 35 deg"):
        env.check(HOME, joint_path(HOME, moved(HOME, "wrist_roll", 40)))
    env.check(HOME, joint_path(HOME, moved(HOME, "wrist_roll", 34)))


def test_excursion_cap_counts_how_far_a_joint_passes_not_where_it_ends():
    env = envelope()
    there = joint_path(HOME, moved(HOME, "wrist_roll", 40))
    back = joint_path(there[-1], moved(HOME, "wrist_roll", 10))
    with pytest.raises(MoveRejected, match="wrist_roll would move 40 deg"):
        env.check(HOME, np.vstack([there, back]))


def test_home_and_rest_are_exempt_from_the_excursion_cap():
    env = envelope()
    path = joint_path(HOME, moved(HOME, "wrist_roll", 90))
    with pytest.raises(MoveRejected):
        env.check(HOME, path)
    env.check(HOME, path, capped=False)


# ── rule 3 ──
def test_hard_floor_rejects_a_path_that_goes_under_the_margin():
    env = envelope(clearance_at_home=0.10)
    path = tool_path(HOME, -0.095, CFG.slow_speed_deg_s)
    with pytest.raises(MoveRejected, match=r"the gripper fingertips to 5 mm above the table \(-\d+ mm allowing for "
                                           r"tracking error and sag\), under the 10 mm hard floor\. Aim"):
        env.check(HOME, path)
    env.check(HOME, tool_path(HOME, -0.09 + ALLOW, CFG.slow_speed_deg_s))


def test_hard_floor_allows_for_tracking_error_and_sag():
    """A path whose nominal clearance clears the floor by less than the allowance is refused: one joint 2 deg off
    its plan would put the arm under the floor."""
    assert ALLOW > 0.01
    env = envelope(clearance_at_home=0.10)
    near = tool_path(HOME, -0.09 + ALLOW / 2, CFG.slow_speed_deg_s)
    assert float(np.min(env.clearance(near))) > CFG.floor_margin_m
    with pytest.raises(MoveRejected, match="allowing for tracking error and sag"):
        env.check(HOME, near)
    loose = SafetyConfig(floor_margin_m=0.010)
    Envelope(MODEL, loose, env.table_z).check(HOME, near)


def test_hard_floor_rejects_even_when_the_end_is_above_it():
    env = envelope(clearance_at_home=0.10)
    down = tool_path(HOME, -0.095, CFG.slow_speed_deg_s)
    up = tool_path(down[-1], 0.09, CFG.slow_speed_deg_s)
    with pytest.raises(MoveRejected, match="hard floor"):
        env.check(HOME, np.vstack([down, up]))


def down_to(xyz, pitch_deg, roll_deg, height=0.05):
    """A slow joint path descending ``height`` onto the tool pose (xyz, pitch, roll), and its start."""
    def solve(p):
        pose = MODEL.pose_from_angles(p, math.radians(pitch_deg), 0, math.radians(roll_deg))
        sol = MODEL.chain.ik(pose, MODEL.home_q, limits=(MODEL.ik.lo, MODEL.ik.hi))
        assert sol.ok
        return np.array(sol.q)
    above, end = solve(np.add(xyz, [0, 0, height])), solve(xyz)
    return above, joint_path(above, end, CFG.slow_speed_deg_s)


@pytest.mark.parametrize("pitch, roll, tool_clearance", [(-90, 0, 0.020), (-45, 90, 0.030)])
def test_hard_floor_watches_the_fingertips_and_the_open_jaws(pitch, roll, tool_clearance):
    xyz = (0.28, 0.0, 0.15)
    above, path = down_to(xyz, pitch, roll)
    assert np.allclose(MODEL.tool_pose(path[-1])[:3, 3], xyz, atol=1e-4)
    table = xyz[2] - tool_clearance
    with pytest.raises(MoveRejected, match=r"gripper fingertips (down )?to -?\d+ mm"):
        Envelope(MODEL, CFG, table).check(above, path)
    exact = SafetyConfig(floor_margin_m=0.010)
    bare = ArmModel(ArmInfo("arm", "metal"), Gripper(0.0, 0.0, 0.0))
    Envelope(bare, exact, table).check(above, path)
    if pitch == -45:
        # The fingertip centre clears the floor here; only the lower open-jaw tip goes under it.
        no_jaws = ArmModel(ArmInfo("arm", "metal"), Gripper(GRIPPERS["metal"].finger_length_m, 0.0, 0.0))
        Envelope(no_jaws, exact, table).check(above, path)
        with pytest.raises(MoveRejected, match="fingertips"):
            Envelope(ArmModel(ArmInfo("arm", "metal"), Gripper(GRIPPERS["metal"].finger_length_m, 0.04, 0.0)),
                     exact, table).check(above, path)


# ── rule 4 ──
def test_recovery_under_the_margin_allows_rising():
    env = envelope(clearance_at_home=0.004)
    env.check(HOME, tool_path(HOME, 0.03, CFG.slow_speed_deg_s))


def test_recovery_under_the_margin_rejects_any_descent():
    env = envelope(clearance_at_home=0.004)
    with pytest.raises(MoveRejected, match=r"4 mm above the table, under the 10 mm hard floor.*lift first"):
        env.check(HOME, tool_path(HOME, -0.002, CFG.slow_speed_deg_s))


def test_recovery_rejects_tiny_descents_that_would_add_up():
    env = envelope(clearance_at_home=0.004)
    with pytest.raises(MoveRejected, match="keep gaining height"):
        env.check(HOME, tool_path(HOME, -0.00005, CFG.slow_speed_deg_s))


def test_recovery_allows_staying_put():
    env = envelope(clearance_at_home=0.004)
    env.check(HOME, np.repeat(HOME[None], 5, axis=0))


def test_recovery_rejects_a_rise_that_descends_again_under_the_floor():
    env = envelope(clearance_at_home=0.004)
    up = tool_path(HOME, 0.004, CFG.slow_speed_deg_s)
    down = tool_path(up[-1], -0.002, CFG.slow_speed_deg_s)
    with pytest.raises(MoveRejected, match="keep gaining height"):
        env.check(HOME, np.vstack([up, down]))


def test_recovery_holds_the_normal_floor_once_above_it():
    env = envelope(clearance_at_home=0.004)
    up = tool_path(HOME, 0.08, CFG.slow_speed_deg_s)
    env.check(HOME, up)
    down = tool_path(up[-1], -0.06, CFG.slow_speed_deg_s)
    with pytest.raises(MoveRejected):
        env.check(HOME, np.vstack([up, down]))


def test_recovery_rejects_a_rise_that_first_dips():
    env = envelope(clearance_at_home=0.004)
    down = tool_path(HOME, -0.001, CFG.slow_speed_deg_s)
    up = tool_path(down[-1], 0.03, CFG.slow_speed_deg_s)
    with pytest.raises(MoveRejected, match="lift first"):
        env.check(HOME, np.vstack([down, up]))


# ── rule 5 and the slow zone ──
def test_speed_for_is_slow_when_the_start_or_any_waypoint_is_in_the_slow_zone():
    high = envelope(clearance_at_home=0.30)
    assert high.speed_for(HOME, joint_path(HOME, moved(HOME, "wrist_roll", 10))) == high.max_speed
    starts_low = envelope(clearance_at_home=0.05)
    assert starts_low.speed_for(HOME, tool_path(HOME, 0.05, 20)) == starts_low.slow_speed
    ends_low = envelope(clearance_at_home=0.10)
    assert ends_low.speed_for(HOME, tool_path(HOME, -0.04, 20)) == ends_low.slow_speed


def test_a_fast_plan_in_the_slow_zone_is_rejected_and_the_slow_replan_passes():
    env = envelope(clearance_at_home=0.10)
    fast = tool_path(HOME, -0.04, CFG.max_speed_deg_s)
    with pytest.raises(MoveRejected, match="deg/s limit near the table"):
        env.check(HOME, fast)
    slow = tool_path(HOME, -0.04, CFG.slow_speed_deg_s)
    env.check(HOME, slow)
    steps = np.abs(np.diff(np.vstack([HOME, slow]), axis=0))
    assert steps.max() <= math.radians(CFG.slow_speed_deg_s) / RATE + 1e-12
    assert len(slow) > 3 * len(fast)


def test_a_slow_plan_ending_in_the_slow_zone_that_started_high_still_runs_slow():
    env = envelope(clearance_at_home=0.09)
    with pytest.raises(MoveRejected, match="near the table"):
        env.check(HOME, tool_path(HOME, -0.03, CFG.max_speed_deg_s))
    env.check(HOME, tool_path(HOME, -0.03, CFG.slow_speed_deg_s))


def test_per_tick_speed_is_checked_far_from_the_table():
    env = envelope()
    target = moved(HOME, "wrist_roll", 10)
    with pytest.raises(MoveRejected, match="wrist_roll at .* over the 20 deg/s limit"):
        env.check(HOME, np.vstack([joint_path(HOME, moved(HOME, "wrist_roll", 2)), target]))
    env.check(HOME, joint_path(HOME, target))


def test_the_default_envelope_lets_the_fingertips_touch_the_table_but_not_go_under_it():
    env = Envelope(MODEL, SafetyConfig(), HOME_LOW - 0.03)
    env.check(HOME, tool_path(HOME, -0.028, 5))
    with pytest.raises(MoveRejected, match="under the 0 mm hard floor"):
        env.check(HOME, tool_path(HOME, -0.032, 5))
    assert env.hold_floor(HOME) == pytest.approx(-0.010)
