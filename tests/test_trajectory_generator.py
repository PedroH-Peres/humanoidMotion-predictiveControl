"""Properties of the analytical walking trajectories (no simulator needed)."""

import numpy as np
import pytest

from mujoco_sim.analytical_walking.trajectory_generator import (
    FootPose,
    Pose2D,
    get_com_pose_at_time,
    get_swing_foot_pose_at_time,
    select_next_poses,
)
from mujoco_sim.analytical_walking.walking_engine import WalkingEngine, WalkState
from mujoco_sim.config import WALK_PARAMS

T, DS, Z_COM, Z_STEP, G = 0.4, 0.12, 0.22, 0.04, 9.81


def _sample_swing(n=4001):
    start = Pose2D(np.array([0.0, 0.054]), 0.0)
    end = Pose2D(np.array([0.04, 0.054]), 0.2)
    ts = np.linspace(0.0, T, n)
    return ts, np.array([get_swing_foot_pose_at_time(t, T, Z_STEP, DS, start, end)[0] for t in ts])


def test_swing_foot_never_goes_below_the_floor():
    _, pos = _sample_swing()
    assert pos[:, 2].min() >= 0.0


def test_swing_foot_height_is_continuous():
    ts, pos = _sample_swing()
    max_jump = np.abs(np.diff(pos[:, 2])).max()
    # Peak vertical speed of the cosine bump is pi * Z_STEP / T_ss; allow 2x margin
    t_ss = T * (1 - DS)
    allowed = 2 * np.pi * Z_STEP / t_ss * (ts[1] - ts[0])
    assert max_jump < allowed


def test_swing_foot_starts_and_ends_on_target():
    _, pos = _sample_swing()
    np.testing.assert_allclose(pos[0], [0.0, 0.054, 0.0], atol=1e-9)
    np.testing.assert_allclose(pos[-1], [0.04, 0.054, 0.0], atol=1e-9)
    assert pos[:, 2].max() == pytest.approx(Z_STEP, rel=0.02)


def test_com_satisfies_boundary_conditions():
    start = Pose2D(np.array([0.0, 0.0]))
    support = Pose2D(np.array([0.0, -0.054]))
    end = Pose2D(np.array([0.02, 0.0]))
    com0, _ = get_com_pose_at_time(0.0, T, DS, Z_COM, G, start, end, support)
    comT, _ = get_com_pose_at_time(T, T, DS, Z_COM, G, start, end, support)
    np.testing.assert_allclose(com0, start.position, atol=1e-9)
    np.testing.assert_allclose(comT, end.position, atol=1e-9)


def test_com_moves_towards_support_foot():
    start = Pose2D(np.array([0.0, 0.0]))
    support = Pose2D(np.array([0.0, -0.054]))
    end = Pose2D(np.array([0.0, 0.0]))
    com_mid, _ = get_com_pose_at_time(T / 2, T, DS, Z_COM, G, start, end, support)
    assert com_mid[1] < -0.01


def test_select_next_poses_rotates_command_into_torso_frame():
    torso = Pose2D(np.array([0.0, 0.0]), np.pi / 2)   # facing +y
    swing = FootPose(np.array([0.0, 0.0]), np.pi / 2, is_left=True)
    next_torso, next_swing = select_next_poses(torso, swing, {"vx": 0.1}, T, 0.054)
    np.testing.assert_allclose(next_torso.position, [0.0, 0.1 * T], atol=1e-12)
    # Left foot lands to the left of the torso, i.e. towards -x when facing +y
    assert next_swing.position[0] < next_torso.position[0]
    assert next_swing.is_left


def test_engine_walks_and_stops():
    engine = WalkingEngine(**WALK_PARAMS)
    dt = 0.008
    engine.set_command(vx=0.05)
    for _ in range(int(2.0 / dt)):
        engine.step(dt)
    assert engine.state == WalkState.WALKING
    assert engine.torso.position[0] > 0.05

    engine.request_stop()
    for _ in range(int(WALK_PARAMS["T"] / dt) + 2):
        target = engine.step(dt)
    assert engine.state == WalkState.IDLE
    assert target.left_sole[2] == 0.0 and target.right_sole[2] == 0.0


def test_march_in_place_starts_from_idle_and_stays_put():
    engine = WalkingEngine(**WALK_PARAMS)
    dt = 0.008
    engine.march_in_place()
    lifted = 0.0
    for _ in range(int(2.0 / dt)):
        target = engine.step(dt)
        lifted = max(lifted, target.left_sole[2], target.right_sole[2])
    assert engine.state == WalkState.IDLE_MARCH
    assert lifted > 0.03                       # the feet do leave the ground
    np.testing.assert_allclose(engine.torso.position, [0.0, 0.0], atol=1e-12)
