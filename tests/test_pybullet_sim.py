"""MPC planner and leg IK of the PyBullet simulation."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from mpc_planner import MPCPlanner  # noqa: E402


@pytest.fixture(scope="module")
def planner():
    return MPCPlanner(h_target=0.2, dt_mpc=0.02)


def test_mpc_respects_zmp_bounds(planner):
    refs = planner.get_references_y(0.5)
    x, _ = planner.solve_mpc(0.0, 0.0, *refs)
    zmp, zmp_min, zmp_max = planner.v_zmp.value, refs[2], refs[3]
    assert np.all(zmp >= zmp_min - 1e-4) and np.all(zmp <= zmp_max + 1e-4)
    # The returned state is the first step of the optimal prediction
    assert x == pytest.approx(planner.v_x.value[1])


def test_mpc_follows_forward_reference(planner):
    x, v, t = 0.0, 0.0, 0.0
    for _ in range(100):  # 2 s
        x, v = planner.solve_mpc(x, v, *planner.get_references_x(t, 0.1))
        t += planner.dt_mpc
    assert x == pytest.approx(0.1 * t, abs=0.03)


def test_feet_alternate_support(planner):
    r0, l0 = planner.get_foot_trajectories(planner.T_STEP * 0.5, 0.1)
    r1, l1 = planner.get_foot_trajectories(planner.T_STEP * 1.5, 0.1)
    assert l0[2] == 0.0 and r0[2] > 0.0   # step 0: left support, right swings
    assert r1[2] == 0.0 and l1[2] > 0.0   # step 1: right support, left swings


def test_leg_ik_moves_ankle_vertically():
    """Raising the IK target by 2 cm raises the PyBullet ankle by 2 cm, without drift."""
    p = pytest.importorskip("pybullet")
    from kinematics import LegIK
    from main import URDF_PATH

    client = p.connect(p.DIRECT)
    try:
        robot = p.loadURDF(URDF_PATH, [0, 0, 1], useFixedBase=True)
        ik = LegIK(robot)

        def ankle_pos(dz):
            ik.reset(ik.solve_leg(0.0, -0.0425, dz, False), ik.solve_leg(0.0, 0.0425, dz, True))
            link_state = p.getLinkState(robot, ik.right_leg[4], computeForwardKinematics=True)
            return np.array(link_state[4])  # world position of the ank_pitch link frame

        delta = ankle_pos(-0.18) - ankle_pos(-0.20)
        np.testing.assert_allclose(delta, [0.0, 0.0, 0.02], atol=2e-3)
    finally:
        p.disconnect(client)
