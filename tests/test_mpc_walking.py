"""MPC walking engine: prediction model, box-QP solver, stability and continuity (no simulator needed)."""

import numpy as np
import pytest

import mujoco_sim.mpc_walking.mpc_walking_engine as mpc_module
from mujoco_sim.config import WALK_PARAMS
from mujoco_sim.mpc_walking.mpc_walking_engine import (
    MPCParams,
    MPCWalkingEngine,
    lipm_matrices,
    prediction_matrices,
    solve_box_qp,
)

DT = 0.008


def test_prediction_matrices_match_stepping_the_model():
    A, B, C = lipm_matrices(0.04, 0.22)
    Ps, Pu = prediction_matrices(A, B, C, 12)
    rng = np.random.default_rng(0)
    s0, U = rng.normal(size=3), rng.normal(size=12)

    s, zmp = s0.copy(), []
    for u in U:
        s = A @ s + B * u
        zmp.append(C @ s)

    np.testing.assert_allclose(Ps @ s0 + Pu @ U, zmp, atol=1e-12)


def test_solve_box_qp_returns_the_unconstrained_optimum_when_it_fits():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(6, 6))
    H, q = R @ R.T + np.eye(6), rng.normal(size=6)
    x = solve_box_qp(H, q, -np.full(6, 1e3), np.full(6, 1e3))
    np.testing.assert_allclose(x, np.linalg.solve(H, -q), atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_solve_box_qp_satisfies_the_optimality_conditions(seed):
    """x is the optimum iff x = clip(x - gradient, lo, hi) (checked with active bounds)."""
    rng = np.random.default_rng(seed)
    n = 20
    R = rng.normal(size=(n, n))
    H, q = R @ R.T / n + np.eye(n), rng.normal(size=n) * 5
    lo, hi = -rng.uniform(0.05, 0.5, n), rng.uniform(0.05, 0.5, n)

    x = solve_box_qp(H, q, lo, hi)

    assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
    assert np.any((x <= lo + 1e-9) | (x >= hi - 1e-9))            # the test does exercise the bounds
    np.testing.assert_allclose(x, np.clip(x - (H @ x + q), lo, hi), atol=1e-8)


def _walk(commands, seconds, params=None):
    """Run the engine; commands maps a time [s] to a callable(engine). Yields (t, target)."""
    engine = MPCWalkingEngine(**WALK_PARAMS, mpc=params or MPCParams())
    pending = sorted(commands.items(), key=lambda kv: kv[0])
    for i in range(int(seconds / DT)):
        while pending and pending[0][0] <= i * DT:
            pending.pop(0)[1](engine)
        yield engine, engine.step(DT)


def test_walking_stays_stable_and_close_to_the_nominal_torso():
    cmds = {0.0: lambda e: e.set_command(vx=0.05, vy=0.02),
            10.0: lambda e: e.set_command(wz=0.5),
            20.0: lambda e: e.set_command(vx=-0.05)}
    worst = 0.0
    for engine, target in _walk(cmds, 30):
        assert np.all(np.isfinite(target.body_pos))
        worst = max(worst, np.abs(engine._state[:, 0] - engine.torso.position).max())
    assert worst < 0.08            # the CoM sways around the nominal torso, it does not drift away


def test_planned_zmp_never_leaves_the_support_polygon():
    narrow = MPCParams(foot_half_width=0.0125, foot_half_length=0.03, zmp_margin=0.01)
    for engine, _ in _walk({0.0: lambda e: e.set_command(vx=0.1)}, 8, narrow):
        if engine.debug:
            d = engine.debug
            assert np.all(d["zmp_plan"] >= d["p_min"] - 1e-9)
            assert np.all(d["zmp_plan"] <= d["p_max"] + 1e-9)


def test_stopping_does_not_make_the_pelvis_jump():
    cmds = {0.0: lambda e: e.set_command(vx=0.05), 4.0: lambda e: e.request_stop()}
    prev, worst = None, 0.0
    for _, target in _walk(cmds, 8):
        if prev is not None:
            worst = max(worst, np.linalg.norm(target.body_pos[:2] - prev))
        prev = target.body_pos[:2].copy()
    assert worst < 0.005           # per 8 ms tick; snapping to the torso used to jump ~2 cm


def test_com_settles_on_the_torso_after_a_stop():
    cmds = {0.0: lambda e: e.set_command(vx=0.05), 4.0: lambda e: e.request_stop()}
    for engine, target in _walk(cmds, 8):
        pass
    np.testing.assert_allclose(target.body_pos[:2], engine.torso.position, atol=2e-3)


def test_idle_engine_matches_the_standing_pose():
    engine = MPCWalkingEngine(**WALK_PARAMS)
    target, standing = engine.step(DT), engine.standing_pose()
    np.testing.assert_allclose(target.body_pos, standing.body_pos)


def test_mpc_period_must_be_a_multiple_of_the_timestep():
    engine = MPCWalkingEngine(**WALK_PARAMS)
    engine.set_command(vx=0.05)
    with pytest.raises(ValueError):
        engine.step(0.012)


def test_qp_solve_is_fast():
    """A solve has to fit comfortably in the 40 ms MPC period."""
    import time
    times = []
    original = mpc_module.solve_box_qp

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        x = original(*args, **kwargs)
        times.append(time.perf_counter() - t0)
        return x

    mpc_module.solve_box_qp = timed
    try:
        for _ in _walk({0.0: lambda e: e.set_command(vx=0.05)}, 4):
            pass
    finally:
        mpc_module.solve_box_qp = original
    assert np.percentile(times, 99) < 0.01
