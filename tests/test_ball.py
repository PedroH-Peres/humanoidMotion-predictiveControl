"""RoboCup soccer ball: parameters from the Webots proto, resting and rolling in MuJoCo."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mujoco_sim.ball import BALL_SIZES, add_ball, ball_parameters, configure_ball  # noqa: E402


def _build(size=1, position=(0.4, 0.0)):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1],
                            contype=1, conaffinity=1)
    add_ball(spec, size, position)
    model = spec.compile()
    configure_ball(model)
    return model, mujoco.MjData(model)


def test_size_one_matches_the_webots_default():
    assert ball_parameters(1) == (0.07, 0.205)


def test_unknown_size_is_rejected():
    with pytest.raises(ValueError):
        ball_parameters(6)


@pytest.mark.parametrize("size", sorted(BALL_SIZES))
def test_mass_radius_and_centre_of_mass(size):
    model, _ = _build(size)
    radius, mass = ball_parameters(size)
    body = model.body("ball")
    assert body.mass[0] == pytest.approx(mass)
    assert model.geom("ball_geom").size[0] == pytest.approx(radius)
    np.testing.assert_allclose(body.ipos, 0.0)   # CoM at the sphere's centre
    np.testing.assert_allclose(body.inertia, 2 / 3 * mass * radius ** 2)


def test_ball_rests_on_the_floor_without_drifting():
    model, data = _build()
    for _ in range(500):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[:2], [0.4, 0.0], atol=1e-3)
    assert data.qpos[2] == pytest.approx(0.07, abs=2e-3)
    assert np.abs(data.qvel).max() < 1e-3


def test_kicked_ball_rolls_and_slows_down():
    model, data = _build()
    data.qvel[0] = 1.0
    speeds = []
    for _ in range(int(3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        speeds.append(data.qvel[0])
    assert data.qpos[0] > 0.4                       # moved forward
    assert speeds[int(1 / model.opt.timestep)] < 0.5   # lost most of its speed in 1 s
    assert min(speeds) >= -1e-3                     # never reverses
