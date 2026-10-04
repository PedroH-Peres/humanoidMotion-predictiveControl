"""IK against MuJoCo forward kinematics, and a full walking smoke test."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from mujoco_sim.analytical_walking.walking_engine import WalkTarget  # noqa: E402
from mujoco_sim.kinematics.ik_solver import D_ANKLE_SOLE  # noqa: E402
from mujoco_sim.sim import Simulation  # noqa: E402


@pytest.fixture(scope="module")
def kinematic_sim():
    return Simulation(physics=False)


@pytest.mark.parametrize(
    "body_pos, body_yaw, sole_pos, sole_yaw, leg",
    [
        ((0.0, 0.0, 0.22), 0.0, (0.0, 0.054, 0.0), 0.0, "left"),
        ((0.0, 0.0, 0.22), 0.0, (0.0, -0.054, 0.0), 0.0, "right"),
        ((0.01, 0.0, 0.21), 0.2, (0.03, 0.06, 0.02), 0.3, "left"),
        ((0.0, 0.01, 0.20), 0.0, (-0.03, -0.05, 0.01), -0.2, "right"),
    ],
)
def test_ik_matches_forward_kinematics(kinematic_sim, body_pos, body_yaw, sole_pos, sole_yaw, leg):
    """Set the joints to the IK solution and check where MuJoCo puts the foot."""
    sim = kinematic_sim
    body_pos, sole_pos = np.array(body_pos), np.array(sole_pos)
    other_sole = np.array([body_pos[0], -sole_pos[1], 0.0])
    if leg == "left":
        target = WalkTarget(body_pos, body_yaw, sole_pos, sole_yaw, other_sole, 0.0)
    else:
        target = WalkTarget(body_pos, body_yaw, other_sole, 0.0, sole_pos, sole_yaw)

    sim._set_free_joint(body_pos, body_yaw)
    leg_angles = sim._solve_legs(target)
    assert len(leg_angles) == 2
    for angles in leg_angles:
        sim._set_joints(angles)
    mujoco.mj_forward(sim.model, sim.data)

    # IK places the ankle point (ank_pitch joint) D_ANKLE_SOLE above the sole target.
    # 5 mm tolerance: the IK ignores the hip/ankle roll-axis offsets (see ik_solver.py)
    side = leg[0]
    ankle = sim.data.xanchor[sim.model.joint(f"{side}_ank_pitch").id]
    np.testing.assert_allclose(ankle, sole_pos + [0, 0, D_ANKLE_SOLE], atol=5e-3)

    R_foot = sim.data.xmat[sim.model.body(f"{side}_ank_roll_link").id].reshape(3, 3)
    assert np.arctan2(R_foot[1, 0], R_foot[0, 0]) == pytest.approx(sole_yaw, abs=1e-3)
    assert R_foot[2, 2] == pytest.approx(1.0, abs=1e-4)  # sole parallel to the floor


@pytest.mark.slow
@pytest.mark.parametrize("command", [{"vx": 0.05}, {"vy": 0.03}, {"wz": 0.5}])
def test_robot_walks_without_falling(command):
    sim = Simulation(physics=True)
    sim.engine.set_command(**command)
    start = sim.body_pos
    while sim.time < 5.0:
        sim.step()
        assert sim.body_pos[2] > 0.18, f"robot fell at t={sim.time:.2f}s"
    assert sum(sim.ik_failures.values()) == 0
    moved = np.linalg.norm(sim.body_pos[:2] - start[:2]) + abs(sim.body_yaw)
    assert moved > 0.1
