"""
RoboCup soccer ball for MuJoCo.

Physical parameters (radius and mass per FIFA ball size, damping) come from the
Webots `RobocupSoccerBall.proto` (R2025a). The proto's 3D mesh is not used (it is
licensed for use only with Webots): the ball is a sphere with a checker texture.

The ball is a free body added after the robot, so the robot's free joint stays
the first joint of the model.
"""

import numpy as np
import mujoco

# FIFA ball size → (radius [m], mass [kg])
BALL_SIZES = {
    1: (0.07, 0.205),
    2: (0.085, 0.250),
    3: (0.095, 0.310),
    4: (0.1025, 0.370),
    5: (0.1125, 0.430),
}

# Webots Damping node: fraction of the speed lost every second (linear and angular)
DAMPING_FRACTION = 0.77

BALL_FRICTION = [0.8, 0.005, 0.0005]   # sliding, torsional, rolling
BODY_NAME = "ball"


def ball_parameters(size: int) -> tuple[float, float]:
    """(radius [m], mass [kg]) of a ball of FIFA size 1..5."""
    if size not in BALL_SIZES:
        raise ValueError(f"ball size must be one of {sorted(BALL_SIZES)}, got {size!r}")
    return BALL_SIZES[size]


def add_ball(spec, size: int = 1, position: tuple[float, float] = (0.0, 0.0)) -> None:
    """Add the ball, resting on the floor at the given (x, y), to an `mujoco.MjSpec`."""
    radius, mass = ball_parameters(size)
    # Hollow thin-walled sphere: I = 2/3 m r²
    inertia = 2.0 / 3.0 * mass * radius ** 2

    spec.add_texture(name="ball_texture", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.05, 0.05, 0.05], rgb2=[0.95, 0.95, 0.95], width=64, height=64)
    material = spec.add_material(name="ball_material")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ball_texture"

    body = spec.worldbody.add_body(name=BODY_NAME, pos=[position[0], position[1], radius])
    body.add_freejoint()
    body.mass = mass
    body.inertia = [inertia] * 3
    body.ipos = [0.0, 0.0, 0.0]            # centre of mass at the sphere's centre
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    body.explicitinertial = True
    body.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0.0, 0.0],
                  material="ball_material", friction=BALL_FRICTION, contype=1, conaffinity=1)


def configure_ball(model: mujoco.MjModel) -> None:
    """Set the ball's velocity damping on a compiled model (see DAMPING_FRACTION)."""
    body = model.body(BODY_NAME)
    dof = model.jnt_dofadr[body.jntadr[0]]
    # Webots drops a fraction d of the speed per second, i.e. v(t) = v0 * exp(-k t)
    # with k = -ln(1 - d); in MuJoCo the damping coefficient is k times the inertia.
    k = -np.log(1.0 - DAMPING_FRACTION)
    model.dof_damping[dof:dof + 3] = k * body.mass[0]
    model.dof_damping[dof + 3:dof + 6] = k * body.inertia
