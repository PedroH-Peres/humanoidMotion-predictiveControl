"""
ROBOTIS OP3 — simulação MuJoCo via robot_descriptions (MuJoCo Menagerie).
"""

import os
import mujoco
import mujoco.viewer

os.chdir(os.path.join(os.path.dirname(__file__), ".."))


def build_model() -> mujoco.MjModel:
    from robot_descriptions import op3_mj_description

    spec = mujoco.MjSpec.from_file(op3_mj_description.MJCF_PATH)
    spec.option.timestep         = 0.006
    spec.option.gravity          = [0.0, 0.0, -9.81]
    spec.option.integrator       = mujoco.mjtIntegrator.mjINT_IMPLICIT
    spec.option.solver            = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.impratio          = 50.0
    spec.option.iterations        = 50
    spec.option.noslip_iterations = 10
    spec.option.tolerance         = 1e-5

    _SOLREF = [0.012, 2.0]
    _SOLIMP = [0.99, 0.9999, 0.0001, 0.5, 3]

    floor = spec.worldbody.add_geom()
    floor.name        = "floor"
    floor.type        = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size        = [0, 0, 0.1]
    floor.pos         = [0, 0, 0]
    floor.rgba        = [0.5, 0.5, 0.5, 1.0]
    floor.contype     = 1
    floor.conaffinity = 1
    floor.solref      = _SOLREF
    floor.solimp      = _SOLIMP

    for geom in spec.geoms:
        if geom.name == "floor":
            continue
        if geom.contype > 0 or geom.conaffinity > 0:
            geom.conaffinity = 0
            geom.solref = _SOLREF
            geom.solimp = _SOLIMP

    model = spec.compile()

    dof_start = model.jnt_dofadr[0]
    model.dof_damping[dof_start:dof_start+3]   = 3.0
    model.dof_damping[dof_start+3:dof_start+6] = 1.0

    return model


def main():
    model = build_model()
    data  = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulando OP3... feche a janela para sair.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
