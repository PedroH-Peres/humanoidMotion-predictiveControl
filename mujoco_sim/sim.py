# Simulação do robô Robotis OP3 via MuJoCo
# Pedro H. Peres

import os
import sys
import time
import msvcrt
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from mujoco_sim.Utils.ik_solver import IKSolver
from mujoco_sim.AnalyticalWalking.walking_engine import WalkingEngine
from mujoco_sim.AnalyticalWalking.config import JOINT_SIGNS, WALK_PARAMS, ARM_POSE, PHYSICS, joint_name as _joint_name

_IK_KEYS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll"]

def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """[w, x, y, z] para rotação pura em Z."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

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



def _build_maps(model: mujoco.MjModel) -> tuple[dict, dict]:
    """
    Retorna (qpos_addr, dof_addr): joint_name → índice em qpos/qvel.
    Funciona para qualquer MJCF sem depender de nomes de atuadores.
    """
    qpos_addr: dict[str, int] = {}
    dof_addr:  dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_addr[name] = model.jnt_qposadr[i]
            dof_addr[name]  = model.jnt_dofadr[i]
    return qpos_addr, dof_addr


def _set_free_joint(data: mujoco.MjData, model: mujoco.MjModel,
                    pos: np.ndarray, quat: np.ndarray) -> None:
    adr = model.jnt_qposadr[0]
    data.qpos[adr:adr+3] = pos
    data.qpos[adr+3:adr+7] = quat


def _apply_joints(data: mujoco.MjData,
                  qpos_addr: dict, dof_addr: dict,
                  joint_angles: dict[str, float],
                  zero_vel: bool) -> None:
    """Escreve ângulos em qpos e, opcionalmente, zera qvel para cada junta."""
    for jname, angle in joint_angles.items():
        if jname in qpos_addr:
            data.qpos[qpos_addr[jname]] = angle
            if zero_vel:
                data.qvel[dof_addr[jname]] = 0.0


def _solve_leg(ik: IKSolver,
               body_pos: np.ndarray, body_yaw: float,
               sole_pos: np.ndarray, sole_yaw: float,
               leg: str, side: str) -> dict[str, float] | None:
    """Roda IK e devolve dict {joint_name: angle} ou None se falhar."""
    ok, angles = ik.solve(body_pos, _rot_z(body_yaw),
                          sole_pos, _rot_z(sole_yaw), leg)
    if not ok:
        return None
    return {
        _joint_name(side, key): JOINT_SIGNS[key][side] * angles[idx]
        for idx, key in enumerate(_IK_KEYS)
    }


_KEY_HELP = (
    "Teclas (terminal em foco):\n"
    "  W=frente  S=lateral  R=rotação  M=marcha  X=parar  Q=sair"
)

def _poll_key(engine: WalkingEngine) -> None:
    if not msvcrt.kbhit():
        return
    key = msvcrt.getwch().lower()
    if   key == "w": engine.set_command(vx=0.08);  print("[CMD] Frente")
    elif key == "s": engine.set_command(vy=0.03);  print("[CMD] Lateral")
    elif key == "r": engine.set_command(wz=0.5);   print("[CMD] Rotação")
    elif key == "m": engine.set_command();          print("[CMD] Marcha no lugar")
    elif key == "x": engine.request_stop();         print("[CMD] Parando...")
    elif key == "q": os._exit(0)



def main():
    model = build_model()
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    qpos_addr, dof_addr = _build_maps(model)
    ik = IKSolver()

    engine = WalkingEngine(**WALK_PARAMS)

    mode_label = "FÍSICA (mj_step)" if PHYSICS else "CINEMÁTICA (mj_forward)"
    print(f"\n{'='*50}")
    print(f"  MODO: {mode_label}")
    print(f"  Joints disponíveis ({len(qpos_addr)}): {sorted(qpos_addr)}")
    print(f"{'='*50}\n")

    # Sempre via qpos — garante que o robô nasce na postura correta
    body_pos, body_yaw, l_sole, l_yaw, r_sole, r_yaw = engine._standing_pose()
    _set_free_joint(data, model, body_pos, _yaw_to_quat(body_yaw))

    l_angles = _solve_leg(ik, body_pos, body_yaw, l_sole, l_yaw, "left",  "l")
    r_angles = _solve_leg(ik, body_pos, body_yaw, r_sole, r_yaw, "right", "r")
    if l_angles: _apply_joints(data, qpos_addr, dof_addr, l_angles, zero_vel=False)
    if r_angles: _apply_joints(data, qpos_addr, dof_addr, r_angles, zero_vel=False)
    _apply_joints(data, qpos_addr, dof_addr, ARM_POSE, zero_vel=False)
    mujoco.mj_forward(model, data)

    dt         = model.opt.timestep
    sim_time   = 0.0
    real_start = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
        print(f"[MODO: {mode_label}]")
        print(_KEY_HELP, "\n")

        while viewer.is_running():
            _poll_key(engine)

            body_pos, body_yaw, l_sole, l_yaw, r_sole, r_yaw = engine.step(dt)

            l_angles = _solve_leg(ik, body_pos, body_yaw, l_sole, l_yaw, "left",  "l")
            r_angles = _solve_leg(ik, body_pos, body_yaw, r_sole, r_yaw, "right", "r")

            if PHYSICS:
                if l_angles: _apply_joints(data, qpos_addr, dof_addr, l_angles, zero_vel=True)
                if r_angles: _apply_joints(data, qpos_addr, dof_addr, r_angles, zero_vel=True)
                _apply_joints(data, qpos_addr, dof_addr, ARM_POSE, zero_vel=True)
                mujoco.mj_step(model, data)
            else:
                _set_free_joint(data, model, body_pos, _yaw_to_quat(body_yaw))
                if l_angles: _apply_joints(data, qpos_addr, dof_addr, l_angles, zero_vel=False)
                if r_angles: _apply_joints(data, qpos_addr, dof_addr, r_angles, zero_vel=False)
                _apply_joints(data, qpos_addr, dof_addr, ARM_POSE, zero_vel=False)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()

            sim_time  += dt
            elapsed    = time.perf_counter() - real_start
            sleep_time = sim_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
