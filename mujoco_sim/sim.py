# Simulação do robô Robotis OP3 via MuJoCo
# Pedro H. Peres

import os
import sys
import time
import select
import termios
import tty
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from mujoco_sim.Utils.ik_solver import IKSolver
from mujoco_sim.Utils.torso_controller import SimulatedIMU, TorsoOrientationController
from mujoco_sim.AnalyticalWalking.walking_engine import WalkingEngine
from mujoco_sim.AnalyticalWalking.config import (
    JOINT_SIGNS, WALK_PARAMS, ARM_POSE, PHYSICS, USE_MPC,
    USE_TORSO_CTRL, TORSO_CTRL_PARAMS,
    ACTUATOR_KP, ACTUATOR_KV, ACTUATOR_FORCE_RANGE,
    joint_name as _joint_name,
)

_IK_KEYS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll"]

def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

def build_model() -> mujoco.MjModel:
    from robot_descriptions import op3_mj_description

    spec = mujoco.MjSpec.from_file(op3_mj_description.MJCF_PATH)
    spec.option.timestep         = 0.008
    spec.option.gravity          = [0.0, 0.0, -9.81]
    spec.option.integrator       = mujoco.mjtIntegrator.mjINT_IMPLICIT
    spec.option.solver            = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.impratio          = 50.0
    spec.option.iterations        = 40
    spec.option.noslip_iterations = 10
    spec.option.tolerance         = 1e-4

    _SOLREF = [0.010, 2.0]
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

    model.actuator_gainprm[:, 0]  =  ACTUATOR_KP
    model.actuator_biasprm[:, 1]  = -ACTUATOR_KP
    model.actuator_biasprm[:, 2]  = -ACTUATOR_KV
    model.actuator_forcerange[:, 0] = -ACTUATOR_FORCE_RANGE
    model.actuator_forcerange[:, 1] =  ACTUATOR_FORCE_RANGE

    return model



def _build_maps(model: mujoco.MjModel) -> tuple[dict, dict, dict]:
    qpos_addr: dict[str, int] = {}
    dof_addr:  dict[str, int] = {}
    ctrl_addr: dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_addr[name] = model.jnt_qposadr[i]
            dof_addr[name]  = model.jnt_dofadr[i]
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            jnt_id = model.actuator_trnid[i, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if name:
                ctrl_addr[name] = i
    return qpos_addr, dof_addr, ctrl_addr

def _set_free_joint(data: mujoco.MjData, model: mujoco.MjModel,
                    pos: np.ndarray, quat: np.ndarray) -> None:
    adr = model.jnt_qposadr[0]
    data.qpos[adr:adr+3] = pos
    data.qpos[adr+3:adr+7] = quat


def _apply_ctrl(data: mujoco.MjData,
                ctrl_addr: dict,
                joint_angles: dict[str, float]) -> None:
    for jname, angle in joint_angles.items():
        if jname in ctrl_addr:
            data.ctrl[ctrl_addr[jname]] = angle


def _apply_joints(data: mujoco.MjData,
                  qpos_addr: dict, dof_addr: dict,
                  joint_angles: dict[str, float],
                  zero_vel: bool) -> None:
    for jname, angle in joint_angles.items():
        if jname in qpos_addr:
            data.qpos[qpos_addr[jname]] = angle
            if zero_vel:
                data.qvel[dof_addr[jname]] = 0.0


def _solve_leg(ik: IKSolver,
               body_pos: np.ndarray, body_yaw: float,
               sole_pos: np.ndarray, sole_yaw: float,
               leg: str, side: str) -> dict[str, float] | None:
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

_term_fd = sys.stdin.fileno()
_term_old = None

def _setup_terminal():
    global _term_old
    _term_old = termios.tcgetattr(_term_fd)
    tty.setcbreak(_term_fd)

def _restore_terminal():
    if _term_old is not None:
        termios.tcsetattr(_term_fd, termios.TCSADRAIN, _term_old)

def _poll_key(engine: WalkingEngine) -> None:
    if not select.select([sys.stdin], [], [], 0)[0]:
        return
    key = sys.stdin.read(1).lower()
    if   key == "w": engine.set_command(vx=0.05);  print("[CMD] Frente")
    elif key == "s": engine.set_command(vy=0.03);  print("[CMD] Lateral")
    elif key == "r": engine.set_command(wz=0.5);   print("[CMD] Rotacao")
    elif key == "m": engine.set_command();          print("[CMD] Marcha no lugar")
    elif key == "x": engine.request_stop();         print("[CMD] Parando...")
    elif key == "q": _restore_terminal(); os._exit(0)

def main():
    model = build_model()
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    qpos_addr, dof_addr, ctrl_addr = _build_maps(model)
    ik = IKSolver()

    if USE_MPC:
        from mujoco_sim.MPCWalking.walking_engine import MPCWalkingEngine
        from mujoco_sim.MPCWalking.config import MPC_PARAMS
        engine = MPCWalkingEngine(dt=model.opt.timestep, **MPC_PARAMS)
    else:
        engine = WalkingEngine(**WALK_PARAMS)

    imu         = SimulatedIMU()
    torso_ctrl  = TorsoOrientationController(**TORSO_CTRL_PARAMS)

    mode_label  = "FÍSICA (mj_step)" if PHYSICS else "CINEMÁTICA (mj_forward)"
    ctrl_label  = "ON" if USE_TORSO_CTRL else "OFF"
    print(f"\n{'='*50}")
    print(f"  MODO: {mode_label}")
    print(f"  Torso controller: {ctrl_label}")
    print(f"  Joints disponíveis ({len(qpos_addr)}): {sorted(qpos_addr)}")
    print(f"{'='*50}\n")

    body_pos, body_yaw, l_sole, l_yaw, r_sole, r_yaw = engine._standing_pose()
    _set_free_joint(data, model, body_pos, _yaw_to_quat(body_yaw))

    l_angles = _solve_leg(ik, body_pos, body_yaw, l_sole, l_yaw, "left",  "l")
    r_angles = _solve_leg(ik, body_pos, body_yaw, r_sole, r_yaw, "right", "r")
    if l_angles:
        _apply_joints(data, qpos_addr, dof_addr, l_angles, zero_vel=False)
        _apply_ctrl(data, ctrl_addr, l_angles)
    if r_angles:
        _apply_joints(data, qpos_addr, dof_addr, r_angles, zero_vel=False)
        _apply_ctrl(data, ctrl_addr, r_angles)
    _apply_joints(data, qpos_addr, dof_addr, ARM_POSE, zero_vel=False)
    _apply_ctrl(data, ctrl_addr, ARM_POSE)
    mujoco.mj_forward(model, data)

    dt         = model.opt.timestep
    sim_time   = 0.0
    real_start = time.perf_counter()

    _setup_terminal()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
        print(f"[MODO: {mode_label}]")
        print(_KEY_HELP, "\n")

        while viewer.is_running():
            _poll_key(engine)

            body_pos, body_yaw, l_sole, l_yaw, r_sole, r_yaw = engine.step(dt)

            l_angles = _solve_leg(ik, body_pos, body_yaw, l_sole, l_yaw, "left",  "l")
            r_angles = _solve_leg(ik, body_pos, body_yaw, r_sole, r_yaw, "right", "r")

            if USE_TORSO_CTRL:
                imu_reading = imu.read(model, data)
                corrections = torso_ctrl.compute(imu_reading)
                if l_angles:
                    for joint, delta in corrections.items():
                        if joint in l_angles:
                            l_angles[joint] += delta
                if r_angles:
                    for joint, delta in corrections.items():
                        if joint in r_angles:
                            r_angles[joint] += delta

            if PHYSICS:
                if USE_MPC:
                    # Open-loop: impõe a posição do torso diretamente (trajetória do LIPM).
                    # As pernas usam física para contato; o corpo segue o MPC sem lag.
                    _set_free_joint(data, model, body_pos, _yaw_to_quat(body_yaw))
                if l_angles: _apply_ctrl(data, ctrl_addr, l_angles)
                if r_angles: _apply_ctrl(data, ctrl_addr, r_angles)
                _apply_ctrl(data, ctrl_addr, ARM_POSE)
                mujoco.mj_step(model, data)
            else:
                _set_free_joint(data, model, body_pos, _yaw_to_quat(body_yaw))
                if l_angles: _apply_joints(data, qpos_addr, dof_addr, l_angles, zero_vel=False)
                if r_angles: _apply_joints(data, qpos_addr, dof_addr, r_angles, zero_vel=False)
                _apply_joints(data, qpos_addr, dof_addr, ARM_POSE, zero_vel=False)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()

            if int(sim_time / 0.5) != int((sim_time + dt) / 0.5):
                adr = model.jnt_qposadr[0]
                px, py, pz = data.qpos[adr], data.qpos[adr+1], data.qpos[adr+2]
                if USE_TORSO_CTRL:
                    r = imu.read(model, data)
                    imu_str = f"  imu=({np.degrees(r['pitch']):.1f}°p, {np.degrees(r['roll']):.1f}°r)"
                else:
                    imu_str = ""
                print(f"[t={sim_time:.1f}s] pos=({px:.3f},{py:.3f},{pz:.3f})  state={engine.state}{imu_str}")

            sim_time  += dt
            elapsed    = time.perf_counter() - real_start
            sleep_time = sim_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    _restore_terminal()
