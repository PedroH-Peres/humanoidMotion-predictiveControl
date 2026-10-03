"""
Robotis OP3 walking simulation in MuJoCo.

Pipeline per control tick:

    WalkingEngine.step()  →  desired pelvis + sole poses (world frame)
    IKSolver.solve()      →  6 joint angles per leg
    data.ctrl             →  PD position actuators  →  mj_step

Run from the repository root:

    python -m mujoco_sim

Author: Pedro H. Peres
"""

import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

from .analytical_walking.walking_engine import WalkingEngine, WalkTarget
from .config import (
    ACTUATOR_FORCE_RANGE, ACTUATOR_KP, ACTUATOR_KV, ARM_POSE, JOINT_SIGNS,
    KEY_COMMANDS, PHYSICS, WALK_PARAMS, joint_name,
)
from .kinematics.ik_solver import IKSolver

# Order of the angles returned by IKSolver.solve
_IK_KEYS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll"]
_LEGS = (("left", "l"), ("right", "r"))


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of `yaw` about the world z axis."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def build_model() -> mujoco.MjModel:
    """Load the OP3 MJCF, add a floor and tune solver, contacts and actuators."""
    from robot_descriptions import op3_mj_description

    spec = mujoco.MjSpec.from_file(op3_mj_description.MJCF_PATH)
    spec.option.timestep = 0.008
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.impratio = 50.0          # stiffer friction cone → less foot slip
    spec.option.iterations = 40
    spec.option.noslip_iterations = 10
    spec.option.tolerance = 1e-4

    # Stiff, nearly hard contacts between the feet and the floor
    solref = [0.010, 2.0]
    solimp = [0.99, 0.9999, 0.0001, 0.5, 3]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.1]
    floor.pos = [0, 0, 0]
    floor.rgba = [0.5, 0.5, 0.5, 1.0]
    floor.contype = 1
    floor.conaffinity = 1
    floor.solref = solref
    floor.solimp = solimp

    # Robot geoms collide with the floor only (no self-collision)
    for geom in spec.geoms:
        if geom.name == "floor":
            continue
        if geom.contype > 0 or geom.conaffinity > 0:
            geom.conaffinity = 0
            geom.solref = solref
            geom.solimp = solimp

    model = spec.compile()

    # Damping on the floating base (3 translational + 3 rotational DoFs)
    dof_start = model.jnt_dofadr[0]
    model.dof_damping[dof_start:dof_start + 3] = 3.0
    model.dof_damping[dof_start + 3:dof_start + 6] = 1.0

    # Position actuators: torque = KP * (ctrl - q) - KV * q_dot
    model.actuator_gainprm[:, 0] = ACTUATOR_KP
    model.actuator_biasprm[:, 1] = -ACTUATOR_KP
    model.actuator_biasprm[:, 2] = -ACTUATOR_KV
    model.actuator_forcerange[:, 0] = -ACTUATOR_FORCE_RANGE
    model.actuator_forcerange[:, 1] = ACTUATOR_FORCE_RANGE

    return model


def _build_maps(model: mujoco.MjModel) -> tuple[dict, dict, dict]:
    """Joint name → qpos address, dof address and actuator index."""
    qpos_addr: dict[str, int] = {}
    dof_addr: dict[str, int] = {}
    ctrl_addr: dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_addr[name] = model.jnt_qposadr[i]
            dof_addr[name] = model.jnt_dofadr[i]
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            jnt_id = model.actuator_trnid[i, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if name:
                ctrl_addr[name] = i
    return qpos_addr, dof_addr, ctrl_addr


class Simulation:
    """OP3 model + walking engine + IK, advanced one control tick at a time."""

    def __init__(self, physics: bool = PHYSICS):
        self.physics = physics
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.qpos_addr, self.dof_addr, self.ctrl_addr = _build_maps(self.model)
        self.ik = IKSolver()
        self.engine = WalkingEngine(**WALK_PARAMS)
        self.dt = self.model.opt.timestep
        self.time = 0.0
        self.ik_failures = {"left": 0, "right": 0}
        self.reset()

    @property
    def body_pos(self) -> np.ndarray:
        adr = self.model.jnt_qposadr[0]
        return self.data.qpos[adr:adr + 3].copy()

    @property
    def body_yaw(self) -> float:
        adr = self.model.jnt_qposadr[0]
        w, _, _, z = self.data.qpos[adr + 3:adr + 7]
        return float(2.0 * np.arctan2(z, w))

    def reset(self) -> None:
        """Place the robot in the engine's standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        target = self.engine.standing_pose()
        self._set_free_joint(target.body_pos, target.body_yaw)
        for joint_angles in (*self._solve_legs(target), ARM_POSE):
            self._set_joints(joint_angles)
            self._set_ctrl(joint_angles)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance the engine and the simulation by one timestep."""
        target = self.engine.step(self.dt)
        leg_angles = self._solve_legs(target)

        if self.physics:
            for joint_angles in (*leg_angles, ARM_POSE):
                self._set_ctrl(joint_angles)
            mujoco.mj_step(self.model, self.data)
        else:
            # Kinematic playback: teleport the pelvis and set joints directly
            self._set_free_joint(target.body_pos, target.body_yaw)
            for joint_angles in (*leg_angles, ARM_POSE):
                self._set_joints(joint_angles)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self.time += self.dt

    def _solve_legs(self, target: WalkTarget) -> list[dict[str, float]]:
        """IK for both legs. A leg whose IK fails is left out (keeps its last command)."""
        R_body = _rot_z(target.body_yaw)
        soles = {"left": (target.left_sole, target.left_yaw),
                 "right": (target.right_sole, target.right_yaw)}
        result = []
        for leg, side in _LEGS:
            sole_pos, sole_yaw = soles[leg]
            ok, angles = self.ik.solve(target.body_pos, R_body,
                                       sole_pos, _rot_z(sole_yaw), leg)
            if not ok:
                if self.ik_failures[leg] == 0:
                    print(f"[IK] {leg} leg target out of reach at t={self.time:.2f}s "
                          f"(further failures are only counted)")
                self.ik_failures[leg] += 1
                continue
            result.append({
                joint_name(side, key): JOINT_SIGNS[key][side] * angles[idx]
                for idx, key in enumerate(_IK_KEYS)
            })
        return result

    def _set_free_joint(self, pos: np.ndarray, yaw: float) -> None:
        adr = self.model.jnt_qposadr[0]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = _yaw_to_quat(yaw)

    def _set_ctrl(self, joint_angles: dict[str, float]) -> None:
        for name, angle in joint_angles.items():
            if name in self.ctrl_addr:
                self.data.ctrl[self.ctrl_addr[name]] = angle

    def _set_joints(self, joint_angles: dict[str, float]) -> None:
        for name, angle in joint_angles.items():
            if name in self.qpos_addr:
                self.data.qpos[self.qpos_addr[name]] = angle


# -----------------------------------------------------------------------------
#  Keyboard control (keys pressed with the viewer window focused)
#  Every letter is already a viewer shortcut, so we use arrows and nav keys.
#  The arrows only step the simulation in the viewer while it is paused.
# -----------------------------------------------------------------------------

_GLFW_KEYS = {
    265: "up", 264: "down", 263: "left", 262: "right",
    266: "page_up", 267: "page_down", 268: "home", 261: "delete",
}

_KEY_HELP = (
    "Keys (viewer window focused):\n"
    "  ↑ / ↓        forward / backward\n"
    "  ← / →        sidestep left / right\n"
    "  PgUp / PgDn  turn left / right\n"
    "  Home         march in place\n"
    "  Delete       stop\n"
    "  Close the window to quit."
)


def _apply_key(engine: WalkingEngine, key: str) -> None:
    vx = KEY_COMMANDS["forward"]["vx"]
    vy = KEY_COMMANDS["lateral"]["vy"]
    wz = KEY_COMMANDS["turn"]["wz"]
    actions = {
        "up":        ("Forward",        lambda: engine.set_command(vx=vx)),
        "down":      ("Backward",       lambda: engine.set_command(vx=-vx)),
        "left":      ("Sidestep left",  lambda: engine.set_command(vy=vy)),
        "right":     ("Sidestep right", lambda: engine.set_command(vy=-vy)),
        "page_up":   ("Turn left",      lambda: engine.set_command(wz=wz)),
        "page_down": ("Turn right",     lambda: engine.set_command(wz=-wz)),
        "home":      ("March in place", engine.march_in_place),
        "delete":    ("Stopping...",    engine.request_stop),
    }
    label, action = actions[key]
    action()
    print(f"[CMD] {label}")


def main() -> None:
    sim = Simulation()

    mode_label = "PHYSICS (mj_step)" if sim.physics else "KINEMATICS (mj_forward)"
    print(f"\n{'=' * 50}")
    print(f"  MODE: {mode_label}")
    print(f"  Joints ({len(sim.qpos_addr)}): {sorted(sim.qpos_addr)}")
    print(f"{'=' * 50}\n")
    print(_KEY_HELP, "\n")

    # The viewer calls key_callback from its own thread; queue the keys and
    # apply them in the simulation loop.
    pressed: deque[str] = deque()

    def key_callback(keycode: int) -> None:
        if keycode in _GLFW_KEYS:
            pressed.append(_GLFW_KEYS[keycode])

    real_start = time.perf_counter()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.2]

        while viewer.is_running():
            while pressed:
                _apply_key(sim.engine, pressed.popleft())

            with viewer.lock():
                sim.step()
            viewer.sync()

            if int(sim.time / 0.5) != int((sim.time - sim.dt) / 0.5):
                px, py, pz = sim.body_pos
                print(f"[t={sim.time:.1f}s] body pos = ({px:.4f}, {py:.4f}, {pz:.4f})  "
                      f"state={sim.engine.state.name}")

            # Keep the simulation in sync with wall-clock time
            sleep_time = sim.time - (time.perf_counter() - real_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    failures = sum(sim.ik_failures.values())
    if failures:
        print(f"[IK] {failures} failed solves during the run: {sim.ik_failures}")
