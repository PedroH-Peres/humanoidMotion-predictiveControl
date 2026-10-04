"""
Aurea robot walking in PyBullet, with an MPC-planned CoM and analytical IK.

Run (from any directory):

    python pybullet_sim/main.py                 # GUI, cmd_vx = 0.1 m/s
    python pybullet_sim/main.py --vx 0.05
    python pybullet_sim/main.py --headless --duration 10

Note: the MPC runs in open loop. The CoM state fed back into the MPC is its
own prediction, not a measurement from the simulator, so the planner acts as a
trajectory generator and the PD joint controllers do the stabilisation.
"""

import argparse
import os
import time

import pybullet as p
import pybullet_data

from kinematics import LegIK
from mpc_planner import MPCPlanner

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "aurea", "aurea.urdf")

DT_SIM = 1 / 240.   # physics timestep [s]
DT_MPC = 0.02       # MPC timestep [s]

# KNOWN ISSUE: DT_MPC is not a multiple of DT_SIM. int(0.02 / (1/240)) = 4, so
# each MPC tick advances the trajectory clock t by 0.020 s while physics only
# advances 4/240 ≈ 0.0167 s: the gait runs 1.2× faster than planned (steps of
# ~0.25 s instead of T_STEP = 0.3 s, ~0.12 m/s for cmd_vx = 0.1). The current
# gains were tuned with this mismatch; making the clocks consistent (e.g.
# DT_MPC = 5 * DT_SIM) makes the robot fall until the planner is re-tuned.
PHYSICS_STEPS_PER_MPC = int(DT_MPC / DT_SIM)

# Fixed arm pose [rad] held while walking
ARM_POSE = {
    "r_sho_roll": 1.5, "l_sho_roll": -1.5,
    "r_sho_pitch": 0.6, "l_sho_pitch": 0.6,
    "r_el": 2.1, "l_el": -2.1,
}


def setup_first_pose(ik: LegIK, planner: MPCPlanner, realtime: bool) -> None:
    """Crouch to the walking height with the arms fixed, then let it settle."""
    for joint, angle in ARM_POSE.items():
        ik.set_fixed_joint(joint, angle)

    print("Moving to the initial pose...")
    right = ik.solve_leg(0.0, -planner.Y_SEP / 2, -planner.H_TARGET, is_left=False)
    left = ik.solve_leg(0.0, planner.Y_SEP / 2, -planner.H_TARGET, is_left=True)
    ik.reset(right, left)

    for _ in range(360):  # 1.5 s
        ik.apply(right, left)
        p.stepSimulation()
        if realtime:
            time.sleep(DT_SIM)
    print("Initial pose settled.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--vx", type=float, default=0.1, help="forward speed [m/s]")
    parser.add_argument("--headless", action="store_true", help="run without GUI, as fast as possible")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    args = parser.parse_args()

    p.connect(p.DIRECT if args.headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.35], useFixedBase=False)

    ik = LegIK(robot_id)
    planner = MPCPlanner(h_target=0.2, dt_mpc=DT_MPC)
    realtime = not args.headless

    setup_first_pose(ik, planner, realtime)

    # MPC state (open loop, see module docstring)
    com_x, com_vx = 0.0, 0.0
    com_y, com_vy = 0.0, 0.0

    t = 0.0

    print(f"Walking with MPC at cmd_vx = {args.vx} m/s")
    while args.duration is None or t < args.duration:
        com_ref_x, zmp_ref_x, zmp_min_x, zmp_max_x = planner.get_references_x(t, args.vx)
        com_ref_y, zmp_ref_y, zmp_min_y, zmp_max_y = planner.get_references_y(t)

        next_com_x, next_com_vx = planner.solve_mpc(com_x, com_vx, com_ref_x, zmp_ref_x, zmp_min_x, zmp_max_x)
        next_com_y, next_com_vy = planner.solve_mpc(com_y, com_vy, com_ref_y, zmp_ref_y, zmp_min_y, zmp_max_y)

        for i in range(PHYSICS_STEPS_PER_MPC):
            # Linear interpolation of the CoM between two MPC samples
            alpha = (i + 1) / PHYSICS_STEPS_PER_MPC
            com_now_x = com_x + (next_com_x - com_x) * alpha
            com_now_y = com_y + (next_com_y - com_y) * alpha
            foot_r, foot_l = planner.get_foot_trajectories(t + i * DT_SIM, args.vx)

            # Feet relative to the torso → IK
            right = ik.solve_leg(foot_r[0] - com_now_x, foot_r[1] - com_now_y,
                                 foot_r[2] - planner.H_TARGET, is_left=False)
            left = ik.solve_leg(foot_l[0] - com_now_x, foot_l[1] - com_now_y,
                                foot_l[2] - planner.H_TARGET, is_left=True)

            ik.apply(right, left)
            p.stepSimulation()
            if realtime:
                time.sleep(DT_SIM)

        com_x, com_vx = next_com_x, next_com_vx
        com_y, com_vy = next_com_y, next_com_vy
        t += DT_MPC

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"Finished at t={t:.1f}s, base = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    p.disconnect()


if __name__ == "__main__":
    main()
