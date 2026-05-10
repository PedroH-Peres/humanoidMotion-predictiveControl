import pybullet as p
import pybullet_data
import time

from ik_solver import RelativeAnalyticalIK, NumericalIK
from trajectory_generator import TrajectoryGenerator

def setup_first_pose(robotId, ik, brain):
    """ Configura a pose inicial de agachamento e braços fixos """
    ID_R_SHO_PITCH, ID_L_SHO_PITCH = 5, 2
    ID_R_SHO_ROLL, ID_L_SHO_ROLL   = 6, 3
    ID_R_ELBOW, ID_L_ELBOW         = 7, 4

    ik.set_fixed_joint(ID_R_SHO_ROLL, 1.5)
    ik.set_fixed_joint(ID_L_SHO_ROLL, -1.5)
    ik.set_fixed_joint(ID_R_SHO_PITCH, 0.6)
    ik.set_fixed_joint(ID_L_SHO_PITCH, 0.6)
    ik.set_fixed_joint(ID_R_ELBOW, 2.1) 
    ik.set_fixed_joint(ID_L_ELBOW, -2.1)

    print("Iniciando a First Pose...")
    ang_d_init = ik.solve_leg(0.0, -brain.Y_SEP/2, -brain.H_TARGET, is_left=False)
    ang_e_init = ik.solve_leg(0.0,  brain.Y_SEP/2, -brain.H_TARGET, is_left=True)

    for i, j in enumerate(ik.JUNTAS_DIR): p.resetJointState(robotId, j, ang_d_init[i])
    for i, j in enumerate(ik.JUNTAS_ESQ): p.resetJointState(robotId, j, ang_e_init[i])
    for j_arm, ang_arm in ik.fixed_joints.items(): p.resetJointState(robotId, j_arm, ang_arm)

    for _ in range(360):
        ik.apply(ang_d_init, ang_e_init)
        p.stepSimulation()
        time.sleep(1/240.)
    print("First Pose estabilizada!")

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    robotId = p.loadURDF("urdf/aurea_urdf_pkg.urdf", [0, 0, 0.35], useFixedBase=False)

    dt_mpc = 0.02
    ik = RelativeAnalyticalIK(robotId)
    brain = TrajectoryGenerator(h_target=0.2, dt_mpc=dt_mpc)

    setup_first_pose(robotId, ik, brain)


    com_x, com_vx = 0.0, 0.0
    com_y, com_vy = 0.0, 0.0
    
    t_global = 0.0
    dt_sim = 1/240.
    passos_fisica_por_mpc = int(dt_mpc / dt_sim) 
    
    cmd_vx = 0.06

    print("Iniciando Caminhada Controlada por MPC!")
    while True:
        

        com_ref_x, zmp_ref_x, zmp_min_x, zmp_max_x = brain.get_references_x(t_global, cmd_vx)
        com_ref_y, zmp_ref_y, zmp_min_y, zmp_max_y = brain.get_references_y(t_global)
        
        next_com_x, next_com_vx = brain.solve_mpc(com_x, com_vx, com_ref_x, zmp_ref_x, zmp_min_x, zmp_max_x)
        next_com_y, next_com_vy = brain.solve_mpc(com_y, com_vy, com_ref_y, zmp_ref_y, zmp_min_y, zmp_max_y)


        for step_fisica in range(passos_fisica_por_mpc):
            t_instante = t_global + (step_fisica * dt_sim)
            progresso_mpc = (step_fisica + 1) / passos_fisica_por_mpc
            
            com_math_x = com_x + (next_com_x - com_x) * progresso_mpc
            com_math_y = com_y + (next_com_y - com_y) * progresso_mpc
            foot_r_math, foot_l_math = brain.get_foot_trajectories(t_instante, cmd_vx)

            dx_r, dy_r, dz_r = foot_r_math[0] - com_math_x, foot_r_math[1] - com_math_y, foot_r_math[2] - brain.H_TARGET
            dx_l, dy_l, dz_l = foot_l_math[0] - com_math_x, foot_l_math[1] - com_math_y, foot_l_math[2] - brain.H_TARGET

            _, orn_b = p.getBasePositionAndOrientation(robotId)
            roll, pitch, _ = p.getEulerFromQuaternion(orn_b)

            angulos_d = ik.solve_leg(dx_r, dy_r, dz_r, is_left=False)
            angulos_e = ik.solve_leg(dx_l, dy_l, dz_l, is_left=True)

            ik.apply(angulos_d, angulos_e)
            p.stepSimulation()
            time.sleep(dt_sim)
            
        com_x, com_vx = next_com_x, next_com_vx
        com_y, com_vy = next_com_y, next_com_vy
        t_global += dt_mpc

if __name__ == "__main__":
    main()