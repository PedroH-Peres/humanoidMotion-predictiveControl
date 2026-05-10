import pybullet as p
import pybullet_data
import time

# Importações dos nossos módulos
from ik_solver import RelativeAnalyticalIK
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

    ik = RelativeAnalyticalIK(robotId)
    brain = TrajectoryGenerator(h_target=0.22)

    setup_first_pose(robotId, ik, brain)

    t = 0.0
    dt_sim = 1/240.
    cmd_vx = 0.05 

    print("Iniciando a Caminhada...")
    while True:
        com_math, foot_r_math, foot_l_math = brain.get_desired_state(t, cmd_vx)

        dx_r = foot_r_math[0] - com_math[0]
        dy_r = foot_r_math[1] - com_math[1]
        dz_r = foot_r_math[2] - brain.H_TARGET
        
        dx_l = foot_l_math[0] - com_math[0]
        dy_l = foot_l_math[1] - com_math[1]
        dz_l = foot_l_math[2] - brain.H_TARGET

        angulos_d = ik.solve_leg(dx_r, dy_r, dz_r, is_left=False)
        angulos_e = ik.solve_leg(dx_l, dy_l, dz_l, is_left=True)

        ik.apply(angulos_d, angulos_e)
        p.stepSimulation()
        
        t += dt_sim
        time.sleep(dt_sim)

if __name__ == "__main__":
    main()