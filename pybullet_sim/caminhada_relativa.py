import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import cvxpy as cp

class RelativeAnalyticalIK:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # --- PARÂMETROS DO ROBÔ ---
        self.L_COXA = 0.12   
        self.L_TIBIA = 0.085  
        self.D_QUADRIL_OFFSET = 0.0425 
        self.D_TORNOZELO_SOLA = 0.048 
        
        self.JUNTAS_DIR = [8, 9, 10, 11, 12, 13]
        self.JUNTAS_ESQ = [14, 15, 16, 17, 18, 19]
        self.fixed_joints = {}

    def set_fixed_joint(self, joint_id, target_angle):

        self.fixed_joints[joint_id] = target_angle

    def solve_leg(self, dx, dy, dz, is_left):
        """
        dx, dy, dz: Distância do pé em relação ao centro do torso (0,0,0)
        """

        sinal_y = 1.0 if is_left else -1.0
        offset_y = self.D_QUADRIL_OFFSET * sinal_y

        rx = -dx
        ry = -(dy - offset_y)
        rz = -(dz + self.D_TORNOZELO_SOLA)
        C = math.sqrt(rx**2 + ry**2 + rz**2)
        
      
        C = min(C, self.L_COXA + self.L_TIBIA - 0.001)

        cos_q5 = (C**2 - self.L_COXA**2 - self.L_TIBIA**2) / (2 * self.L_COXA * self.L_TIBIA)
        cos_q5 = max(-1.0, min(1.0, cos_q5))
        q5_knee = math.acos(cos_q5) 


        q7_ankle_roll = math.atan2(ry, rz)
        
        alpha = math.asin(max(-1.0, min(1.0, (self.L_COXA / C) * math.sin(q5_knee))))
        q6_ankle_pitch = -math.atan2(rx, math.copysign(1.0, rz) * math.sqrt(ry**2 + rz**2)) - alpha

        q1_hip_yaw = 0.0
        q2_hip_roll = -q7_ankle_roll
        q3_hip_pitch = -q6_ankle_pitch - q5_knee

        return [q1_hip_yaw, q2_hip_roll, q3_hip_pitch, q5_knee, q6_ankle_pitch, q7_ankle_roll]

    def apply(self, angulos_dir, angulos_esq):
        for i, angle in enumerate(angulos_dir):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_DIR[i], p.POSITION_CONTROL, 
                                    targetPosition=angle, force=5000, positionGain=1.0)
        for i, angle in enumerate(angulos_esq):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_ESQ[i], p.POSITION_CONTROL, 
                                    targetPosition=angle, force=5000, positionGain=1.0)
        
        for j in range(p.getNumJoints(self.robot_id)):
            if j not in self.JUNTAS_DIR and j not in self.JUNTAS_ESQ:
                target = self.fixed_joints.get(j, 0.0) # Busca o ângulo desejado (padrão é 0)
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=5000, positionGain=1.0)


class WalkingMath:
    def __init__(self, h_target=0.2, t_step=0.6, y_sep=0.09):
        self.H_TARGET = h_target
        self.T_STEP = t_step
        self.Y_SEP = y_sep
        
    def get_desired_state(self, t, cmd_vx):
        """ 
        Gera as coordenadas matemáticas puras do CoM e dos pés ao longo do tempo.
        """
        step_idx = int(t / self.T_STEP)
        t_in_step = t % self.T_STEP
        progresso = t_in_step / self.T_STEP
        
        is_left_support = (step_idx % 2 == 0)

        avanco_total = cmd_vx * t
        
        if is_left_support:
            math_foot_l_x = avanco_total
            math_foot_l_y = self.Y_SEP / 2.0
            math_foot_l_z = 0.0 
            
            passo_anterior = cmd_vx * (step_idx * self.T_STEP)
            passo_futuro = cmd_vx * ((step_idx + 1) * self.T_STEP)
            math_foot_r_x = passo_anterior + (passo_futuro - passo_anterior) * self.suavizar(progresso)
            math_foot_r_y = -self.Y_SEP / 2.0
            math_foot_r_z = 0.05 * math.sin(math.pi * progresso) # Parábola
        else:
            math_foot_r_x = avanco_total
            math_foot_r_y = -self.Y_SEP / 2.0
            math_foot_r_z = 0.0
            
            passo_anterior = cmd_vx * (step_idx * self.T_STEP)
            passo_futuro = cmd_vx * ((step_idx + 1) * self.T_STEP)
            math_foot_l_x = passo_anterior + (passo_futuro - passo_anterior) * self.suavizar(progresso)
            math_foot_l_y = self.Y_SEP / 2.0
            math_foot_l_z = 0.05 * math.sin(math.pi * progresso)

        math_com_x = avanco_total
        ginga_amplitude = self.Y_SEP / 2.0 - 0.01 
        math_com_y = ginga_amplitude * math.sin(math.pi * (t / self.T_STEP))

        return (math_com_x, math_com_y), (math_foot_r_x, math_foot_r_y, math_foot_r_z), (math_foot_l_x, math_foot_l_y, math_foot_l_z)

    def suavizar(self, x):
        return 0.5 * (1.0 - math.cos(math.pi * x))


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    robotId = p.loadURDF("urdf/aurea_urdf_pkg.urdf", [0, 0, 0.45], useFixedBase=False)

    ik = RelativeAnalyticalIK(robotId)
    brain = WalkingMath(h_target=0.2)

    ID_R_SHO_PITCH = 5
    ID_L_SHO_PITCH = 2
    ID_R_SHO_ROLL  = 6
    ID_L_SHO_ROLL  = 3
    ID_R_ELBOW     = 7
    ID_L_ELBOW     = 4

    ik.set_fixed_joint(ID_R_SHO_ROLL, 1.4)
    ik.set_fixed_joint(ID_L_SHO_ROLL, -1.4)
    
    ik.set_fixed_joint(ID_R_SHO_PITCH, 0.4)
    ik.set_fixed_joint(ID_L_SHO_PITCH, 0.4)
    

    ik.set_fixed_joint(ID_R_ELBOW, 1.9) 
    ik.set_fixed_joint(ID_L_ELBOW, -1.9)

    print("Iniciando a First Pose...")
    
    ang_d_init = ik.solve_leg(0.0, -brain.Y_SEP/2, -brain.H_TARGET, is_left=False)
    ang_e_init = ik.solve_leg(0.0,  brain.Y_SEP/2, -brain.H_TARGET, is_left=True)

    for i, j in enumerate(ik.JUNTAS_DIR):
        p.resetJointState(robotId, j, ang_d_init[i])
    for i, j in enumerate(ik.JUNTAS_ESQ):
        p.resetJointState(robotId, j, ang_e_init[i])
        
    for j_arm, ang_arm in ik.fixed_joints.items():
        p.resetJointState(robotId, j_arm, ang_arm)

    for _ in range(360):
        ik.apply(ang_d_init, ang_e_init)
        p.stepSimulation()
        time.sleep(1/240.)

    print("First Pose estabilizada! Iniciando a Caminhada...")
    # ========================================

    for _ in range(100):
        ang_d = ik.solve_leg(0.0, -brain.Y_SEP/2, -brain.H_TARGET, is_left=False)
        ang_e = ik.solve_leg(0.0,  brain.Y_SEP/2, -brain.H_TARGET, is_left=True)
        ik.apply(ang_d, ang_e)
        p.stepSimulation()

    print("Iniciando Caminhada Relativa!")
    t = 0.0
    dt_sim = 1/240.
    
    cmd_vx = 0.07 

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