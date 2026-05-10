import math
import pybullet as p

class RelativeAnalyticalIK:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # --- PARÂMETROS DO ROBÔ ---
        self.L_COXA = 0.12  
        self.L_TIBIA = 0.085  
        self.D_QUADRIL_OFFSET = 0.0425
        self.D_TORNOZELO_SOLA = 0.048
        
        # IDs das juntas
        self.JUNTAS_DIR = [8, 9, 10, 11, 12, 13]
        self.JUNTAS_ESQ = [14, 15, 16, 17, 18, 19]

        self.fixed_joints = {}

    def set_fixed_joint(self, joint_id, target_angle):
        self.fixed_joints[joint_id] = target_angle

    def solve_leg(self, dx, dy, dz, is_left):
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
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_DIR[i], p.POSITION_CONTROL, targetPosition=angle, force=6000, positionGain=1.0)
        for i, angle in enumerate(angulos_esq):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_ESQ[i], p.POSITION_CONTROL, targetPosition=angle, force=6000, positionGain=1.0)
        
        for j in range(p.getNumJoints(self.robot_id)):
            if j not in self.JUNTAS_DIR and j not in self.JUNTAS_ESQ:
                target = self.fixed_joints.get(j, 0.0) 
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=6000, positionGain=1.0)

class NumericalIK:

    def __init__(self, robot_id, foot_r_link_idx=12, foot_l_link_idx=18):
        self.robot_id = robot_id
        self.D_TORNOZELO_SOLA = 0.048
        
        self.JUNTAS_DIR = [8, 9, 10, 11, 12, 13]
        self.JUNTAS_ESQ = [14, 15, 16, 17, 18, 19]

        self.FOOT_R_LINK = foot_r_link_idx
        self.FOOT_L_LINK = foot_l_link_idx

        self.fixed_joints = {}
        
        self.joint_to_dof = {}
        dof_idx = 0
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED:
                self.joint_to_dof[j] = dof_idx
                dof_idx += 1

    def set_fixed_joint(self, joint_id, target_angle):
        self.fixed_joints[joint_id] = target_angle

    def solve_leg(self, dx, dy, dz, is_left, h_target=0.22):

        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        target_z = (base_pos[2] - h_target) + (dz + self.D_TORNOZELO_SOLA)

        target_pos = [base_pos[0] + dx, base_pos[1] + dy, target_z]
        target_orn = p.getQuaternionFromEuler([0, 0, 0]) # Pé paralelo ao mundo
        
        link_alvo = self.FOOT_L_LINK if is_left else self.FOOT_R_LINK

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            link_alvo,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )

        juntas_da_perna = self.JUNTAS_ESQ if is_left else self.JUNTAS_DIR
        angulos = []
        for j in juntas_da_perna:
            dof_index = self.joint_to_dof[j]
            angulos.append(joint_poses[dof_index])
            
        return angulos

    def apply(self, angulos_dir, angulos_esq, force=5000, gain=1.0):
        for i, angle in enumerate(angulos_dir):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_DIR[i], p.POSITION_CONTROL, targetPosition=angle, force=force, positionGain=gain)
        for i, angle in enumerate(angulos_esq):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_ESQ[i], p.POSITION_CONTROL, targetPosition=angle, force=force, positionGain=gain)
        
        for j in range(p.getNumJoints(self.robot_id)):
            if j not in self.JUNTAS_DIR and j not in self.JUNTAS_ESQ:
                target = self.fixed_joints.get(j, 0.0) 
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=force, positionGain=gain)