import math
import pybullet as p

class RelativeAnalyticalIK:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # --- PARÂMETROS DO ROBÔ ---
        self.L_COXA = 0.12  
        self.L_TIBIA = 0.085  
        self.D_QUADRIL_OFFSET = 0.044 
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
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_DIR[i], p.POSITION_CONTROL, targetPosition=angle, force=5000, positionGain=1.0)
        for i, angle in enumerate(angulos_esq):
            p.setJointMotorControl2(self.robot_id, self.JUNTAS_ESQ[i], p.POSITION_CONTROL, targetPosition=angle, force=5000, positionGain=1.0)
        
        for j in range(p.getNumJoints(self.robot_id)):
            if j not in self.JUNTAS_DIR and j not in self.JUNTAS_ESQ:
                target = self.fixed_joints.get(j, 0.0) 
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=5000, positionGain=1.0)