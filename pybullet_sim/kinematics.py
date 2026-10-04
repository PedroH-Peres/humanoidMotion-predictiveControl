"""
Analytical leg IK for the Aurea robot, relative to the torso.

The foot target is given as an offset (dx, dy, dz) from the torso centre, so
the walking code never needs the robot's pose in the world.
"""

import math

import pybullet as p

# Joint order inside each leg, as returned by LegIK.solve_leg
LEG_JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll"]


class LegIK:
    # ── Robot dimensions [m] ──────────────────────────────────────────────
    L_THIGH = 0.12          # hip_pitch → knee
    L_SHIN = 0.085          # knee → ank_pitch
    HIP_OFFSET_Y = 0.0425   # torso centre → hip, lateral
    ANKLE_TO_SOLE = 0.045   # ankle joints → foot sole

    # ── Position controller settings ──────────────────────────────────────
    MOTOR_FORCE = 100.0
    POSITION_GAIN = 1.0

    def __init__(self, robot_id: int):
        self.robot_id = robot_id

        name_to_id = {
            p.getJointInfo(robot_id, j)[1].decode(): j
            for j in range(p.getNumJoints(robot_id))
        }
        self.right_leg = [name_to_id[f"r_{name}"] for name in LEG_JOINTS]
        self.left_leg = [name_to_id[f"l_{name}"] for name in LEG_JOINTS]
        self._name_to_id = name_to_id

        # Target angles for every joint that is not part of a leg (default 0)
        self.fixed_joints: dict[int, float] = {}

    def set_fixed_joint(self, joint_name: str, target_angle: float) -> None:
        self.fixed_joints[self._name_to_id[joint_name]] = target_angle

    def solve_leg(self, dx: float, dy: float, dz: float, is_left: bool) -> list[float]:
        """
        Joint angles that place the foot sole at (dx, dy, dz) from the torso centre.

        The foot is kept parallel to the ground and hip_yaw is fixed at 0, so the
        problem reduces to a planar 2-link chain (thigh + shin) plus a roll.
        Targets out of reach are clamped to a fully stretched leg.
        """
        offset_y = self.HIP_OFFSET_Y * (1.0 if is_left else -1.0)

        # Vector from the ankle to the hip
        rx = -dx
        ry = -(dy - offset_y)
        rz = -(dz + self.ANKLE_TO_SOLE)

        C = math.sqrt(rx**2 + ry**2 + rz**2)
        C = min(C, self.L_THIGH + self.L_SHIN - 0.001)

        # Law of cosines for the knee
        cos_knee = (C**2 - self.L_THIGH**2 - self.L_SHIN**2) / (2 * self.L_THIGH * self.L_SHIN)
        knee = math.acos(max(-1.0, min(1.0, cos_knee)))

        ankle_roll = math.atan2(ry, rz)
        alpha = math.asin(max(-1.0, min(1.0, (self.L_THIGH / C) * math.sin(knee))))
        ankle_pitch = -math.atan2(rx, math.copysign(1.0, rz) * math.sqrt(ry**2 + rz**2)) - alpha

        # Foot parallel to the ground → hip compensates the ankle and knee
        hip_yaw = 0.0
        hip_roll = -ankle_roll
        hip_pitch = -ankle_pitch - knee

        return [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]

    def apply(self, right_angles: list[float], left_angles: list[float]) -> None:
        """Send leg angles and hold every other joint at its fixed target."""
        targets = dict(zip(self.right_leg, right_angles))
        targets.update(zip(self.left_leg, left_angles))
        for j in range(p.getNumJoints(self.robot_id)):
            target = targets.get(j, self.fixed_joints.get(j, 0.0))
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=target,
                                    force=self.MOTOR_FORCE,
                                    positionGain=self.POSITION_GAIN)

    def reset(self, right_angles: list[float], left_angles: list[float]) -> None:
        """Teleport the joints to the given angles (used for the initial pose)."""
        for j, angle in zip(self.right_leg, right_angles):
            p.resetJointState(self.robot_id, j, angle)
        for j, angle in zip(self.left_leg, left_angles):
            p.resetJointState(self.robot_id, j, angle)
        for j, angle in self.fixed_joints.items():
            p.resetJointState(self.robot_id, j, angle)
