"""
Analytical IK solver for the Robotis OP3 leg.

6-DOF chain per leg:
  hip_yaw (z) → hip_roll (x) → hip_pitch (y) → knee (y) → ank_pitch (y) → ank_roll (x)

Constants measured from the robot_descriptions OP3 MJCF
(mujoco_menagerie / op3_mj_description).

Approximation: the solver treats the hip_roll and ank_roll axes as passing
through the hip_pitch / ank_pitch points. On the OP3 they are offset by
[-0.024, ∓0.019] m, so the foot lands off target whenever the roll angles
are non-zero. With the lateral CoM sway of the default gait the ankle error
reaches ~4-5 mm (measured against MuJoCo forward kinematics). Modelling the
offsets exactly would remove it; it is a good exercise.
"""

import numpy as np

# ── Kinematic constants (metres) ──────────────────────────────────────────────
# Accumulated body_link → hip_pitch offset (at zero pose)
#   body → hip_yaw:   [0,  ±0.035, 0]
#   hip_yaw → roll:   [-0.024, 0, -0.0285]
#   hip_roll → pitch: [+0.0241, ±0.019, 0]
HIP_Y   = 0.054    # lateral offset body → hip_pitch  (sign varies per leg)
HIP_Z   = -0.0285  # vertical offset body → hip_pitch

L_COXA  = 0.1102   # hip_pitch → knee (vertical)
L_TIBIA = 0.1100   # knee      → ank_pitch (vertical)

# Distance from the ankle point to the foot sole. The "ankle point" is the
# ank_pitch joint (the ank_roll axis sits at the same height, offset by
# [-0.0241, ∓0.019], mirroring the hip). Foot contact boxes: z-half = 0.004,
# centre 0.0266 below ank_pitch → sole 0.0305 below it.
D_ANKLE_SOLE = 0.0305
# ─────────────────────────────────────────────────────────────────────────────


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


class IKSolver:
    """Analytical IK for one OP3 leg."""

    def solve(
        self,
        p_body: np.ndarray,
        R_body: np.ndarray,
        p_foot_target: np.ndarray,
        R_foot_target: np.ndarray,
        leg: str,
    ) -> tuple[bool, list[float]]:
        """
        Solve IK for one leg.

        Args:
            p_body:        (3,) body torso position in world frame.
            R_body:        (3,3) body rotation matrix in world frame.
            p_foot_target: (3,) foot **sole** target position in world frame.
            R_foot_target: (3,3) foot target orientation in world frame.
            leg:           'left' or 'right'.

        Returns:
            (success, angles) where angles = [hip_yaw, hip_roll, hip_pitch,
                                               knee, ank_pitch, ank_roll]
            Angles are in radians, following the geometric convention of the
            analytical solution.  Apply the joint-axis sign from the MJCF
            (hip_yaw/roll axes are negative) when commanding the robot.
        """
        R1 = R_body
        R7 = R_foot_target

        # Ankle point (ank_pitch joint) from the sole target
        p7_sole = p_foot_target
        p7 = p7_sole - R7 @ np.array([0.0, 0.0, -D_ANKLE_SOLE])

        sign = 1.0 if leg == "left" else -1.0
        hip_offset = np.array([0.0, HIP_Y * sign, HIP_Z])
        p2 = p_body + R1 @ hip_offset  # hip_pitch joint in world frame

        # Vector from ankle to hip expressed in foot frame
        r_vec = R7.T @ (p2 - p7)
        rx, ry, rz = r_vec

        C = float(np.linalg.norm(r_vec))
        if C > L_COXA + L_TIBIA or C < 1e-9:  # out of reach / degenerate
            return False, []

        # ── Knee angle (q5) ───────────────────────────────────────────────
        cos_q5 = (C**2 - L_COXA**2 - L_TIBIA**2) / (2.0 * L_COXA * L_TIBIA)
        cos_q5 = float(np.clip(cos_q5, -1.0, 1.0))
        q5 = np.arccos(cos_q5)

        # ── Ankle roll (q7) ───────────────────────────────────────────────
        q7 = np.arctan2(ry, rz)

        # ── Ankle pitch (q6) ──────────────────────────────────────────────
        sin_alpha = float(np.clip((L_COXA / C) * np.sin(q5), -1.0, 1.0))
        alpha = np.arcsin(sin_alpha)
        q6 = -np.arctan2(rx, np.copysign(1.0, rz) * np.sqrt(ry**2 + rz**2)) - alpha

        # ── Hip angles from remaining rotation ────────────────────────────
        R_q7_inv  = _rot_x(-q7)
        R_q56_inv = _rot_y(-q5 - q6)
        R_target  = R1.T @ R7 @ R_q7_inv @ R_q56_inv

        R12, R22        = R_target[0, 1], R_target[1, 1]
        R31, R32, R33   = R_target[2, 0], R_target[2, 1], R_target[2, 2]

        q2 = np.arctan2(-R12, R22)               # hip_yaw
        s2, c2 = np.sin(q2), np.cos(q2)
        q3 = np.arctan2(R32, -R12 * s2 + R22 * c2)  # hip_roll
        q4 = np.arctan2(-R31, R33)               # hip_pitch

        return True, [q2, q3, q4, q5, q6, q7]
