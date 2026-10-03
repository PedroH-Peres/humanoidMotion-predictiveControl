"""
Trajectory generator for analytical bipedal walking.

Port of trajectory_generator.cpp from the aurea_walk ROS2 package, with two
fixes to the swing-foot landing profile (see get_swing_foot_pose_at_time).

Conventions
-----------
- All positions are in the world frame, in metres. Yaw is in radians.
- A step lasts T seconds and is split into three phases:

      |-- double support --|------ single support ------|-- double support --|
      0                    tb                           te                   T

  with tb = T * ds_ratio / 2 and te = T - tb.
- phi = t / T is the normalised phase of the step (0 → 1).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pose2D:
    """Planar pose on the ground: (x, y) position and yaw."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0

    def copy(self) -> "Pose2D":
        return Pose2D(self.position.copy(), self.yaw)


@dataclass
class FootPose(Pose2D):
    """Planar pose of a foot, tagged with which leg it belongs to."""

    is_left: bool = True

    def copy(self) -> "FootPose":
        return FootPose(self.position.copy(), self.yaw, self.is_left)


def _h_phi(phi: float, phi_b: float, phi_e: float) -> float:
    """Horizontal blend: 1 before phi_b, 0 after phi_e, cosine in between."""
    if phi < phi_b:
        return 1.0
    if phi >= phi_e:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * (phi - phi_b) / (phi_e - phi_b)))


def _v_phi(phi: float, phi_b: float, phi_e: float) -> float:
    """Vertical bump: 0 outside [phi_b, phi_e], peaks at 1 in the middle."""
    if phi < phi_b or phi >= phi_e:
        return 0.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (phi - phi_b) / (phi_e - phi_b)))


def get_com_pose_at_time(
    t: float,
    T: float,
    ds_ratio: float,
    z_com: float,
    g: float,
    p_start: Pose2D,
    p_end: Pose2D,
    p_support: Pose2D,
) -> tuple[np.ndarray, float]:
    """
    CoM (x, y) position and yaw at time t within a step.

    The ZMP reference is piecewise linear: it moves from p_start to the support
    foot during the first double support, stays on the support foot during
    single support, and moves to p_end during the last double support. The CoM
    is the closed-form solution of the Linear Inverted Pendulum Model (LIPM)

        x_ddot = lambda^2 * (x - zmp),   lambda = sqrt(g / z_com)

    with boundary conditions x(0) = p_start and x(T) = p_end.
    """
    tb = (T * ds_ratio) / 2.0
    te = T - tb
    phi = t / T

    h = _h_phi(phi, tb / T, te / T)
    com_yaw = h * p_start.yaw + (1.0 - h) * p_end.yaw

    if z_com <= 0.0:
        return np.array(p_start.position, dtype=float), com_yaw

    lam = np.sqrt(g / z_com)

    # ZMP slopes during the two double-support phases
    m_d1 = (p_support.position - p_start.position) / tb if tb > 1e-6 else np.zeros(2)
    m_d2 = (p_end.position - p_support.position) / (T - te) if (T - te) > 1e-6 else np.zeros(2)

    k_d1 = (1.0 / lam) * m_d1 * np.sinh(-lam * tb)
    k_d2 = (1.0 / lam) * m_d2 * np.sinh(lam * (T - te))

    exp_pos = np.exp(lam * T)
    exp_neg = np.exp(-lam * T)
    denom = exp_pos - exp_neg
    if abs(denom) < 1e-6:
        return np.array(p_start.position, dtype=float), com_yaw

    # Homogeneous-solution coefficients that satisfy the boundary conditions
    c1 = (k_d2 - k_d1 * exp_neg) / denom
    c2 = (k_d1 * exp_pos - k_d2) / denom

    if t < tb:
        zmp = p_start.position + m_d1 * t
        sinh_t = (1.0 / lam) * m_d1 * np.sinh(lam * (t - tb))
        com_pos = zmp + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t) - sinh_t
    elif t >= te:
        zmp = p_support.position + m_d2 * (t - te)
        sinh_t = (1.0 / lam) * m_d2 * np.sinh(lam * (t - te))
        com_pos = zmp + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t) - sinh_t
    else:
        com_pos = p_support.position + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t)

    return com_pos, com_yaw


def get_swing_foot_pose_at_time(
    t: float,
    T: float,
    z_step: float,
    ds_ratio: float,
    p_start: Pose2D,
    p_end: Pose2D,
) -> tuple[np.ndarray, float]:
    """
    Swing foot (x, y, z) position and yaw at time t within a step.

    The foot follows a cosine blend horizontally and a cosine bump vertically.
    Near the end of single support the vertical bump is replaced by a linear
    descent so the foot lands with a constant, controlled vertical speed.
    """
    tb = (T * ds_ratio) / 2.0
    te = T - tb
    phi = t / T
    phi_b = tb / T
    phi_e = te / T

    h = _h_phi(phi, phi_b, phi_e)
    v = _v_phi(phi, phi_b, phi_e)

    pos_2d = h * p_start.position + (1.0 - h) * p_end.position
    yaw = h * p_start.yaw + (1.0 - h) * p_end.yaw

    landing_phase_start = phi_e * 0.85
    if phi > landing_phase_start:
        # Fix 1: the C++ original passed a time (landing_phase_start * T) where
        # _v_phi expects a phase, so the foot jumped up when landing started.
        z_land_start = z_step * _v_phi(landing_phase_start, phi_b, phi_e)
        # Fix 2: clamp so the foot stays on the ground (z = 0) during the final
        # double support instead of being commanded below the floor.
        landing_progress = min(
            (phi - landing_phase_start) / (phi_e - landing_phase_start), 1.0
        )
        z = (1.0 - landing_progress) * z_land_start
    else:
        z = z_step * v

    return np.array([pos_2d[0], pos_2d[1], z], dtype=float), yaw


def select_next_poses(
    current_torso: Pose2D,
    swing_foot: FootPose,
    v_cmd: dict,
    T: float,
    y_sep: float,
) -> tuple[Pose2D, FootPose]:
    """
    Plan the next torso pose and swing-foot landing pose.

    The torso moves by (vx, vy, wz) * T, expressed in its own frame. The swing
    foot lands y_sep to the side of the new torso pose and half a stride ahead.
    """
    dx = v_cmd.get("vx", 0.0) * T
    dy = v_cmd.get("vy", 0.0) * T
    d_yaw = v_cmd.get("wz", 0.0) * T

    yaw = current_torso.yaw
    c, s = np.cos(yaw), np.sin(yaw)
    world_disp = np.array([c * dx - s * dy, s * dx + c * dy])

    next_torso = Pose2D(current_torso.position + world_disp, yaw + d_yaw)

    foot_offset_y = y_sep if swing_foot.is_left else -y_sep

    ny = next_torso.yaw
    cn, sn = np.cos(ny), np.sin(ny)
    local_offset = np.array([v_cmd.get("vx", 0.0) * T / 2.0, foot_offset_y])
    world_offset = np.array([cn * local_offset[0] - sn * local_offset[1],
                             sn * local_offset[0] + cn * local_offset[1]])

    next_swing = FootPose(next_torso.position + world_offset, ny, swing_foot.is_left)

    return next_torso, next_swing
