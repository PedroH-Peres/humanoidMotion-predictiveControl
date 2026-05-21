"""
Trajectory generator for analytical bipedal walking.
All poses are dicts: {'position': np.ndarray(2,), 'yaw': float, 'is_left': bool (optional)}
"""

import numpy as np


def _h_phi(phi: float, phi_b: float, phi_e: float) -> float:
    """Cosine interpolation from 1→0 during single support phase."""
    if phi < phi_b:
        return 1.0
    if phi >= phi_e:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * (phi - phi_b) / (phi_e - phi_b)))


def _v_phi(phi: float, phi_b: float, phi_e: float) -> float:
    """Smooth vertical profile for foot lift/land."""
    if phi < phi_b or phi >= phi_e:
        return 0.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (phi - phi_b) / (phi_e - phi_b)))


def get_com_pose_at_time(
    t: float,
    T: float,
    ds_ratio: float,
    z_com: float,
    g: float,
    p_start: dict,
    p_end: dict,
    p_support: dict,
) -> tuple[np.ndarray, float]:
    """
    Compute COM (x, y) position and yaw at time t within a step of period T.
    Uses LIPM dynamics for lateral/forward motion, cosine blend for yaw.

    Returns:
        (com_pos_2d, com_yaw)
    """
    tb = (T * ds_ratio) / 2.0
    te = T - tb
    phi = t / T

    h = _h_phi(phi, tb / T, te / T)
    com_yaw = h * p_start["yaw"] + (1.0 - h) * p_end["yaw"]

    if z_com <= 0.0:
        return np.array(p_start["position"], dtype=float), com_yaw

    lam = np.sqrt(g / z_com)

    m_d1 = (p_support["position"] - p_start["position"]) / tb if tb > 1e-6 else np.zeros(2)
    m_d2 = (p_end["position"] - p_support["position"]) / (T - te) if (T - te) > 1e-6 else np.zeros(2)

    k_d1 = (1.0 / lam) * m_d1 * np.sinh(-lam * tb)
    k_d2 = (1.0 / lam) * m_d2 * np.sinh(lam * (T - te))

    exp_pos = np.exp(lam * T)
    exp_neg = np.exp(-lam * T)
    denom = exp_pos - exp_neg
    if abs(denom) < 1e-6:
        return np.array(p_start["position"], dtype=float), com_yaw

    c1 = (k_d2 - k_d1 * exp_neg) / denom
    c2 = (k_d1 * exp_pos - k_d2) / denom

    if t < tb:
        zmp = p_start["position"] + m_d1 * t
        sinh_t = (1.0 / lam) * m_d1 * np.sinh(lam * (t - tb))
        com_pos = zmp + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t) - sinh_t
    elif t >= te:
        zmp = p_support["position"] + m_d2 * (t - te)
        sinh_t = (1.0 / lam) * m_d2 * np.sinh(lam * (t - te))
        com_pos = zmp + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t) - sinh_t
    else:
        com_pos = p_support["position"] + c1 * np.exp(lam * t) + c2 * np.exp(-lam * t)

    return com_pos, com_yaw


def get_swing_foot_pose_at_time(
    t: float,
    T: float,
    z_step: float,
    ds_ratio: float,
    p_start: dict,
    p_end: dict,
) -> tuple[np.ndarray, float]:
    """
    Compute swing foot position (x, y, z) and yaw in world frame at time t.

    Returns:
        (swing_pos_3d, swing_yaw)
    """
    tb = (T * ds_ratio) / 2.0
    te = T - tb
    phi = t / T
    phi_b = tb / T
    phi_e = te / T

    h = _h_phi(phi, phi_b, phi_e)
    v = _v_phi(phi, phi_b, phi_e)

    pos_2d = h * p_start["position"] + (1.0 - h) * p_end["position"]
    yaw = h * p_start["yaw"] + (1.0 - h) * p_end["yaw"]

    landing_phase_start = phi_e * 0.85
    if phi > landing_phase_start:
        phase_phi = min((phi - landing_phase_start) / (phi_e - landing_phase_start), 1.0)
        # Height at start of landing phase (faithful port of C++ code)
        z_land_start = z_step * _v_phi(landing_phase_start * T, phi_b, phi_e)
        z = (1.0 - phase_phi) * z_land_start
    else:
        z = z_step * v

    return np.array([pos_2d[0], pos_2d[1], z], dtype=float), yaw


def select_next_poses(
    current_torso: dict,
    swing_foot: dict,
    v_cmd: dict,
    T: float,
    y_sep: float,
) -> tuple[dict, dict]:
    """
    Plan the next torso and swing-foot target poses given a velocity command.

    v_cmd keys: 'vx', 'vy', 'wz'
    Returns:
        (next_torso, next_swing) as pose dicts
    """
    dx = v_cmd.get("vx", 0.0) * T
    dy = v_cmd.get("vy", 0.0) * T
    d_yaw = v_cmd.get("wz", 0.0) * T

    yaw = current_torso["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    world_disp = np.array([c * dx - s * dy, s * dx + c * dy])

    next_torso = {
        "position": current_torso["position"] + world_disp,
        "yaw": yaw + d_yaw,
    }

    is_left = swing_foot.get("is_left", True)
    foot_offset_y = y_sep if is_left else -y_sep

    ny = next_torso["yaw"]
    cn, sn = np.cos(ny), np.sin(ny)
    local_offset = np.array([v_cmd.get("vx", 0.0) * T / 2.0, foot_offset_y])
    world_offset = np.array([cn * local_offset[0] - sn * local_offset[1],
                              sn * local_offset[0] + cn * local_offset[1]])

    next_swing = {
        "position": next_torso["position"] + world_offset,
        "yaw": ny,
        "is_left": is_left,
    }

    return next_torso, next_swing
