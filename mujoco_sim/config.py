"""
Configuration for the MuJoCo OP3 simulation.

Edit the values here to change how the robot is simulated and how it walks.
"""

# -----------------------------------------------------------------------------
#  SIMULATION MODE
#  True  → mj_step: full physics (contacts, gravity, PD actuators)
#  False → mj_forward: kinematics only; joints are set directly, no contacts
# -----------------------------------------------------------------------------

PHYSICS = True

# -----------------------------------------------------------------------------
#  POSITION ACTUATOR GAINS (applied to every joint)
#  ACTUATOR_KP          — proportional gain [N·m/rad]
#  ACTUATOR_KV          — derivative (damping) gain [N·m·s/rad]
#  ACTUATOR_FORCE_RANGE — torque limit per joint [N·m]
# -----------------------------------------------------------------------------

ACTUATOR_KP = 120.0
ACTUATOR_KV = 12.0
ACTUATOR_FORCE_RANGE = 80.0

# -----------------------------------------------------------------------------
#  WALKING ENGINE PARAMETERS
# -----------------------------------------------------------------------------

WALK_PARAMS = {
    "T":        0.4,    # Step period [s] — smaller = faster steps
    "z_com":    0.22,   # Torso height [m]
    "z_step":   0.04,   # Maximum swing-foot height [m]
    "ds_ratio": 0.12,   # Fraction of the step in double support (0 to 0.5)
    "y_sep":    0.054,  # Lateral distance from torso centre to each foot [m]
}

# Velocity commands bound to the keyboard keys (see sim.py)
KEY_COMMANDS = {
    "forward": {"vx": 0.05},  # [m/s]
    "lateral": {"vy": 0.03},  # [m/s]
    "turn":    {"wz": 0.5},   # [rad/s]
}

# -----------------------------------------------------------------------------
#  ROBOT-SPECIFIC MAPPING
# -----------------------------------------------------------------------------

# Sign that converts the IK's geometric angle convention into the joint axes
# defined in the OP3 MJCF (some axes point in the opposite direction).
JOINT_SIGNS = {
    #  Joint           Left        Right
    "hip_yaw":   {"l": -1.0,  "r": -1.0},
    "hip_roll":  {"l": -1.0,  "r": -1.0},
    "hip_pitch": {"l":  1.0,  "r": -1.0},
    "knee":      {"l":  1.0,  "r": -1.0},
    "ank_pitch": {"l": -1.0,  "r":  1.0},
    "ank_roll":  {"l":  1.0,  "r":  1.0},
}

# Fixed arm pose [rad] held during walking
ARM_POSE = {
    "l_sho_pitch":  0.0,
    "r_sho_pitch":  0.0,
    "l_sho_roll":   1.4,
    "r_sho_roll":  -1.4,
    "l_el":         0.0,
    "r_el":         0.0,
}


def joint_name(side: str, key: str) -> str:
    """MJCF joint name, e.g. joint_name("l", "knee") → "l_knee"."""
    return f"{side}_{key}"
