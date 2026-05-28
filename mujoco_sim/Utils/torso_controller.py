"""
Simulated IMU + Torso Orientation Controller for the OP3.
"""

import numpy as np


class SimulatedIMU:
    def read(self, model, data) -> dict:
        """
        Returns a dict with:
            pitch      [rad] — torso pitch (positive = forward lean)
            roll       [rad] — torso roll  (positive = lean left)
            pitch_rate [rad/s]
            roll_rate  [rad/s]
        """
        adr = model.jnt_qposadr[0]
        w, x, y, z = data.qpos[adr+3:adr+7] 

        roll  = float(np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y)))
        pitch = float(np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0)))

        dof = model.jnt_dofadr[0]
        wx, wy = float(data.qvel[dof+3]), float(data.qvel[dof+4])

        return {"pitch": pitch, "roll": roll, "pitch_rate": wy, "roll_rate": wx}


class TorsoOrientationController:
    """
    PD controller that keeps the torso upright by correcting hip angles.

    Args:
        kp_pitch, kd_pitch: proportional / derivative gains for pitch axis
        kp_roll,  kd_roll:  proportional / derivative gains for roll  axis
        max_delta:          hard clamp on each correction output [rad]
    """

    def __init__(
        self,
        kp_pitch: float = 1.5,
        kd_pitch: float = 0.05,
        kp_roll:  float = 1.5,
        kd_roll:  float = 0.05,
        max_delta: float = 0.25,
        target_pitch: float = 0.0,  
        target_roll:  float = 0.0,  
    ):
        self.kp_pitch     = kp_pitch
        self.kd_pitch     = kd_pitch
        self.kp_roll      = kp_roll
        self.kd_roll      = kd_roll
        self.max_delta    = max_delta
        self.target_pitch = target_pitch
        self.target_roll  = target_roll

    def compute(self, imu: dict) -> dict:
        """
        Compute hip angle corrections from IMU readings.
        """
        err_pitch  = imu["pitch"] - self.target_pitch
        err_roll   = imu["roll"]  - self.target_roll
        pitch_rate = imu["pitch_rate"]
        roll_rate  = imu["roll_rate"]

        d_pitch = float(np.clip(
            self.kp_pitch * err_pitch + self.kd_pitch * pitch_rate,
            -self.max_delta, self.max_delta,
        ))
        d_roll = float(np.clip(
            self.kp_roll * err_roll + self.kd_roll * roll_rate,
            -self.max_delta, self.max_delta,
        ))

        return {
            "l_hip_pitch":  d_pitch,
            "r_hip_pitch": -d_pitch,
            "l_hip_roll":  -d_roll,
            "r_hip_roll":  -d_roll,
        }
