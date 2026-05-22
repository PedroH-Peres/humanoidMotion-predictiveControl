"""
Analytical walking state machine for bipedal robots.

Usage:
    engine = WalkingEngine()
    engine.set_command(vx=0.1)          # start walking forward
    body_pos, body_yaw, lp, ly, rp, ry = engine.step(dt)
    # body_pos: np.ndarray(3,) – desired pelvis position in world frame
    # body_yaw: float           – desired pelvis yaw
    # lp / rp:  np.ndarray(3,) – left/right sole target in world frame
    # ly / ry:  float           – left/right foot yaw
"""

import numpy as np
from .trajectory_generator import (
    get_com_pose_at_time,
    get_swing_foot_pose_at_time,
    select_next_poses,
)


class WalkingEngine:
    IDLE       = 0
    WALKING    = 1
    IDLE_MARCH = 2
    STOPPING   = 3

    def __init__(
        self,
        T: float         = 0.3,
        z_com: float     = 0.22,
        z_step: float    = 0.03,
        ds_ratio: float  = 0.1,
        y_sep: float     = 0.054,
        g: float         = 9.81,
    ):
        self.T        = T
        self.z_com    = z_com
        self.z_step   = z_step
        self.ds_ratio = ds_ratio
        self.y_sep    = y_sep
        self.g        = g

        self.state  = self.IDLE
        self.t_step = 0.0

        # Foot states (world frame 2-D positions)
        self.left_foot  = {"position": np.array([0.0,  y_sep]), "yaw": 0.0, "is_left": True}
        self.right_foot = {"position": np.array([0.0, -y_sep]), "yaw": 0.0, "is_left": False}

        self.support_foot    = self.right_foot
        self.swing_foot_ref  = self.left_foot

        # Torso state (world frame)
        self.torso = {"position": np.array([0.0, 0.0]), "yaw": 0.0}

        # Step plan (set by start_new_step)
        self.torso_start  = None
        self.torso_target = None
        self.swing_start  = None
        self.swing_target = None

        # Velocity command
        self._cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self._cmd = {"vx": vx, "vy": vy, "wz": wz}

    def request_stop(self) -> None:
        if self.state in (self.WALKING, self.IDLE_MARCH):
            self.state = self.STOPPING

    def step(self, dt: float) -> tuple:
        """
        Advance the engine by dt seconds and return the desired kinematic state.

        Returns:
            (body_pos, body_yaw, left_sole, left_yaw, right_sole, right_yaw)
            body_pos   – np.ndarray(3,): desired pelvis world position [x, y, z_com]
            body_yaw   – float: desired pelvis yaw
            left_sole  – np.ndarray(3,): desired left foot sole world position
            left_yaw   – float: desired left foot yaw
            right_sole – np.ndarray(3,): desired right foot sole world position
            right_yaw  – float: desired right foot yaw
        """
        self._update_state()

        if self.state == self.IDLE:
            return self._standing_pose()

        self.t_step += dt

        com_2d, com_yaw = get_com_pose_at_time(
            self.t_step, self.T, self.ds_ratio, self.z_com, self.g,
            self.torso_start, self.torso_target, self.support_foot,
        )

        swing_3d, swing_yaw = get_swing_foot_pose_at_time(
            self.t_step, self.T, self.z_step, self.ds_ratio,
            self.swing_start, self.swing_target,
        )

        body_pos = np.array([com_2d[0], com_2d[1], self.z_com])
        sup_pos  = np.array([self.support_foot["position"][0],
                              self.support_foot["position"][1], 0.0])

        if self.swing_foot_ref["is_left"]:
            left_pos, left_yaw   = swing_3d, swing_yaw
            right_pos, right_yaw = sup_pos, self.support_foot["yaw"]
        else:
            right_pos, right_yaw = swing_3d, swing_yaw
            left_pos, left_yaw   = sup_pos, self.support_foot["yaw"]

        if self.t_step >= self.T:
            self._end_step()

        return body_pos, com_yaw, left_pos, left_yaw, right_pos, right_yaw

    def _update_state(self) -> None:
        should_walk = (abs(self._cmd["vx"]) > 0.01 or
                       abs(self._cmd["vy"]) > 0.01 or
                       abs(self._cmd["wz"]) > 0.01)

        prev = self.state

        if self.state == self.STOPPING:
            pass  # state changes happen in _end_step
        elif self.state == self.IDLE:
            if should_walk:
                self.state = self.WALKING
        elif self.state in (self.WALKING, self.IDLE_MARCH):
            self.state = self.WALKING if should_walk else self.IDLE_MARCH

        if prev == self.IDLE and self.state != self.IDLE:
            self._start_new_step()

    def _start_new_step(self) -> None:
        self.t_step = 0.0

        # Swap support / swing
        if self.support_foot["is_left"]:
            self.support_foot   = self.right_foot
            self.swing_foot_ref = self.left_foot
        else:
            self.support_foot   = self.left_foot
            self.swing_foot_ref = self.right_foot

        cmd = self._cmd if self.state == self.WALKING else {"vx": 0.0, "vy": 0.0, "wz": 0.0}

        self.torso_target, self.swing_target = select_next_poses(
            self.torso, self.swing_foot_ref, cmd, self.T, self.y_sep
        )
        self.torso_start = {
            "position": self.torso["position"].copy(),
            "yaw": self.torso["yaw"],
        }
        self.swing_start = {
            "position": self.swing_foot_ref["position"].copy(),
            "yaw": self.swing_foot_ref["yaw"],
            "is_left": self.swing_foot_ref["is_left"],
        }

    def _end_step(self) -> None:
        self.torso["position"] = self.torso_target["position"].copy()
        self.torso["yaw"]      = self.torso_target["yaw"]
        self.swing_foot_ref["position"] = self.swing_target["position"].copy()
        self.swing_foot_ref["yaw"]      = self.swing_target["yaw"]

        if self.state == self.STOPPING:
            self.state = self.IDLE
        else:
            self._start_new_step()

    def _standing_pose(self) -> tuple:
        body_pos   = np.array([self.torso["position"][0],
                                self.torso["position"][1], self.z_com])
        body_yaw   = self.torso["yaw"]
        left_pos   = np.array([self.left_foot["position"][0],
                                self.left_foot["position"][1], 0.0])
        right_pos  = np.array([self.right_foot["position"][0],
                                self.right_foot["position"][1], 0.0])
        return (body_pos, body_yaw,
                left_pos,  self.left_foot["yaw"],
                right_pos, self.right_foot["yaw"])
