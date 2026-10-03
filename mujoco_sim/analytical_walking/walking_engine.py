"""
Analytical walking state machine for bipedal robots.

Usage:
    engine = WalkingEngine()
    engine.set_command(vx=0.1)          # start walking forward
    target = engine.step(dt)
    target.body_pos     # np.ndarray(3,) – desired pelvis position in world frame
    target.body_yaw     # float          – desired pelvis yaw
    target.left_sole    # np.ndarray(3,) – left sole target in world frame
    target.left_yaw     # float          – left foot yaw
    target.right_sole / target.right_yaw

State machine:

    IDLE ──set_command(≠0)──▶ WALKING ◀──set_command(≠0 / 0)──▶ IDLE_MARCH
      │                          │                                   ▲
      └──────────────────────────┼────── march_in_place() ──────────┘
      ▲                          ▼                                   │
      └──(step ends)──── STOPPING ◀──────── request_stop() ─────────┘
"""

from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from .trajectory_generator import (
    FootPose,
    Pose2D,
    get_com_pose_at_time,
    get_swing_foot_pose_at_time,
    select_next_poses,
)


class WalkState(Enum):
    IDLE = auto()        # standing still, both feet on the ground
    WALKING = auto()     # stepping according to the velocity command
    IDLE_MARCH = auto()  # stepping in place (command is zero)
    STOPPING = auto()    # finishing the current step, then IDLE


class WalkTarget(NamedTuple):
    """Desired kinematic state for one control tick (all in world frame)."""

    body_pos: np.ndarray
    body_yaw: float
    left_sole: np.ndarray
    left_yaw: float
    right_sole: np.ndarray
    right_yaw: float


class WalkingEngine:
    def __init__(
        self,
        T: float = 0.5,
        z_com: float = 0.22,
        z_step: float = 0.04,
        ds_ratio: float = 0.13,
        y_sep: float = 0.054,
        g: float = 9.81,
    ):
        self.T = T
        self.z_com = z_com
        self.z_step = z_step
        self.ds_ratio = ds_ratio
        self.y_sep = y_sep
        self.g = g

        self.state = WalkState.IDLE
        self.t_step = 0.0

        self.left_foot = FootPose(np.array([0.0, y_sep]), 0.0, is_left=True)
        self.right_foot = FootPose(np.array([0.0, -y_sep]), 0.0, is_left=False)

        # References to left_foot / right_foot; swapped at every step
        self.support_foot = self.right_foot
        self.swing_foot = self.left_foot

        self.torso = Pose2D(np.array([0.0, 0.0]), 0.0)

        # Current step plan (set by _start_new_step)
        self.torso_start: Pose2D | None = None
        self.torso_target: Pose2D | None = None
        self.swing_start: FootPose | None = None
        self.swing_target: FootPose | None = None

        self._cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """Set the velocity command [m/s, m/s, rad/s] in the torso frame."""
        self._cmd = {"vx": vx, "vy": vy, "wz": wz}

    def march_in_place(self) -> None:
        """Keep stepping without moving (also starts stepping from IDLE)."""
        self._cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        if self.state == WalkState.IDLE:
            self.state = WalkState.IDLE_MARCH
            self._start_new_step()
        elif self.state == WalkState.STOPPING:
            self.state = WalkState.IDLE_MARCH

    def request_stop(self) -> None:
        """Finish the current step and go back to IDLE."""
        # Clear the command, otherwise IDLE would switch straight back to WALKING
        self._cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        if self.state in (WalkState.WALKING, WalkState.IDLE_MARCH):
            self.state = WalkState.STOPPING

    def step(self, dt: float) -> WalkTarget:
        """Advance the engine by dt seconds and return the desired kinematic state."""
        self._update_state()

        if self.state == WalkState.IDLE:
            return self.standing_pose()

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
        sup_pos = np.array([*self.support_foot.position, 0.0])

        if self.swing_foot.is_left:
            left_pos, left_yaw = swing_3d, swing_yaw
            right_pos, right_yaw = sup_pos, self.support_foot.yaw
        else:
            right_pos, right_yaw = swing_3d, swing_yaw
            left_pos, left_yaw = sup_pos, self.support_foot.yaw

        if self.t_step >= self.T:
            self._end_step()

        return WalkTarget(body_pos, com_yaw, left_pos, left_yaw, right_pos, right_yaw)

    def standing_pose(self) -> WalkTarget:
        """Kinematic state with both feet on the ground under the current torso."""
        return WalkTarget(
            body_pos=np.array([*self.torso.position, self.z_com]),
            body_yaw=self.torso.yaw,
            left_sole=np.array([*self.left_foot.position, 0.0]),
            left_yaw=self.left_foot.yaw,
            right_sole=np.array([*self.right_foot.position, 0.0]),
            right_yaw=self.right_foot.yaw,
        )

    def _update_state(self) -> None:
        should_walk = any(abs(v) > 0.01 for v in self._cmd.values())

        prev = self.state

        if self.state == WalkState.STOPPING:
            pass  # the transition to IDLE happens in _end_step
        elif self.state == WalkState.IDLE:
            if should_walk:
                self.state = WalkState.WALKING
        elif self.state in (WalkState.WALKING, WalkState.IDLE_MARCH):
            self.state = WalkState.WALKING if should_walk else WalkState.IDLE_MARCH

        if prev == WalkState.IDLE and self.state != WalkState.IDLE:
            self._start_new_step()

    def _start_new_step(self) -> None:
        self.t_step = 0.0

        # Swap support / swing
        if self.support_foot.is_left:
            self.support_foot, self.swing_foot = self.right_foot, self.left_foot
        else:
            self.support_foot, self.swing_foot = self.left_foot, self.right_foot

        cmd = self._cmd if self.state == WalkState.WALKING else {"vx": 0.0, "vy": 0.0, "wz": 0.0}

        self.torso_target, self.swing_target = select_next_poses(
            self.torso, self.swing_foot, cmd, self.T, self.y_sep
        )
        self.torso_start = self.torso.copy()
        self.swing_start = self.swing_foot.copy()

    def _end_step(self) -> None:
        self.torso.position = self.torso_target.position.copy()
        self.torso.yaw = self.torso_target.yaw
        self.swing_foot.position = self.swing_target.position.copy()
        self.swing_foot.yaw = self.swing_target.yaw

        if self.state == WalkState.STOPPING:
            self.state = WalkState.IDLE
        else:
            self._start_new_step()
