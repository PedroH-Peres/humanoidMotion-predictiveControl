"""
Walking engine whose CoM trajectory comes from a model predictive controller (MPC).

It has the same interface as the analytical `WalkingEngine` (`set_command`,
`step`, ...) and reuses its state machine and its swing-foot trajectories.
The only thing that changes is how the CoM (x, y) is computed.

The idea, in four steps
-----------------------
1. MODEL. The CoM is a Linear Inverted Pendulum (LIPM). Each axis (x and y,
   independent) has the state s = [position, velocity, acceleration] and is
   driven by the jerk u. The ZMP is a linear function of the state:

       s[k+1] = A s[k] + B u[k]            p[k] = C s[k] = x - (z_c / g) * acceleration

2. REFERENCE. From the planned footsteps we know where the ZMP should be (p_ref)
   and where it is allowed to be (p_min, p_max: inside the support polygon)
   for the next N samples.

3. OPTIMISATION. The unknowns are the future ZMPs P = [p[1] ... p[N]]. Stacking
   the model gives  P = Ps s0 + Pu U  (U = the jerks). Pu can be inverted, so the
   jerks follow from P:  U = Pu^-1 (P - Ps s0). We look for the P that stays close
   to p_ref, with few jerks, inside the support polygon:

       minimise    w_zmp * |P - p_ref|^2 + w_jerk * |Pu^-1 (P - Ps s0)|^2
       subject to  p_min <= P <= p_max

   This is a quadratic program whose only constraints are bounds on each unknown.

4. RECEDING HORIZON. Only the first jerk, u[0] = (P[0] - Ps[0] s0) / Pu[0, 0], is
   applied, for one MPC period. The model advances and the problem is solved again.

Frames: world frame, metres, yaw in radians (same as the analytical engine).
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..analytical_walking.trajectory_generator import (
    FootPose,
    Pose2D,
    get_com_pose_at_time,
    get_swing_foot_pose_at_time,
    select_next_poses,
)
from ..analytical_walking.walking_engine import WalkingEngine, WalkState, WalkTarget


# -----------------------------------------------------------------------------
#  1. Model
# -----------------------------------------------------------------------------

def lipm_matrices(delta: float, z_com: float, g: float = 9.81):
    """
    LIPM with jerk input, discretised with period `delta`.

    Returns (A, B, C) such that  s[k+1] = A s[k] + B u[k]  and  zmp = C s.
    """
    A = np.array([[1.0, delta, delta ** 2 / 2.0],
                  [0.0, 1.0,   delta],
                  [0.0, 0.0,   1.0]])
    B = np.array([delta ** 3 / 6.0, delta ** 2 / 2.0, delta])
    C = np.array([1.0, 0.0, -z_com / g])
    return A, B, C


def prediction_matrices(A, B, C, N: int):
    """
    Predictions over N steps as one matrix product:  P = Ps @ s0 + Pu @ U.

    P[i] is the ZMP after i+1 steps, so
        Ps[i]    = C A^(i+1)
        Pu[i, j] = C A^(i-j) B     (j <= i; zero above the diagonal)
    """
    Ps = np.zeros((N, 3))
    Pu = np.zeros((N, N))
    powers = [np.eye(3)]                      # powers[k] = A^k
    for _ in range(N):
        powers.append(A @ powers[-1])
    for i in range(N):
        Ps[i] = C @ powers[i + 1]
        for j in range(i + 1):
            Pu[i, j] = C @ powers[i - j] @ B
    return Ps, Pu


# -----------------------------------------------------------------------------
#  3. The quadratic program
# -----------------------------------------------------------------------------

def solve_box_qp(H: np.ndarray, q: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 max_iter: int = 60) -> np.ndarray:
    """
    minimise 1/2 x'Hx + q'x   subject to   lo <= x <= hi     (H symmetric positive definite)

    Projected Newton method. It starts from the unconstrained optimum clipped to the
    bounds. Each iteration pins the variables that sit on a bound and are pushed
    outwards by the gradient, takes a Newton step on the others, and clips the result
    back into the bounds. It normally converges in a handful of iterations.
    """
    def cost(v):
        return 0.5 * v @ H @ v + q @ v

    x = np.clip(np.linalg.solve(H, -q), lo, hi)
    for _ in range(max_iter):
        grad = H @ x + q
        pinned = ((x <= lo + 1e-9) & (grad > 0)) | ((x >= hi - 1e-9) & (grad < 0))
        free = ~pinned
        if not free.any():
            break

        step = np.zeros_like(x)
        step[free] = -np.linalg.solve(H[np.ix_(free, free)], grad[free])
        if np.abs(step).max() < 1e-10:
            break

        t = 1.0                                       # shorten the step until the cost drops
        while True:
            x_new = np.clip(x + t * step, lo, hi)
            if cost(x_new) <= cost(x) + 1e-4 * grad @ (x_new - x) or t < 1e-8:
                break
            t *= 0.5
        x = x_new
    return x


# -----------------------------------------------------------------------------
#  Parameters and footstep preview
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MPCParams:
    """
    Tuning of the controller.

    The horizon is short on purpose: the LIPM forgets the future quickly (time constant
    1 / sqrt(g / z_c) = 0.15 s), so 0.8 s already holds all the useful information. Much
    longer horizons make the QP badly conditioned: 1.6 s is ~40000 times worse.
    """

    dt: float = 0.04                # period of the MPC [s]; a whole multiple of the sim timestep
    horizon: int = 20               # N samples ahead (N * dt = 0.8 s, two steps)
    w_zmp: float = 1.0              # weight on the ZMP tracking error
    w_jerk: float = 1e-6            # weight on the jerk (smoothness)
    foot_half_length: float = 0.05  # half of the sole length [m] (adjust to the OP3 foot)
    foot_half_width: float = 0.03   # half of the sole width [m]
    zmp_margin: float = 0.01        # keep the ZMP this far inside the sole [m]


class PlannedStep(NamedTuple):
    """One step of the footstep preview. Positions are 2D, in the world frame."""

    t0: float                # start of the step, relative to the start of the current step [s]
    torso_start: np.ndarray  # nominal torso position at the start ...
    torso_end: np.ndarray    # ... and at the end of the step
    support: np.ndarray      # support foot
    previous: np.ndarray     # foot that was the support in the previous step (leaves the ground)
    landing: np.ndarray      # where the swing foot lands
    standing: bool = False   # robot stands still on `support` and `landing`


# -----------------------------------------------------------------------------
#  Engine
# -----------------------------------------------------------------------------

class MPCWalkingEngine(WalkingEngine):
    def __init__(
        self,
        T: float = 0.5,
        z_com: float = 0.22,
        z_step: float = 0.04,
        ds_ratio: float = 0.13,
        y_sep: float = 0.054,
        g: float = 9.81,
        *,
        mpc: MPCParams | None = None,
    ):
        super().__init__(T, z_com, z_step, ds_ratio, y_sep, g)
        self.mpc = mpc or MPCParams()

        # Matrices that depend only on the parameters: build them once
        A, B, C = lipm_matrices(self.mpc.dt, z_com, g)
        self._Ps, self._Pu = prediction_matrices(A, B, C, self.mpc.horizon)
        Pu_inv = np.linalg.inv(self._Pu)
        self._M = Pu_inv.T @ Pu_inv                              # |Pu^-1 x|^2 = x' M x
        self._H = (self.mpc.w_zmp * np.eye(self.mpc.horizon)
                   + self.mpc.w_jerk * self._M)                  # Hessian of the QP

        self.debug: dict = {}      # last reference, bounds and ZMP plan (for plotting)
        self._reset()

    @property
    def zmp(self) -> np.ndarray:
        """ZMP (x, y) of the current model state."""
        c = lipm_matrices(self.mpc.dt, self.z_com, self.g)[2]
        return self._state @ c

    def step(self, dt: float) -> WalkTarget:
        """Advance the engine by dt seconds and return the desired kinematic state."""
        self._update_state()

        if self.state == WalkState.IDLE:
            if self._is_settling():
                # After a stop the CoM is still a little off the torso: let the MPC bring it
                # to rest instead of snapping it there in a single tick
                com_2d = self._advance_mpc(dt)
                return self.standing_pose()._replace(body_pos=np.array([*com_2d, self.z_com]))
            self._reset()
            return self.standing_pose()

        com_2d = self._advance_mpc(dt)
        self.t_step += dt

        # Only the yaw is taken from the analytical generator
        _, com_yaw = get_com_pose_at_time(
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

    # ------------------------------------------------------------------ MPC loop

    def _reset(self) -> None:
        """Back to standing still; the first walking tick starts the MPC again."""
        self._state = np.array([[self.torso.position[0], 0.0, 0.0],    # x: pos, vel, acc
                                [self.torso.position[1], 0.0, 0.0]])   # y: pos, vel, acc
        self._jerk = np.zeros(2)     # jerk being applied (x, y)
        self._tick = 0

    def _is_settling(self) -> bool:
        """True while the model CoM has not come to rest over the standing torso."""
        off_center = np.abs(self._state[:, 0] - self.torso.position).max()
        return off_center > 1e-3 or np.abs(self._state[:, 1]).max() > 1e-3

    def _starting_state(self) -> np.ndarray:
        """
        CoM state to start walking from, taken from the analytical generator at t = 0.

        Starting from rest does not work: the first step is already on a single foot, the ZMP
        is limited to that foot, and a CoM at rest between the feet can only accelerate away
        from it (the LIPM diverges). The analytical solution starts the step with the lateral
        velocity that carries the CoM over the support foot, so we begin from that state.
        """
        h = 1e-4
        args = (self.T, self.ds_ratio, self.z_com, self.g,
                self.torso_start, self.torso_target, self.support_foot)
        pos, _ = get_com_pose_at_time(0.0, *args)
        ahead, _ = get_com_pose_at_time(h, *args)
        vel = (ahead - pos) / h
        # The ZMP starts on the CoM (the reference goes from the torso to the support foot),
        # so the LIPM gives zero acceleration
        return np.column_stack([pos, vel, np.zeros(2)])

    def _advance_mpc(self, dt: float) -> np.ndarray:
        """Solve the QP every `mpc.dt`, advance the model by `dt`, return the CoM (x, y)."""
        ticks_per_cycle = round(self.mpc.dt / dt)
        if abs(ticks_per_cycle * dt - self.mpc.dt) > 1e-9:
            raise ValueError(f"mpc.dt ({self.mpc.dt} s) must be a whole multiple of the "
                             f"simulation timestep ({dt} s)")

        if self._tick == 0:
            self._state = self._starting_state()
        if self._tick % ticks_per_cycle == 0:
            self._solve()
        self._tick += 1

        # 4. Receding horizon: the first jerk is held until the next solve
        A, B, _ = lipm_matrices(dt, self.z_com, self.g)
        for axis in (0, 1):
            self._state[axis] = A @ self._state[axis] + B * self._jerk[axis]
        return self._state[:, 0].copy()

    def _solve(self) -> None:
        """Solve the QP of each axis from the current state and keep the first jerk."""
        p_ref, p_min, p_max = self._horizon_references()
        w, w_jerk = self.mpc.w_zmp, self.mpc.w_jerk
        plan = np.zeros_like(p_ref)
        for axis in (0, 1):
            free = self._Ps @ self._state[axis]        # ZMP if the jerk were zero from now on
            q = -w * p_ref[:, axis] - w_jerk * self._M @ free
            P = solve_box_qp(self._H, q, p_min[:, axis], p_max[:, axis])
            self._jerk[axis] = (P[0] - free[0]) / self._Pu[0, 0]
            plan[:, axis] = P
        self.debug = {"p_ref": p_ref, "p_min": p_min, "p_max": p_max, "zmp_plan": plan}

    # ------------------------------------------- 2. reference and support polygon

    def _footstep_preview(self) -> list[PlannedStep]:
        """
        Footsteps that cover the horizon, starting with the step in progress.

        Assumes the velocity command stays constant, as the analytical engine does when a
        step starts. After a stop request the current step is the last one: the robot then
        stands on both feet (a single "standing" step, which is also the plan while idle).
        """
        T = self.T
        if self.state == WalkState.IDLE:        # standing: settling after a stop
            torso = self.torso.position
            return [PlannedStep(0.0, torso, torso, self.left_foot.position,
                                self.right_foot.position, self.right_foot.position, standing=True)]

        n_steps = int(np.ceil(self.mpc.dt * self.mpc.horizon / T)) + 1
        walking = self.state == WalkState.WALKING
        cmd = self._cmd if walking else {"vx": 0.0, "vy": 0.0, "wz": 0.0}

        # Start with the step in progress...
        torso_start = self.torso_start.position
        torso_end = self.torso_target.position
        support = self.support_foot.position
        previous = self.swing_start.position
        landing = self.swing_target.position
        support_is_left = self.support_foot.is_left
        yaw = self.torso_target.yaw

        steps = []
        for k in range(n_steps):
            steps.append(PlannedStep(k * T, torso_start, torso_end, support, previous, landing))

            if self.state == WalkState.STOPPING:
                steps.append(PlannedStep((k + 1) * T, torso_end, torso_end, support, landing,
                                         landing, standing=True))
                break

            # ... then plan the next one: the foot that just landed becomes the support
            swing = FootPose(support, yaw, is_left=support_is_left)
            next_torso, next_swing = select_next_poses(
                Pose2D(torso_end, yaw), swing, cmd, T, self.y_sep)
            torso_start, torso_end = torso_end, next_torso.position
            previous, support, landing = support, landing, next_swing.position
            support_is_left = not support_is_left
            yaw = next_torso.yaw

        return steps

    def _zmp_reference(self, step: PlannedStep, t_local: float):
        """
        Where the ZMP should be, and which feet are on the ground, t_local seconds into a step.

            double support  ZMP moves from the torso to the support foot
            single support  ZMP stays on the support foot
            double support  ZMP moves from the support foot to the next torso
        """
        T = self.T
        t_b = T * self.ds_ratio / 2.0       # end of the first double support
        t_e = T - t_b                       # start of the last double support

        if step.standing:
            return step.torso_end, (step.support, step.landing)
        if t_local < t_b:
            a = t_local / t_b
            return step.torso_start + a * (step.support - step.torso_start), (step.previous, step.support)
        if t_local >= t_e:
            a = (t_local - t_e) / (T - t_e)
            return step.support + a * (step.torso_end - step.support), (step.support, step.landing)
        return step.support, (step.support,)

    def _horizon_references(self):
        """
        ZMP reference and bounds at the N times t + dt, t + 2 dt, ... (t = now in the step).

        Returns (p_ref, p_min, p_max), each of shape (N, 2): columns are x and y. The bounds
        are the bounding box of the feet on the ground, shrunk by the sole margin. The foot
        yaw is ignored, which is conservative while turning.
        """
        N, dt, T = self.mpc.horizon, self.mpc.dt, self.T
        half = np.array([self.mpc.foot_half_length, self.mpc.foot_half_width]) - self.mpc.zmp_margin

        steps = self._footstep_preview()
        p_ref = np.zeros((N, 2))
        p_min = np.zeros((N, 2))
        p_max = np.zeros((N, 2))

        for i in range(N):
            t = self.t_step + (i + 1) * dt
            step = steps[min(int(t // T), len(steps) - 1)]
            t_local = min(t - step.t0, T - 1e-9)
            p_ref[i], feet = self._zmp_reference(step, t_local)
            feet = np.array(feet)
            p_min[i] = feet.min(axis=0) - half
            p_max[i] = feet.max(axis=0) + half

        return p_ref, p_min, p_max
