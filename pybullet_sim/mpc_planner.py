"""
CoM trajectory planner using Model Predictive Control (MPC) on the LIPM.

For each horizontal axis (x and y separately) the planner solves the QP

    minimise   Σ_k  W_COM·(x_k - x_ref_k)² + W_ZMP·(zmp_k - zmp_ref_k)² + W_VEL·v_k²
    subject to x_{k+1} = x_k + v_k·dt
               v_{k+1} = v_k + ω²·(x_k - zmp_k)·dt       (LIPM, ω = √(g/h))
               zmp_min_k ≤ zmp_k ≤ zmp_max_k               (ZMP inside the support polygon)

over a horizon of N steps, and returns the first predicted state.

It also generates the reference footstep plan: a fixed step period, alternating
support feet, stride length = cmd_vx · T_STEP.
"""

import math

import cvxpy as cp
import numpy as np


class MPCPlanner:
    # Cost weights
    W_COM = 800.0   # track the CoM reference
    W_ZMP = 2.0     # keep the ZMP close to the centre of the support foot
    W_VEL = 10.0    # penalise CoM velocity (smoothness)

    def __init__(self, h_target=0.22, t_step=0.3, ds_ratio=0.1, y_sep=0.085,
                 dt_mpc=0.02, n_horizon=25):
        self.H_TARGET = h_target                # CoM height [m]
        self.T_STEP = t_step                    # step period [s]
        self.T_DS = t_step * ds_ratio           # double-support duration [s]
        self.T_SS = t_step * (1.0 - ds_ratio)   # single-support duration [s]
        self.Y_SEP = y_sep                      # distance between the feet [m]
        self.SWING_HEIGHT = 0.05                # max swing-foot height [m]

        self.dt_mpc = dt_mpc
        self.N = n_horizon
        self.g = 9.81
        self.omega = math.sqrt(self.g / self.H_TARGET)

        # Half-size of the ZMP support region around the foot centre [m]
        self.FOOT_MARGIN_X = 0.07
        self.FOOT_MARGIN_Y = 0.03

        self._setup_mpc_problem()

    def _setup_mpc_problem(self):
        # Parameters change every solve; the problem structure is built once
        self.p_x0 = cp.Parameter()
        self.p_vx0 = cp.Parameter()
        self.p_com_ref = cp.Parameter(self.N)
        self.p_zmp_ref = cp.Parameter(self.N)
        self.p_zmp_min = cp.Parameter(self.N)
        self.p_zmp_max = cp.Parameter(self.N)

        self.v_x = cp.Variable(self.N + 1)
        self.v_vx = cp.Variable(self.N + 1)
        self.v_zmp = cp.Variable(self.N)

        cost = 0
        constraints = [self.v_x[0] == self.p_x0, self.v_vx[0] == self.p_vx0]

        for k in range(self.N):
            ax = (self.omega**2) * (self.v_x[k] - self.v_zmp[k])
            constraints += [self.v_x[k + 1] == self.v_x[k] + self.v_vx[k] * self.dt_mpc]
            constraints += [self.v_vx[k + 1] == self.v_vx[k] + ax * self.dt_mpc]

            constraints += [self.v_zmp[k] >= self.p_zmp_min[k]]
            constraints += [self.v_zmp[k] <= self.p_zmp_max[k]]

            cost += self.W_COM * cp.sum_squares(self.v_x[k] - self.p_com_ref[k])
            cost += self.W_ZMP * cp.sum_squares(self.v_zmp[k] - self.p_zmp_ref[k])
            cost += self.W_VEL * cp.sum_squares(self.v_vx[k])

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve_mpc(self, x0, vx0, com_ref, zmp_ref, zmp_min, zmp_max):
        """Return the CoM (position, velocity) one MPC step ahead."""
        self.p_x0.value = x0
        self.p_vx0.value = vx0
        self.p_com_ref.value = com_ref
        self.p_zmp_ref.value = zmp_ref
        self.p_zmp_min.value = zmp_min
        self.p_zmp_max.value = zmp_max

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError as exc:
            print(f"[MPC] solver error: {exc}")

        if self.v_x.value is not None and self.prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return self.v_x.value[1], self.v_vx.value[1]

        # No solution: keep moving at constant velocity for this step
        print(f"[MPC] no solution (status={self.prob.status}); extrapolating")
        return x0 + vx0 * self.dt_mpc, vx0

    def get_foot_trajectories(self, t, cmd_vx):
        """World (x, y, z) of the right and left foot soles at time t."""
        step_idx = int(t / self.T_STEP)
        t_in_step = t % self.T_STEP
        is_left_support = (step_idx % 2 == 0)

        current_x = cmd_vx * (step_idx * self.T_STEP)
        next_x = cmd_vx * ((step_idx + 1) * self.T_STEP)
        previous_x = cmd_vx * ((step_idx - 1) * self.T_STEP) if step_idx > 0 else 0.0

        swing_progress = 0.0 if t_in_step < self.T_DS else (t_in_step - self.T_DS) / self.T_SS
        swing_z = self.SWING_HEIGHT * math.sin(math.pi * swing_progress) if swing_progress > 0 else 0.0
        swing_x = previous_x + (next_x - previous_x) * self._s_curve(swing_progress)

        if is_left_support:
            r_foot = (swing_x, -self.Y_SEP / 2.0, swing_z)
            l_foot = (current_x, self.Y_SEP / 2.0, 0.0)
        else:
            r_foot = (current_x, -self.Y_SEP / 2.0, 0.0)
            l_foot = (swing_x, self.Y_SEP / 2.0, swing_z)

        return r_foot, l_foot

    def get_references_y(self, t_current):
        """Lateral CoM/ZMP references and ZMP bounds over the horizon."""
        com_ref, zmp_ref = np.zeros(self.N), np.zeros(self.N)
        zmp_min, zmp_max = np.zeros(self.N), np.zeros(self.N)
        for k in range(self.N):
            t_future = t_current + k * self.dt_mpc
            step_idx = int(t_future / self.T_STEP)
            t_in_step = t_future % self.T_STEP
            is_left_support = (step_idx % 2 == 0)

            support_y = self.Y_SEP / 2.0 if is_left_support else -self.Y_SEP / 2.0

            if step_idx == 0:
                previous_y = 0.0
            else:
                previous_y = -self.Y_SEP / 2.0 if is_left_support else self.Y_SEP / 2.0

            if t_in_step < self.T_DS:
                # Double support: shift the CoM towards the new support foot;
                # the ZMP may lie anywhere between both feet
                com_ref[k] = previous_y + (support_y - previous_y) * self._s_curve(t_in_step / self.T_DS)
                zmp_ref[k] = support_y
                zmp_min[k] = -self.Y_SEP / 2.0 - self.FOOT_MARGIN_Y
                zmp_max[k] = self.Y_SEP / 2.0 + self.FOOT_MARGIN_Y
            else:
                # Single support: the ZMP must stay under the support foot
                com_ref[k], zmp_ref[k] = support_y, support_y
                zmp_min[k] = support_y - self.FOOT_MARGIN_Y
                zmp_max[k] = support_y + self.FOOT_MARGIN_Y
        return com_ref, zmp_ref, zmp_min, zmp_max

    def get_references_x(self, t_current, cmd_vx):
        """Forward CoM/ZMP references and ZMP bounds over the horizon."""
        com_ref, zmp_ref = np.zeros(self.N), np.zeros(self.N)
        zmp_min, zmp_max = np.zeros(self.N), np.zeros(self.N)
        for k in range(self.N):
            t_future = t_current + k * self.dt_mpc
            step_idx = int(t_future / self.T_STEP)
            t_in_step = t_future % self.T_STEP

            current_x = cmd_vx * (step_idx * self.T_STEP)
            previous_x = cmd_vx * ((step_idx - 1) * self.T_STEP) if step_idx > 0 else 0.0

            com_ref[k] = cmd_vx * t_future
            zmp_ref[k] = current_x

            if t_in_step < self.T_DS:
                zmp_min[k], zmp_max[k] = previous_x - self.FOOT_MARGIN_X, current_x + self.FOOT_MARGIN_X
            else:
                zmp_min[k], zmp_max[k] = current_x - self.FOOT_MARGIN_X, current_x + self.FOOT_MARGIN_X
        return com_ref, zmp_ref, zmp_min, zmp_max

    @staticmethod
    def _s_curve(x):
        """Smooth 0 → 1 cosine ramp for x in [0, 1]."""
        return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, x))))
