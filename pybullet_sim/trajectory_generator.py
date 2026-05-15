import math
import numpy as np
import cvxpy as cp

class TrajectoryGenerator:
    def __init__(self, h_target=0.22, t_step=0.3, ds_ratio=0.1, y_sep=0.085, dt_mpc=0.02, n_horizon=25):
        self.H_TARGET = h_target
        self.T_STEP = t_step
        self.T_DS = t_step * ds_ratio       
        self.T_SS = t_step * (1.0 - ds_ratio) 
        self.Y_SEP = y_sep
                
        self.dt_mpc = dt_mpc
        self.N = n_horizon
        self.g = 9.81
        self.omega = math.sqrt(self.g / self.H_TARGET)
        
        self.FOOT_MARGIN_X = 0.07
        self.FOOT_MARGIN_Y = 0.03
        
        self._setup_mpc_problem()

    def _setup_mpc_problem(self):
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
            constraints += [self.v_x[k+1] == self.v_x[k] + self.v_vx[k] * self.dt_mpc]
            constraints += [self.v_vx[k+1] == self.v_vx[k] + ax * self.dt_mpc]
            
            constraints += [self.v_zmp[k] >= self.p_zmp_min[k]]
            constraints += [self.v_zmp[k] <= self.p_zmp_max[k]]
            
            cost += 800.0 * cp.sum_squares(self.v_x[k] - self.p_com_ref[k])  
            cost += 2.0 * cp.sum_squares(self.v_zmp[k] - self.p_zmp_ref[k])   
            cost += 10.0 * cp.sum_squares(self.v_vx[k])                
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve_mpc(self, x0, vx0, com_ref, zmp_ref, zmp_min, zmp_max):
        self.p_x0.value = x0
        self.p_vx0.value = vx0
        self.p_com_ref.value = com_ref
        self.p_zmp_ref.value = zmp_ref
        self.p_zmp_min.value = zmp_min
        self.p_zmp_max.value = zmp_max
        
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.v_x.value is not None: 
                return self.v_x.value[1], self.v_vx.value[1]
        except: pass
        return x0 + vx0 * self.dt_mpc, vx0

    def get_foot_trajectories(self, t, cmd_vx):
        step_idx = int(t / self.T_STEP)
        t_in_step = t % self.T_STEP
        is_left_support = (step_idx % 2 == 0)
        
        avanco_atual = cmd_vx * (step_idx * self.T_STEP)
        avanco_futuro = cmd_vx * ((step_idx + 1) * self.T_STEP)
        passo_anterior = cmd_vx * ((step_idx - 1) * self.T_STEP) if step_idx > 0 else 0.0
        
        progresso_swing = 0.0 if t_in_step < self.T_DS else (t_in_step - self.T_DS) / self.T_SS 
        
        z_math = 0.05 * math.sin(math.pi * progresso_swing) if progresso_swing > 0 else 0.0

        if is_left_support:
            r_foot = (passo_anterior + (avanco_futuro - passo_anterior) * self._s_curve(progresso_swing), 
                      -self.Y_SEP / 2.0, 
                      z_math)
            l_foot = (avanco_atual, self.Y_SEP / 2.0, 0.0)
        else:
            r_foot = (avanco_atual, -self.Y_SEP / 2.0, 0.0)
            l_foot = (passo_anterior + (avanco_futuro - passo_anterior) * self._s_curve(progresso_swing), 
                      self.Y_SEP / 2.0, 
                      z_math)
                      
        return r_foot, l_foot

    def get_references_y(self, t_current):
        com_ref, zmp_ref, zmp_min, zmp_max = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        for k in range(self.N):
            t_futuro = t_current + k * self.dt_mpc
            step_idx = int(t_futuro / self.T_STEP)
            t_in_step = t_futuro % self.T_STEP
            is_left_support = (step_idx % 2 == 0)
            
            alvo_y = self.Y_SEP / 2.0 if is_left_support else -self.Y_SEP / 2.0
            
            if step_idx == 0:
                pe_anterior_y = 0.0
            else:
                pe_anterior_y = -self.Y_SEP / 2.0 if is_left_support else self.Y_SEP / 2.0
            
            if t_in_step < self.T_DS:
                com_ref[k] = pe_anterior_y + (alvo_y - pe_anterior_y) * self._s_curve(t_in_step / self.T_DS)
                zmp_ref[k] = alvo_y
                zmp_min[k], zmp_max[k] = -self.Y_SEP/2.0 - self.FOOT_MARGIN_Y, self.Y_SEP/2.0 + self.FOOT_MARGIN_Y
            else:
                com_ref[k], zmp_ref[k] = alvo_y, alvo_y
                zmp_min[k], zmp_max[k] = alvo_y - self.FOOT_MARGIN_Y, alvo_y + self.FOOT_MARGIN_Y
        return com_ref, zmp_ref, zmp_min, zmp_max

    def get_references_x(self, t_current, cmd_vx):
        com_ref, zmp_ref, zmp_min, zmp_max = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        for k in range(self.N):
            t_futuro = t_current + k * self.dt_mpc
            step_idx = int(t_futuro / self.T_STEP)
            t_in_step = t_futuro % self.T_STEP
            
            avanco_atual = cmd_vx * (step_idx * self.T_STEP)
            passo_anterior = cmd_vx * ((step_idx - 1) * self.T_STEP) if step_idx > 0 else 0.0
            
            com_ref[k] = cmd_vx * t_futuro 
            zmp_ref[k] = avanco_atual    
            
            if t_in_step < self.T_DS:
                zmp_min[k], zmp_max[k] = passo_anterior - self.FOOT_MARGIN_X, avanco_atual + self.FOOT_MARGIN_X
            else:
                zmp_min[k], zmp_max[k] = avanco_atual - self.FOOT_MARGIN_X, avanco_atual + self.FOOT_MARGIN_X
        return com_ref, zmp_ref, zmp_min, zmp_max

    def _s_curve(self, x): 
        return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, x))))