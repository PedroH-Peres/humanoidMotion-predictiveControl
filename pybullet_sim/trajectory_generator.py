import math

class TrajectoryGenerator:
    def __init__(self, h_target=0.20, t_step=0.4, y_sep=0.09):
        self.H_TARGET = h_target
        self.T_STEP = t_step
        self.Y_SEP = y_sep
        
    def get_desired_state(self, t, cmd_vx):
        """ 
        Gera o estado desejado baseado puramente no tempo t e na velocidade alvo cmd_vx.
        """
        step_idx = int(t / self.T_STEP)
        t_in_step = t % self.T_STEP
        progresso = t_in_step / self.T_STEP
        
        is_left_support = (step_idx % 2 == 0)
        avanco_total = cmd_vx * t
        
        if is_left_support:
            math_foot_l_x = avanco_total
            math_foot_l_y = self.Y_SEP / 2.0
            math_foot_l_z = 0.0 
            
            passo_anterior = cmd_vx * (step_idx * self.T_STEP)
            passo_futuro = cmd_vx * ((step_idx + 1) * self.T_STEP)
            math_foot_r_x = passo_anterior + (passo_futuro - passo_anterior) * self.suavizar(progresso)
            math_foot_r_y = -self.Y_SEP / 2.0
            math_foot_r_z = 0.05 * math.sin(math.pi * progresso) 
        else:
            math_foot_r_x = avanco_total
            math_foot_r_y = -self.Y_SEP / 2.0
            math_foot_r_z = 0.0
            
            passo_anterior = cmd_vx * (step_idx * self.T_STEP)
            passo_futuro = cmd_vx * ((step_idx + 1) * self.T_STEP)
            math_foot_l_x = passo_anterior + (passo_futuro - passo_anterior) * self.suavizar(progresso)
            math_foot_l_y = self.Y_SEP / 2.0
            math_foot_l_z = 0.05 * math.sin(math.pi * progresso)

        math_com_x = avanco_total
        ginga_amplitude = self.Y_SEP / 2.0 - 0.01 
        math_com_y = ginga_amplitude * math.sin(math.pi * (t / self.T_STEP))

        return (math_com_x, math_com_y), (math_foot_r_x, math_foot_r_y, math_foot_r_z), (math_foot_l_x, math_foot_l_y, math_foot_l_z)

    def suavizar(self, x):
        return 0.5 * (1.0 - math.cos(math.pi * x))