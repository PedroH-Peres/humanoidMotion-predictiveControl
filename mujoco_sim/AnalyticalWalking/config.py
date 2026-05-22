JOINT_SIGNS = {
    #  Junta          Esquerda   Direita
    "hip_yaw":   {"l": -1.0,  "r": -1.0},
    "hip_roll":  {"l": -1.0,  "r": -1.0},
    "hip_pitch": {"l":  1.0,  "r":  -1.0},
    "knee":      {"l":  1.0,  "r":  -1.0},
    "ank_pitch": {"l":  -1.0,  "r":  1.0},
    "ank_roll":  {"l":  1.0,  "r":  1.0},
}

PHYSICS = True

# -----------------------------------------------------------------------------
#  GANHOS DOS ATUADORES DE POSIÇÃO
#  ACTUATOR_KP          — ganho proporcional [N·m/rad]
#  ACTUATOR_KV          — amortecimento derivativo [N·m·s/rad]
#  ACTUATOR_FORCE_RANGE — limite de torque por junta [N·m]
# -----------------------------------------------------------------------------

ACTUATOR_KP          = 120.0
ACTUATOR_KV          = 12.0
ACTUATOR_FORCE_RANGE = 5.0

# -----------------------------------------------------------------------------
#  PARÂMETROS DO MOTOR DE CAMINHADA
# -----------------------------------------------------------------------------

WALK_PARAMS = {
    "T":        0.3,    # Período de um passo [s] — menor = mais rápido
    "z_com":    0.22,   # Altura do torso [m]
    "z_step":   0.025,  # Altura máxima do pé no ar [m]
    "ds_ratio": 0.1,    # Fração do passo em apoio duplo (0 a 0.5)
    "y_sep":    0.054,  # Separação lateral dos pés [m]
}

ARM_POSE = {
    "l_sho_pitch":  -0.0,    # ombro esquerdo pitch
    "r_sho_pitch":  0.0,    # ombro direito pitch
    "l_sho_roll":  1.4,    # ombro esquerdo roll
    "r_sho_roll":   -1.4,    # ombro direito roll 
    "l_el":        0.0,    # cotovelo esquerdo
    "r_el":         -0.0,    # cotovelo direito   
}

def joint_name(side: str, key: str) -> str:
    return f"{side}_{key}"
