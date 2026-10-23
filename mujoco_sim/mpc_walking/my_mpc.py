import numpy as np
from mujoco_sim.analytical_walking.walking_engine import WalkState
from mujoco_sim.analytical_walking.trajectory_generator import Pose2D, FootPose, select_next_poses

g, zc = 9.81, 0.22
w = np.sqrt(g / zc)                      # omega


def closed(t, x0, v0, p):
    #Solução fechada do pêndulo com ZMP constante em p.
    return p + (x0 - p) * np.cosh(w * t) + v0 / w * np.sinh(w * t)

def lipm_matrices(delta, z_com, g=9.81):
    A = np.array([
        [1.0, delta, (delta**2)/2],
        [0.0, 1, delta],
        [0.0, 0.0, 1.0]
    ])
    B = np.array([delta ** 3 / 6, delta**2 / 2, delta])
    C = np.array([1.0, 0.0, -z_com / g])
    return A, B, C

def prediction_matrices(A, B, C, N):
    potencias = [np.eye(3)]
    Ps = np.zeros((N, 3))
    Pu = np.zeros((N, N))
    for _ in range(N):
        potencias.append(A @ potencias[-1])
    for i in range(N):
        Ps[i] = C @ potencias[i+1]
        for j in range(i+1):
            Pu[i, j] = C @ potencias[i-j] @ B
    return Ps, Pu

def mpc_sem_restricoes(wj, wz=1.0, N=20, delta=0.04, t_total=3.0):
    A, B, C = lipm_matrices(delta, zc)
    Ps, Pu = prediction_matrices(A, B, C, N)
    H = wz * Pu.T @ Pu + wj * np.eye(N)

    s=np.zeros(3)
    tempos, zmps, coms = [],[],[]
    for k in range(int(t_total/delta)):
        t = k*delta
        p_ref = np.array([0.05 if t + (i + 1) * delta >= 1.0 else 0.0 for i in range(N)])
        grad = wz * Pu.T @ (Ps @ s - p_ref)
        U = -np.linalg.solve(H, grad)
        s = A @ s + B * U[0]
        tempos.append(t+delta)
        zmps.append(C @ s)
        coms.append(s[0])
    return np.array(tempos), np.array(zmps), np.array(coms)

for wj in (1e-6, 1e-4, 1e-2):
    t, zmp, com = mpc_sem_restricoes(wj)
    inicio = t[np.argmax(np.abs(zmp) > 1e-3)]
    i14, i25 = np.searchsorted(t, 1.4), np.searchsorted(t, 2.5)
    print(f"w_jerk={wj:.0e}: ZMP começa a mover em t={inicio:.2f} s | ZMP em 1.4 s = {zmp[i14]*100:.1f} cm "
          f"| ZMP em 2.5 s = {zmp[i25]*100:.2f} cm | CoM final = {com[-1]*100:.1f} cm")

def qp_em_P(Pu, wz, wj):
    Pu_inv = np.linalg.inv(Pu)
    M = Pu_inv.T @ Pu_inv
    H_P = wz * np.eye(len(Pu)) + wj * M
    return H_P, M

def solve_box_qp(H, q, lo, hi, max_iter=60):
    custo = lambda v: 0.5 * v @ H @ v + q @ v
    x = np.clip(np.linalg.solve(H, -q), lo, hi)
    for _ in range(max_iter):
        grad = H @ x + q
        preso = ((x <= lo + 1e-9) & (grad > 0)) | ((x >= hi - 1e-9) & (grad < 0)) 
        livre = ~preso
        if not livre.any():
            break
        passo = np.zeros_like(x)
        passo[livre] = -np.linalg.solve(H[np.ix_(livre, livre)], grad[livre])
        if np.abs(passo).max() < 1e-10:
            break
        t = 1.0                               # busca de linha: encurta o passo até o custo cair
        while True:
            x_novo = np.clip(x + t * passo, lo, hi)
            if custo(x_novo) <= custo(x) + 1e-4 * grad @ (x_novo - x) or t < 1e-8:
                break
            t *= 0.5
        x = x_novo
    return x

def mpc_com_limites(wj, limites, wz=1.0, N=20, delta=0.04, t_total=4.0):
    A, B, C = lipm_matrices(delta, zc)
    Ps, Pu = prediction_matrices(A, B, C, N)
    H_P, M = qp_em_P(Pu, wz, wj)
    s = np.zeros(3); zmps, coms, ativos = [], [], 0
    for k in range(int(t_total / delta)):
        t = k * delta
        p_ref = np.array([0.05 if t + (i + 1) * delta >= 1.0 else 0.0 for i in range(N)])
        livre = Ps @ s
        q = -wz * p_ref - wj * M @ livre
        lo = np.full(N, limites[0] if limites else -1e9)
        hi = np.full(N, limites[1] if limites else 1e9)
        P = solve_box_qp(H_P, q, lo, hi)
        ativos += int(np.any((P <= lo + 1e-9) | (P >= hi - 1e-9)))
        u0 = (P[0] - livre[0]) / Pu[0, 0]
        s = A @ s + B * u0
        zmps.append(C @ s); coms.append(s[0])
    return np.array(zmps), np.array(coms), ativos

def zmp_no_passo(t_local, T, ds_ratio, torso_ini, apoio, torso_fim):
    t_b = T * ds_ratio / 2                  # fim do primeiro apoio duplo
    t_e = T - t_b                           # começo do último apoio duplo
    if t_local < t_b:                       # apoio duplo 1: desliza do torso ao pé de apoio
        a = t_local / t_b
        return torso_ini + a * (apoio - torso_ini)
    if t_local >= t_e:                      # apoio duplo 2: desliza do pé de apoio ao próximo torso
        a = (t_local - t_e) / (T - t_e)
        return apoio + a * (torso_fim - apoio)
    return apoio   

def pes_em_contato(t_local, T, ds_ratio, anterior, apoio, pouso):
    t_b = T * ds_ratio / 2
    t_e = T - t_b
    if t_local < t_b:
        return [anterior, apoio]
    if t_local >= t_e:
        return [apoio, pouso]
    return [apoio]

def limites(pes, meio_comprimento, meia_largura, margem):
    meio = np.array([meio_comprimento, meia_largura]) - margem
    pts = np.array(pes)
    return pts.min(axis=0) - meio, pts.max(axis=0) + meio

def plano_de_passos(engine, n_passos):
    zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    comando = engine._cmd if engine.state == WalkState.WALKING else zero
    torso_ini = engine.torso_start.position
    torso_fim = engine.torso_target.position
    apoio = engine.support_foot.position
    anterior = engine.swing_start.position
    pouso = engine.swing_target.position
    apoio_e_esq = engine.support_foot.is_left
    yaw = engine.torso_target.yaw
    passos = []
    for k in range(n_passos):
        passos.append(dict(t0=k * engine.T, torso_ini=torso_ini, torso_fim=torso_fim,
                           apoio=apoio, anterior=anterior, pouso=pouso))
        balanco = FootPose(apoio, yaw, is_left=apoio_e_esq)
        prox_torso, prox_pouso = select_next_poses(Pose2D(torso_fim, yaw), balanco, comando,
                                                   engine.T, engine.y_sep)
        torso_ini, torso_fim = torso_fim, prox_torso.position
        anterior, apoio, pouso = apoio, pouso, prox_pouso.position
        apoio_e_esq = not apoio_e_esq
        yaw = prox_torso.yaw
    return passos

def referencias(engine, N, delta, meio_comprimento=0.05, meia_largura=0.03, margem=0.01):
    T = engine.T
    n_passos = int(np.ceil(delta * N / T)) + 1
    passos = plano_de_passos(engine, n_passos)
    p_ref = np.zeros((N, 2)); p_min = np.zeros((N, 2)); p_max = np.zeros((N, 2))
    for i in range(N):
        t = engine.t_step + (i + 1) * delta
        passo = passos[min(int(t // T), len(passos) - 1)]
        t_local = min(t - passo["t0"], T - 1e-9)
        p_ref[i] = zmp_no_passo(t_local, T, engine.ds_ratio, passo["torso_ini"], passo["apoio"], passo["torso_fim"])
        pes = pes_em_contato(t_local, T, engine.ds_ratio, passo["anterior"], passo["apoio"], passo["pouso"])
        p_min[i], p_max[i] = limites(pes, meio_comprimento, meia_largura, margem)
    return p_ref, p_min, p_max, passos

from mujoco_sim.config import WALK_PARAMS
from mujoco_sim.mpc_walking.mpc_walking_engine import MPCWalkingEngine


def ticks_por_ciclo(dt_mpc, dt_sim):
    n = round(dt_mpc / dt_sim)
    if(abs(n*dt_sim - dt_mpc) > 1e-9):
        raise ValueError(f"dt_mpc ({dt_mpc}) precisa ser múltiplo inteiro de dt_sim ({dt_sim})")
    return n

def roda_ciclos(s0, jerks, z_com, dt_sim=0.008, dt_mpc=0.04):
    # roda o relógio duplo. jerks[c] é o jerk do ciclo c.
    # devolve (estados no fim de cada ciclo, ZMP de cada tick)
    n = ticks_por_ciclo(dt_mpc, dt_sim)
    A, B, C = lipm_matrices(dt_sim, z_com)
    s = s0.copy(); fins, zmps = [], []
    for k in range(n * len(jerks)):
        if k % n == 0:
            jerk = jerks[k // n]
        s = A @ s + B * jerk 
        zmps.append(C @ s)
        if (k + 1) % n == 0:
            fins.append(s.copy())
    return np.array(fins), np.array(zmps)

