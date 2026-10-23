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

def prepara_mpc(delta, N, z_com, wz, wj):
    A, B, C = lipm_matrices(delta, z_com)
    Ps, Pu = prediction_matrices(A, B, C, N)
    H_P, M = qp_em_P(Pu, wz, wj)
    return dict(A=A, B=B, C=C, Ps=Ps, Pu=Pu, H_P=H_P, M=M, wz=wz, wj=wj)


def ciclo_mpc(s, p_ref, p_min, p_max, mpc):
    # s: estado de UM eixo [x, v, a];  p_ref, p_min, p_max: vetores de tamanho N
    livre = mpc["Ps"] @ s
    q = -mpc["wz"] * p_ref - mpc["wj"] * mpc["M"] @ livre 
    P = solve_box_qp(mpc["H_P"], q, p_min, p_max) 
    u0 = (P[0] - livre[0]) / mpc["Pu"][0, 0]
    return u0, P

from mujoco_sim.mpc_walking.mpc_walking_engine import MPCParams


def captura_ciclo(engine, ticks):
    """Anda `ticks` ticks no engine do repositório e guarda o ÚLTIMO ciclo de MPC que ele resolveu."""
    cap = {}
    original = engine._solve
    def embrulhado():
        cap["s"] = engine._state.copy()                 # estado ANTES de resolver
        cap["ref"] = engine._horizon_references()       # p_ref, p_min, p_max
        original()
        cap["jerk"] = engine._jerk.copy()
        cap["P"] = engine.debug["zmp_plan"].copy()
    engine._solve = embrulhado
    for _ in range(ticks):
        engine.step(0.008)
    return cap

import time
from mujoco_sim.analytical_walking.walking_engine import WalkingEngine, WalkTarget
from mujoco_sim.analytical_walking.trajectory_generator import get_com_pose_at_time, get_swing_foot_pose_at_time


class MeuMPCEngine(WalkingEngine):
    def __init__(self, T=0.5, z_com=0.22, z_step=0.04, ds_ratio=0.13, y_sep=0.054, g=9.81,
                 delta=0.04, N=20, wz=1.0, wj=1e-6):
        super().__init__(T, z_com, z_step, ds_ratio, y_sep, g)
        self.delta, self.N = delta, N
        self.mpc = prepara_mpc(delta, N, z_com, wz, wj)
        self._reseta()

    def _reseta(self):
        """Volta ao repouso sob o torso; o próximo tick andando recomeça o MPC."""
        self.estado = np.array([[self.torso.position[0], 0.0, 0.0],
                            [self.torso.position[1], 0.0, 0.0]])
        self.jerk = np.zeros(2)
        self.tick = 0

    def _estado_inicial(self):
        # posição e velocidade do CoM no instante 0 do passo, vindas do gerador analítico
        h = 1e-4
        args = (self.T, self.ds_ratio, self.z_com, self.g,
                self.torso_start, self.torso_target, self.support_foot)
        pos, _ = get_com_pose_at_time(0.0, *args)
        a_frente, _ = get_com_pose_at_time(h, *args)
        vel = (a_frente - pos) / h     
        return np.column_stack([pos, vel, np.zeros(2)])    # aceleração zero: o ZMP começa sobre o CoM

    def _avanca_mpc(self, dt):
        n = ticks_por_ciclo(self.delta, dt)
        if self.tick == 0:
            self.estado = self._estado_inicial()
        if self.tick % n == 0:
            p_ref, p_min, p_max, _ = referencias(self, self.N, self.delta)  
            for ax in (0, 1):
                self.jerk[ax], _ = ciclo_mpc(self.estado[ax], p_ref[:, ax], p_min[:, ax], p_max[:, ax], self.mpc)
        self.tick += 1
        A, B, _ = lipm_matrices(dt, self.z_com)                              
        for ax in (0, 1):
            self.estado[ax] = A @ self.estado[ax] + B * self.jerk[ax]
        return self.estado[:, 0].copy()   # a posição (x, y) do CoM

    def step(self, dt):                   # <- já vem pronto
        self._update_state()
        if self.state == WalkState.IDLE:
            self._reseta()
            return self.standing_pose()

        com_2d = self._avanca_mpc(dt)
        self.t_step += dt

        _, com_yaw = get_com_pose_at_time(self.t_step, self.T, self.ds_ratio, self.z_com, self.g,
                                          self.torso_start, self.torso_target, self.support_foot)
        swing_3d, swing_yaw = get_swing_foot_pose_at_time(self.t_step, self.T, self.z_step, self.ds_ratio,
                                                          self.swing_start, self.swing_target)
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

def compara(nome, comandos, segundos):
    """Roda o seu engine e o do repositório com os mesmos comandos e compara tick a tick."""
    meu, ref = MeuMPCEngine(**WALK_PARAMS), MPCWalkingEngine(**WALK_PARAMS)
    pendentes = sorted(comandos, key=lambda c: c[0])
    dif = {"body": 0.0, "pés": 0.0, "yaw": 0.0}; t_meu = []; anterior = None; salto = 0.0; desvio = 0.0
    for i in range(int(segundos / 0.008)):
        while pendentes and pendentes[0][0] <= i * 0.008:
            kw = pendentes.pop(0)[1]
            meu.set_command(**kw); ref.set_command(**kw)
        t0 = time.perf_counter(); a = meu.step(0.008); t_meu.append(time.perf_counter() - t0); b = ref.step(0.008)
        dif["body"] = max(dif["body"], np.abs(a.body_pos - b.body_pos).max())
        dif["pés"] = max(dif["pés"], np.abs(a.left_sole - b.left_sole).max(), np.abs(a.right_sole - b.right_sole).max())
        dif["yaw"] = max(dif["yaw"], abs(a.body_yaw - b.body_yaw))
        if anterior is not None: salto = max(salto, np.linalg.norm(a.body_pos[:2] - anterior))
        anterior = a.body_pos[:2].copy()
        if meu.state != WalkState.IDLE: desvio = max(desvio, np.abs(meu.estado[:, 0] - meu.torso.position).max())
    t = np.array(t_meu) * 1e3
    print(f"  {nome:28s} dif corpo {dif['body']:.0e} | pés {dif['pés']:.0e} | yaw {dif['yaw']:.0e} | maior salto da pelve {salto*1000:.1f} mm | |CoM-torso| máx {desvio*100:.1f} cm | tempo/tick média {t.mean():.2f} ms, máx {t.max():.1f} ms")


print("\n== Teste 7c: o seu engine == o engine do repositório")
compara("frente 0.05 por 8 s", [(0.0, dict(vx=0.05))], 8)
compara("lateral 0.03 por 8 s", [(0.0, dict(vy=0.03))], 8)
compara("giro 0.5 por 8 s", [(0.0, dict(wz=0.5))], 8)
compara("mudanças de comando, 16 s", [(0.0, dict(vx=0.05)), (4.0, dict(vy=0.03)), (8.0, dict(wz=0.5)), (12.0, dict(vx=-0.05))], 16)

print("\n== Teste 7c-2: o robô no MuJoCo (física completa) andando com o SEU engine")
from mujoco_sim.sim import Simulation
for nome, cmd in (("frente 0.05", dict(vx=0.05)), ("lateral 0.03", dict(vy=0.03)), ("giro 0.5", dict(wz=0.5))):
    sim = Simulation(); sim.engine = MeuMPCEngine(**WALK_PARAMS); sim.engine.set_command(**cmd)
    inclinacao = 0.0
    for _ in range(int(8 / sim.dt)):
        sim.step()
        r = sim.imu.read(sim.model, sim.data); inclinacao = max(inclinacao, abs(r["pitch"]), abs(r["roll"]))
    print(f"  {nome:13s} em 8 s: x={sim.body_pos[0]:+.3f} m  y={sim.body_pos[1]:+.3f} m  yaw={np.degrees(sim.body_yaw):+7.1f} graus | altura {sim.body_pos[2]:.3f} m | inclinação máx {np.degrees(inclinacao):.1f} graus | falhas de IK {sum(sim.ik_failures.values())}")
