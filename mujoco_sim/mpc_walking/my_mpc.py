import numpy as np

g, zc = 9.81, 0.22
w = np.sqrt(g / zc)                      # omega


def closed(t, x0, v0, p):
    """Solução fechada do pêndulo com ZMP constante em p."""
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
    x = np.clip(-np.linalg.solve(H, -q), lo, hi)                         
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

for nome, lim in (("sem restrições", None), ("limites [-2, +4] cm", (-0.02, 0.04)), ("limites [0, +7] cm", (0.0, 0.07))):
    z, x, a = mpc_com_limites(1e-6, lim)
    print(f"{nome:20s}: ZMP mín {z.min()*100:6.2f} | ZMP máx {z.max()*100:5.2f} | CoM final {x[-1]*100:5.2f} cm | ciclos com limite ativo {a}/100")