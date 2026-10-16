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