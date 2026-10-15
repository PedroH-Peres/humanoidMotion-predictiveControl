import numpy as np

g, zc = 9.81, 0.22
w = np.sqrt(g / zc)                      # omega


def closed(t, x0, v0, p):
    """Solução fechada do pêndulo com ZMP constante em p."""
    return p + (x0 - p) * np.cosh(w * t) + v0 / w * np.sinh(w * t)
x0, v0, p, t = 0.012, 0.0, 0.01, 0.5
print(closed(t, x0, v0, p))

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

delta, N = 0.04, 20
A, B, C = lipm_matrices(delta, zc)
Ps, Pu = prediction_matrices(A, B, C, N)

rng = np.random.default_rng(0)
s0 = rng.normal(size=3)
U = rng.normal(size=N)

# Teste 1: o modelo passo a passo contra a conta de matrizes
s, zmp = s0.copy(), []
for u in U:
    s = A @ s + B * u
    zmp.append(C @ s)
print("Teste 1: dif %.1e" % np.abs(np.array(zmp) - (Ps @ s0 + Pu @ U)).max())

# Teste 2: a estrutura de Pu
print("Teste 2: formas Ps", Ps.shape, " Pu", Pu.shape)
print("         triangular inferior:", np.allclose(Pu, np.tril(Pu)))
print("         Pu[0,0] = %.4e   C@B = %.4e" % (Pu[0, 0], C @ B))
print("         diagonal constante :", np.allclose(np.diag(Pu), C @ B))
print("         Toeplitz (Pu[3,1] == Pu[5,3]):", np.isclose(Pu[3, 1], Pu[5, 3]))
print("         posto de Pu:", np.linalg.matrix_rank(Pu))

# Teste 3: cada coluna de Pu é a resposta a um jerk isolado
j = 4
U_imp = np.zeros(N); U_imp[j] = 1.0
s, resp = np.zeros(3), []
for u in U_imp:
    s = A @ s + B * u
    resp.append(C @ s)
print("Teste 3: coluna %d de Pu: dif %.1e" % (j, np.abs(np.array(resp) - Pu[:, j]).max()))
print("         ZMP antes do jerk (linhas 0 a %d): %s" % (j - 1, np.round(Pu[:j, j], 6)))