# Construindo o MPC de caminhada, passo a passo

Guia para você escrever, **por conta própria**, o controle preditivo que está em
[`mpc_walking_engine.py`](../mujoco_sim/mpc_walking/mpc_walking_engine.py), entendendo cada conta, e
terminar com a **realimentação do estado real do robô**.

Todos os números marcados com ✔ foram medidos neste repositório (δ = 40 ms, z_c = 0,22 m,
g = 9,81, passo da física de 8 ms). Use-os para saber se a sua versão está certa.

## Como usar este guia

1. Deixe o `mpc_walking_engine.py` como **gabarito**. Não copie: abra só para comparar depois de tentar.
2. Escreva o seu em outro arquivo, por exemplo `mujoco_sim/mpc_walking/meu_mpc.py`, **uma etapa por vez**.
3. Cada etapa termina em *Como verificar*. Só passe para a seguinte quando a verificação passar.
4. Os testes `tests/test_mpc_walking.py` conferem funções com estes nomes: `lipm_matrices`,
   `prediction_matrices`, `solve_box_qp` e a classe `MPCWalkingEngine`. Se usar os mesmos nomes, basta
   trocar o `import` do teste para o seu módulo.
5. Para gráficos: `pip install matplotlib` (ou `pip install -e ".[tools]"`) dentro do `.venv`.

| # | Etapa | Você produz |
|---|-------|-------------|
| 1 | O pêndulo e o ZMP | simulação + solução fechada |
| 2 | Discretização com jerk | `lipm_matrices` |
| 3 | Predição em horizonte | `prediction_matrices` |
| 4 | Custo → QP, MPC sem restrições | laço de MPC simples |
| 5 | Restrições e o solver | `solve_box_qp` |
| 6 | Plano de passos → referência e limites | pré-visualização de passos |
| 7 | O laço de recuo e os relógios | `step()` do engine |
| 8 | Estado inicial viável (DCM) | partida sem divergir |
| 9 | Continuidade na parada | sem saltos |
| 10 | Realimentação do robô real | MPC em malha fechada |

---

## Etapa 1. O pêndulo invertido e o ZMP

**Objetivo.** Entender de onde vem `ẍ = ω²(x − p)`.

**Teoria.** Uma massa `m` a altura constante `z_c`, apoiada por uma perna sem massa num ponto `p` do chão.
O momento da gravidade em torno de `p` é `m·g·(x − p)`. A força de inércia horizontal, a altura `z_c`,
dá `m·z_c·ẍ`. No ZMP esses momentos se anulam:

```
m g (x − p) = m z_c ẍ     ⇒     ẍ = ω² (x − p),    ω = √(g / z_c)
p = x − (z_c / g) ẍ        (o "ZMP", a saída do modelo)
```

Leitura física: o CoM **acelera para longe** do ZMP. Por isso andar é "cair controladamente".

**O que escrever.**
```python
w = np.sqrt(g / zc)
def closed(t, x0, v0, p):                      # solução fechada com ZMP constante
    return p + (x0 - p) * np.cosh(w * t) + v0 / w * np.sinh(w * t)
```
e um integrador numérico (Euler fica ruim; use RK4 com passo de 0,1 ms).

**Como verificar.**
- ✔ `ω = 6,678 rad/s` e `1/ω = 0,1498 s`.
- ✔ Com `x₀ − p = 1 cm` e `v₀ = 0`, depois de 0,5 s o afastamento é `14,11 cm` (×14,11), nos dois métodos
  (`0,141107` simulado e fechado).
- Trace `x(t)` para `p` à esquerda e à direita do CoM e veja o CoM fugir do ZMP.

**Erros comuns.** Esquecer que o sinal é `(x − p)` e não `(p − x)`; usar passo grande no Euler.

---

## Etapa 2. Discretizar com jerk como entrada

**Objetivo.** Chegar em `s[k+1] = A·s[k] + B·u[k]` e `p[k] = C·s[k]`.

**Teoria.** Tome o estado `s = [x, ẋ, ẍ]` e como entrada o **jerk** `u = x⃛`. É um triplo integrador:

```
ṡ = F s + G u,    F = [[0,1,0],[0,0,1],[0,0,0]],    G = [0,0,1]ᵀ
```

Como `F³ = 0`, a exponencial de matriz tem poucos termos:

```
A = e^{Fδ} = I + Fδ + F²δ²/2
B = ∫₀^δ e^{Fτ} G dτ = (Iδ + Fδ²/2 + F²δ³/6) G = [δ³/6, δ²/2, δ]ᵀ
C = [1, 0, −z_c/g]                              (do ZMP da etapa 1)
```

O jerk fica constante dentro de cada período δ (*zero-order hold*). Por que jerk e não o ZMP como
entrada? Com jerk, a trajetória do CoM é suave (aceleração contínua), e o ZMP continua sendo uma função
linear do estado.

**O que escrever.** `lipm_matrices(delta, z_com, g) -> (A, B, C)`.

**Como verificar.**
- ✔ `A = [[1, 0,04, 8e-4],[0, 1, 0,04],[0, 0, 1]]`.
- ✔ `B = [1,0667e-5, 8e-4, 0,04]`.
- ✔ `C = [1, 0, −0,02243]` e `C·B = −8,864e-4`.
- Aplique um jerk constante por δ e compare com integrar `ṡ = Fs + Gu` com RK4.

**Erros comuns.** Esquecer o termo `δ³/6`; usar `A = I + Fδ` (Euler) em vez da série completa.

---

## Etapa 3. Predição em horizonte

**Objetivo.** Prever o ZMP de N instantes à frente como **uma conta matricial**.

**Teoria.** Desenrole o modelo:

```
s[1] = A s₀ + B u₀
s[2] = A² s₀ + A B u₀ + B u₁
s[k] = Aᵏ s₀ + Σ_{j<k} A^{k−1−j} B u_j
```

O ZMP após `i+1` passos é `P[i] = C s[i+1]`. Empilhando `P = [P[0] … P[N−1]]` e `U = [u₀ … u_{N−1}]`:

```
P = Ps · s₀ + Pu · U
Ps[i]    = C A^{i+1}                      (N × 3)
Pu[i, j] = C A^{i−j} B   se j ≤ i,  0 se j > i       (N × N, triangular inferior)
```

`Pu` é invertível porque a diagonal vale `C·B ≠ 0`. Isso será útil na etapa 5.

**O que escrever.** `prediction_matrices(A, B, C, N) -> (Ps, Pu)`. Guarde as potências `A^k` numa lista
para não recalcular.

**Como verificar.** Sorteie `s₀` e `U`, passe o modelo passo a passo guardando `C·s` e compare com
`Ps @ s0 + Pu @ U` (`np.allclose`, tolerância 1e-12). ✔ Com N=20: `Pu[0,0] = −8,864e-4`,
triangular inferior, posto 20.

**Erros comuns.** Deslocamento de índice: `P[0]` usa `A¹`, não `A⁰`.

---

## Etapa 4. Do custo ao QP: o MPC sem restrições

**Objetivo.** Escolher os jerks que fazem o ZMP seguir uma referência, e fechar o laço.

**Teoria.** Custo:

```
J(U) = w_zmp ‖P − p_ref‖² + w_jerk ‖U‖²,       P = Ps s₀ + Pu U
```

Substitua `P` e expanda. Dividindo por 2, e chamando `livre = Ps s₀` (o ZMP se o jerk fosse zero):

```
J/2 = ½ Uᵀ H U + gᵀ U + const
H = w_zmp PuᵀPu + w_jerk I           (constante: calcule uma vez)
g = w_zmp Puᵀ (livre − p_ref)         (muda a cada ciclo)
```

Sem restrições, derive e iguale a zero: `U = −H⁻¹ g` (use `np.linalg.solve`, não `inv`).

**Recuo (receding horizon).** Aplique só `u₀`, avance o modelo `s ← A s + B u₀`, e resolva de novo no
período seguinte.

**O que escrever.** Um script solto: referência de ZMP que vale 0 e muda para 5 cm em `t = 1 s`; a cada
δ resolva `U`, aplique `U[0]`, registre ZMP e CoM; trace tudo.

**Como verificar.** ✔ (N = 20, `w_zmp = 1`):

| `w_jerk` | o ZMP começa a mover (>1 mm) | ZMP final | observação |
|---|---|---|---|
| 1e-6 | em 0,88 s (0,12 s antes) | 5,0 cm | segue com antecipação curta |
| 1e-4 | em 0,40 s (0,60 s antes) | 4,98 cm | mais suave, antecipa mais |
| 1e-2 | em 0,72 s | **8,4 cm** em 2,5 s | jerk caro demais: o controlador não segura o modo instável e passa do alvo |

A antecipação é o *preview*: o MPC "vê" a referência mudar e prepara o CoM antes. É o que o gerador
analítico, que só conhece o passo atual, não faz.

**Erros comuns.** Referência indexada errado (`p_ref[i]` é o ZMP em `t + (i+1)δ`); usar `H` com `w_jerk` tão
alto que o sistema instável não é controlado.

---

## Etapa 5. Restrições e o solver

**Objetivo.** Manter o ZMP dentro do polígono de apoio: `p_min ≤ P ≤ p_max`.

### 5a. Reformular para que só haja limites simples

Na forma anterior, as restrições envolvem `Pu·U` (N desigualdades acopladas). Como `Pu` é invertível,
troque a variável: use **o próprio ZMP futuro `P`** como incógnita e recupere os jerks por
`U = Pu⁻¹ (P − livre)`:

```
J/2 = ½ Pᵀ H_P P + qᵀ P + const
H_P = w_zmp I + w_jerk M,        M = Pu⁻ᵀ Pu⁻¹
q   = −w_zmp p_ref − w_jerk M · livre
restrição:   p_min ≤ P ≤ p_max       (só limites, uma caixa)
```

O jerk aplicado sai direto: `u₀ = (P[0] − livre[0]) / Pu[0,0]` (porque `Pu` é triangular inferior).

### 5b. Resolver um QP com limites: Newton projetado

```
x = clip(−solve(H, q), lo, hi)                          # ótimo sem restrições, cortado nos limites
repita (até 60 vezes):
    grad = H x + q
    preso[i] = (x[i] ≤ lo[i] e grad[i] > 0)  ou  (x[i] ≥ hi[i] e grad[i] < 0)
    livre = não preso
    passo[livre] = −solve(H[livre, livre], grad[livre]);  passo[preso] = 0
    se max|passo| < 1e-10: pare
    t = 1;  enquanto custo(clip(x + t·passo)) não cair o bastante: t /= 2     # busca de linha
    x = clip(x + t·passo)
```

Ideia: variáveis encostadas num limite e empurradas para fora pelo gradiente **ficam presas**. As demais
recebem um passo de Newton. O `clip` garante que o resultado é sempre viável.

### 5c. O certificado de otimalidade (use sempre)

`x` é o ótimo **se e somente se** `x == clip(x − (H x + q), lo, hi)`. Meça
`max|x − clip(x − (Hx + q), lo, hi)|`: dispensa qualquer outro solver para conferir a resposta.

**O que escrever.** `solve_box_qp(H, q, lo, hi)`.

**Como verificar.**
- Problema com limites largos: deve devolver `−H⁻¹q`.
- Problemas aleatórios com limites ativos: resíduo KKT ≲ 1e-8. ✔ No engine real: ~1e-13.
- ✔ Tempo por QP (N=20): **0,15 ms** em média.

### 5d. O horizonte e o condicionamento (a lição mais cara)

✔ Número de condição de `H_P`:

| horizonte | N | cond(H_P) |
|---|---|---|
| 0,8 s | 20 | 1,9e2 |
| 1,0 s | 25 | 3e3 |
| 1,2 s | 30 | 4e4 |
| 1,6 s | 40 | 8,6e6 |

Com N=40 o meu primeiro solver (ADMM) não convergia: o ZMP saía do pé em até 4 cm. Horizonte maior não
ajuda, porque o LIPM esquece o futuro rápido (`1/ω = 0,15 s`). Mas horizonte curto demais **diverge**:
✔ com `N ≤ 12` (≤ 0,48 s) o seu MPC diverge no andar para frente, e com `N = 15` (0,6 s) já fica estável.
Use N = 20 para ter margem.

**Erros comuns.** Comparar com OSQP e achar que "infactível" é verdade: com a variável original (jerks)
e N grande ele devolveu "primal infeasible" em problemas perfeitamente factíveis, por má escala.

---

## Etapa 6. Plano de passos → referência e limites

**Objetivo.** Para os N instantes do horizonte, saber **onde o ZMP deve estar** (`p_ref`) e **onde pode
estar** (`p_min`, `p_max`).

**Teoria.** Em cada passo de duração `T`, com `t_b = T·ds_ratio/2` e `t_e = T − t_b` (✔ com `T=0,4` e
`ds_ratio=0,12`: `t_b = 0,024 s`):

```
0 ≤ t < t_b      apoio duplo:   p = torso_ini + (apoio − torso_ini)·t / t_b
t_b ≤ t < t_e    apoio simples: p = apoio
t_e ≤ t < T      apoio duplo:   p = apoio + (torso_fim − apoio)·(t − t_e)/(T − t_e)
```

Os pés em contato (o polígono) em cada fase: apoio duplo 1 → (pé anterior, apoio); apoio simples → (apoio);
apoio duplo 2 → (apoio, pé que vai pousar). O limite é a **caixa envolvente** desses pés, aumentada de
`±(meio_comprimento − margem, meia_largura − margem)` (✔ gabarito: `0,05`, `0,03` e margem `0,01`).

**Passos futuros.** O engine analítico só planeja o passo atual. Para o horizonte, repita a regra dele:

```
próximo apoio = pé que acabou de pousar
próximo "pé anterior" = apoio atual
(próximo torso, próximo pouso) = select_next_poses(Pose2D(torso_fim, yaw),
                                  FootPose(apoio_atual, yaw, is_left=apoio_é_esq), comando, T, y_sep)
```

Para N = 20 e δ = 0,04 s (0,8 s) com T = 0,4 s, você precisa de 3 passos. Se o comando for parar,
o passo atual é o último e depois entra um segmento "parado" com `p = torso_fim` e o polígono dos dois pés.

**O que escrever.** Uma função que devolve `p_ref`, `p_min`, `p_max` com forma `(N, 2)`: x e y.

**Como verificar.** Trace `p_ref` e os limites para 3 passos andando para frente: no apoio simples o
`p_ref` tem de ficar parado no pé, nas transições ele se desloca rápido, e `p_ref` está sempre dentro
dos limites. Confira o instante `t_local` com `t // T`.

**Erros comuns.** Esquecer de trocar o pé de apoio a cada passo; misturar `t` global com o `t` local do
passo.

---

## Etapa 7. O laço de recuo e os relógios

**Objetivo.** Rodar o MPC a cada 40 ms enquanto a física roda a 8 ms.

**Teoria.** O modelo do MPC avança em fatias de δ = 40 ms, mas o simulador chama o engine a cada 8 ms. Mantenha o
jerk **constante durante os 5 ticks** e avance o modelo com `A₈, B₈` (as mesmas matrizes da etapa 2 com
δ = 0,008). Isso é exato (*zero-order hold*), não uma aproximação.

```
a cada tick (dt = 8 ms):
    se tick % 5 == 0:
        resolve o QP do eixo x e do eixo y          → P
        jerk[eixo] = (P[0] − livre[0]) / Pu[0,0]
    s ← A₈ s + B₈ · jerk                             # para cada eixo
    devolve s[:, 0]                                  # posição do CoM (x, y)
```

x e y são **independentes**: mesmas matrizes, dois QPs.

**Como verificar.** ✔ Cinco passos de 8 ms com jerk constante dão exatamente o mesmo estado que um passo de
40 ms (diferença `1,4e-17`). Se `δ_MPC / dt` não for inteiro, os relógios descasam; lance um erro.

**Erros comuns.** Resolver o QP a cada tick (gasta tempo à toa); usar `dt` do MPC para avançar o modelo a
cada tick de 8 ms.

---

## Etapa 8. Estado inicial viável (o DCM)

**Objetivo.** Entender por que partir do repouso com restrição **diverge**, e como corrigir.

**Teoria.** Defina o *ponto de captura* (DCM): `ξ = x + ẋ/ω`. Derivando e usando a etapa 1:

```
ξ̇ = ẋ + ẍ/ω = ẋ + ω(x − p) = ω (ξ − p)        ⇒     ξ(t) = p + (ξ₀ − p) e^{ωt}
```

O DCM **sempre se afasta** do ZMP, exponencialmente. Para o robô não cair, o ZMP tem de ficar
perto do DCM. Com o ZMP preso dentro de um pé, o DCM só fica limitado se **já começa** perto dele.

✔ Experimento: CoM parado em `x = 0`, ZMP fixo em `+3,4 cm` por 0,4 s → o CoM vai a `−21,3 cm` e o DCM a
`−45,7 cm`, para longe do pé. É exatamente o que o primeiro passo em apoio simples faz com o robô parado.

**Correção.** Comece o MPC no estado que o gerador analítico tem em `t = 0` do passo: posição `torso_ini`,
velocidade por diferença finita de `get_com_pose_at_time` (com `h = 1e-4 s`) e aceleração zero (o ZMP
começa sobre o CoM).

**Como verificar.** Tire essa correção e rode: ✔ o eixo y cresce ×27 a cada 0,5 s (`e^{ω·0,5}`).
Recoloque: o CoM oscila ±2,5 cm em y e o ZMP alterna entre os pés (±5,5 cm).

---

## Etapa 9. Continuidade na parada

**Objetivo.** Não dar um "tranco" na pelve ao parar.

**O problema.** Quando o estado vira `IDLE`, se o engine devolver a pose nominal do torso, a pelve salta:
✔ **19,6 mm em um tick de 8 ms** (≈2,5 m/s), contra no máximo 2,5 mm no analítico.

**Correção.** Em `IDLE`, continue rodando o MPC com um plano de "parado" (`p_ref = torso`, polígono dos
dois pés) até o CoM assentar (`|pos − torso| < 1 mm` e `|vel| < 1 mm/s`), só então congele.

**Como verificar.** Meça `max‖Δbody_pos‖` por tick durante uma parada. ✔ Alvo: ≈2,4 mm. O assentamento
leva ≈0,7 s.

---

## Etapa 10. Realimentação do estado real do robô

Até aqui o estado `s₀` do MPC vem do próprio modelo (malha aberta). Aqui ele passa a vir **do robô**.

### 10a. Medir o CoM e a velocidade no MuJoCo

```python
import mujoco
mujoco.mj_subtreeVel(model, data)             # calcula as velocidades das subárvores
com = data.subtree_com[1][:2]                 # CoM do robô inteiro, x e y (mundo)
vel = data.subtree_linvel[1][:2]              # velocidade do CoM
```

O corpo 1 é a base do robô (`body_link`). ✔ O robô pesa 3,147 kg e o CoM real está a 0,228 m do chão
(o modelo usa `z_com = 0,22`).

### 10b. O deslocamento entre a pelve e o CoM

O MPC calcula uma trajetória da **pelve**, mas o CoM real é outro ponto. ✔ Medido:

| situação | CoM real − pelve (x) | (y) |
|---|---|---|
| parado | −5,4 mm | +0,1 mm |
| andando (média, 8 s) | −4,7 mm | −0,1 mm |
| andando (desvio) | ±1,1 mm | ±3,2 mm (9,6 mm pico a pico) |

Há um **deslocamento fixo de ~5 mm** e uma variação dinâmica de poucos milímetros. Se você ignorar o
deslocamento, o MPC passa a "corrigir" um erro que não existe. Calibre-o com o robô parado e gire-o pelo
yaw:

```python
offset_body = com_parado - pelve_parada                        # ≈ [-0.0054, 0.0001]
offset_world = rot_z(yaw) @ offset_body
com_medido = com_real - offset_world                           # "CoM equivalente da pelve"
```

Alimente o MPC com `com_medido`: assim continua planejando a trajetória da pelve, sem mudar a saída.

### 10c. Como entra no MPC

No início de cada `_solve`, em vez de usar só `s₀` do modelo, misture:

```
s0 = s_modelo + K · (s_medido − s_modelo)           K = 0: malha aberta   …   K = 1: só confia na medida
```

- `s_medido = [x_medido, v_medido]`. A **aceleração não é medida**: mantenha a do modelo.
- Comece com `K` pequeno (0,1 ou 0,2) e aumente.
- Crie no engine um método `update_measurement(com_xy, vel_xy)` e chame-o no `Simulation.step`, antes de
  `engine.step`.

### 10d. Ruído e atraso

- No simulador a velocidade é exata. Para ver o efeito real, some ruído (por exemplo `N(0, 5 mm/s)`) e
  veja o MPC reagir ao ruído. Depois filtre a velocidade (passa-baixa) ou use um Kalman sobre o mesmo modelo
  `[x, ẋ, ẍ]`.
- O robô responde com atraso ao comando de pelve (IK + atuadores PD). Ganho alto com atraso instabiliza.
  É por isso que se começa com `K` pequeno.

### 10e. Como testar a realimentação

**Empurrão.** O MuJoCo permite aplicar uma força na base:

```python
data.xfrc_applied[1, :3] = [fx, fy, 0]          # no corpo 1; zere de volta depois de ~0,1 s
```

✔ Calibração (malha aberta, andando a 0,05 m/s, força aplicada por 0,1 s em `t = 3 s`):

| direção | aguenta | cai |
|---|---|---|
| lateral (y) | 10 N | 20 N |
| frontal (x) | 20 N | 40 N |

(analítico e MPC em malha aberta se comportam igual, como esperado). **Seu objetivo:** que o limiar suba
com a realimentação. Meça também a inclinação máxima do tronco e o tempo até voltar ao padrão.

**Centro de pressão real.** Para conferir se a hipótese "o ZMP fica dentro do pé" vale no robô de
verdade:

```python
def centro_de_pressao(model, data):
    floor = model.geom("floor").id
    f6, num, den = np.zeros(6), np.zeros(2), 0.0
    for k in range(data.ncon):
        c = data.contact[k]
        if floor not in (c.geom1, c.geom2): continue
        mujoco.mj_contactForce(model, data, k, f6)          # f6[0] = força normal
        num += f6[0] * c.pos[:2];  den += f6[0]
    return None if den < 1e-6 else num / den
```

(Exclua os contatos da bola, se houver.) ✔ Parado: CoP em `x = −6,3 mm`, coerente com o CoM 5,4 mm
atrás da pelve.

**Limite esperado.** O robô obedece a posições via IK e atuadores PD, então corrigir só o CoM resolve
perturbações pequenas. Empurrões fortes pedem também **mudar onde o próximo passo pousa** (Diedam et al.,
2008). Isso é a etapa seguinte e fica fora deste guia.

---

## Lista de conferência final

| Verificação | Alvo |
|---|---|
| `Pu[0,0]` | −8,864e-4 |
| Resíduo KKT | ≲ 1e-8 (gabarito: 1e-13) |
| Tempo por QP (2 eixos) | < 1 ms |
| Horizonte mínimo estável | N = 15 (use 20) |
| Maior salto de pelve por tick, andando e parando | ≈ 2,4 mm |
| CoM do MPC vs. analítico, 30 s | ≲ 2 cm |
| ZMP planejado dentro dos limites | sempre |
| Empurrão lateral que derruba (malha aberta) | 20 N, 0,1 s |

## Leitura

- Herdt et al. (2010), *Walking without thinking about it*: esta formulação. Comece por ela.
- Kajita et al. (2003), *Biped walking pattern generation by using preview control of ZMP*.
- Wieber (2006), *Trajectory free linear model predictive control for stable walking*.
- Englsberger et al. (2015), *Three-dimensional bipedal walking control based on divergent component of motion*:
  o DCM da etapa 8.
- Diedam et al. (2008), *Online walking gait generation with adaptive foot positioning*.
- Boyd & Vandenberghe, *Convex Optimization*, caps. 4 e 5: QP e condições KKT.
