# Teoria

Este documento explica os modelos usados no código e aponta onde cada equação está implementada. A ideia é poder ler a teoria e o código lado a lado.

- [1. Pêndulo Invertido Linear (LIPM)](#1-pêndulo-invertido-linear-lipm)
- [2. Zero Moment Point (ZMP)](#2-zero-moment-point-zmp)
- [3. Gerador analítico de caminhada (MuJoCo)](#3-gerador-analítico-de-caminhada-mujoco)
- [4. MPC do CoM (PyBullet)](#4-mpc-do-com-pybullet)
- [5. Cinemática inversa analítica da perna](#5-cinemática-inversa-analítica-da-perna)
- [6. Do alvo cinemático ao torque](#6-do-alvo-cinemático-ao-torque)
- [Limitações conhecidas](#limitações-conhecidas)
- [Referências](#referências)

---

## 1. Pêndulo Invertido Linear (LIPM)

Um humanoide tem dezenas de graus de liberdade, mas para planejar o equilíbrio basta um modelo bem mais simples: toda a massa concentrada no centro de massa (CoM), apoiada num ponto do chão por uma "perna" sem massa e de comprimento variável. Se o CoM é mantido a uma **altura constante** $z_c$, a dinâmica horizontal fica **linear** e desacoplada entre x e y:

$$
\ddot{x} = \omega^2 \,(x - p_x), \qquad \ddot{y} = \omega^2 \,(y - p_y), \qquad \omega = \sqrt{g / z_c}
$$

onde $(p_x, p_y)$ é o ZMP (próxima seção). Intuição: o CoM é "empurrado para longe" do ponto de apoio, e com mais força quanto mais longe estiver. Por isso andar é uma sequência de quedas controladas: a cada passo o ponto de apoio é colocado de forma a frear e redirecionar essa queda.

| Onde | O quê |
|------|-------|
| [`trajectory_generator.py`](../mujoco_sim/analytical_walking/trajectory_generator.py) | `lam = np.sqrt(g / z_com)`, solução fechada |
| [`mpc_planner.py`](../pybullet_sim/mpc_planner.py) | `self.omega`, dinâmica discretizada nas restrições do QP |

## 2. Zero Moment Point (ZMP)

O ZMP é o ponto do chão onde o momento horizontal resultante das forças de contato é nulo, ou seja, o "centro de pressão" sob os pés. A condição física de equilíbrio é:

> **O ZMP precisa estar dentro do polígono de suporte** (a envoltória convexa dos pés em contato).

Se o planejador pede um ZMP fora do pé, o pé real não consegue gerar essa força e o robô tomba em torno da borda do pé. Na prática:

- **Apoio simples:** o ZMP fica sob o pé de apoio.
- **Apoio duplo:** o ZMP pode estar em qualquer lugar entre os dois pés, e é nessa fase que ele "viaja" de um pé para o outro.

Invertendo a equação do LIPM, dá para recuperar o ZMP a partir de uma trajetória do CoM: $p = x - \ddot{x}/\omega^2$. É isso que [`tools/plot_trajectories.py`](../tools/plot_trajectories.py) faz para desenhar o ZMP.

## 3. Gerador analítico de caminhada (MuJoCo)

Arquivo: [`mujoco_sim/analytical_walking/trajectory_generator.py`](../mujoco_sim/analytical_walking/trajectory_generator.py)

Aqui o problema é invertido: **primeiro** escolhe-se a trajetória do ZMP e **depois** calcula-se o CoM que a realiza, resolvendo a EDO do LIPM em forma fechada.

### Fases de um passo

Cada passo dura $T$ e tem três fases, com $t_b = T \cdot \text{ds\_ratio}/2$ e $t_e = T - t_b$:

```
|-- apoio duplo --|---------- apoio simples ----------|-- apoio duplo --|
0                 tb                                  te               T
ZMP: início → pé de apoio      ZMP parado no pé de apoio      pé de apoio → fim
```

O ZMP de referência é linear por partes: sai da posição inicial do torso, vai até o pé de apoio, fica lá durante o apoio simples e segue para a posição final do torso.

### Solução fechada

Para um ZMP linear por partes, a solução de $\ddot{x} = \omega^2(x - p)$ é

$$
x(t) = p(t) + c_1 e^{\omega t} + c_2 e^{-\omega t} - \text{(termo de correção da rampa do ZMP)}
$$

As constantes $c_1, c_2$ são escolhidas para que $x(0)$ seja a pose inicial do torso e $x(T)$, a pose final (`c1`, `c2` no código). O resultado é um CoM que balança suavemente na direção do pé de apoio. Veja o gráfico gerado por `tools/plot_trajectories.py`.

> **Detalhe didático:** a solução casa a *posição* do CoM entre passos, mas não a *velocidade*. Na troca de passo a aceleração dá um salto, e o ZMP recuperado tem picos nesses instantes. Um gerador "de verdade" impõe também continuidade de velocidade, que é uma das coisas que o MPC resolve naturalmente.

### Pé em balanço

- **Horizontal:** blend cosseno $h(\phi)$ de 1 → 0 entre a posição inicial e a final do pé.
- **Vertical:** bump cosseno $v(\phi)$ com altura máxima `z_step`. Perto do fim do apoio simples ele vira uma descida linear, para o pé pousar com velocidade vertical controlada.

### Planejamento de passos

`select_next_poses` converte o comando de velocidade $(v_x, v_y, \omega_z)$ num deslocamento do torso de $(v_x, v_y, \omega_z)\cdot T$, no referencial do próprio torso. O pé em balanço pousa a `y_sep` para o lado do novo torso e meia passada à frente.

### Máquina de estados

[`walking_engine.py`](../mujoco_sim/analytical_walking/walking_engine.py) alterna pé de apoio e pé de balanço a cada passo e controla as transições `IDLE → WALKING ⇄ IDLE_MARCH → STOPPING → IDLE`.

## 4. MPC do CoM (PyBullet)

Arquivo: [`pybullet_sim/mpc_planner.py`](../pybullet_sim/mpc_planner.py)

Em vez de fixar o ZMP e resolver o CoM analiticamente, o **Model Predictive Control** deixa o otimizador escolher o ZMP a cada instante, dentro de limites. Para cada eixo (x e y separadamente), num horizonte de $N$ passos de $\Delta t$:

$$
\begin{aligned}
\min_{x,\,v,\,p}\quad & \sum_{k=0}^{N-1} W_{com}\,(x_k - x^{ref}_k)^2 + W_{zmp}\,(p_k - p^{ref}_k)^2 + W_{vel}\,v_k^2 \\
\text{s.a.}\quad & x_{k+1} = x_k + v_k\,\Delta t \\
& v_{k+1} = v_k + \omega^2 (x_k - p_k)\,\Delta t \\
& p^{min}_k \le p_k \le p^{max}_k \\
& x_0 = x_{atual},\; v_0 = v_{atual}
\end{aligned}
$$

- As duas primeiras restrições são o LIPM discretizado por Euler.
- A terceira garante que **o ZMP fica dentro do pé** (o polígono de suporte vira uma caixa de ±`FOOT_MARGIN` em torno do pé). É ela que dá garantia de equilíbrio *no modelo*.
- É um QP convexo, resolvido pelo OSQP via `cvxpy`. O problema é montado uma vez com `cp.Parameter` e só os valores mudam a cada ciclo.
- Só o primeiro estado previsto é usado e o problema é resolvido de novo no ciclo seguinte (*receding horizon*).

As referências (`get_references_x/y`) e os limites do ZMP vêm de um plano de passos fixo: período `T_STEP`, pés alternados e passada `cmd_vx * T_STEP`.

> **Importante:** neste código o MPC roda em **malha aberta**. O estado $(x_0, v_0)$ passado ao MPC é a própria previsão do ciclo anterior, não uma medição do simulador. O MPC funciona, portanto, como um **gerador de trajetória**, e quem estabiliza o robô são os controladores PD das juntas. Fechar a malha (estimar o CoM real e realimentar) é o passo natural seguinte.

## 5. Cinemática inversa analítica da perna

Arquivos: [`mujoco_sim/kinematics/ik_solver.py`](../mujoco_sim/kinematics/ik_solver.py) (OP3, 6 DoF completos) e [`pybullet_sim/kinematics.py`](../pybullet_sim/kinematics.py) (Aurea, versão simplificada com pé paralelo ao chão e `hip_yaw = 0`).

A perna tem a cadeia `hip_yaw (z) → hip_roll (x) → hip_pitch (y) → knee (y) → ank_pitch (y) → ank_roll (x)`. Como os três eixos do quadril se cruzam (aproximadamente) num ponto e os dois do tornozelo também, o problema tem solução fechada. É o método do livro de Kajita et al.:

1. **Ponto do tornozelo:** a partir do alvo da sola, sobe-se `D_ANKLE_SOLE` no referencial do pé.
2. **Vetor tornozelo → quadril** $r$, expresso no referencial do pé. Sua norma $C$ é a distância entre quadril e tornozelo.
3. **Joelho** pela lei dos cossenos: $\cos q_{knee} = \dfrac{C^2 - L_1^2 - L_2^2}{2 L_1 L_2}$. Se $C > L_1 + L_2$, o alvo está fora de alcance.
4. **Tornozelo roll:** $q_{ar} = \operatorname{atan2}(r_y, r_z)$.
5. **Tornozelo pitch:** ângulo de $r$ no plano sagital, menos o ângulo $\alpha$ que a coxa faz com a linha quadril–tornozelo ($\sin\alpha = \frac{L_1}{C}\sin q_{knee}$).
6. **Quadril:** a rotação que sobra, $R = R_{body}^\top R_{foot} R_x(-q_{ar}) R_y(-(q_{knee}+q_{ap}))$, é decomposta em $R_z(q_{hy}) R_x(q_{hr}) R_y(q_{hp})$.

O IK devolve ângulos numa convenção geométrica. A tabela `JOINT_SIGNS` em [`config.py`](../mujoco_sim/config.py) converte para o sentido real de cada eixo no MJCF do OP3.

O teste [`tests/test_mujoco_sim.py`](../tests/test_mujoco_sim.py) confere o IK contra a cinemática direta do MuJoCo: aplica os ângulos e mede onde o tornozelo foi parar.

## 6. Do alvo cinemático ao torque

Nenhum dos dois simuladores recebe torques calculados pelo planejador. O laço é sempre:

```
gerador/MPC → poses desejadas (torso + pés) → IK → ângulos → controlador PD de posição → torque → física
```

No MuJoCo os atuadores são `position` com $\tau = K_p (q_{des} - q) - K_v \dot{q}$, saturado em `ACTUATOR_FORCE_RANGE` ([`config.py`](../mujoco_sim/config.py)). No PyBullet é o `POSITION_CONTROL` do motor de cada junta.

Com `PHYSICS = False`, o MuJoCo só faz cinemática: o torso é teleportado para a pose planejada e as juntas são escritas diretamente. Serve para depurar o gerador e o IK isolados da dinâmica.

---

## Limitações conhecidas

Estes pontos são intencionais (simplificações de estudo) ou problemas já identificados. São bons exercícios.

| Onde | Limitação |
|------|-----------|
| PyBullet | **Relógio do MPC ≠ relógio da física.** `int(DT_MPC / DT_SIM) = int(4,8) = 4`, então cada ciclo avança a trajetória 0,020 s, mas a física só 0,0167 s: a caminhada roda 1,2× mais rápido que o planejado. Os ganhos atuais foram ajustados assim; com relógios consistentes o robô cai até o planejador ser reajustado. Detalhes em [`main.py`](../pybullet_sim/main.py). |
| PyBullet | MPC em malha aberta (seção 4). |
| MuJoCo | O IK ignora o deslocamento dos eixos de roll do quadril e do tornozelo (−24, ∓19 mm no OP3). Com o balanço lateral da marcha padrão, o tornozelo erra o alvo em ~4–5 mm. |
| MuJoCo | Velocidade do CoM descontínua entre passos (seção 3). |
| Ambos | O CoM é aproximado pela posição do torso/pelve; a massa real das pernas e dos braços não entra no modelo. |

## Referências

- S. Kajita, H. Hirukawa, K. Harada, K. Yokoi. *Introduction to Humanoid Robotics*. Springer, 2014. LIPM, ZMP, preview control e o IK analítico da seção 5.
- S. Kajita et al. "Biped walking pattern generation by using preview control of zero-moment point". *ICRA*, 2003.
- S. Kajita et al. "The 3D Linear Inverted Pendulum Mode: a simple modeling for a biped walking pattern generation". *IROS*, 2001.
- P.-B. Wieber. "Trajectory free linear model predictive control for stable walking in the presence of strong perturbations". *Humanoids*, 2006. A formulação de MPC com restrição de ZMP.
- M. Vukobratović, B. Borovac. "Zero-Moment Point — thirty five years of its life". *International Journal of Humanoid Robotics*, 2004.
- [MuJoCo Menagerie: Robotis OP3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotis_op3), o modelo usado na simulação MuJoCo.
- [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py), que baixa o modelo automaticamente.
