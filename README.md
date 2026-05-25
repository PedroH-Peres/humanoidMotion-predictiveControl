# humanoidMotion-predictiveControl

Simulações de caminhada bípede para robôs humanoides usando cinemática inversa analítica e controle preditivo (MPC). O projeto tem dois ambientes: um em PyBullet com o robô **Aurea** (URDF próprio), e um mais recente em MuJoCo com o **Robotis OP3**.

---

## Estrutura

```
├── pybullet_sim/   # Simulação do Aurea — MPC + IK analítica
└── mujoco_sim/     # Simulação do OP3  — caminhada analítica ou MPC
    ├── AnalyticalWalking/
    └── Utils/
```

---

## pybullet_sim

Caminhada controlada por MPC com geração de trajetória do CoM (Linear Inverted Pendulum Model). A IK é analítica relativa ao CoM — os pés são posicionados no espaço do corpo, não no mundo.

### Dependências

```
pip install pybullet numpy scipy
```

### Como rodar

```bash
cd pybullet_sim
python main.py
```

O robô passa por uma pose inicial de estabilização e em seguida começa a caminhar com `cmd_vx = 0.1 m/s`. Para mudar a velocidade, edite essa variável em `main.py`.

---

## mujoco_sim

Simulação com física completa (contato, torques, amortecimento). Usa o modelo oficial do Robotis OP3 via `robot_descriptions`. Suporta dois modos de caminhada, configurados em `mujoco_sim/AnalyticalWalking/config.py`:

| Flag | Efeito |
|------|--------|
| `PHYSICS = True` | Roda `mj_step` com atuadores PD — física real |
| `PHYSICS = False` | Roda `mj_forward` — só cinemática, sem contato |
| `USE_MPC = True` | Motor de caminhada baseado em MPC |
| `USE_MPC = False` | Motor de caminhada analítico (puro) |

### Dependências

```
pip install mujoco robot_descriptions numpy
```

> O `robot_descriptions` baixa o MJCF do OP3 automaticamente na primeira execução.

### Como rodar

```bash
# da raiz do projeto
python -m mujoco_sim
```

Uma janela do viewer MuJoCo abre. Com o **terminal em foco** (não a janela do viewer), use as teclas:

| Tecla | Comando |
|-------|---------|
| `W` | Andar para frente |
| `S` | Passo lateral |
| `R` | Rotação no lugar |
| `M` | Marcha estacionária |
| `X` | Parar |
| `Q` | Sair |

> **Atenção:** o input de teclado usa `msvcrt` (Windows). No Linux/macOS o polling de tecla não funciona — o robô executa a pose padrão (marcha no lugar) sem comandos interativos.

### Parâmetros de caminhada

Tudo em `mujoco_sim/AnalyticalWalking/config.py`:

```python
WALK_PARAMS = {
    "T":        0.4,    # período de um passo [s]
    "z_com":    0.22,   # altura do torso [m]
    "z_step":   0.04,   # altura máxima do pé no ar [m]
    "ds_ratio": 0.12,   # fração em apoio duplo
    "y_sep":    0.054,  # separação lateral dos pés [m]
}
```

Os ganhos dos atuadores (`ACTUATOR_KP`, `ACTUATOR_KV`, `ACTUATOR_FORCE_RANGE`) também ficam no mesmo arquivo.

---

## Requisitos gerais

- Python 3.10+
- As duas sims são independentes — instale só o que for usar
