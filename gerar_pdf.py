from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import datetime

OUT = "/home/pedrohperes/Codes/humanoidMotion-predictiveControl/resumo_tecnico_MPC_walking.pdf"

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.2*cm, bottomMargin=2*cm,
    title="Resumo Tecnico - MPC Walking",
    author="Pedro H. Peres",
)

W = A4[0] - 4*cm  # largura util

# ── Estilos ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

AZUL    = colors.HexColor("#1a3c8c")
AZUL_S  = colors.HexColor("#1450a0")
CINZA   = colors.HexColor("#555555")
BGCODE  = colors.HexColor("#f0f0f5")
BGTAB   = colors.HexColor("#dce6f5")

titulo_pg = ParagraphStyle("titulo_pg",
    fontSize=22, leading=28, textColor=AZUL, alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold")

subtit_pg = ParagraphStyle("subtit_pg",
    fontSize=13, leading=18, textColor=CINZA, alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold")

data_pg = ParagraphStyle("data_pg",
    fontSize=10, textColor=colors.HexColor("#999999"), alignment=TA_CENTER, spaceAfter=16)

h1 = ParagraphStyle("h1",
    fontSize=14, leading=18, textColor=colors.black, fontName="Helvetica-Bold",
    spaceBefore=14, spaceAfter=4)

h2 = ParagraphStyle("h2",
    fontSize=11, leading=15, textColor=AZUL_S, fontName="Helvetica-Bold",
    spaceBefore=8, spaceAfter=3)

corpo = ParagraphStyle("corpo",
    fontSize=10, leading=14, textColor=colors.HexColor("#1e1e1e"),
    spaceBefore=2, spaceAfter=4, fontName="Helvetica")

code_s = ParagraphStyle("code_s",
    fontSize=9, leading=13, fontName="Courier",
    textColor=colors.HexColor("#282828"),
    backColor=BGCODE,
    borderPad=6, spaceBefore=4, spaceAfter=4,
    leftIndent=8, rightIndent=8)

bullet_s = ParagraphStyle("bullet_s",
    fontSize=10, leading=14, fontName="Helvetica",
    leftIndent=14, firstLineIndent=-10,
    textColor=colors.HexColor("#1e1e1e"),
    spaceBefore=1, spaceAfter=1)


def H1(txt):
    elems = [
        Spacer(1, 0.1*cm),
        Paragraph(txt, h1),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333333")),
        Spacer(1, 0.15*cm),
    ]
    return elems

def H2(txt):  return [Paragraph(txt, h2)]
def P(txt):   return [Paragraph(txt, corpo)]
def SP():     return [Spacer(1, 0.25*cm)]

def CODE(txt):
    return [Preformatted(txt, code_s)]

def BULLET(items):
    return [Paragraph(f"<bullet>&bull;</bullet> {i}", bullet_s) for i in items]

def TABELA(header, rows, col_w=None):
    if col_w is None:
        n = len(header)
        col_w = [W / n] * n
    data = [header] + rows
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  BGTAB),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.HexColor("#141414")),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fc")]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",     (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    return [t, Spacer(1, 0.2*cm)]

def HR():
    return [HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#cccccc")), Spacer(1, 0.1*cm)]

# ── Conteudo ──────────────────────────────────────────────────────────────────
story = []

# Capa
story += [
    Spacer(1, 1*cm),
    Paragraph("Caminhada Bipede com MPC", titulo_pg),
    Paragraph("Resumo Tecnico da Discussao", subtit_pg),
    Paragraph(f"Pedro H. Peres  |  {datetime.date.today().strftime('%B %Y')}", data_pg),
    HRFlowable(width="70%", thickness=1.5, color=AZUL, hAlign="CENTER"),
    Spacer(1, 0.8*cm),
]

# 1. Arquitetura
story += H1("1. Arquitetura do Projeto")
story += P(
    "O repositorio humanoidMotion-predictiveControl contem duas simulacoes independentes "
    "de caminhada bipede para robos humanoides, ambas baseadas em cinematica inversa analitica "
    "combinada com geracao de trajetoria do CoM."
)
story += TABELA(
    ["Simulador", "Robo", "Motor de caminhada", "Fisica"],
    [
        ["pybullet_sim", "Aurea (URDF)", "MPC (CVXPY + OSQP)", "PyBullet"],
        ["mujoco_sim",   "Robotis OP3",  "Analitico ou MPC",   "MuJoCo (mj_step)"],
    ],
    col_w=[3.5*cm, 3.5*cm, 6*cm, 3.2*cm]
)

# 2. MPC do pybullet
story += H1("2. MPC do pybullet_sim")

story += H2("2.1 Modelo fisico - LIPM")
story += P(
    "O MPC nao modela o robo completo. Trata o CoM como um pendulo invertido linear (LIPM), "
    "cuja dinamica e:"
)
story += CODE("  x'' = omega^2 * (x - zmp)      onde omega = sqrt(g / h)")
story += P(
    "A integracao e Euler explicito com dt = 0.02 s (50 Hz). O estado completo e [x, vx] - "
    "posicao e velocidade do CoM. Esta simplificacao e valida enquanto a altura do CoM (h) "
    "for aproximadamente constante."
)

story += H2("2.2 Problema de otimizacao (QP)")
story += P(
    "O QP tem horizonte N = 25 steps (0.5 s de antecipacao) e e resolvido com OSQP via CVXPY. "
    "Os eixos X e Y sao tratados como dois QPs independentes com a mesma estrutura."
)
story += P("Variaveis de decisao:")
story += BULLET([
    "v_x[0..N]     - posicao do CoM ao longo do horizonte",
    "v_vx[0..N]    - velocidade do CoM",
    "v_zmp[0..N-1] - posicao do ZMP (variavel de planejamento)",
])
story += P("Funcao de custo:")
story += CODE(
    "  800 * (x[k] - com_ref[k])^2   # rastrear referencia do CoM\n"
    "    2 * (zmp[k] - zmp_ref[k])^2  # ZMP proximo do centro do pe\n"
    "   10 * vx[k]^2                  # suavizar velocidade"
)
story += P(
    "O peso alto no CoM (800) e baixo no ZMP (2) faz o otimizador mover o ZMP livremente "
    "para que o CoM siga a referencia - o ZMP e o que tem grau de liberdade real no planejamento."
)

story += H2("2.3 ZMP como grandeza fisica")
story += P(
    "O ZMP nao e uma entrada de controle - e uma consequencia do movimento. A relacao no LIPM e:"
)
story += CODE("  ZMP = x - x'' / omega^2")
story += P(
    "O 'ZMP' no QP e uma variavel de planejamento. O MPC encontra uma trajetoria de CoM tal que "
    "o ZMP implicito fique dentro do poligono de suporte. O ZMP real do PyBullet nunca e "
    "medido nem comparado - e puramente planejamento."
)

story += H2("2.4 Necessidade do vx_next")
story += P(
    "O LIPM e segunda ordem - o estado completo e (x, vx). Descartar vx_next e estimar "
    "velocidade por diferenca finita resulta em:"
)
story += CODE("  vx_estimado = (x_next - x_atual) / dt = vx[0]  # velocidade ATUAL, nao a proxima")
story += P(
    "Isso perde o efeito da aceleracao gerada pelo ZMP. Com feedback real do estado do robo "
    "(medicao da posicao do CoM), seria possivel estimar vx externamente - mas isso exige "
    "um estimador de estado robusto (ex: filtro de Kalman)."
)

# 3. Herdt
story += H1("3. MPC Classico vs. Herdt 2010")

story += H2("3.1 MPC Classico - Kajita 2003")
story += P(
    "O esquema original usa jerk do CoM como variavel de controle. As posicoes dos passos "
    "precisam ser decididas por um foot step planner separado antes da execucao."
)
story += CODE(
    "  min  alpha*||Jerk||^2 + gamma*||CoP - CoP_ref||^2\n"
    "  CoP_ref = centro do poligono de suporte pre-planejado"
)

story += H2("3.2 Herdt - Automatic Foot Step Placement")
story += P(
    "A modificacao central e adicionar as posicoes dos proximos passos como variaveis de "
    "decisao do proprio QP. O robo recebe apenas uma velocidade de referencia e decide "
    "automaticamente onde pousar o pe."
)
story += CODE(
    "  u_k = [Jerk_x, X_footsteps, Jerk_y, Y_footsteps]\n\n"
    "  min  alpha*||Jerk||^2 + beta*||CoM_vel - vel_ref||^2 + gamma*||CoP - CoP_ref(steps)||^2\n"
    "  s.t. CoP dentro do poligono de suporte (restricao explicita hard)"
)
story += P(
    "A referencia do CoP passa a depender dos proprios footsteps que o QP otimiza - CoP e "
    "passos se co-adaptam numa unica otimizacao."
)
story += TABELA(
    ["", "MPC Classico", "Herdt"],
    [
        ["Variavel de controle", "Jerk do CoM", "Jerk + posicao dos passos"],
        ["Footsteps",            "Pre-definidos", "Otimizados online"],
        ["Restricao CoP",        "Soft (custo)", "Hard (restricao explicita)"],
        ["Input ao robo",        "vel_ref + passos fixos", "Apenas vel_ref"],
        ["Robustez",             "Limitada", "Adapta passos a perturbacoes"],
    ],
    col_w=[4.5*cm, 5.5*cm, 6.2*cm]
)

# 4. Feedback
story += H1("4. Feedback Necessario por Metodo")
story += TABELA(
    ["Sensor", "MPC Classico", "Herdt"],
    [
        ["IMU acelerometro",  "x'' do CoM",          "x'' do CoM"],
        ["IMU + cinematica",  "Estimar vx (deriva)",  "Estimar vx (deriva)"],
        ["Encoders + FK",     "Posicao dos pes",      "Posicao dos pes (critico)"],
        ["F/T no pe",         "CoP real (ideal)",     "CoP real (necessario)"],
        ["Deteccao contato",  "Timing do passo",      "Atualizar X_fc no QP"],
    ],
    col_w=[4.2*cm, 5.5*cm, 6.5*cm]
)
story += P(
    "No Herdt, X_fc (posicao do pe de suporte fixo) entra como parametro do QP. Um erro "
    "nessa leitura contamina o planejamento de todos os passos futuros - a sensibilidade "
    "a erro de localizacao dos pes e muito maior que no classico."
)

# 5. MPC vs Analitico
story += H1("5. MPC Classico vs. Metodo Analitico")
story += P(
    "O metodo analitico define explicitamente onde o CoM estara em funcao do tempo. "
    "Funciona bem em regime estacionario (velocidade constante, chao plano, sem perturbacoes). "
    "O MPC classico adiciona valor em dois cenarios especificos:"
)
story += BULLET([
    "Mudanca brusca de velocidade - encontra automaticamente a trajetoria de aceleracao "
    "que mantem o CoP feasivel durante a transicao.",
    "Perturbacao externa com feedback - replana a partir do estado real do robo. "
    "Sem feedback, os dois metodos tem o mesmo problema.",
])
story += P(
    "Em regime estacionario sem feedback, o MPC classico nao justifica sua complexidade "
    "comparado a um analitico bem calibrado. O valor aparece nas bordas do movimento."
)

# 6. Hardware real
story += H1("6. Hardware Real - Dynamixel MX-106 + IMU")

story += H2("6.1 O que os sensores realmente fornecem")
story += TABELA(
    ["Sensor", "O que da", "Limitacao"],
    [
        ["MX-106 encoder",   "Posicao da junta",      "Folga 0.5-2 graus por junta"],
        ["MX-106 corrente",  "Proxy de contato",      "Ruidoso apos 6 anos de uso"],
        ["IMU pitch/roll",   "Orientacao do torso",   "Melhor leitura disponivel"],
        ["IMU acelerometro", "Aceleracao linear",     "Integrar deriva rapidamente"],
    ],
    col_w=[4*cm, 5*cm, 7.2*cm]
)
story += P(
    "Com folga acumulada em 6 juntas por perna, o erro na posicao do pe pode atingir 5-15 mm. "
    "Isso inviabiliza estimacao confiavel do CoM via cinematica direta e elimina o Herdt "
    "como opcao pratica neste hardware."
)

story += H2("6.2 Teto realista de qualidade")
story += P("Caminhada analitica com estabilizacao de torso via IMU:")
story += CODE(
    "  Analitico         -> trajetoria base deterministica, sem acumulacao de erro\n"
    "  IMU pitch/roll    -> controlador de torso no quadril (~200-500 Hz)\n"
    "  Corrente motor    -> deteccao de contato, corrigir timing do passo"
)
story += P(
    "O controlador de torso soma correcoes por cima dos angulos calculados pela IK. "
    "O IMU e o feedback mais valioso porque nao tem folga mecanica - leitura limpa "
    "e de alta frequencia."
)

# 7. Robustez
story += H1("7. Robustez a Perturbacoes e Empurroes")

story += H2("7.1 Hierarquia de resposta")
story += CODE(
    "  Perturbacao pequena -> Ankle strategy (tornozelo)\n"
    "  Perturbacao media   -> Hip strategy (quadril/torso via IMU)\n"
    "  Perturbacao grande  -> Foot placement (captura de passo)"
)

story += H2("7.2 Ankle Strategy")
story += P(
    "Malha proporcional simples: erro de pitch do IMU gera torque extra no tornozelo. "
    "Roda a alta frequencia, reage antes do MPC. Implementavel diretamente com o hardware atual."
)

story += H2("7.3 Capture Point / DCM")
story += P("O Capture Point e o ponto onde, se o pe pousar, o robo para de cair:")
story += CODE("  CP = x + vx / omega")
story += P(
    "Com IMU e possivel estimar vx por integracao curta (funciona por alguns segundos antes "
    "de derivar). Ao detectar perturbacao (pico no IMU), calcula-se onde o pe deve pousar "
    "e redireciona-se o passo em andamento."
)
story += P(
    "O DCM (Divergent Component of Motion) generaliza o Capture Point como variavel de "
    "estado dinamica - base do controle de robos como Atlas e Valkyrie. Voce pode fazer "
    "uma versao simplificada so com planejamento de passos dado o seu hardware."
)

# 8. RL
story += H1("8. Reinforcement Learning e Aprendizado")

story += H2("8.1 Onde o aprendizado agrega valor")
story += P(
    "O problema central e que o modelo (LIPM + IK analitica) nao representa bem o robo real: "
    "folgas, desgaste, flexibilidade estrutural. O aprendizado pode aprender a diferenca "
    "entre o modelo e a realidade sem precisar de um modelo mais complexo."
)

story += H2("8.2 Residual Policy")
story += CODE(
    "  q_final = q_analitico(t) + pi(observacao)\n\n"
    "  observacao: [IMU pitch/roll, corrente motores, fase do passo, erro de junta]\n"
    "  saida:      [delta_q por junta - pequenas correcoes]"
)
story += P(
    "A politica so aprende o erro do modelo, nao a caminhada inteira. Se falhar, o analitico "
    "continua operando - degradacao graciosa. Treina no MuJoCo com domain randomization."
)

story += H2("8.3 Domain Randomization para Sim2Real")
story += CODE(
    "  kp_gain     ~ Uniform(0.7, 1.3)    # ganho degradado pelo desgaste\n"
    "  backlash    ~ Uniform(0, 2 graus)  # folga por junta\n"
    "  delay       ~ Uniform(8, 20 ms)    # latencia serial Dynamixel\n"
    "  joint_frict ~ Uniform(0, 0.3)      # atrito extra pelo desgaste\n"
    "  imu_noise   ~ Normal(0, 0.02 gr)   # ruido do IMU"
)

story += H2("8.4 Ordem de implementacao recomendada")
story += CODE(
    "  1. System Identification do robo real\n"
    "  2. Analitico + estabilizacao IMU com modelo calibrado\n"
    "  3. Residual Policy treinada no MuJoCo com domain randomization\n"
    "  4. Fine-tuning leve no hardware real (episodios curtos, safe exploration)"
)

# 9. Referencias
story += H1("9. Referencias")
story += BULLET([
    "Kajita et al. (2003) - Biped walking pattern generation by using preview control of ZMP. ICRA.",
    "Herdt et al. (2010) - Online Walking Motion Generation with Automatic Foot Step Placement. Advanced Robotics, Taylor & Francis.",
    "Wieber (2006) - Trajectory free linear MPC for stable walking in the presence of strong perturbations. Humanoids.",
    "Wieber (2008) - Viability and predictive control for safe locomotion. IROS.",
    "Diedam et al. (2008) - Online walking gait generation with adaptive foot positioning through LMPC. IROS.",
])

doc.build(story)
print(f"PDF gerado: {OUT}")
