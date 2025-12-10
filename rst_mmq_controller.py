import numpy as np
import matplotlib.pyplot as plt

def sistema(u, y1):
    """Função que define o sistema dinâmico (não-linear)"""
    # Exemplo: y(t) = 0.08 * y(t-1)^2 + 0.56 * y(t-1) + 0.1 * u(t-1)
    return 0.08 * y1**2 + 0.56 * y1 + 0.1 * u

# --- Configurações Iniciais ---
np.random.seed(0)
nit = 5000 # Número de iterações

# --- Geração de Ruído (Opcional, se precisar de identificação estocástica) ---
eta = np.zeros(nit)
for i in range(nit):
    eta[i] = 1 if np.random.rand() > 0.5 else -1
ruido = eta * 0.00 # Ruído definido como 0.00, ou seja, sistema determinístico

# --- Parâmetros Iniciais do MMQ ---
theta = np.zeros(4) # [a1, a2, a3, a4]
P = 1000 * np.eye(4) # Matriz de Covariância inicial
# Fatores de esquecimento (Adaptativo com Fator de Esquecimento Variável)
lambds0 = 1.1 
lambds1 = .75
lambda_ = lambds0 / lambds1 

# --- Sinais de Entrada e Perturbação ---
# Sinal de referência (degraus)
uc = np.concatenate([
    1.00 * np.ones(nit//4),
    0.50 * np.ones(nit//4),
    0.75 * np.ones(nit//4),
    1.15 * np.ones(nit - 3*(nit//4)) # Ajuste para garantir o tamanho correto
]) 
# Perturbação (comentada no laço principal, mas inicializada)
dist = 0.2 * np.concatenate([np.zeros(nit - 1000), np.ones(1000)])

# --- Inicialização de Arrays ---
y = np.zeros(nit) # Saída real do sistema (controlada)
erro = np.zeros(nit) # Erro de previsão (para MMQ)
# Parâmetros RST resultantes:
bt, t0, s0, s1 = np.ones(nit), np.ones(nit), np.ones(nit), np.ones(nit)
# Parâmetros identificados:
a1, a2, a3, a4 = np.ones(nit), np.ones(nit), np.ones(nit), np.ones(nit) 
# Saída do sistema sem controle (ysc) e de referência (yr - desnecessária aqui, mas mantida)
yr, ysc = np.zeros(nit), np.zeros(nit)
e = np.zeros(nit) # Erro de controle
u = np.zeros(nit) # Sinal de controle
u[0], u[1] = 0, 0 # Condições iniciais
y[0], y[1] = 0, 0 # Condições iniciais

# --- Laço Principal de Simulação e Controle ---
for t in range(2, nit):
    # 1. Simulação do sistema
    yr[t] = sistema(u[t-1], yr[t-1]) 
    ysc[t] = sistema(uc[t-1], ysc[t-1]) # Simulação sob referência constante
    y[t] = yr[t] # Saída Real (Sistema controlado) # + dist[t] 
    
    # 2. Erro de Controle
    e[t] = uc[t] - y[t]
    
    # 3. Identificação MMQ Direto
    # Vetor de Regressores (Baseado no modelo ARX do seu sistema RST/MMQ)
    # phi = [u(t-1), u(t-2), -y(t-1), -y(t-2)]
    fi = np.array([uc[t-1], uc[t-2], y[t-1], -y[t-2]]) 
    
    # Resíduo (erro de previsão)
    erro[t] = y[t] - fi.T @ theta
    
    # Ganho MMQ (k_gain)
    k_gain = P @ fi / (lambda_ + fi.T @ P @ fi)
    
    # Atualização dos Parâmetros (theta)
    theta = theta + k_gain * erro[t]
    
    # Atualização da Matriz de Covariância (P)
    P = (P - np.outer(k_gain, fi.T) @ P) / lambds0
    
    # 4. Cálculo dos Parâmetros do Controlador RST
    a1[t], a2[t], a3[t], a4[t] = theta
    
    # Parâmetros do controlador T, S e R (implícitos)
    # Determinar T*b (denominador comum)
    bt[t] = 1 + (a1[t] + a2[t]) - (a3[t] + a4[t])
    # Determinar T_0, S_0, S_1
    s0[t] = (a1[t] / bt[t])
    s1[t] = (a2[t] / bt[t])
    t0[t] = (s0[t] + s1[t]) # T(z^-1) = t0 + t1*z^-1 + ... (simplificado aqui)
    
    # 5. Lei de Controle RST (Implementação simplificada)
    # u(t) = [T0, S0, S1, R1] @ [uc(t), -y(t), -y(t-1), u(t-1)]
    # O seu código tem 4 termos, o que sugere que o T0 é o coeficiente de uc(t)
    # e o u(t-1) vem de um R(z^-1) = 1 + r1*z^-1 (onde r1 é -1)
    # Aqui, os parâmetros são remapeados para T0, S0, S1 e R1 (implícito)
    u[t] = np.array([t0[t], s0[t], s1[t], 1]) @ np.array([uc[t], -y[t], -y[t-1], u[t-1]])

# --- Plotagem de Resultados ---
t = np.arange(nit)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(t, s0, 'r', linewidth=1.5)
plt.title('Parâmetro S0 (Identificado)')
plt.ylim(0, .2)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, s1, 'r', linewidth=1.5)
plt.title('Parâmetro S1 (Identificado)')
plt.ylim(0, .2)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, t0, 'k', linewidth=1.5)
plt.title('Parâmetro T0 (Identificado)')
plt.ylim(0, .2)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b', label='Resposta Controlada (y)', linewidth=1.5)
plt.plot(t, ysc, 'k', label='Resposta Não-Controlada (ysc)', linewidth=1.5)
plt.plot(t, uc, '--r', label='Referência (uc)', linewidth=1.5)
plt.ylim(0, 2.2)
plt.grid(True)
plt.title('Comparação: Saída Controlada vs. Referência')
plt.xlabel('Tempo (amostras)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
