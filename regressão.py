import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# carregando dos dados
dados = pd.read_csv("/content/Housing.csv")

#  One-Hot Encoding

def one_hot_encoding(df):
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)
    return df_encoded

dados_encoded = one_hot_encoding(dados)

# Separação treino/teste

def dividir_treino_teste(df, proporcao_teste=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    tamanho_teste = int(len(df) * proporcao_teste)
    indices_teste = indices[:tamanho_teste]
    indices_treino = indices[tamanho_teste:]
    return df.iloc[indices_treino], df.iloc[indices_teste]

df_treino, df_teste = dividir_treino_teste(dados_encoded)

X_treino = df_treino.drop('price', axis=1).values
y_treino = df_treino['price'].values

X_teste = df_teste.drop('price', axis=1).values
y_teste = df_teste['price'].values

# Funções de métricas

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 5. Regressão Linear OLS

def regressao_ols(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    return beta

def prever(X, beta):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return X_bias @ beta

beta_ols = regressao_ols(X_treino, y_treino)
y_pred_ols = prever(X_teste, beta_ols)

mse_ols = mse(y_teste, y_pred_ols)
r2_ols = r2(y_teste, y_pred_ols)

print(f"OLS - MSE: {mse_ols:.2f}, R²: {r2_ols:.4f}")

# Gradiente Descendente (GD)

def gradiente_descendente(X, y, alpha=0.01, epocas=1000):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    m, n = X_bias.shape
    beta = np.zeros(n)
    historico_mse = []

    for _ in range(epocas):
        y_pred = X_bias @ beta
        erro = y_pred - y
        gradiente = (2 / m) * (X_bias.T @ erro)
        beta -= alpha * gradiente
        historico_mse.append(mse(y, y_pred))

    return beta, historico_mse

# Normalizações
def min_max_normalizar(X):
    min_v = X.min(axis=0)
    max_v = X.max(axis=0)
    return (X - min_v) / (max_v - min_v + 1e-8), min_v, max_v

def zscore_normalizar(y):
    media = y.mean()
    desvio = y.std()
    return (y - media) / (desvio + 1e-8), media, desvio

X_treino_norm, X_min, X_max = min_max_normalizar(X_treino)
y_treino_norm, y_media, y_std = zscore_normalizar(y_treino)

X_teste_norm = (X_teste - X_min) / (X_max - X_min + 1e-8)

# GD com normalização

beta_gd, historico_gd = gradiente_descendente(X_treino_norm, y_treino_norm, alpha=0.1, epocas=1000)

y_pred_norm = prever(X_teste_norm, beta_gd)
y_pred_gd = y_pred_norm * y_std + y_media

mse_gd = mse(y_teste, y_pred_gd)
r2_gd = r2(y_teste, y_pred_gd)

print(f"GD - MSE: {mse_gd:.2f}, R²: {r2_gd:.4f}")

plt.plot(historico_gd)
plt.xlabel("Época")
plt.ylabel("MSE (treino)")
plt.title("Curva de Aprendizagem - GD")
plt.show()


# SGD (Gradiente Descendente Estocástico)

def sgd(X, y, alpha=0.01, epocas=1000):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    m, n = X_bias.shape
    beta = np.zeros(n)
    historico_mse = []

    for _ in range(epocas):
        idx = np.random.randint(m)
        Xi = X_bias[idx:idx+1]
        yi = y[idx:idx+1]
        erro = (Xi @ beta) - yi
        gradiente = 2 * Xi.T @ erro
        beta -= alpha * gradiente.flatten()
        y_pred = X_bias @ beta
        historico_mse.append(mse(y, y_pred))

    return beta, historico_mse

beta_sgd, historico_sgd = sgd(X_treino_norm, y_treino_norm, alpha=0.1, epocas=1000)

y_pred_sgd_norm = prever(X_teste_norm, beta_sgd)
y_pred_sgd = y_pred_sgd_norm * y_std + y_media

mse_sgd = mse(y_teste, y_pred_sgd)
r2_sgd = r2(y_teste, y_pred_sgd)

print(f"SGD - MSE: {mse_sgd:.2f}, R²: {r2_sgd:.4f}")

plt.plot(historico_sgd, label="SGD")
plt.plot(historico_gd, label="GD")
plt.xlabel("Época")
plt.ylabel("MSE (treino)")
plt.legend()
plt.title("Curva de Aprendizagem - GD vs SGD")
plt.show()


# treinos múltiplos e comparação


def treinar_varias_vezes(modelo_func, X, y, X_teste, y_teste, n=10):
    mses = []
    r2s = []
    for _ in range(n):
        beta, _ = modelo_func(X, y, alpha=0.1, epocas=1000)
        y_pred = prever(X_teste, beta)
        y_pred = y_pred * y_std + y_media
        mses.append(mse(y_teste, y_pred))
        r2s.append(r2(y_teste, y_pred))
    return mses, r2s

mses_gd, r2s_gd = treinar_varias_vezes(gradiente_descendente, X_treino_norm, y_treino_norm, X_teste_norm, y_teste)
mses_sgd, r2s_sgd = treinar_varias_vezes(sgd, X_treino_norm, y_treino_norm, X_teste_norm, y_teste)

plt.plot(mses_gd, label="MSE GD")
plt.plot(mses_sgd, label="MSE SGD")
plt.axhline(mse_ols, color='black', linestyle='--', label="MSE OLS")
plt.xlabel("Execução")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE em múltiplas execuções")
plt.show()

plt.plot(r2s_gd, label="R² GD")
plt.plot(r2s_sgd, label="R² SGD")
plt.axhline(r2_ols, color='black', linestyle='--', label="R² OLS")
plt.xlabel("Execução")
plt.ylabel("R²")
plt.legend()
plt.title("R² em múltiplas execuções")
plt.show()

print("Os valores variam devido à sensibilidade à inicialização e à ordem dos dados (SGD).")
print("Uma sugestão: usar taxa de aprendizado adaptativa ou momento para maior estabilidade.")
