import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados do CSV
caminho = 'petr4.csv'
dados = pd.read_csv(caminho, sep=';', decimal=',')

# Converter a coluna 'DateTime' para datetime
dados['DateTime'] = pd.to_datetime(dados['DateTime'], format='%Y/%m/%d')

# Filtrar os dados dos últimos 3 meses
data_fim = dados['DateTime'].max()
data_inicio = data_fim - pd.DateOffset(months=3)
dados_filtrados = dados[(dados['DateTime'] >= data_inicio) & (dados['DateTime'] <= data_fim)]

# Preparar os dados
x = np.arange(len(dados_filtrados)).reshape(-1, 1)  # Usar os índices como valores de x
y = dados_filtrados['Series 1'].values

print(dados_filtrados.head())

def regressao_polinomial(x, y, grau):

    x = np.array(x).flatten()
    y = np.array(y)
    
    # Ajustar o polinômio usando numpy's polyfit
    coeficientes = np.polyfit(x, y, grau)
    
    return coeficientes

def plot_regressao_polinomial(x, y, coeficientes):
   
    plt.scatter(x, y, color='blue', label='Dados Originais')
    
    # Calcular os valores ajustados
    x_fit = np.linspace(min(x), max(x) + 15, 100)  # Estender para 15 dias além dos dados originais
    y_fit = np.polyval(coeficientes, x_fit)
    
    # Plotar a linha de regressão
    plt.plot(x_fit, y_fit, color='red', label='Regressão Polinomial (grau {})'.format(len(coeficientes)-1))
    
    plt.xlabel('DIAS')
    plt.ylabel('PREÇO EM R$')
    plt.title('Regressão Polinomial')
    plt.legend()
    plt.grid(True)
    plt.show()


grau_do_polinomio = 8 # Escolha o grau do polinômio desejado
coeficientes = regressao_polinomial(x, y, grau_do_polinomio)

# Printar os coeficientes
print(f'Coeficientes do polinômio de grau {grau_do_polinomio}:')
print(coeficientes)

plot_regressao_polinomial(x, y, coeficientes)

# Calcular os resíduos acumulados
y_pred = np.polyval(coeficientes, x)
residuos = y - y_pred
residuos_acumulados = np.cumsum(residuos)

# Plotar o gráfico dos resíduos acumulados
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuos_acumulados)), residuos_acumulados, marker='o', linestyle='-', color='green')
plt.title('Resíduos Acumulados da Regressão Polinomial')
plt.xlabel('Índice')
plt.ylabel('Resíduos Acumulados')
plt.grid(True)
plt.show()

#previsão do preço
dias_a_estender = 15
x_estendido = np.arange(np.max(x)+1, np.max(x)+1 + dias_a_estender).reshape(-1, 1)
y_estendido = np.polyval(coeficientes, x_estendido)

# Printar os valores previstos para os próximos 15 dias
print(f'Previsão de preço da ação para os próximos {dias_a_estender} dias:')
for i in range(dias_a_estender):
    print(f'Dia {i+1}: {y_estendido[i]}')
