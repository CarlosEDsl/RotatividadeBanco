# Bibliotecas usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import AuxFunctions as aux

pd.set_option('display.max_columns', None)
filename = "06_rotatividade_clientes_bancários.csv"


df = pd.read_csv(filename, header=0)
print('\033[95m\nPrimeiros dados do dataset:\033[0m\n')
print(df.head())

# Analise inicial tecnica da base

print('\033[94m\nInformações sobre o dataset:\033[0m\n')

df.info()

print(f'\033[92m\nO dataset possui {df.shape[0]} entradas (registros) e {df.shape[1]} atributos (variáveis).\033[0m\n')

# Tipos de dados dos atributos
print(f'\033[93m\nOs tipos dos atributos são do tipo:\n{df.dtypes}\033[0m')

# Verificando se há dados ausentes
missing_data = ((df.isnull().sum() / df.shape[0]) * 100).sort_values(ascending=False)
print(f'\033[91m\nDados faltando no dataset em porcentagem: \n{missing_data}\033[0m')

# Verificando valores ausentes e mostrando as linhas com valores ausentes

print(df[df.isnull().any(axis=1)])

# Este comando da dropb em todas as linhas com valores ausentes

df = df.dropna()

# Resolvemos dar drop em todos os valores ausentes para não prejudicar a analise, ja que são poucas linhas e os dados são importantes

df.hist(bins=15, figsize=(20, 16));

# Primeiras estatísticas descritivas

print('\033[95m\nPrimeiras estatísticas descritivas:\033[0m\n')
print(df.describe())

# Verificando valores zeros (Que podem indicar ausencia de alguns dados de medição)
print(f"\033[96m\n\nVerificando valores zeros (Que podem indicar ausencia de alguns dados de medição)\033[0m")

cols_with_zeros = ['CreditScore', 'Age']
for col in cols_with_zeros:
    print(f"\033[93m{col} - Total de zeros: { (df[col] == 0).sum() }\033[0m")

print(f"\033[96m\n\nVerificando valores booleanos se estão integros (0 ou 1)\033[0m")
boolean_cols = ['HasCrCard', 'IsActiveMember', 'Exited']
for col in boolean_cols:
  print(f"\033[93m{col} - Diferente de 1 e 0: {((df[col] != 0) & (df[col] != 1)).sum()}\033[0m")
    
print(f"\033[96m\n\nVerificando Valores específicos\033[0m")
s_cols = ['Geography', 'Gender']
for col in s_cols:
    print(f"\033[93m{col} - {df[col].unique()}\033[0m")

print(f"\033[92m\nComo não encontramos nenhuma coluna com ausencia de dados não precisamos limpar essas colunas\n\033[0m")
print(df.head())

colunas_outliers = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
print(f"\033[96m\n\nVerificando outliers nas colunas {colunas_outliers}\033[0m")
for col in colunas_outliers:
  outliers = aux.detectar_outliers_iqr(df, col)
  tabela_outliers = pd.DataFrame({
      'ID': outliers.index,
      col: outliers[col]
  })

  print(f"\033[93m\n===== Outliers na coluna {col} (Primeiros 10) =====\033[0m")
  print(tabela_outliers.head(10).to_string(index=False))
  print(f"\033[93mTotal de outliers: {len(outliers)}\033[0m")

print(f"\033[92m\n Encontramos alguns outliers nas idades, contudo são idades realistas, alguns valores do creditScore estão baixos, contudo ainda são realistas\n\033[0m")


print(f"\033[91m\n Derrubando linhas duplicadas, total: {df.duplicated().sum()}\n\033[0m")
df = df.drop_duplicates()

# Gerando os histogramas e boxplots com seaborn
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited']

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Histograma - {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot - {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# ITS A BEAUTIFUL DAY FOR PIE

print("\n\033[92m✅ Análise concluída com sucesso!\033[0m")
print("\033[93m📊 Para visualizar os gráficos, abra a aba VNC do Replit.\033[0m")
print("\033[96m🔄 Mantendo aplicação ativa...\033[0m")

# PARA VER OS GRAFICOS, ABRA A ABA VNC DO REPLIT

# Mantém a aplicação rodando para evitar recovery mode
import time
try:
    while True:
        time.sleep(30)  # Aguarda 30 segundos
        print(f"\033[90m⏰ Aplicação rodando... {pd.Timestamp.now().strftime('%H:%M:%S')}\033[0m")
except KeyboardInterrupt:
    print("\n\033[91m⏹️ Aplicação finalizada.\033[0m")
