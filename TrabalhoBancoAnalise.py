# Bibliotecas usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

def detectar_outliers_iqr(df, coluna):
  Q1 = df[coluna].quantile(0.25)
  Q3 = df[coluna].quantile(0.75)
  IQR = Q3 - Q1
  limite_inferior = Q1 - 1.5 * IQR
  limite_superior = Q3 + 1.5 * IQR
  outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
  return outliers

def shapiro(amostra):
    stat, p = stats.shapiro(amostra)
    print(f"Shapiro-Wilk: Estatística={stat:.4f}, p-value={p:.4f}")
    print("Os dados são normais?", "Sim" if p > 0.05 else "Não")

def quiQuadrada(contigencia):
  # Criar a tabela de contingência
  print("Tabela de Contingência:\n", contingencia)

  # Aplicar o teste qui-quadrado
  chi2, p, dof, expected = stats.chi2_contingency(contingencia)

  print(f"\nResultado do Teste Qui-Quadrado:")
  print(f"Chi2: {chi2}")
  print(f"P-valor: {p}")
  print(f"Grau de liberdade: {dof}")
  print(f"Frequências esperadas:\n{expected}")

  # Interpretar
  alpha = 0.05  # nível de significância de 5%

  if p < alpha:
      print("\nExiste uma associação significativa entre Geography e Exited.")
  else:
      print("\nNão há evidência de associação significativa entre Geography e Exited.")

pd.set_option('display.max_columns', None)
filename = "/content/drive/MyDrive/Analise/06_rotatividade_clientes_bancários.csv"


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

null_counts = df.isnull().sum()
tabela_nulos = pd.DataFrame({'Variável': null_counts.index,
                              'Quantidade de Nulos': null_counts.values})

print('\n\033[94mQuantidade de Dados Nulos por Variável:\033[0m\n')
print(tabela_nulos.to_string(index=False))

# Verificando valores ausentes e mostrando as linhas com valores ausentes

print('\n\033[94mLinhas com valores nulos:\033[0m\n')

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
  outliers = detectar_outliers_iqr(df, col)
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









# !!!!!!Análise Estatística!!!!!!
medianas = df.median(numeric_only=True)
print("\nMedianas das variáveis numéricas:\n")
print(medianas)

# Percentual de clientes presentes no banco (Variavel principal)
percentual = df['Exited'].value_counts(normalize=True) * 100
print("\nDistribuição percentual da variável Exited:\n")
print(percentual)

# Criando faixas etárias
bins = [18, 30, 45, 60, 100]
labels = ['18-29', '30-44', '45-59', '60+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Taxa de churn por faixa etária
churn_rate_age = df.groupby('Age_Group')['Exited'].mean().reset_index()
print("\nTaxa de Churn por Faixa Etária:\n", churn_rate_age)

# Correlação entre Idade e Exited
correlation_age_exited = df['Age'].corr(df['Exited'])
print(f"\nCorrelação entre Idade e Exited: {correlation_age_exited:.3f}")


# Gerando os histogramas e boxplots com seaborn
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited']

#Histogramas
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Histograma - {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Boxplots
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot - {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Heatmap
plt.figure(figsize=(10, 8))
correlation = df[numeric_cols].corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Mapa de Correlação')
plt.show()

#Analisando variavel principal
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df, palette='pastel')
plt.title('Distribuição da Variável Alvo - Exited')
plt.xlabel('Exited (0 = Permaneceu, 1 = Saiu)')
plt.ylabel('Quantidade')
plt.grid(True)
plt.show()

# Análise Bivariada: Variáveis Numéricas vs. Exited
print("\n\033[94mAnálise Bivariada: Variáveis Numéricas vs Exited\033[0m\n")
numeric_cols_no_exited = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for col in numeric_cols_no_exited:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Exited', y=col, data=df, palette='viridis')
    plt.title(f'Boxplot de {col} por Exited')
    plt.xlabel('Exited (0 = Permaneceu, 1 = Saiu)')
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

# Análise Bivariada: Variáveis de categoria vs. Exited
print("\n\033[94mAnálise Bivariada: Variáveis de Categoria vs Exited\033[0m\n")
categorical_cols = ['Geography', 'Gender']
for col in categorical_cols:
    # Gráfico de barras empilhadas (contagem)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, hue='Exited', data=df, palette='pastel')
    plt.title(f'Contagem de Exited por {col}')
    plt.xlabel(col)
    plt.ylabel('Contagem')
    plt.legend(title='Exited', labels=['Permaneceu', 'Saiu'])
    plt.grid(axis='y')
    plt.show()

    # Gráfico de barras de proporção
    churn_rate = df.groupby(col)['Exited'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=col, y='Exited', data=churn_rate, palette='rocket')
    plt.title(f'Taxa de Churn por {col}')
    plt.xlabel(col)
    plt.ylabel('Taxa de Churn')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()


# !!!!!!!!!!!!!!!HIPOTESES!!!!!!!!!!!!!!!!!
# 1- Clientes na faixa dos 49-60 anos tem maior probabilidade de sair do banco.

# --- 1 Histograma mostrando a distribuição das idades ---
#Separando a faixa etaria
bins = [0, 30, 40, 49, 60, 70, 100]
labels = ['<30', '30-39', '40-48', '49-60', '61-70', '71+']
df['FaixaEtariaAgrupada'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

churn_por_faixa = df.groupby('FaixaEtariaAgrupada')['Exited'].mean().reset_index()

#Criando o histograma
plt.figure(figsize=(10, 6))
sns.barplot(x='FaixaEtariaAgrupada', y='Exited', data=churn_por_faixa, palette='rocket')
plt.title('Taxa de Churn por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Taxa de Churn (Proporção de Clientes que Saíram)')
plt.ylim(0, 0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 2. Teste de hipótese, porcentagem de clientes que sairam de cada gupo etario ---
churn_por_faixa = df.groupby('FaixaEtariaAgrupada')['Exited'].mean().reset_index()

print("Porcentagem de Churn por Faixa Etária:")
print(churn_por_faixa)

# ---3 Teste Qui Quadrado, para ver se existe associação entre a faixa etária e o churn ---
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Valor Chi-quadrado: {chi2:.4f}")
print(f"Valor p: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Existe uma associação estatisticamente significativa entre a faixa etária e o churn.")
else:
    print("Não há uma associação estatisticamente significativa entre a faixa etária e o churn.")

# 2- Clientes que residem na alemanha tem maior probabilidade de sair do banco.

paises = df['Geography'].unique()
num_paises = len(paises)
cols = 2
rows = (num_paises + 1) // cols

plt.figure(figsize=(cols * 6, rows * 6))
for i, pais in enumerate(paises, 1):
    df_pais = df[df['Geography'] == pais]
    counts = df_pais['Exited'].value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=0)
    labels = ['Ativo', 'Inativo']
    explode = [0, 0.1]  # destaca inativos

    plt.subplot(rows, cols, i)
    plt.pie(counts, labels=labels, autopct='%1.1f%%',
            colors=sns.color_palette('pastel')[:2],
            explode=explode, shadow=True)
    plt.title(f'Clientes Ativos vs Inativos - {pais}')
plt.tight_layout()
plt.show()

# --- 2. Cálculo da taxa de churn por país ---

total_por_pais = df['Geography'].value_counts()
churn_por_pais = df[df['Exited'] == 1]['Geography'].value_counts()
taxa_churn = churn_por_pais / total_por_pais

print("Taxa de churn por país:")
print(taxa_churn)

# --- 3. Visualização comparativa da taxa de churn ---

taxa_churn_sorted = taxa_churn.sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=taxa_churn_sorted.index, y=taxa_churn_sorted.values, palette='pastel')
plt.ylabel('Taxa de Churn')
plt.title('Taxa de Churn por País')
plt.ylim(0, taxa_churn_sorted.max() + 0.05)
plt.show()

# --- 4. Teste de hipótese:

contingencia = pd.crosstab(df['Geography'], df['Exited'])
quiQuadrada(contingencia)


# 3- Pessoas inativas têm uma chance maior de sair

#Histograma mostrando que mais de 25% dos inativos sairam e menos de 15% dos ativos sairam
plt.figure(figsize=(8, 6))
sns.barplot(x='IsActiveMember', y='Exited', data=churn_rate_by_activity, palette='viridis')
plt.title('Taxa de Rotatividade por Status de Atividade do Cliente')
plt.xlabel('Status de Atividade do Cliente')
plt.ylabel('Taxa de Rotatividade (Proporção de Exited = 1)')
plt.ylim(0, churn_rate_by_activity['Exited'].max() * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#TESTE Z PARA PROPORÇÕES
churn_counts = df.groupby(['IsActiveMember', 'Exited']).size().unstack(fill_value=0)

inactive_exited = churn_counts.loc[0, 1]
inactive_total = churn_counts.loc[0, 0] + churn_counts.loc[0, 1]

active_exited = churn_counts.loc[1, 1]
active_total = churn_counts.loc[1, 0] + churn_counts.loc[1, 1]

count = np.array([inactive_exited, active_exited])
nobs = np.array([inactive_total, active_total])

z_statistic, p_value = proportions_ztest(count, nobs)

print("=== Resultado do Teste de Hipótese ===")
print(f"Z-Statistic: {z_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"\nCom um p-value ({p_value:.4f}) menor que o nível de significância ({alpha}), rejeitamos a hipótese nula.")
    print("Há evidências estatísticas significativas de que a proporção de clientes que saem do banco é diferente entre clientes inativos e ativos.")
    print("Especificamente, parece que clientes inativos têm uma tendência maior a sair do banco.")
else:
    print(f"\nCom um p-value ({p_value:.4f}) maior que o nível de significância ({alpha}), não rejeitamos a hipótese nula.")
    print("Não há evidências estatísticas significativas para afirmar que a proporção de clientes que saem do banco é diferente entre clientes inativos e ativos.")


churn_rate_by_activity = df.groupby('IsActiveMember')['Exited'].mean().reset_index()

churn_rate_by_activity['IsActiveMember'] = churn_rate_by_activity['IsActiveMember'].map({0: 'Inativo', 1: 'Ativo'})
#!!!!!!!!!!!!!!!!!!!!!!!!!Análise Preditiva!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X_simple = df[['Age']]
y = df['Exited']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_s = model_simple.predict(X_test_s)

rmse_simple = sqrt(mean_squared_error(y_test_s, y_pred_s))
print(f"RMSE da Regressão Simples (Age → Exited): {rmse_simple:.4f}")

print(f"Intercepto: {model_simple.intercept_}")
print(f"Coeficiente de Age: {model_simple.coef_[0]}")




# ITS A BEAUTIFUL DAY FOR PIE

#!!!!!!!!!!!!!REGRESSAO LINEAR MULTIPLA!!!!!!!!!!!!!




# PARA VER OS GRAFICOS, ABRA A ABA VNC DO REPLIT