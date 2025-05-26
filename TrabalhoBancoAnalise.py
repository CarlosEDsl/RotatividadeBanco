# Bibliotecas usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import AuxFunctions as aux
from scipy import stats
from scipy.stats import chi2_contingency

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

# 1️⃣ Criar a tabela de contingência
contingencia = pd.crosstab(df['Geography'], df['Exited'])

print("Tabela de Contingência:\n", contingencia)

# 2️⃣ Aplicar o teste qui-quadrado
chi2, p, dof, expected = chi2_contingency(contingencia)

print(f"\nResultado do Teste Qui-Quadrado:")
print(f"Chi2: {chi2}")
print(f"P-valor: {p}")
print(f"Grau de liberdade: {dof}")
print(f"Frequências esperadas:\n{expected}")

# 3️⃣ Interpretar
alpha = 0.05  # nível de significância de 5%

if p < alpha:
    print("\n✅ Existe uma associação significativa entre Geography e Exited.")
else:
    print("\n❌ Não há evidência de associação significativa entre Geography e Exited.")

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
# 2- Clientes que residem na alemanha tem maior probabilidade de sair do banco.
# Conta as categorias apenas onde Exited == 1
counts = df[df['Exited'] == 1]['Geography'].value_counts()

# Garante que o explode tenha o tamanho correto
explode = [0.1] + [0] * (len(counts) - 1)  # Só destaca o primeiro setor

# Cria o gráfico
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 2)
plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=sns.color_palette('pastel')[:len(counts)],
        explode=explode, shadow=True)
plt.title('Gráfico de Pizza - Distribuição por país (Exited = 1)')
plt.tight_layout()
plt.show()

# 3- Pessoas inativas têm uma chance maior de sair


# ITS A BEAUTIFUL DAY FOR PIE





# PARA VER OS GRAFICOS, ABRA A ABA VNC DO REPLIT

print("\n\033[92m✅ Análise concluída! A aplicação continuará rodando...\033[0m")
print("\033[93m📊 Para visualizar os gráficos, abra a aba VNC do Replit.\033[0m")

# Mantém a aplicação rodando para evitar o recovery mode
import time
try:
    while True:
        time.sleep(60)  # Aguarda 60 segundos antes de verificar novamente
        print(f"\033[96m🔄 Aplicação ainda rodando... {pd.Timestamp.now().strftime('%H:%M:%S')}\033[0m")
except KeyboardInterrupt:
    print("\n\033[91m⏹️  Aplicação finalizada pelo usuário.\033[0m")
