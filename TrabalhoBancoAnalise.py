#Bibliotecas usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = "06_rotatividade_clientes_bancários.csv"
df = pd.read_csv(filename)

#Analise inicial tecnica da base

df.info()

#Verificando valores ausentes e mostrando as linhas com valores ausentes

print(df[df.isnull().any(axis=1)])

#Este comando da dropb em todas as linhas com valores ausentes

df = df.dropna()

#Resolvemos dar drop em todos os valores ausentes para não prejudicar a analise, ja que são poucas linhas e os dados são importantes

#Gerando os histogramas e boxplots com seaborn
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

#PARA VER OS GRAFICOS, ABRA A ABA VNC DO REPLIT