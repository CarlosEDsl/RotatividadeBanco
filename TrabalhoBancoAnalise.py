#Bibliotecas usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = "06_rotatividade_clientes_bancários.csv"
df = pd.read_csv(filename)

#Analise inicial tecnica da base

df.info()

#Verificando valores ausentes

df.isnull().sum()
nulos = df.isnull().any(axis=1)
