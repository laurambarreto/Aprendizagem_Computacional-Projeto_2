# Import das bibliotecas necessárias
import time
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report
import optuna

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_01.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Função que retorna as métricas de avaliação
def metricas(y_pred, y_true):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- Análise inicial ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Distribuição de diabetes e não diabetes do dataset
ax = sns.countplot(x = y, color = '#73D7FF') 
plt.title("Diabetes binary distribution before", fontsize = 20)
plt.xlabel("Diabetes_binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Aumentar o tamanho dos números dos eixos
plt.tick_params(axis = 'both', which = 'major', labelsize = 13)

# Definir os ticks do eixo Y de 25.000 em 25.000
max_val = y.value_counts().max()
plt.yticks(ticks=range(0, max_val + 25000, 25000))

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

print(df['Diabetes_binary'].value_counts(), "\n")

##---------- Pré-processamento ----------##
# Remover duplicados
df = df.drop_duplicates()
print("Distribution after removing duplicates:")
print(df['Diabetes_binary'].value_counts(), "\n")

df_l0 = df[df['Diabetes_binary'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_binary'] == 1]  # classe minoritária

# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis=1)
df_l0_clean = df_l0[cond]

# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1], axis=0)

# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Distribuição de diabetes e não diabetes nos dados de treino depois da remoção de outliers e linhas duplicadas
ax = sns.countplot(x = y, color = '#73D7FF')
plt.title("Diabetes binary distribution after", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

print(y_train.value_counts(), "\n")

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()

# Reduzir o número de exemplos da classe dominante (undersampling) nos dados de treino
undersampler = RandomUnderSampler(sampling_strategy = 'auto', random_state = 42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com undersampling
ax = sns.countplot(x = y_train_under, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with undersampling)", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

print(y_train_under.value_counts(), "\n")

