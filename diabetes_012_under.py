# Import das bibliotecas necessárias
import time
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_012.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("Diabetes_012", axis = 1)

# Seleção da coluna target
y = df.Diabetes_012

# Função que retorna as métricas de avaliação
def metricas(y_pred, y_true):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- Análise inicial ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Correlações entre todas as colunas 
correlation_matrix = df.corr()
plt.figure(figsize = (6, 4))
sns.heatmap(correlation_matrix,cmap = 'coolwarm', annot = False)
plt.title('Correlation Matrix Heatmap')
plt.xticks(ticks = np.arange(len(df.columns)) + 0.5, labels = df.columns, rotation = 45, ha = 'right', fontsize = 8)
plt.yticks(ticks = np.arange(len(df.columns)) + 0.5, labels = df.columns, rotation = 0, fontsize = 8)
plt.tight_layout()
plt.show()

# Distribuição de diabetes e não diabetes do dataset
ax = sns.countplot(x = y, color = '#73D7FF')
plt.title("Diabetes distribution before")

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

print(df['Diabetes_012'].value_counts(), "\n")


##---------- Pré-processamento ----------##
# Verificar dados nulos (NÃO HÁ NENHUM DADO A FALTAR)
print("Missing data per column:")
print(df.isnull().sum(), "\n")

# Verificar réplicas completas (linhas idênticas)
duplicated = df[df.duplicated(keep = False)]  # `keep = False` marca todas as ocorrências
print(f"Number of duplicated lines: {len(duplicated)}") 

# Agrupa linhas idênticas e conta ocorrências
count_duplicated = df.groupby(df.columns.tolist()).size().reset_index(name = 'Count')

# Mostra as linhas repetidas
print(count_duplicated.sort_values('Count', ascending = False))

# Remover duplicados
df = df.drop_duplicates()
print("Distribution after removing duplicates:")
print(df['Diabetes_012'].value_counts(), "\n")

df_l0 = df[df['Diabetes_012'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_012'] == 1]  # classe minoritária
df_l2 = df[df['Diabetes_012'] == 2]  # classe minoritária

# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis=1)
df_l0_clean = df_l0[cond]

# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1,df_l2], axis = 0)

# Seleção das colunas das características
X = df.drop("Diabetes_012", axis = 1)

# Seleção da coluna target
y = df.Diabetes_012

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Distribuição de diabetes e não diabetes nos dados de treino depois da remoção de outliers e linhas duplicadas
ax = sns.countplot(x = y_train, color = '#73D7FF')
plt.title("Diabetes multiclass distribution after", fontsize = 20)
plt.xlabel("Diabetes 012", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()


print(y_train.value_counts(), "\n")

# Reduzir o número de exemplos da classe dominante (undersampling) nos dados de treino
undersampler = RandomUnderSampler(sampling_strategy = 'auto', random_state = 42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino depois do SMOTE 
ax = sns.countplot(x = y_train_under, color = '#73D7FF')
plt.title("Diabetes multiclass distribution (Undersampling)", fontsize = 20)
plt.xlabel("Diabetes 012", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

print(y_train_under.value_counts(), "\n")