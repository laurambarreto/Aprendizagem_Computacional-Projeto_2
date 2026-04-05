# Import das bibliotecas necessárias
import time
import optuna
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
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

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_01.csv', delimiter = ",")


# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

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

# Verificar se são linearmente separáveis
# Normalizar os dados
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(X_norm)

# Define um array de cores fixas: ex. vermelho, verde, azul
colors = {0: '#ffc0dc', 1: '#ffff00'}

# Mapeia as cores com base nas classes
maped_colors = [colors[classe] for classe in y]

# Cria os elementos da legenda manualmente
legenda_cores = [Patch(color = cor, label = f'Class {classe}') for classe, cor in colors.items()]
 
# Visualizar os dados reduzidos em 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = maped_colors, alpha = 0.5)
plt.title("Real classes (0 vs. 1)", fontsize = 20)
plt.xlabel("1st component", fontsize = 14)
plt.ylabel("2nd component", fontsize = 14)
plt.legend(handles = legenda_cores, fontsize = 12, title_fontsize = 13)
plt.show()

# Variância explicada pelas duas e três primeiras componentes
var_2 = sum(pca.explained_variance_ratio_[:2])
print(f"Variance explained by the first two components: {var_2:.4f}")

##---------- Pré-processamento ----------##
# Verificar dados nulos (NÃO HÁ NENHUM DADO A FALTAR)
print("Missing data per column:")
print(df.isnull().sum(), "\n")

# Verificar duplicatas completas (linhas idênticas)
duplicated = df[df.duplicated(keep = False)]  # `keep = False` marca todas as ocorrências
print(f"Number of duplicated lines: {len(duplicated)}") 

# Agrupa linhas idênticas e conta ocorrências
count_duplicated = df.groupby(df.columns.tolist()).size().reset_index(name = 'Count')

# Mostra as linhas repetidas
print(count_duplicated.sort_values('Count', ascending = False))

# Remover duplicados
df = df.drop_duplicates()
df_l0 = df[df['Diabetes_binary'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_binary'] == 1]  # classe minoritária

# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis = 1)
df_l0_clean = df_l0[cond]
# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1], axis = 0)

X = df.drop("Diabetes_binary", axis = 1)
y = df.Diabetes_binary

# EM 3D DEPOIS DE TIRAR OUTLIERS E DUPLICADOS
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 3)
X_reduced = pca.fit_transform(X_norm)

# Define um array de cores fixas: ex. vermelho, verde, azul
colors = {0: '#ffc0dc', 1: '#ffff00'}

# Mapeia as cores com base nas classes
maped_colors = [colors[classe] for classe in y]

# Cria os elementos da legenda manualmente
legend_colors = [Patch(color = cor, label = f'Class {classe}') for classe, cor in colors.items()]

# Criar figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = maped_colors, alpha=0.6)

# Títulos e legendas
ax.set_title("Real classes (0 vs. 1) after removing outliers and duplicates", fontsize=18)
ax.set_xlabel("1st component", fontsize=12)
ax.set_ylabel("2nd component", fontsize=12)
ax.set_zlabel("3rd component", fontsize=12)
ax.legend(handles=legend_colors, fontsize=10)
plt.show()

# Variância explicada pelas duas e três primeiras componentes
var_2 = sum(pca.explained_variance_ratio_[:3])
print(f"Variance explained by the first three components: {var_2:.4f}")

# Distribuição de diabetes e não diabetes nos dados de treino antes da remo
ax = sns.countplot(x = y, color = '#73D7FF')
plt.title("Diabetes binary distribution before", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 160000)
plt.show()


# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Distribuição das classes nos dados de treino no final da remoção de outliers e linhas duplicadas
ax = sns.countplot(x = y_train, color = '#73D7FF')
plt.title("Diabetes binary distribution after")

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()
print(y_train.value_counts(), "\n")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar SMOTE aos dados de treino
smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_scaled, y_train)

# Distribuição das classes nos dados de treino depois de usar SMOTE 
ax = sns.countplot(x = y_train_SMOTE, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with SMOTE)", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()

print(y_train_SMOTE.value_counts(), "\n")

##---------- MODELIZAÇÃO ----------##
#---------- REDES NEURONAIS ----------#
# Teste com os hiperparâmetros que trouxeram melhores resultados
kernel = 'poly'
C = 21.60310384162283
degree = 3
gamma = 0.35156928389335196
svm = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)

# Treinar
svm.fit(X_train_SMOTE, y_train_SMOTE)

# Previsões no teste
y_pred_svm = svm.predict(X_test_scaled)

# Avaliação final do modelo
print("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Recall: %.2f' % recall_score(y_test, y_pred_svm))
print('Precision: %.2f' % precision_score(y_test, y_pred_svm))
print('F1: %.2f' % f1_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using SVM')
plt.show()

# Testes com os hiperparâmetros
def objective(trial):
    # Parâmetros a testar
    hidden_layer_1 = trial.suggest_int('hidden_layer_1', 5, 100) # nº de neuronios na 1ª camada
    hidden_layer_2 = trial.suggest_int('hidden_layer_2', 0, 100)  # nº de neuronios na 2ª camada
    activation = trial.suggest_categorical('activation', ['logistic', 'tanh', 'relu'])
    solver = trial.suggest_categorical('solver', ['adam'])
    alpha = trial.suggest_float('alpha', 1e-4, 1e-1, log=True) # termos de regularização
    learning_rate_init = trial.suggest_float('learning_rate_init', 5e-3, 1e-1, log=True)


    # Estrutura da rede
    if hidden_layer_2 == 0:
        hidden_layers = (hidden_layer_1,)
    else:
      hidden_layers = (hidden_layer_1, hidden_layer_2)
    # Criar o classificador
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
    )

    # Treinar e prever
    mlp.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = mlp.predict(X_test_scaled)

    # Métrica a maximizar
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"[TRIAL] alpha={alpha}, learning_rate={learning_rate_init}, F1={macro_f1}")
    return macro_f1


# === GUARDAR ESTUDO NUM FICHEIRO SQLITE === #
storage = "sqlite:///binary_mlp_study2_pc1.db"
study = optuna.create_study(
    direction="maximize",
    study_name="mlp_study4",
    storage=storage,
    load_if_exists=True
)

# === OTIMIZAR ===
study.optimize(objective, n_trials=50)

# === RESULTADOS ===
print("\n🔍 Melhores parâmetros encontrados:")
print(study.best_params)
print(f"🎯 Melhor F1-score: {study.best_value:.4f}")


# Teste com os hiperparâmetros que trouxeram melhores resultados
for i in range (10):
    hidden_layer_1 = 61 # nº de neuronios na 1ª camada
    hidden_layer_2 = 31  # nº de neuronios na 2ª camada
    activation = 'logistic'
    solver = 'adam'
    alpha = 0.01814681810781384 # termo de regularização
    learning_rate_init = 0.0029843831320328232
    hidden_layers = (hidden_layer_1, hidden_layer_2)

    # Criar o classificador
    mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    activation=activation,
    solver=solver,
    alpha=alpha,
    learning_rate_init=learning_rate_init,
    max_iter=1000
    )

    # Treinar e prever
    mlp.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred_mlp = mlp.predict(X_test_scaled)

    # Avaliar o classifier
    # Macro-Average (igual peso para todas classes)
    macro_precision = precision_score(y_test, y_pred_mlp, average = 'macro')
    macro_recall = recall_score(y_test, y_pred_mlp, average = 'macro')
    macro_f1 = f1_score(y_test, y_pred_mlp, average = 'macro')

    print ("MLP CLASSIFIER RESULTS - ", i + 1)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}\n")
    print(classification_report(y_test, y_pred_mlp))

    # Matriz de confusão do classificador MLP
    cm = confusion_matrix(y_test, y_pred_mlp)
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix - Using MLP classifier')
    plt.show()

# Valores das métricas durante as 10 execuções
accuracy = [0.9200, 0.9200, 0.9200, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9200]
recall = [0.8991, 0.8988, 0.8990, 0.8998, 0.9009, 0.9005, 0.8991, 0.9003, 0.9010, 0.9007]
precision = [0.9356, 0.9318, 0.9229, 0.9369, 0.9355, 0.9385, 0.9440, 0.9361, 0.9426, 0.9336]

# Criar o boxplot
plt.boxplot([accuracy, recall, precision], labels=['Accuracy', 'Recall', 'Precision'],patch_artist=True,
            boxprops = dict(facecolor='#73D7FF', edgecolor='#73D7FF'),)
plt.title('MLP Binary Metrics Boxplot')
plt.ylabel('Score')
plt.show()

#---------- SVM ----------# 
def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    C = trial.suggest_float('C', 1e-1, 60, log=True)

    if kernel == 'rbf':
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)

    elif kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 3)
        coef0 = trial.suggest_float('coef0', 0.0, 0.5)
        svm = SVC(kernel='poly', C=C, degree=degree, coef0=coef0, random_state=42)

    else:  # linear
        svm = SVC(kernel='linear', C=C, random_state=42)

    # Treinar
    svm.fit(X_train_SMOTE, y_train_SMOTE)

    # Prever
    y_pred = svm.predict(X_test_scaled)

    # Métrica a maximizar
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"[TRIAL] kernel={kernel}, C={C:.5f}, F1={f1:.4f}")
    return f1

storage = "sqlite:///binary_SVM_study_pc1.db"
study = optuna.create_study(
    direction="maximize",
    study_name="SVM_study3",
    storage=storage,
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

# Resultados
print("\n🔍 Melhores parâmetros encontrados:")
print(study.best_params)
print(f"🎯 Melhor F1-score: {study.best_value:.4f}")

# Treinar o modelo final com melhores parâmetros em treino + validação
best_params = study.best_params

if best_params['kernel'] == 'poly':
    svm = SVC(kernel=best_params['kernel'], C=best_params['C'], degree=best_params['degree'], max_iter=best_params['max_iter'], random_state=42)
else:
    svm = SVC(kernel=best_params['kernel'], C=best_params['C'], max_iter=best_params['max_iter'], random_state=42)

start_time = time.time()
svm.fit(X_train_SMOTE, y_train_SMOTE)
time_total = time.time() - start_time
print(f"Total training time: {time_total:.2f} seconds")

# Previsões no teste
y_pred_svm = svm.predict(X_test_scaled)

# Avaliação final do modelo
print("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Recall: %.2f' % recall_score(y_test, y_pred_svm))
print('Precision: %.2f' % precision_score(y_test, y_pred_svm))
print('F1: %.2f' % f1_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using SVM')
plt.show()


#---------- REGRESSÃO LOGÍSTICA ----------#
# Inicializar o modelo
model = LogisticRegression()

# Medir o tempo de treino
start_time = time.time()
# Treinar o modelo
model.fit(X_train_SMOTE, y_train_SMOTE)
training_time = time.time() - start_time

# Fazer previsões com os dados de teste
y_test_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_SMOTE)

# Matriz de confusão
conf_matrix = confusion_matrix (y_test, y_test_pred)
ax = sns.heatmap(conf_matrix, annot = True, cmap = 'Blues',fmt = "g", annot_kws = {"size": 18})
ax.set_title('Confusion Matrix with LR (SMOTE)', fontsize = 20)
ax.set_xlabel('\nPredicted Values', fontsize = 14)
ax.set_ylabel('Actual Values ', fontsize = 14)
plt.show()

# Analisar as métricas 
accuracy_train, recall_train, precision_train = metricas(y_train_pred, y_train_SMOTE)   
accuracy_test, recall_test, precision_test = metricas(y_test_pred, y_test)

print("Logistic Regression (SMOTE):")
print('Accuracy: %.2f' % accuracy_score(y_test, y_test_pred))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(classification_report(y_test, y_test_pred))


