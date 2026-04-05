# Import das bibliotecas necessárias
import time
import optuna
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_012.csv', delimiter = ",")

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
df = pd.concat([df_l0_clean, df_l1,df_l2], axis=0)

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

# Escolher o novo tamanho alvo para todas as classes
objective_len = 40000

# 1. Normalizar os dados ANTES de aplicar underSampling e SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  

# 2. Aplica UNDERSAMPLING à classe 0 (nos dados já normalizados)
under = RandomUnderSampler(sampling_strategy={0: objective_len}, random_state=42)
X_under, y_under = under.fit_resample(X_train_scaled, y_train)

# 3. Aplica SMOTE às classes 1 e 2 (nos dados normalizados e após RUS)
smote = BorderlineSMOTE(sampling_strategy={1: objective_len, 2: objective_len}, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_under, y_under)

# Juntar tudo num novo DataFrame
df_final = pd.concat([pd.DataFrame(X_train_bal, columns = X.columns),
                      pd.Series(y_train_bal, name = "Diabetes_012")], axis = 1)

# Distribuição de diabetes e não diabetes nos dados de treino depois do SMOTE 
ax = sns.countplot(x = y_train_bal, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with SMOTE + under)", fontsize = 20)
plt.xlabel("Diabetes 012", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

print(y_train_bal.value_counts(), "\n")

##---------- REDES NEURONAIS ----------##

# Teste com os hiperparâmetros que trouxeram melhores resultados
kernel = 'sigmoid'
C = 11.625528864773162
svm = SVC(kernel=kernel, C=C)

# Treinar
svm.fit(X_train_bal, y_train_bal)

# Previsões no teste
y_pred_svm = svm.predict(X_test_scaled)

# Avaliação final do modelo
print("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Recall: %.2f' % recall_score(y_test, y_pred_svm, average ="macro"))
print('Precision: %.2f' % precision_score(y_test, y_pred_svm, average ="macro"))
print('F1: %.2f' % f1_score(y_test, y_pred_svm,average ="macro"))
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


for i in range (10):
    mlp = MLPClassifier(
        hidden_layer_sizes=(80, 5),
        activation='tanh',
        solver='adam',
        alpha=0.02244792272294147,
        learning_rate_init=0.0776713732712586
    )

    mlp.fit(X_train_bal, y_train_bal)
    y_pred_mlp = mlp.predict(X_test_scaled)

    # Avaliar o classifier
    # Macro-Average (igual peso para todas classes)
    macro_precision = precision_score(y_test, y_pred_mlp, average = 'macro')
    macro_recall = recall_score(y_test, y_pred_mlp, average = 'macro')
    macro_f1 = f1_score(y_test, y_pred_mlp, average = 'macro')

    # Weighted-Average (ponderado pelo número de amostras)
    weighted_precision = precision_score(y_test, y_pred_mlp, average = 'weighted')
    weighted_recall = recall_score(y_test, y_pred_mlp, average = 'weighted')
    weighted_f1 = f1_score(y_test, y_pred_mlp, average = 'weighted')

    print ("MLP Classifier - ", i + 1)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}\n")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
    print(classification_report(y_test, y_pred_mlp))

    cm = confusion_matrix(y_test, y_pred_mlp)
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix - Using MLP classifier')
    plt.show()

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
    mlp.fit(X_train_bal, y_train_bal)
    y_pred = mlp.predict(X_test_scaled)

    # Métrica a maximizar
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"[TRIAL] alpha={alpha}, learning_rate={learning_rate_init}, F1={macro_f1}")
    return macro_f1

# === GUARDAR ESTUDO NUM FICHEIRO SQLITE === #
storage = "sqlite:///multi_mlp_study_pc1.db"
study = optuna.create_study(
    direction="maximize",
    study_name="mlp_study5",
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

mlp = MLPClassifier(
    hidden_layer_sizes=(78, 56),
    activation='logistic',
    solver='adam',
    alpha=0.002187564015159477,
    learning_rate_init=0.014724654480469402,
    max_iter=300,
    random_state=42
)
start_time = time.time()
mlp.fit(X_train_scaled, y_train_bal)
tempo_total = time.time() - start_time
print(f"Total training time: {tempo_total:.2f} seconds")

y_pred_mlp = mlp.predict(X_test)

# Avaliar o classifier
# Macro-Average (igual peso para todas classes)
macro_precision = precision_score(y_test, y_pred_mlp, average = 'macro')
macro_recall = recall_score(y_test, y_pred_mlp, average = 'macro')
macro_f1 = f1_score(y_test, y_pred_mlp, average = 'macro')

# Weighted-Average (ponderado pelo número de amostras)
weighted_precision = precision_score(y_test, y_pred_mlp, average = 'weighted')
weighted_recall = recall_score(y_test, y_pred_mlp, average = 'weighted')
weighted_f1 = f1_score(y_test, y_pred_mlp, average = 'weighted')

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
print(classification_report(y_test, y_pred_mlp))

cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using MLP classifier')
plt.show()

# Valores das métricas durante as 10 execuções
accuracy = [0.6100,0.5400,0.6200,0.6100,0.5200,0.7400,0.6200,0.5600,0.4800,0.6100]
recall = [0.5622,0.5294,0.5630,0.5611,0.5232,0.6047,0.5676,0.5418,0.5055,0.5691]
precision = [0.4460, 0.4272,0.4657,0.4458,0.4398,0.5657,0.4576,0.4274,0.4198,0.4876]

# Criar o boxplot
plt.boxplot([accuracy, recall, precision], labels=['Accuracy', 'Recall', 'Precision'],patch_artist=True,
            boxprops = dict(facecolor='#73D7FF', edgecolor='#73D7FF'),)
plt.title('MLP Binary Metrics Boxplot')
plt.ylabel('Score')
plt.show()

##---------- SVM ----------## 

def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    C = trial.suggest_float('C', 0.1, 100, log=True)

    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 3)
    else:
        degree = 3  # default

    if kernel in ['rbf', 'poly', 'sigmoid']:
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    else:
        gamma = 'scale'  # default, mas não será usado

    if kernel in ['poly', 'sigmoid']:
        coef0 = trial.suggest_float('coef0', 0.0, 0.5)
    else:
        coef0 = 0.0  # default

    # Criar o modelo SVC com os parâmetros sugeridos
    svm = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0, random_state=42, max_iter=10000)

    # Treinar
    svm.fit(X_train_bal, y_train_bal)

    # Prever
    y_pred = svm.predict(X_test_scaled)

    # Métrica a maximizar
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"[TRIAL] kernel={kernel}, C={C:.5f}, F1={f1:.4f}")
    return f1

storage = "sqlite:///multi_SVM_study_pc1.db"
study = optuna.create_study(
    direction="maximize",
    study_name="SVM_study4",
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
svm.fit(X_train, y_train_bal)
time_total = time.time() - start_time
print(f"Total training time: {time_total:.2f} seconds")

# Previsões no teste
y_pred_svm = svm.predict(X_test)

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
