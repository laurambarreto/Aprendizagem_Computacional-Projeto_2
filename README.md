# Aprendizagem Computacional: Classificação de Diabetes

**Autores:** Beatriz Martins e Laura Barreto

## Resumo do Projeto
Este trabalho tem como objetivo o estudo de dois datasets que se relacionam com o diagnóstico da doença diabetes. O projeto visa resolver a previsão de diagnóstico dividindo-se em dois problemas distintos de *Machine Learning*:
* **Classificação Binária:** O objetivo é prever o diagnóstico distinguindo simplesmente entre os pacientes sem diabetes e as restantes classes (diabetes/pré-diabetes agrupados).
* **Classificação Multiclasse:** O objetivo é prever o diagnóstico abrangendo três classes individuais, nomeadamente doentes sem diabetes, pré-diabetes e diabetes.

Os dados utilizados provêm da pesquisa *Behavioral Risk Factor Surveillance System* (BRFSS), disponibilizada através da plataforma Kaggle.

---

## Modelos e Avaliação
Para analisar e classificar os dados de saúde, foram testados diversos modelos consoante o tipo de problema:
* **Dataset Binário:** Foram aplicados os algoritmos *Support Vector Machines* (SVM), Redes Neuronais e Regressão Logística.
* **Dataset Multiclasse:** Foram utilizados os modelos *Support Vector Machines* (SVM), Redes Neuronais e *Naive Bayes*.

A biblioteca Optuna foi utilizada ao longo das experiências para otimização, guardando a evolução das métricas (como o *F1-score*) em ficheiros de base de dados `.db` após a aplicação dos modelos com diferentes hiperparâmetros.

---

## Tratamento e Balanceamento de Dados
Uma vez que os dados de saúde exibem frequentemente um número díspar de amostras por classe, aplicaram-se metodologias de balanceamento:
* **SMOTE:** Esta técnica de geração sintética é utilizada ativamente para o balanceamento dos dados e posterior teste nas etapas de modelização e resultados.
* **Undersampling:** Esta abordagem foi aplicada em ficheiros próprios para efeitos focados na visualização gráfica do redimensionamento da classe maioritária.

---

## Estrutura de Ficheiros e Diretórios

### Ficheiros de Dados (`.csv`)
* **`diabetes_01.csv`:** Contém o dataset relativo à classificação de carácter binário.
* **`diabetes_02.csv`:** Contém o dataset correspondente ao problema de caráter multiclasse.

### Scripts de Código (`.py`)
* **Problema Binário:**
    * `diabetes_01_SMOTE.py`: Ficheiro usado para testar a técnica SMOTE e executar o resto das etapas, englobando a modelização e os resultados.
    * `diabetes_01_under.py`: Script utilizado apenas para visualizar a técnica de *undersampling*.
* **Problema Multiclasse:**
    * `diabetes_012_u_S.py`: Ficheiro principal desta vertente, utilizado para testar as duas técnicas de balanceamento e para o cálculo dos modelos e extração de resultados finais.
    * `diabetes_012_SMOTE.py` e `diabetes_012_under.py`: Scripts utilizados exclusivamente para observar a visualização das técnicas de SMOTE e *undersampling*, respetivamente.

### Registo do Optuna (`.db`)
* Bases de dados locais geradas pelo programa, como `binary_mlp_study_pc1.db` ou `multi_SVM_study_pc1.db`, que guardam a evolução da métrica *F1-score* após os vários testes de otimização de parâmetros dos modelos.

---

## Requisitos e Como Executar

1.  **Pré-requisitos:** O ambiente de execução do programa exige a instalação de dependências específicas de análise de dados e *Machine Learning*. As bibliotecas necessárias presentes no ficheiro de dependências são: `matplotlib`, `optuna`, `imblearn`, `numpy`, `pandas`, `scikit-learn` e `seaborn`. Pode instalar todas via terminal:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execução:** Para executar o projeto, selecione e corra o script Python correspondente à tarefa desejada.
