1. Introdução

    Este trabalho tem como objetivo o estudo de dois datasets: um binário e outro multiclasse e ambos se relacionam com a 
doença diabetes. Este ficheiro README.txt serve de guia para o entendimento e a execução dos ficheiros.

2. Ficheiros

    O nosso trabalho consiste em: 
        -> 2 ficheiros .csv: contêm os datasets
            . Binário: diabetes_01.csv
            . Multiclasse: diabetes_02.csv

        -> 5 ficheiros .py:
            . Dataset binário:
                -> diabetes_01_SMOTE.py: usado para testar a técnica SMOTE e para o resto das etapas (modelização e resultados)
                -> diabetes_01_under.py: usado apenas para visualizar a técnica de undersampling
            
            . Dataset multiclasse:
                -> diabetes_012_SMOTE.py: utilizado apenas para visualizar a técnica SMOTE
                -> diabetes_012_u_S.py: utilizado para testar as duas técnicas e para o resto das etapas (modelização e resultados)
                -> diabetes_012_under.py: utilizado apenas para visualizar a técnica de undersampling

        -> ... ficheiros .db:
            Este PC (PC1):
            . Modelo da Rede Neuronal:
                -> binary_mlp_study_pc1.db: guarda evolução da métrica F1-score após aplicação de MLP aplicado ao dataset binário depois 
                de vários testes com diferentes parâmetros
                -> binary_mlp_study2_pc1.db: guarda mais dados
                -> multi_mlp_study_pc1.db: guarda evolução da métrica F1-score após aplicação de MLP aplicado ao dataset multiclasse depois 
                de vários testes com diferentes parâmetros
    
            . Modelo SVM:
                -> binary_SVM_study_pc1.db: guarda evolução da métrica F1-score após aplicação de SVM aplicado ao dataset binário depois 
                de vários testes com diferentes parâmetros
                -> multi_SVM_study_pc1.db: guarda evolução da métrica F1-score após aplicação de SVM aplicado ao dataset multiclasse depois 
                de vários testes com diferentes parâmetros
            
            Pasta "Resultados noutros PCs":
            . PC2:
                -> binary_mlp_study_pc2.py
                -> binary_SVM_study_pc2.py
                -> multi_mlp_study_pc2.py
                -> multi_SVM_study_pc2.py
            . PC3:
                -> binary_mlp_study_pc3.py
                -> binary_SVM_study_pc3.py
                -> multi_mlp_study_pc3.py
                -> multi_SVM_study_pc3.py


3. Como correr os Ficheiros 

    Antes de correr os ficheiros, existem bibliotecas que são necessárias ter instaladas para que tudo funcione corretamente.
    Desta forma, criámos o ficheiro requirements.txt onde se encontram essas bibliotecas e para a sua instalação abre-se o terminal
e digita-se o comando "pip install -r requirements.txt". Assim os ficheiros .py estão prontos para serem executados sem erros.

    Atenção: a biblioteca Optuna não funciona em todas as versões de Python, no entanto foi usada com sucesso nas versões
3.10.12, 3.12.7 e 3.12.2.

    Para correr os ficheiros, abra o terminal e use o comando "python <nome_do_ficheiro.py>".


