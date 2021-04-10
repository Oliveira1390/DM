## Classificação através de Regressão Logística
# %% Importação das bibliotecas padrão
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% Importação da base de dados e criação do Dataframe
df = pd.read_csv('./titanic/train.csv')
df.head()
# %% Exclusão das linhas com missing
df = df.dropna(subset=['Age'])
df = df.reset_index(drop=True)
# %% Separando em Entrada e Saída
X = df.iloc[:,[2,4,5]].values
y = df.Survived.values
# %% Alterando Gênero
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
# %% Separando em base de treino e de teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %% Escalonamento dos atributos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %% Treinando Modelo - Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# %% Prevendo base de testes
y_pred = classifier.predict(X_test)
# %% Matriz Confusão e Acc
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %% Matriz Confusão e Acc/Kappa/F1
def pred_and_evalue(X_test, y_test):

    y_pred = classifier.predict(X_test)

    # Acc
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia:', acuracia)

    # Kappa
    from sklearn.metrics import cohen_kappa_score
    k = cohen_kappa_score(y_test, y_pred)
    print('Kappa:', k)

    # F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    print('F1:', f1)

    # Matriz Confusão
    from sklearn.metrics import confusion_matrix
    confMatrix = confusion_matrix(y_pred, y_test)
    
    ax = plt.subplot()
    sns.heatmap(confMatrix, annot=True, fmt=".0f")
    plt.xlabel('Real')
    plt.ylabel('Previsto')
    plt.title('Matriz Confusão')

    ## Colocar nomes
    ax.xaxis.set_ticklabels(['1', '0'])
    ax.yaxis.set_ticklabels(['1', '0'])
    plt.show()
# %% Execução do Método
pred_and_evalue(X_test, y_test)
# %% Visualização do resultado de treino

# %% Visualização do resultado de teste
