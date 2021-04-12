## Trabalho de DM
##====================================================================##
##===============EXCLUINDO AS LINHAS QUE CONTEM MISSING===============##
##====================================================================##
# %% Importação das bibliotecas padrão
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% Importação da base de dados e criação do Dataframe
df = pd.read_csv('./titanic/train.csv')
df.head()
# %% Análise Exploratória
import pandas_profiling as pp
profile = pp.ProfileReport(df=df, title='Análise Exploratória')
# %%
profile.to_file(output_file='Profile.html')
# %% Separando em Entrada e Saída
X = df.iloc[:,[2,4,5]]
y = df.Survived
# %% Verificação de Missing
print(f"A entrada X possui missing? {X.isnull().values.any()}")
print(f"Quantos? {X.isnull().sum().sum()}")
print(f"Quais colunas possuem missing? \n {X.isnull().sum()}")
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
# %%
##====================================================================##
# %% Treinando Modelo - Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
# %% Prevendo base de testes
y_pred = model.predict(X_test)
# %% Matriz Confusão e Acc
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %% Criação de um dataframe para resultados
df_comp = pd.DataFrame( columns=['Modelo', 'ExclusaoMissing', 'GridSearch', 'Acurácia', 'Kappa', 'F1'])
# %% Método para inserção dos valores do obtidos
def inclui_resultado(modelo, exclusao_missing, grid_search, acuracia, k, f1):
    i = df_comp.shape[0]
    df_comp.loc[i] = [modelo, exclusao_missing, grid_search, acuracia, k, f1]
# %% Matriz Confusão e Acc/Kappa/F1
def pred_and_evalue(X_test, y_test, model):

    y_pred = model.predict(X_test)

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
    confMatrix = confusion_matrix(y_test, y_pred)
    
    ax = plt.subplot()
    sns.heatmap(confMatrix, annot=True, fmt=".0f")
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz Confusão')

    ## Colocar nomes
    ax.xaxis.set_ticklabels(['Sobreviveu', 'Morreu'])
    ax.yaxis.set_ticklabels(['Sobreviveu', 'Morreu'])
    plt.show()
    return acuracia, k, f1
# %% Execução do Método
print('Classificação: Regressão Logística:')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Regressão Logística', 1, 0, acuracia, k, f1)
# %% Visualização do resultado de treino

# %% Visualização do resultado de teste

# %%
##====================================================================##
# %% Treinando Modelo - K-NN (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

# %% Execução do Método de Matriz Confusão e Acc
print('Classificação: KNN')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('KNN', 1, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Parâmetros a serem testados
tuned_parameters = [{'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]
print("# Tuning hyper-parâmetros parar F1 score")
print()
model = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring='f1')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print()
# %% Matriz Consfusão e Acc do Modelo GridSearch/KNN
print('Classificação: KNN com GridSearch')
acuracia, k, f1 = pred_and_evalue(X_test,y_test,model)
inclui_resultado('KNN_GS', 1, 1, acuracia, k, f1)
# %%
##====================================================================##
# %% Treinando Modelo - SVM (Support Vector Machine)
from sklearn.svm import SVC

model = SVC(random_state=0)
model.fit(X_train, y_train)
# %% Avaliação do Modelo SVM
print('Classificação: SVM')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('SVM', 1, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Trainamento Modelo - Árvore de Decisão - Gini
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(min_samples_leaf=5, random_state=0, criterion='gini')
model.fit(X_train, y_train)

# %% Árvore de Decisão
from sklearn import tree
fig, ax = plt.subplots(figsize=(20,10))
tree.plot_tree(model, class_names=['Não Sobreviveu', 'Sobreviveu'],filled=True, rounded=True)

# %% Avaliação do Modelo Árvore de Decisão (Gini - impurezas)
print('Classificação: Árvore de Decisão (Gini)')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Árvore Decisão', 1, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Grid Search - Árvore de Decisão (Gini)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Set the parameters by cross-validation
tuned_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8,10,12],
                     'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10]}]

print("# Tuning hyper-parameters for F1 score")
print()

model = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring='f1')
model.fit(X_train, y_train)

y_true, y_pred = y_test, model.predict(X_test)
print(classification_report(y_true, y_pred))
print()
# %% Avaliação do GridSearch - Árvore de Decisão (Gini)
print('Classificação: GridSearch Árvore de Decisão (Gini)')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Árvore Decisão_GS', 1, 1, acuracia, k, f1)
# %% Melhores Parâmetros
print('Melhores Parâmetros:', model.best_params_)
# %% 
fig, ax = plt.subplots(figsize=(20, 10)) # Definir tamanho da imagem a ser gerada
tree.plot_tree(model.best_estimator_, class_names=['Não Sobreviveu', 'Sobreviveu'], 
               filled=True, rounded=True) ##, feature_names=data.columns); Ajustar para incluir esse parâmetro
##====================================================================##
# %% Treinamento do Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(min_samples_leaf=5, random_state=0)
model.fit(X_train, y_train)
# %% Avaliação do modelo
print('Classificação: Random Forest')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Random Forest', 1, 0, acuracia, k, f1)
# %%
##====================================================================##
##===========SUSTITUINDO OS MISSINGS PELA MÉDIA DAS IDADES============##
##====================================================================##
# %% Importação da biblioteca
df = pd.read_csv('./titanic/train.csv')
df.head()
# %% Separando em Entrada e Saída
X = df.iloc[:,[2,4,5]]
y = df.Survived
# %% Verificação de Missing
print(f"A entrada X possui missing? {X.isnull().values.any()}")
print(f"Quantos? {X.isnull().sum().sum()}")
print(f"Quais colunas possuem missing? \n {X.isnull().sum()}")
# %% Separando em Entrada e Saída
X = df.iloc[:,[2,4,5]].values
y = df.Survived.values
# %% Substituicao pela media das idades
from sklearn.impute import SimpleImputer
si1 = SimpleImputer(missing_values=np.nan, strategy='mean')
si1.fit(X[:,2:])
X[:,2:] = si1.transform(X[:,2:])
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
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
# %% Prevendo base de testes
y_pred = model.predict(X_test)
# %% Matriz Confusão e Acc
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %% Execução do Método de Matriz Confusão e Acc
print('Classificação: Regressão Logística:')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Regressão Logística', 0, 0, acuracia, k, f1)
# %% Treinando Modelo - K-NN (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

# %% Execução do Método de Matriz Confusão e Acc
print('Classificação: KNN')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('KNN', 0, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Parâmetros a serem testados
tuned_parameters = [{'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]
print("# Tuning hyper-parâmetros parar F1 score")
print()
model = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring='f1')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print()
# %% Matriz Consfusão e Acc do Modelo GridSearch/KNN
print('Classificação: KNN com GridSearch')
acuracia, k, f1 = pred_and_evalue(X_test,y_test,model)
inclui_resultado('KNN_GS', 0, 1, acuracia, k, f1)
# %%
##====================================================================##
# %% Treinando Modelo - SVM (Support Vector Machine)
from sklearn.svm import SVC

model = SVC(random_state=0)
model.fit(X_train, y_train)
# %% Avaliação do Modelo SVM
print('Classificação: SVM')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('SVM', 0, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Trainamento Modelo - Árvore de Decisão - Gini
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(min_samples_leaf=5, random_state=0, criterion='gini')
model.fit(X_train, y_train)

# %% Árvore de Decisão
from sklearn import tree
fig, ax = plt.subplots(figsize=(20,10))
tree.plot_tree(model, class_names=['Não Sobreviveu', 'Sobreviveu'],filled=True, rounded=True)

# %% Avaliação do Modelo Árvore de Decisão (Gini - impurezas)
print('Classificação: Árvore de Decisão (Gini)')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Árvore Decisão', 0, 0, acuracia, k, f1)
# %%
##====================================================================##
# %% Grid Search - Árvore de Decisão (Gini)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Set the parameters by cross-validation
tuned_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8,10,12],
                     'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10]}]

print("# Tuning hyper-parameters for F1 score")
print()

model = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring='f1')
model.fit(X_train, y_train)

y_true, y_pred = y_test, model.predict(X_test)
print(classification_report(y_true, y_pred))
print()
# %% Avaliação do GridSearch - Árvore de Decisão
print('Classificação: GridSearch Árvore de Decisão')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Árvore Decisão_GS', 0, 1, acuracia, k, f1)
# %% Melhores Parâmetros
print('Melhores Parâmetros:', model.best_params_)
# %% 
fig, ax = plt.subplots(figsize=(20, 10)) # Definir tamanho da imagem a ser gerada
tree.plot_tree(model.best_estimator_, class_names=['Não Sobreviveu', 'Sobreviveu'], 
               filled=True, rounded=True) ##, feature_names=data.columns); Ajustar para incluir esse parâmetro
##====================================================================##
# %% Treinamento do Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(min_samples_leaf=5, random_state=0)
model.fit(X_train, y_train)
# %% Avaliação do modelo
print('Classificação: Random Forest')
acuracia, k, f1 = pred_and_evalue(X_test, y_test, model)
inclui_resultado('Random Forest', 0, 0, acuracia, k, f1)
# %%
print(df_comp)