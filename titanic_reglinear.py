# %%
# Tratamento dos dados
## Importação das bibliotecas padrão
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
## Importação da base de dados e criação do Dataframe
df = pd.read_csv('./titanic/train.csv')
df.head()
# %%
## Atribuindo Entradas e Saídas (X e y)
X = df.iloc[:,[2,4,5]] #df.iloc[:,[0,2,4,5]]
Xmean = X.values ## Transformando em array
y = df.Survived
ya = y.values ## Transformando em array
# %%
## Verificação de Missing
print(f"A entrada X possui missing? {X.isnull().values.any()}")
print(f"Quantos? {X.isnull().sum().sum()}")
print(f"Quais colunas possuem missing? \n {X.isnull().sum()}")
# %%
## Tratamento do Male/Female para 1/0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Xmean[:,1] = le.fit_transform(Xmean[:,1])
# %%
## Tratamento do Missing através da substituição pela Média
from sklearn.impute import SimpleImputer
si1 = SimpleImputer(missing_values=np.nan, strategy='mean')
si1.fit(Xmean[:,2:])
Xmean[:,2:] = si1.transform(Xmean[:,2:])
# %%
## Separando dados de Teste e Treino
from sklearn.model_selection import train_test_split
Xmean_train, Xmean_test, ya_train, ya_test = train_test_split(Xmean, ya, test_size=0.2, random_state=1)
# %%
## Regressão Linear Simples da base de treino
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(Xmean_train,ya_train)

# %%
## Predizendo os resultados da base de Teste
ya_pred = reg.predict(Xmean_test)
ya_pred = np.round(ya_pred)
# %%
## Visualizando Resultados de Treino
plt.scatter(Xmean_train[:,2], ya_train, color='red')
#plt.plot(Xmean_train[:,3], reg.predict(Xmean_train), color='blue')
plt.scatter(Xmean_train[:,2], np.round(reg.predict(Xmean_train)), color='blue', alpha=0.2)
plt.title('Sobreviveram x Idade (Treino)')
plt.xlabel('Idade')
plt.ylabel('Sobreviveram (1=Sim;0=Não)')
plt.show()
# %%
## Visualizando Resultados de Testes
plt.scatter(Xmean_test[:,2], ya_test, color='red')
#plt.plot(Xmean_train[:,3], reg.predict(Xmean_train), color='blue')
plt.scatter(Xmean_train[:,2], np.round(reg.predict(Xmean_train)), color='blue', alpha=0.2)
plt.title('Sobreviveram x Idade (Teste)')
plt.xlabel('Idade')
plt.ylabel('Sobreviveram (1=Sim;0=Não)')
plt.show()
# %% Verificando o score do modelo
reg.score(Xmean_train, ya_train) ## Muito baixo 0.39
# %%
from sklearn.feature_selection import f_regression
## Análise excluindo as linhas NaN
# %%
df_exc = df
df_exc = df_exc.dropna(subset=['Age'])
df_exc = df_exc.reset_index(drop=True)
# %% Separando Entradas / Saídas
X_exc = df_exc.iloc[:,[2,4,5]]
y_exc = df_exc.Survived
# %% Codificando Genero Masc/Fem (1/0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_exc['Sex'] = le.fit_transform(X_exc['Sex'])
# %% Divisão do treino x teste
from sklearn.model_selection import train_test_split
Xexc_train, Xexc_test, yexc_train, yexc_test = train_test_split(X_exc,y_exc, test_size=0.2, random_state=1)
print('Xexc treino:', Xexc_train.shape, 'Xexc teste:', Xexc_test.shape, 'yexc treino:', yexc_train.shape, 'yexc teste:',yexc_test.shape)
# %% Método para treino e modelo
from sklearn.linear_model import LinearRegression

def train_predict(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(Xexc_train, yexc_train)
    yexc_pred = np.round(model.predict(X_test))

    return model,yexc_pred
# %% Execução do método acima
model, yexc_pred = train_predict(Xexc_train, yexc_train, Xexc_test)
# %%
model.score(Xexc_train, yexc_train)
# %%
def evaluate(y_test, y_pred, X_test):
    # R2
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print('R2:', r2)

    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print('R2 Ajustado:', adj_r2)
# %%
evaluate(yexc_test, yexc_pred, Xexc_test)
# %%
Xexc_test.shape[0]
