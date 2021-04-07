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
X = df.iloc[:,[0,2,4,5]]
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
Xmean[:,2] = le.fit_transform(Xmean[:,2])
# %%
## Tratamnto do Missing através da substituição pela Média
from sklearn.impute import SimpleImputer
si1 = SimpleImputer(missing_values=np.nan, strategy='mean')
si1.fit(Xmean[:,3:])
Xmean[:,3:] = si1.transform(Xmean[:,3:])
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
plt.scatter(Xmean_train[:,3], ya_train, color='red')
#plt.plot(Xmean_train[:,3], reg.predict(Xmean_train), color='blue')
plt.scatter(Xmean_train[:,3], np.round(reg.predict(Xmean_train)), color='blue', alpha=0.2)
plt.title('Sobreviveram x Idade (Treino)')
plt.xlabel('Idade')
plt.ylabel('Sobreviveram')
plt.show()
# %%
## Visualizando Resultados de Testes
plt.scatter(Xmean_test[:,3], ya_test, color='red')
#plt.plot(Xmean_train[:,3], reg.predict(Xmean_train), color='blue')
plt.scatter(Xmean_train[:,3], np.round(reg.predict(Xmean_train)), color='blue', alpha=0.2)
plt.title('Sobreviveram x Idade (Teste)')
plt.xlabel('Idade')
plt.ylabel('Sobreviveram')
plt.show()
# %% Verificando o score do modelo
reg.score(Xmean_train, ya_train) ## Muito baixo 0.39

# %%
