# %%
# Tratamento dos dados
## Importação das bibliotecas padrão
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
## Importação da base de dados e criação do Dataframe
df = pd.read_csv('./titanic/train.csv')
# %%
df.head()
# %%
## Atribuindo Entradas e Saídas (X e y)
X = df.iloc[:,[0,2,3,4,5]]
y = df.Survived
print(X)
print(y)
# %%
## Verificação de Missing
print(f"A entrada X possui missing? {X.isnull().values.any()}")
print(f"Quantos? {X.isnull().sum().sum()}")
print(f"Quais colunas possuem missing? \n {X.isnull().sum()}")
# %%
## Tratamento do Missing através da exclusão dos itens
#Xexc = X.dropna()
#Xexc = Xexc.reset_index(drop=True)
# %%
## Tratamnto do Missing através da substituição pela Média
from sklearn.impute import SimpleImputer
si1 = SimpleImputer(missing_values=np.nan, strategy='mean')
Xmean = X.values
si1.fit(Xmean[:,4:])
Xmean[:,4:] = si1.transform(Xmean[:,4:])
print(Xmean)
# %%
## Transformando Saída em Array
ya = y.values
# %%
## Separando em Teste e Treino
from sklearn.model_selection import train_test_split
Xmean_train, Xmean_test, ya_train, ya_test = train_test_split(Xmean, ya, test_size=0.2, random_state=1)
print(Xmean_train)
print(Xmean_test)
print(ya_train)
print(ya_test)
