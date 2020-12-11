"""
Redes Neurais - Regressão
Base de dados de video games
Regressão com múltiplas saídas
Previsão do valor de venda de alguns jogos em países diferentes
"""


import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np

""" Carregar a variável com a base de dados """
base = pd.read_csv('games.csv')

# Apagar algumas colunas de parametros que não serão importantes
base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

# Apagar algumas linhas de parametros no qual há valores não preenchidos (nan)
# Axis = 0 será apagado as linhas e axis = 1 será apagado as colunas da base
base = base.dropna(axis=0)

# Apagar todas as linhas no qual os preços (Em NA e EU) são menores do que 1 para que não tenha problemas no treinamento da rede neural
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]
# Observação: a base de dados fica pequena após todos esses pré-processamentos reliazados

# Retirar a coluna 'Name" para não dar problemas e pq não há uma relação com os preços
nome_jogos = base.Name
base = base.drop('Name', axis=1)

""" Carregando as variáveis previsores e saídas """
previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values


""" Tranformação dos atributos categóricos em atributos numéricos """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

# Linhas de código utilizadas no curso da udemy porém que não existe mais. Problema no 'categorical_features'
# Linhas de código atualizada para a criação da variável do tipo dummy (encontrado na internet)
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [0, 2, 3, 8])], remainder='passthrough')
previsores = np.array(ct.fit_transform(previsores))


""" Estrutura da rede neural """
camada_entrada = Input(shape=(61,))
# Units = (entrada + saída)/2 = (61 + 3)/2 = 32
camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)
# Activation = linear indica que a rede não faz nada, aoenas deixa passar o valor recebido
camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer='adam', loss='mse')   # mse = mean square error (aceita como abreviação)
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs=5000, batch_size=100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)


print('Fim')