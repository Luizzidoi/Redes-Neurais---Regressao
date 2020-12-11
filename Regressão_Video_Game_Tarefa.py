"""
Redes Neurais - Regressão
Base de dados de video games
Regressão com múltiplas saídas
Previsão do valor de venda de alguns jogos em países diferentes
Tarefa: Previsão do valor total das vendas dos jogos
"""


import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregar a variável com a base de dados """
base = pd.read_csv('games.csv')

# Apagar algumas colunas de parametros que não serão importantes para essa tarefa
base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis=1)
base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)

# Apagar algumas linhas de parametros no qual há valores não preenchidos (nan)
base = base.dropna(axis=0)

# Retirar da base de dados vendas com valores menores do que 1
base = base.loc[base['Global_Sales'] > 1]

# Retirar a coluna 'Name" para não dar problemas e pq não há uma relação com os preços
nome_jogos = base.Name
base = base.drop('Name', axis=1)


""" Começa as configurações da rede neural - Carregamento das variáveis """
previsores = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
valor_vendas = base.iloc[:, 4].values


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

# Estrutura da rede neural
# A rede possui 99 neuronios na entrada
camada_entrada = Input(shape=(99,))
# ativacao = Activation(activation='sigmoid')
# Units = (entrada + saída)/2 = (99 + 1)/2 = 50
camada_oculta1 = Dense(units=50, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=50, activation='sigmoid')(camada_oculta1)
d = Dropout(0.3)(camada_oculta1)
camada_saida = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida])
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(previsores, [valor_vendas], epochs=5000, batch_size=100, verbose=2)

previsoes = regressor.predict(previsores)

print('Fim')