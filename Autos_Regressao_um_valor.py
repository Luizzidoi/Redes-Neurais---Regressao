""""
Redes Neurais
Problema de Regressão
Base de dados de carros usados (Kaggle.com)
Com base em vários atributos de um determinado veículo, é possível determinar o seu preço
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Colocar o 'Encoding' devido às caracteristicas da base de dado por conter várias strings
base = pd.read_csv('autos.csv', encoding='ISO-8859-1')

""" Apagar alguns atributos da base de dados que não serão utilizados """
# Axis = 1 para apagar a coluna inteira
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)


""" Analisar cada atributo da base de dados para ver se realmente ajudará a rede a encontrar um padrão """
# Por exemplo: o atributo 'name' não ajudara a rede a encontrar um padrão, então após essa análise o atributo é removido
base['name'].value_counts()
base = base.drop('name', axis=1)
# Igualmente para os outros dois atributos abaixo:
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

# Localizar na base de dados valores de preços menores que 10 dolares para serem removidos (valores inconsistentes)
i1 = base.loc[base.price <= 10]
# Adicionar à variável 'base' apenas os carros com preços acima de 10 dolares
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

# Mostra os veículos que a variável 'vehicleType' é nula
base.loc[pd.isnull(base['vehicleType'])]
# Mostra a quantidade que há de cada veículo na base de dados. Limousine é o que mais aparece
base['vehicleType'].value_counts()
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()                # 'manuell' é o que mais aparece
base.loc[pd.isnull(base['model'])]
base['model'].value_counts()                  # 'golf' é o que mais aparece
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()               # 'benzin' é o que mais aparece
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()      # 'nein' é o que mais aparece

# Montando uma variável que recebe os atributos que serão trocados na base de dados. Preenchimento daqueles que são naN (não atribuído)
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
# Linha que fará a substituição dos valores na base de dados
base = base.fillna(value=valores)

""" Carrega as variáveis com os atributos da base de dados que serão utilizados """
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values


# Importação da classe que será utilizada para a transformação de atributo categórico em atributo numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()

# Transformação de todos (que são necessários) os atributos categóricos em atributos numéricos
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])
# print(previsores[0:20])


# Linhas de código utilizadas no curso da udemy porém que não existe mais. Problema no 'categorical_features'
# onehot_encoder = OneHotEncoder(categorical_features = [0, 1, 3, 5, 8, 9, 10])     # Criação da variável tipo dummy (parecido com a transformação de um número em um número binário)
# previsores = onehot_encoder.fit_transform(previsores).toarray()                   # Variável previsores recebe o onehot_encoder e armazena em uma matriz


""" Linhas de código atualizadas para a criação da variável do tipo dummy (encontrado na internet) """
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
previsores = np.array(ct.fit_transform(previsores))

""" Criação da rede neural e treinamento """
regressor = Sequential()
# Units = (316 + 1) / 2 = 158
regressor.add(Dense(units=158, activation='relu', input_dim=316))
regressor.add(Dense(units=158, activation='relu'))
# 'linear' para problemas de regressão. Função que não faz nada, já que queremos o retorno do preço
regressor.add(Dense(units=1, activation='linear'))

# 'mean_absolute_error' para regressão. Traz a média do erro absoluto
regressor.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
# A cada 300 (valor do batch_size) registros a rede faz a atualização dos pesos
regressor.fit(previsores, preco_real, batch_size=300, epochs=100)

# A rede pega todos os registros da base de dados -> jogar para o conjunto de pesos e fazer a previsão de um valor
previsoes = regressor.predict(previsores)
print(preco_real.mean())
print(previsoes.mean())

print('Fim')