""""
Redes Neurais
Problema de Regressão
Base de dados de carros usados (Kaggle.com)
Com base em vários atributos de um determinado veículo, é possível determinar o seu preço
Validação Cruzada = após encontrar os melhores parametros com o método Tuning, utiliza-se o método de validação cruzada
para o treinamento da rede.
"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Colocar o 'Encoding' devido às caracteristicas da base de dado por conter várias strings
base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
# Apagar alguns atributos da base de dados que não serão utilizados
# Axis = 1 para apagar a coluna inteira
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

# Adiciona à variável 'base' apenas os carros com preços acima de 10 dolares
base = base[base.price > 10]
base = base[base.price < 350000]

# Montando uma variável que recebe os atributos que serão trocados na base de dados. Preenchimento daqueles que são naN (não atribuído)
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
# Linha que faz a substituição dos valores na base de dados
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

""" Linhas de código atualizada para a criação da variável do tipo dummy (encontrado na internet) """
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
previsores = np.array(ct.fit_transform(previsores))

""" Função de criação da Rede Neural """
def criarRede():
    regressor = Sequential()
    # Units = (316 + 1) / 2 = 158
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    # 'linear' para problemas de regressão. Função que não faz nada, já que queremos o retorno do preço.
    regressor.add(Dense(units=1, activation='linear'))

    # 'mean_absolute_error' para regressão. Traz a média do erro absoluto
    regressor.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    return regressor

""" Chama a função de criação da Rede Neural """
regressor = KerasRegressor(build_fn=criarRede, epochs=100, batch_size=300)

resultados = cross_val_score(estimator=regressor, X=previsores, y=preco_real, cv=10, scoring='neg_mean_absolute_error')

""" Cálcula a média e o desvio. A média serve pq a validação cruzada traz 10 (valor do cv) resultados de probabilidade e 
para ter uma maior rpecisão da % da rede é feito uma média entre esses 10 valores.
O desvio é cálculado para ter uma idea se a rede teve overfiting """
# Função que cálcula a média da variável 'resultados'
media = resultados.mean()
# Função que cálcula o desvio padrão da variável 'resultados'
desvio = resultados.std()

print('Fim')