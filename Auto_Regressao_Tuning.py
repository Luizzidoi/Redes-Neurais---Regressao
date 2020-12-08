""""
Redes Neurais
Problema de Regressão
Base de dados de carros usados (Kaggle.com)
Com base em vários atributos de um determinado veículo, é possível determinar o seu preço
Tuning = Encontrar a melhor confiuração para a rede dentro de uma lista de configurações
"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
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

""" Linhas de código atualizada para a criação da variável do tipo dummy (encontrado na internet) """
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
previsores = np.array(ct.fit_transform(previsores))


""" Função de criação da Rede Neural """
def criarRede(loss):
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))

    regressor.compile(optimizer='adam', loss=loss, metrics=['mean_absolute_error'])

    return regressor


regressor = KerasRegressor(build_fn=criarRede, epochs=100, batch_size=300)
""" Lista com algumas configurações que podem ser utilizadas """
parametros = {'loss': ['mean_squared_error', 'mean_absolute_error',
                       'mean_absolute_percentage_error',
                       'mean_squared_logarithmic_error',
                       'squared_hinge']}

grid_search = GridSearchCV(estimator=regressor, param_grid=parametros, cv=10)
grid_search = grid_search.fit(previsores, preco_real)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros)
print(melhor_precisao)