# %%
# importing basic libraries
import xgboost as xgb
import pandas as pd
import numpy as np
import asyncio
import json
from memphis import Memphis, MemphisError, MemphisConnectError

# %%
import os
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

# %%
def update_neighbors(df_aux):
    
    df_filled = df_aux.copy()
   
    # Definindo os índices das vizinhanças que queremos preencher
    neighborhood = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]
    fields = ['temp_day-1_0_0', 'temp_day-1_0_1', 'temp_day-1_0_2', 'temp_day-1_1_0', 'temp_day-1_1_2', 'temp_day-1_2_0', 'temp_day-1_2_1', 'temp_day-1_2_2']

    # Iterando sobre os registros do DataFrame original
    for index, row in df_filled.iterrows():
        x = index[0]
        y = index[1]
        default = 90 # value of no fire
        # print(x, y, default)
        for i in range(len(neighborhood)):
            new_x, new_y = neighborhood[i][0] + x, neighborhood[i][1] + y
            # print(new_x, new_y)

            # Verificando se os índices vizinhos estão dentro dos limites do DataFrame
            field = fields[i]
            if 0 <= new_x < 30 and 0 <= new_y < 30:
                value = df_filled.loc[(new_x, new_y), 'temperature']
                df_filled.loc[(x, y), field] = value
            else:
                df_filled.loc[(x, y), field] = default
   
    return df_filled

# %%
# Column list
cols = ['geospatial_x', 'geospatial_y', 'temperature', 
        'temp_day-1_0_0', 'temp_day-1_0_1', 'temp_day-1_0_2',
        'temp_day-1_1_0', 'temp_day-1_1_2',
        'temp_day-1_2_0', 'temp_day-1_2_1', 'temp_day-1_2_2',
        'alarm']
df_base = pd.DataFrame(columns=cols)

# Create matrix  x and y        
x = np.arange(0, 30)
y = np.arange(0, 30)
coords = pd.MultiIndex.from_product([x, y])

# Create dataframe and fill with NaN
# Extrair níveis do MultiIndex em colunas separadas
df_base['geospatial_x'] = coords.get_level_values(0)
df_base['geospatial_y'] = coords.get_level_values(1)

# Definir MultiIndex como índice do DataFrame
df_base.set_index(['geospatial_x', 'geospatial_y'], inplace=True)

df = df_base.copy()

# %%


# %%
# read new day's sensors
query = text("SELECT * FROM memphis2023.sensors where day = :day")
# df_sensors = pd.read_sql(query, engine, ad=actual_day)
parameters = {"day": variables.DAY_BASE + 1}
df_sensors = connection.execute(query, parameters).fetchall()
# update df_base with the new sensors of the date -1 (actual)
for one_sensor in df_sensors:
    day = one_sensor[0]
    x = one_sensor[1]
    y = one_sensor[2]
    temperature = one_sensor[3]
    # print(x, y, temperature)
    idx = (x, y)
    df.loc[idx, 'temperature'] = temperature

# %%
# get the average temperature 
average = df["temperature"].mean()
print(average)

# update df_sensors with the neighbors
df = update_neighbors(df)

# %%
features = ['geospatial_x', 'geospatial_y', 'temperature', 
        'temp_day-1_0_0', 'temp_day-1_0_1', 'temp_day-1_0_2',
        'temp_day-1_1_0', 'temp_day-1_1_2',
        'temp_day-1_2_0', 'temp_day-1_2_1', 'temp_day-1_2_2']


# %%

df = df.reset_index()
df = df[features]
# convert fields to int
df[features] = df[features].astype(int)

# %%
# loading the modelo
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('xgb_model_simples.bin')

# %%
df.head()

# %%
# Obter as probabilidades das previsões para a classe positiva (1)
y_prob = loaded_model.predict_proba(df)[:, 1]

# %%
# Definir o valor de corte personalizado para a classificação
valor_corte = 0.45 # Escolha um valor de corte adequado

# %%
# Transformar as probabilidades em rótulos discretos com base no valor de corte
y_pred = (y_prob >= valor_corte).astype(int)

# %%
# do predictions
# y_pred = loaded_model.predict(df)

# %%
print(y_pred)

# %%
matrix = y_pred.reshape(30, 30)

# %%
# Encontrar índices onde matriz == 1
list_predict = []
idx = np.where(matrix == 1)

print('ALERTS TO BE SENT')
# Imprimir coordenadas 
for i, j in zip(idx[0], idx[1]):
    print(f'Coordinate: ({i}, {j})')
    list_predict.append((i,j))

# %%
# read alarms of that day
query = text("SELECT distinct geospatial_x, geospatial_y, event_day FROM alerts where event_day = :day group by geospatial_x, geospatial_y, event_day order by geospatial_x, geospatial_y, event_day ")
parameters = {"day": variables.DAY_BASE + 1}
results = connection.execute(query, parameters).fetchall()
df_alerts =  pd.DataFrame(results, columns=['geospatial_x', 'geospatial_y', 'event_day'])
df_alerts.head(50)

# %%
list_official = []
for index, row in df_alerts.iterrows():
    x = row['geospatial_x']
    y = row['geospatial_y']
    list_official.append((x,y))
        

# %%
# Criar conjuntos a partir das listas
set_predict = set(list_predict)
set_official = set(list_official)

# Elementos iguais
elements_in_both_lists = set_predict.intersection(set_official)
print("Both lists:")
for element in elements_in_both_lists:
    print(element)

# %%
# Elementos presentes apenas na lista_predict
elements_only_in_predict = set_predict.difference(set_official)
print("\nOnly predict (does not appears at official):")
for element in elements_only_in_predict:
    print(element)

# %%
# Elementos presentes apenas na lista_official
elements_only_in_official = set_official.difference(set_predict)
print("\nOfficial and not predict:")
for element in elements_only_in_official:
    print(element)

# %%
df.set_index(['geospatial_x', 'geospatial_y'], inplace=True)
print(df.loc[(2,10), 'temperature'])


# %%
df.loc[(4,10)]


