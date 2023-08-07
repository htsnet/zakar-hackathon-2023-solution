# importing basic libraries
from __future__ import annotations
import xgboost as xgb
import pandas as pd
import numpy as np
import asyncio
import json
from memphis import Memphis, Headers, MemphisError, MemphisConnectError, MemphisHeaderError, MemphisSchemaError


import os
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

@dataclass
class Config:
    memphis_host: str = os.environ.get("HOST")
    memphis_user: str = os.environ.get("USER")
    memphis_pwd: str = os.environ.get("PWD")
    memphis_id: str = os.environ.get("ID")
    
config = Config()

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
        try:
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
        except:
            df_filled.loc[(x, y), field] = default
   
    return df_filled


async def main():
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
    # get new records of sensors
    df_sensors = pd.DataFrame(columns=['day', 'geospatial_x', 'geospatial_y', 'temperature'])
        
    try:
        memphis = Memphis()
        await memphis.connect(host=config.memphis_host, username=config.memphis_user, password=config.memphis_pwd, account_id=config.memphis_id)
        consumer = await memphis.consumer(station_name="zakar-temperature-readings", consumer_name="prediction")
        
        # while True:
        # get only one day (900 records)
        batch = await consumer.fetch(batch_size=900)
        if batch is not None:
            for msg in batch:
                serialized_record = msg.get_data()
                record = json.loads(serialized_record)
                
                df_aux = pd.DataFrame({'day': [record['day']], 'geospatial_x': [record['geospatial_x']], 'geospatial_y': [record['geospatial_y']], 'temperature': [record['temperature']]})
                df_sensors = pd.concat([df_sensors, df_aux], ignore_index = True)
                # Acknowledge the message to the MEMPHIS queue
                await msg.ack()
        # else:
        #     break 
        
    except (MemphisError, MemphisConnectError) as e:
        print(e)
        
    df_sensors = df_sensors.reset_index(drop=True)
    # update df_base with the new sensors of the date -1 (actual)
    for i, row in df_sensors.iterrows():
        day = row['day']
        day_sensors = day
        x = row['geospatial_x']
        y = row['geospatial_y']
        temperature = row['temperature']
        idx = (x, y)
        df.loc[idx, 'temperature'] = temperature

    # get the average temperature 
    average = df["temperature"].mean()
    print(average)

    # update df_sensors with the neighbors
    df = update_neighbors(df)

    features = ['geospatial_x', 'geospatial_y', 'temperature', 
            'temp_day-1_0_0', 'temp_day-1_0_1', 'temp_day-1_0_2',
            'temp_day-1_1_0', 'temp_day-1_1_2',
            'temp_day-1_2_0', 'temp_day-1_2_1', 'temp_day-1_2_2']

    df = df.reset_index()
    df = df[features]
    # convert fields to int
    df[features] = df[features].astype(int)

    # %%
    # loading the modelo
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('xgb_model.bin')

    # Obter as probabilidades das previsões para a classe positiva (1)
    y_prob = loaded_model.predict_proba(df)[:, 1]

    # Definir o valor de corte personalizado para a classificação
    valor_corte = 0.55 # Escolha um valor de corte adequado

    # Transformar as probabilidades em rótulos discretos com base no valor de corte
    y_pred = (y_prob >= valor_corte).astype(int)

    matrix = y_pred.reshape(30, 30)


    # Encontrar índices onde matriz == 1
    list_predict = []
    idx = np.where(matrix == 1)

    print('ALERTS TO BE SENT')
    # Imprimir coordenadas 
    for i, j in zip(idx[0], idx[1]):
        print(f'Coordinate: ({i}, {j})')
        list_predict.append((i,j))
        # send alert
        try:
            producer = await memphis.producer(station_name="zakar-fire-predictions", producer_name="prediction") # you can send the message parameter as dict as well
            headers = Headers()
            headers.add("key", "value") 
            for alarm in list_predict:
                await producer.produce(
                        message={"event_day": int(day_sensors), "notification_day": int(day_sensors), "geospatial_x": int(i), "geospatial_y": int(j)},
                        headers=headers,
                        ack_wait_sec=30
                        )
           
        except (MemphisError, MemphisConnectError, MemphisHeaderError, MemphisSchemaError) as e:
            print(e)
        
    await memphis.close()
        
if __name__ == "__main__":
    asyncio.run(main())

