#Created on Thu Feb  9 07:33:12 2023
#@author: Gaston Franco

import requests
import pandas as pd
import numpy as np

from datetime import datetime, date
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Función para obtener los precios de Ethereum de la API
def get_eth_price(date):
    test= 1675220400
    today_datetime = datetime.combine(date, datetime.min.time())
    timestamp = int(today_datetime.timestamp())
    url = f'https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=usd&from=1392577232&to={timestamp}'
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    df = df.drop('timestamp', axis=1)
    df = df.iloc[:-1,:]
    return df

# Función para la normalización y escalado + preproceso para entrada a la red LSTM
def preprocessing_data(training_set):
    training_set = training_set.iloc[:,0:1].values
    # Escalado de variables
    sc = MinMaxScaler()
    
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    
    for i in range(60,len(training_set)):
        
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, sc


# Creación y entrenamiento del modelo
def training_model(X_train, y_train):
    tf.keras.backend.set_image_data_format("channels_last")
    model = Sequential()
    model.add(LSTM(64, return_sequences = True, input_shape = (X_train.shape[1],1)))
    model.add(Dropout(0.25))
    model.add(LSTM(64, return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(64, return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(64, return_sequences = False))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.compile(optimizer='Nadam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=32, batch_size=32)
    return model


# Función para hacer la predicción
def predict(model, data, sc):
     dataset = data['price']
     inputs = dataset[len(dataset)-90:].values
     inputs = inputs.reshape(-1,1)
     # Escalado de variables
     
     inputs = sc.transform(inputs)
     X_train = []
     
     for i in range(60,90):
         X_train.append(inputs[i-60:i, 0])

         
     X_train= np.array(X_train)
     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
     
     # Haciendo predicción
     y_pred = sc.inverse_transform(model.predict(X_train))
     return y_pred


# Main loop
try:
    today = date.today()
    data = get_eth_price(date = today)
    X_train, y_train,sc = preprocessing_data(data)
    model = training_model(X_train, y_train)
    prediction = predict(model, data, sc)
except Exception as e:
    print(e)
    
    
# Exportar predicción
import os
import boto3
import json

# Prediccion a formato json
prediction_dict = {i+1: float(j[0]) for i,j in enumerate(prediction)}
prediction_json = json.dumps(prediction_dict)


ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')


# Crea una sesión de AWS con las credenciales
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)

# Crea un objeto S3
s3 = session.resource("s3")

# Nombre del bucket y la carpeta donde se guardará la predicción
bucket_name = 'gfranco'
folder_name = "predicts"
file_name = "prediction-{}.json".format(datetime.now().month)


file_object = s3.Object(bucket_name, f"{folder_name}/{file_name}")

# Escribe la predicción en el archivo
file_object.put(Body=prediction_json.encode("utf-8"))
