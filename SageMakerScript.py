
import requests
import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

import os
import boto3
import json

# Función para obtener los precios de Ethereum de la API
def get_eth_price(date):
    test= 1675220400
    today_datetime = datetime.combine(date, datetime.min.time())
    timestamp = int(today_datetime.timestamp())
    
    url = f'https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=usd&from={1664593200}&to={timestamp}'
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    df = df.drop('timestamp', axis=1)
    df = df.iloc[:-1,:]
    return df


def preprocessing_data(training_set):
    training_set = training_set.iloc[:,0:1].values
    # Escalado de variables
    sc = MinMaxScaler()
    
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    
    for i in range(15, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-15:i, 0])
        y_train.append(training_set_scaled[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, sc


# Creación y entrenamiento del modelo
def training_model(X_train, y_train):
    tf.keras.backend.set_image_data_format("channels_last")
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1000, batch_size=32)
    return model

def predict_prices_for_month(training_set, sc, model, days=30):
    all_predicts = []
    last_days = training_set[-15:]
    for i in range(days):
        last_days_scaled = sc.transform(last_days)
        X_test = np.array([last_days_scaled])
        predicted_price_scaled = model.predict(X_test)
        predicted_price = sc.inverse_transform(predicted_price_scaled)
        all_predicts.append(predicted_price[0,0])
        last_days = np.concatenate([last_days, predicted_price], axis=0)[-15:]
    return all_predicts


def json_to_aws(day_predictions):
    try:
        prediction_json = json.dumps(day_predictions)
        
        # Crea una sesión de AWS con las credenciales
        session = boto3.Session(
            aws_access_key_id='Your Access Key',
            aws_secret_access_key='Your Secret Key'
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
        return 'se ha enviado la predicciones'
    except Exception as e:
        return str(e)




today = date.today()
training_set = get_eth_price(today)
X_train, y_train, sc = preprocessing_data(training_set)
model = training_model(X_train, y_train)
training_set_pred = training_set.iloc[:,0:1].values
month_predict = predict_prices_for_month(training_set_pred, sc, model)
day_predictions = {i+1: float(prediction) for i, prediction in enumerate(month_predict)}
json_to_aws(day_predictions)



import matplotlib.pyplot as plt
df = pd.DataFrame(month_predict, columns=['0'])
# Graficar los datos
df.plot()
plt.show()
