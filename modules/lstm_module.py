# modules/lstm_module.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : (i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


@st.cache_data
def LSTM_ts(signal, epochs=10, forecast_periods=1):
    # Подготовка данных
    data = signal.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    look_back = 3
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential(
        [
            LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(100),
            Dense(1),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Прогноз
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Прогноз на будущие периоды
    last_sequence = data_scaled[-look_back:]
    future_predictions = []
    for _ in range(forecast_periods):
        next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    # Обратное преобразование
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # Подготовка данных для отображения
    max_length = len(data) + forecast_periods
    train_plot = np.full(max_length, np.nan)
    train_plot[look_back : len(train_predict) + look_back] = train_predict.flatten()
    test_plot = np.full(max_length, np.nan)
    test_plot[len(train_predict) + (look_back * 2) : len(data)] = test_predict.flatten()
    future_plot = np.full(max_length, np.nan)
    future_plot[-forecast_periods:] = future_predictions.flatten()
    original_data = np.full(max_length, np.nan)
    original_data[: len(data)] = data.flatten()

    return original_data, train_plot, test_plot, future_plot


def calculate_metrics(original, predicted):
    rmse = np.sqrt(mean_squared_error(original, predicted))
    return rmse
