import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller
import App_descriptions_streamlit as txt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback


st.set_page_config(page_title="LSTM")
st.title(
    "Прогнозирование временных рядов методом LSTM"
)

txt.LSTM_descr()

st.sidebar.header("Настройки модели LSTM")
epochs = int(st.sidebar.number_input("Количество эпох", min_value=1))
# batch_size = int(st.sidebar.number_input("Размер батча", min_value=1, value=1)) # пока не роляет никак
# d = st.sidebar.number_input("Порядок дифференцирования d", min_value=0)

txt.LSTM_epochs_choice()


def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 0])


def new_method_start():
    # st.session_state
    # st.session_state.final_dataframe.empty

    if not st.session_state.final_dataframe.empty:
        time_series = st.session_state.final_dataframe
    else:
        st.warning(
            "Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App»"
        )
        st.stop()
    df_chart_display_iloc(time_series)
    return time_series



# def LSTM_ts (signal, time, epochs):
#     # Подготовка данных
#     data = signal.values.reshape(-1, 1)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     train_size = int(len(data_scaled) * 0.8)
#     train, test = data_scaled[:train_size], data_scaled[train_size:]

#     def create_dataset(dataset, look_back=1):
#         X, Y = [], []
#         for i in range(len(dataset) - look_back):
#             X.append(dataset[i:(i + look_back), 0])
#             Y.append(dataset[i + look_back, 0])
#         return np.array(X), np.array(Y)

#     look_back = 3
#     X_train, y_train = create_dataset(train, look_back)
#     X_test, y_test = create_dataset(test, look_back)

#     # Преобразование данных в форму [samples, time steps, features]
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Создание LSTM модели
#     model = Sequential()
#     model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
#     model.add(LSTM(100))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')

#     # Обучение модели
#     model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)

#     # Прогноз
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)

#     # Обратное преобразование прогнозов
#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)

#     # Создание графиков для обучающих и тестовых прогнозов
#     train_plot = np.empty_like(data)
#     train_plot[:, :] = np.nan
#     train_plot[look_back:len(train_predict) + look_back, :] = train_predict

#     test_plot = np.empty_like(data)
#     test_plot[:, :] = np.nan
#     test_plot[len(train_predict) + (look_back * 2): len(data), :] = test_predict

#     # Построение графика
#     plt.figure(figsize=(10, 6))
#     plt.plot(time, data, label='Original Data')
#     plt.plot(time, train_plot, label='Train Predict')
#     plt.plot(time, test_plot, label='Test Predict')

#     plt.xlabel('Date')
#     plt.ylabel('Concentration (ppm)')
#     plt.title('PPM Forecast with LSTM')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

class MetricsCallback(Callback):
    def on_train_begin(self, logs={}):
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_mae': [],
            'test_mae': []
        }

    def on_epoch_end(self, epoch, logs={}):
        # Обновление метрик ошибок после завершения каждой эпохи
        train_predict = self.model.predict(self.model.validation_data[0])
        test_predict = self.model.predict(self.validation_data[0])

        train_mae = mean_absolute_error(self.model.validation_data[1], train_predict)
        test_mae = mean_absolute_error(self.validation_data[1], test_predict)

        self.history['train_loss'].append(logs.get('loss'))
        self.history['test_loss'].append(logs.get('val_loss'))
        self.history['train_mae'].append(train_mae)
        self.history['test_mae'].append(test_mae)

# def LSTM_ts(signal, time, epochs=10):
#     # Подготовка данных
#     data = signal.values.reshape(-1, 1)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     train_size = int(len(data_scaled) * 0.8)
#     train, test = data_scaled[:train_size], data_scaled[train_size:]

#     def create_dataset(dataset, look_back=1):
#         X, Y = [], []
#         for i in range(len(dataset) - look_back):
#             X.append(dataset[i:(i + look_back), 0])
#             Y.append(dataset[i + look_back, 0])
#         return np.array(X), np.array(Y)

#     look_back = 3
#     X_train, y_train = create_dataset(train, look_back)
#     X_test, y_test = create_dataset(test, look_back)

#     # Преобразование данных в форму [samples, time steps, features]
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Создание LSTM модели
#     model = Sequential()
#     model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
#     model.add(LSTM(100))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')

#     # Создание колбэка для отслеживания метрик
#     metrics_callback = MetricsCallback()

#     # Обучение модели
#     model.fit(X_train, y_train,
#               epochs=epochs,
#               batch_size=1,
#               verbose=2,
#               validation_data=(X_test, y_test),
#               callbacks=[metrics_callback])

#     # Прогноз
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)

#     # Обратное преобразование прогнозов
#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)

#     # Создание графиков для обучающих и тестовых прогнозов
#     train_plot = np.empty_like(data)
#     train_plot[:, :] = np.nan
#     train_plot[look_back:len(train_predict) + look_back, :] = train_predict

#     test_plot = np.empty_like(data)
#     test_plot[:, :] = np.nan
#     test_plot[len(train_predict) + (look_back * 2): len(data), :] = test_predict

#     return data, train_plot, test_plot, time, metrics_callback.history



def LSTM_ts(signal, time, epochs=10):
    # Подготовка данных
    data = signal.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 3
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # Преобразование данных в форму [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Создание LSTM модели
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Обучение модели
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Прогноз
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Обратное преобразование прогнозов
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Создание графиков для обучающих и тестовых прогнозов
    train_plot = np.empty_like(data)
    train_plot[:, :] = np.nan
    train_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_plot = np.empty_like(data)
    test_plot[:, :] = np.nan
    test_plot[len(train_predict) + (look_back * 2): len(data), :] = test_predict

    return data, train_plot, test_plot, time




time_series = new_method_start()

signal = time_series.iloc[:, 0]
time = np.arange(0, time_series.shape[0])

data, train_plot, test_plot, time = LSTM_ts(signal, time, epochs)
# data, train_plot, test_plot, time, metrics_history = LSTM_ts(signal, time, epochs=10)



# Преобразование данных в DataFrame для отображения в Streamlit
results_df = pd.DataFrame({
    'Date': time,
    'Original Data': data.flatten(),
    'Train Predict': train_plot.flatten(),
    'Test Predict': test_plot.flatten()
})

# epochs_range = range(len(metrics_history['train_loss']))
# error_df = pd.DataFrame({
#     'Epochs': epochs_range.flatten(),
#     'Error_MSE': metrics_history['train_loss'].flatten()
# })

results_df.set_index('Date', inplace=True)

st.subheader('LSTM Predictions')
st.line_chart(results_df)


# data = pd.concat([fft_df, power_df])

# alt_chart = alt.Chart(data).mark_line().encode(x="Частота", y="Амплитуда", color="Тип")

# st.altair_chart(alt_chart, use_container_width=True)
