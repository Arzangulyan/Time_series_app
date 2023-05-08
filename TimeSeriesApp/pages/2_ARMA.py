import streamlit as st
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARMA")

st.title("Прогнозирование временных рядов с использованием ARMA")
st.sidebar.header("Настройки ARMA")
# uploaded_file = st.file_uploader("Загрузите файл CSV с данными временных рядов", type="csv")


def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 1])


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
    "Загруженный ряд"
    df_chart_display_iloc(time_series)
    return time_series


time_series = new_method_start()

if time_series is not None:
    data = time_series.iloc[:, 1]
    st.line_chart(data)

    len(data)

    st.sidebar.title("Параметры модели ARMA")
    p = st.sidebar.number_input("Параметр AR (p)", min_value=1)
    q = st.sidebar.number_input("Параметр MA (q)", min_value=1)

    try:
        model = ARIMA(data, order=(p, 0, q)).fit()
        # ПОКА НЕ РАЗБЕРУСЬ, НЕ ВОЗВРАЩАТЬ ЭТО В КОД
        # st.write(f"Оценка AIC для модели ARMA({p}, {q}): {model.aic}")

        forecast_steps = st.sidebar.number_input(
            "Количество шагов прогнозирования в будущее", min_value=1, value=5
        )
        forecast = model.forecast(steps=forecast_steps)

        st.write("Прогноз временного ряда:")
        st.line_chart(pd.concat([data, forecast]))

    except ValueError as e:
        st.write("Не удалось обучить модель ARMA. Проверьте параметры и данные.")
        st.write("Ошибка:", e)
