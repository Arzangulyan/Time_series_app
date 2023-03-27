import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

st.title("Прогнозирование временных рядов с использованием ARMA")

uploaded_file = st.file_uploader("Загрузите файл CSV с данными временных рядов", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.write("Ваши загруженные данные:")
    st.line_chart(data)

    st.sidebar.title("Параметры модели ARMA")
    p = st.sidebar.slider("Параметр AR (p)", 1, 10, 1)
    q = st.sidebar.slider("Параметр MA (q)", 1, 10, 1)

    try:
        model = ARIMA(data, order=(p, 0, q)).fit()
        st.write(f"Оценка AIC для модели ARMA({p}, {q}): {model.aic}")

        forecast_steps = st.sidebar.number_input(
            "Количество шагов прогнозирования в будущее", 1, 100, 5
        )
        forecast = model.forecast(steps=forecast_steps)

        st.write("Прогноз временного ряда:")
        st.line_chart(pd.concat([data, forecast]))

    except ValueError as e:
        st.write("Не удалось обучить модель ARMA. Проверьте параметры и данные.")
        st.write("Ошибка:", e)