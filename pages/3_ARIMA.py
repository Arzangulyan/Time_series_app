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
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import App_descriptions_streamlit as txt

st.set_page_config(page_title="ARIMA")

st.title("Прогнозирование временных рядов с использованием ARIMA")
st.sidebar.header("Настройки ARIMA")
# uploaded_file = st.file_uploader("Загрузите файл CSV с данными временных рядов", type="csv")

txt.ARIMA_descr()

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
    "Загруженный ряд"
    df_chart_display_iloc(time_series)
    return time_series


time_series = new_method_start()


# if time_series is not None:
data = time_series.iloc[:, 0]

# Вычисление значений ACF и PACF
def ACF_PACF_plot(data):
    acf_values = acf(data, nlags = 100)
    pacf_values = pacf(data, nlags = 100)

    # Создание датафреймов для ACF и PACF
    acf_df = pd.DataFrame({"Lag": np.arange(len(acf_values)), "Value": acf_values})
    pacf_df = pd.DataFrame({"Lag": np.arange(len(pacf_values)), "Value": pacf_values})

    # Построение графиков ACF и PACF с использованием Altair
    acf_chart = (
        alt.Chart(acf_df)
        .mark_bar()
        .encode(
            x="Lag",
            y="Value",
        )
        .properties(title="Autocorrelation Function (ACF)")
    )

    pacf_chart = (
        alt.Chart(pacf_df)
        .mark_bar()
        .encode(
            x="Lag",
            y="Value",
        )
        .properties(title="Partial Autocorrelation Function (PACF)")
    )

    st.altair_chart(acf_chart)
    st.altair_chart(pacf_chart)

ACF_PACF_plot(data)

st.line_chart(data)

# d = st.sidebar.number_input("Порядок дифференцирования d", min_value=0)
diff_b = st.checkbox('дифф')
if diff_b == 1:
    data_diff = data.diff().dropna()
    st.write(adfuller(data_diff))
    ACF_PACF_plot(data_diff)
    st.line_chart(data_diff)
    diff_2_b = st.checkbox('дифф_2')
    if diff_2_b == 1:
        data_diff_2 = data_diff.diff().dropna()
        st.write(adfuller(data_diff_2))
        ACF_PACF_plot(data_diff_2)
        st.line_chart(data_diff_2)





# len(data)


st.sidebar.title("Параметры модели ARIMA")
p = st.sidebar.number_input("Параметр AR (p)", min_value=1)
q = st.sidebar.number_input("Параметр MA (q)", min_value=1)
d = st.sidebar.number_input("Порядок дифференцирования d", min_value=0)




@st.cache_data
def arima_proccesing(data, p, d, q):
    model = ARIMA(data, order=(p, d, q)).fit()
    return model


try:
    # model = ARIMA(data, order=(p, 0, q)).fit()
    model = arima_proccesing(data, p, d, q)
    # ПОКА НЕ РАЗБЕРУСЬ, НЕ ВОЗВРАЩАТЬ ЭТО В КОД
    # st.write(f"Оценка AIC для модели ARMA({p}, {q}): {model.aic}")

    forecast_steps = st.sidebar.number_input(
        "Количество шагов прогнозирования в будущее", min_value=1, value=5
    )
    forecast = model.forecast(steps=forecast_steps)

    st.write("Прогноз временного ряда:")
    ARIMA_df = pd.DataFrame()
    # ARMA_df['Время'] = time_series.iloc[:,0]
    ARIMA_df["Прогноз"] = pd.concat([data, forecast])
    ARIMA_df["Исходные данные"] = data
    st.line_chart(ARIMA_df)

except ValueError as e:
    st.write("Не удалось обучить модель ARIMA. Проверьте параметры и данные.")
    st.write("Ошибка:", e)
