import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller
import App_descriptions_streamlit as txt

st.set_page_config(page_title="Fast Fourier Transform")
st.title(
    "Выделение сезонностей во временных рядах с помощью Быстрого Фурье Преобразования"
)

st.sidebar.header("Настройки Фурье преобразования")

txt.Fourier_descr()

def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 1])


def new_method_start():
    # st.session_state
    # st.session_state.final_dataframe.empty

    if not st.session_state.time_series.empty:
        time_series = st.session_state.time_series
    else:
        st.warning(
            "Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App»"
        )
        st.stop()
    df_chart_display_iloc(time_series)
    return time_series


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0 : N // 2])
    return f_values, fft_values


# def plot_fft_plus_power(time, signal):
#     dt = time[1] - time[0]
#     N = len(signal)
#     fs = 1 / dt

#     # fig, ax = plt.subplots(figsize=(5, 3))
#     fig, ax = plt.subplots()
#     variance = np.std(signal) ** 2
#     f_values, fft_values = get_fft_values(signal, dt, N, fs)
#     fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
#     ax.plot(f_values, fft_values, "r-", label="Фурье преобразование")
#     ax.plot(f_values, fft_power, "k--", linewidth=1, label="FFT Power Spectrum")
#     ax.set_xlabel("Частота", fontsize=18)
#     ax.set_ylabel("Амплитуда", fontsize=18)
#     ax.legend()
#     st.pyplot(fig)


def fft_plus_power_dataframe(time, signal):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1 / dt

    variance = np.std(signal) ** 2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum

    fft_df = pd.DataFrame(
        {"Частота": f_values, "Амплитуда": fft_values, "Тип": "Фурье преобразование"}
    )
    power_df = pd.DataFrame(
        {"Частота": f_values, "Амплитуда": fft_power, "Тип": "FFT Power Spectrum"}
    )

    return fft_df, power_df


time_series = new_method_start()

signal = time_series.iloc[:, 1]
time = np.arange(0, time_series.shape[0])
fft_df, power_df = fft_plus_power_dataframe(time, signal)
data = pd.concat([fft_df, power_df])

alt_chart = alt.Chart(data).mark_line().encode(x="Частота", y="Амплитуда", color="Тип")

st.altair_chart(alt_chart, use_container_width=True)
