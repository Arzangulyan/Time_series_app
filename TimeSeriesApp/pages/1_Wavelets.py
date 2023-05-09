import streamlit as st
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Wavelets")
st.title("Выделение сезонностей во временных рядах с помощью Вейвлет преобразования")

st.sidebar.header("Настройки вейвлетов")


def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 1])

@st.cache_data
def wavelet_transform(time_series, mother_wavelet):
    mother_switcher = {"Морле": "morl", "Гаусс": "gaus7", "Мексиканская шляпа": "mexh"}
    freq_switcher = {
        "Морле": 0.8125,
        "Гаусс": 0.6,
        "Мексиканская шляпа": 3,
    }  # Здесь хранятся собственные частоты для каждого вейвлета
    n_samples = len(time_series.iloc[:, 1])
    scales = np.logspace(0, np.log10(n_samples / 4), num=n_samples // 4)

    coef, freqs = pywt.cwt(
        time_series.iloc[:, 1], scales, mother_switcher.get(mother_wavelet)
    )

    return coef, np.divide(
        freqs, freq_switcher.get(mother_wavelet)
    )  # Делим частоты на собственную частоту

    # coef, freqs = pywt.cwt(time_series.iloc[:, 1], np.arange(1, len(time_series)/4), mother_switcher.get(mother_wavelet))
    # return coef, freqs


# def plot_wavelet_transform(time_series, wavelet_select):
#     coef, freq = wavelet_transform(time_series, wavelet_select)

#     fig, ax = plt.subplots()
#     ax.imshow(
#         coef,
#         cmap="copper",
#         aspect="auto",
#         extent=[0, len(time_series.iloc[:, 1]), freq[-1], freq[0]],
#     )
#     ax.set_title("Power Spectrum", fontsize=20)
#     ax.set_ylabel("Период", fontsize=18)
#     ax.set_xlabel("Время", fontsize=18)
#     ax.invert_yaxis()

#     return fig


def plot_wavelet_transform(time_series, wavelet_select):
    coef, freq = wavelet_transform(time_series, wavelet_select)

    time_indexes = time_series.iloc[:, 0].to_list()  # Преобразуем индексы даты и времени в список
    n_samples = len(time_series.iloc[:, 1])

    # Преобразуем datetime объекты в числовые значения
    time_num = mdates.date2num(time_indexes)

    # Вычисляем интервал времени между двумя соседними точками
    dt = time_indexes[1] - time_indexes[0]
    dt_seconds = dt.total_seconds()

    # Преобразуем частоты вейвлет-преобразования в периоды (время)
    periods = 1 / (freq * dt_seconds)

    fig, ax = plt.subplots()
    ax.imshow(
        coef,
        cmap="copper",
        aspect="auto",
        extent=[time_num[0], time_num[-1], periods[-1], periods[0]],
    )
    ax.set_title("Power Spectrum", fontsize=20)
    ax.set_ylabel("Период", fontsize=18)
    ax.set_xlabel("Время", fontsize=18)
    ax.invert_yaxis()

    # Разметка оси X с указанными временными метками
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate(rotation=45, ha='right')

    return fig


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


time_series = new_method_start()


wavelet_select = st.sidebar.selectbox(
    label="Выберите материнский вейвлет",
    options=(["", "Морле", "Гаусс", "Мексиканская шляпа"]),
)

# st.stop()
if wavelet_select == "":
    st.stop()

# #TEST
# x = np.linspace(0, 1000,1000)
# y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900)
# linechart = st.line_chart(y)
# coef, freqs = pywt.cwt(y, np.arange(1,129), 'mexh')
# fig, ax = plt.subplots()
# ax.imshow(coef, cmap = 'copper', aspect = 'auto')
# st.pyplot(fig)
# # sns.heatmap(coef, ax = ax, cmap = 'copper')
# # st.write(fig)
#
# I = np.empty((len(freqs)))
# for j in range(len(freqs)-1):
#     for i in range(len(y)):
#         I[j] += ((coef[j, i])**2 + (coef[j+1,i])**2)/2
# # st.write(I)
# I_s = pd.DataFrame({'I':I, 'Freqs': freqs})
# st.write(I_s)
# fig2, ax2 = plt.subplots()
# ax2.plot(freqs, I)
# ax2.set_aspect('auto')
# st.pyplot(fig2)
#
# # Intergral_spectrum = st.line_chart(I_s)
# #TEST

# REAL

# coef, freq = wavelet_transform(time_series, wavelet_select)
# fig1, ax1 = plt.subplots()
# ax1.imshow(coef, cmap = 'copper', aspect = 'auto')
# # sns.heatmap(coef, ax = ax, cmap = 'copper')
# # st.write(fig)
# ax1.set_title("Power Spectrum", fontsize=20)
# ax1.set_ylabel("Период", fontsize=18)
# ax1.set_xlabel("Время", fontsize=18)
# ax1.invert_yaxis()
# st.pyplot(fig1)

fig1 = plot_wavelet_transform(time_series, wavelet_select)
st.pyplot(fig1)


# I = np.empty((len(freqs)))
# for j in range(len(freqs)-1):
#     for i in range(len(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])-1):
#         I[j] += ((coef[j, i]) + (coef[j+1,i]))/2
# # st.write(I)
# Int_freq = pd.DataFrame({'I':I, 'Freqs': freqs})
# st.write(Int_freq)
# fig2, ax2 = plt.subplots()
# ax2.plot(freqs, I)
# ax2.set_aspect('auto')
# plt.xscale("log")
# st.pyplot(fig2)
# REAL
