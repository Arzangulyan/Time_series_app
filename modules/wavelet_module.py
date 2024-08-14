import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st


# Вейвлет преобразование / 1_Wavelets.py
def plot_wavelet_transform(time_series, mother_wavelet):
    coef, freq = wavelet_transform(time_series, mother_wavelet)

    time_indexes = (
        time_series.index.to_list()
    )  # Преобразуем индексы даты и времени в список
    n_samples = len(time_series.iloc[:, 0])

    # Преобразуем datetime объекты в числовые значения
    time_num = mdates.date2num(time_indexes)

    # Вычисляем интервал времени между двумя соседними точками
    dt = time_indexes[1] - time_indexes[0]
    dt_seconds = dt.total_seconds()

    # Преобразуем частоты вейвлет-преобразования в периоды (время)
    periods = 1 / (freq / dt_seconds)

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M"))
    fig.autofmt_xdate(rotation=45, ha="right")

    return fig


@st.cache_data
def wavelet_transform(time_series, mother_wavelet):
    mother_switcher = {"Морле": "morl", "Гаусс": "gaus7", "Мексиканская шляпа": "mexh"}
    freq_switcher = {
        "Морле": 0.8125,
        "Гаусс": 0.6,
        "Мексиканская шляпа": 3,
    }  # Здесь хранятся собственные частоты для каждого вейвлета
    n_samples = len(time_series.iloc[:, 0])
    scales = np.logspace(np.log10(30 / 365), np.log10(n_samples), num=n_samples)

    coef, freqs = pywt.cwt(
        time_series.iloc[:, 0], scales, mother_switcher.get(mother_wavelet)
    )
    # Делим частоты на собственную частоту
    return coef, np.divide(freqs, freq_switcher.get(mother_wavelet))
