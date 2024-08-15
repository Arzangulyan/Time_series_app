import numpy as np
import pandas as pd
import streamlit as st

def get_fft_values(y_values, fs):
    N = len(y_values)
    fft_values = np.fft.rfft(y_values)
    fft_freq = np.fft.rfftfreq(N, d=1/fs)
    fft_amplitude = np.abs(fft_values) / N  # Нормализация амплитуды
    return fft_freq, fft_amplitude

@st.cache_data
def fft_plus_power_dataframe(time, signal, nfft=None):
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0].values
    elif isinstance(signal, pd.Series):
        signal = signal.values
    elif not isinstance(signal, np.ndarray):
        raise ValueError("signal должен быть pandas DataFrame, Series или numpy array")

    dt = (time[-1] - time[0]) / (len(time) - 1)  # Средний интервал времени
    fs = 1 / dt  # Частота дискретизации

    if nfft is not None and nfft > len(signal):
        signal = np.pad(signal, (0, nfft - len(signal)), 'constant')
    elif nfft is not None:
        signal = signal[:nfft]

    f_values, fft_values = get_fft_values(signal, fs)
    fft_power = fft_values ** 2

    # Расчет периодов
    periods = 1 / f_values
    periods[0] = np.inf  # Избегаем деления на ноль

    fft_df = pd.DataFrame({
        "Частота": f_values,
        "Период": periods,
        "Амплитуда": fft_values,
        "Тип": "Фурье преобразование"
    })
    power_df = pd.DataFrame({
        "Частота": f_values,
        "Период": periods,
        "Амплитуда": fft_power,
        "Тип": "FFT Power Spectrum"
    })

    return fft_df, power_df

def apply_window(signal, window_type='hamming'):
    if isinstance(signal, (pd.Series, pd.DataFrame)):
        signal = signal.values
    
    if window_type == 'hamming':
        window = np.hamming(len(signal))
    elif window_type == 'hann':
        window = np.hanning(len(signal))
    elif window_type == 'blackman':
        window = np.blackman(len(signal))
    else:
        window = np.ones(len(signal))
    return signal * window

def next_power_of_2(n):
    return 2**np.ceil(np.log2(n)).astype(int)