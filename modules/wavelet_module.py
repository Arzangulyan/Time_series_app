import numpy as np
import pandas as pd
import pywt
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def wavelet_transform(time_series, mother_wavelet, max_scales=100):
    mother_switcher = {"Морле": "morl", "Гаусс": "gaus1", "Мексиканская шляпа": "mexh"}
    freq_switcher = {"Морле": 0.8125, "Гаусс": 0.6, "Мексиканская шляпа": 3}
    
    n_samples = len(time_series)
    scales = np.logspace(np.log10(1), np.log10(n_samples/2), num=min(max_scales, n_samples//2))
    
    if isinstance(time_series, pd.DataFrame):
        data = time_series.iloc[:, 0].values
    elif isinstance(time_series, pd.Series):
        data = time_series.values
    else:
        data = time_series
    
    coef, freqs = pywt.cwt(data, scales, mother_switcher.get(mother_wavelet))
    freqs = np.divide(freqs, freq_switcher.get(mother_wavelet))
    return coef, freqs

def get_scale_ticks(min_period, max_period, num_ticks=6):
    log_min, log_max = np.log2(min_period), np.log2(max_period)
    log_ticks = np.linspace(log_min, log_max, num_ticks)
    return np.exp2(log_ticks)

def format_period(period, unit: str = 'days'):
    if unit == 'days':
        if period < 7:
            return f"{period:.0f} дней"
        elif period < 30:
            return f"{period/7:.1f} недель"
        elif period < 365:
            return f"{period/30:.1f} месяцев"
        else:
            return f"{period/365:.1f} лет"
    else:  # measurements
        return f"{period:.0f} измерений"

def plot_wavelet_transform(time_series, coef, freqs, periods, tickvals, ticktext, scale_unit):
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(coef),
        x=time_series.index,
        y=np.log2(periods),
        colorscale='Viridis',
        colorbar=dict(title='Мощность')
    ))
    
    fig.update_layout(
        title="Вейвлет-преобразование",
        xaxis_title="Время",
        yaxis_title=f"Период ({scale_unit})",
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
    )
    
    return fig