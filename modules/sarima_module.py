import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

def plot_acf_pacf(data, lags=1000):
    acf_values = acf(data, nlags=lags)
    pacf_values = pacf(data, nlags=lags)
    
    confidence = 1.96 / np.sqrt(len(data))
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('ACF', 'PACF'))
    
    fig.add_trace(
        go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, lags], y=[confidence, confidence], mode='lines', line=dict(color='red', dash='dash'), name='Upper CI'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, lags], y=[-confidence, -confidence], mode='lines', line=dict(color='red', dash='dash'), name='Lower CI'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, lags], y=[confidence, confidence], mode='lines', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, lags], y=[-confidence, -confidence], mode='lines', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(height=600, width=800, title_text="ACF and PACF Plots")
    return fig

def calculate_acf_pacf(data):
    acf_values = acf(data, nlags=100)
    pacf_values = pacf(data, nlags=100)
    acf_df = pd.DataFrame({'Lag': np.arange(len(acf_values)), 'Value': acf_values})
    pacf_df = pd.DataFrame({'Lag': np.arange(len(pacf_values)), 'Value': pacf_values})
    return acf_df, pacf_df

@st.cache_data
def sarima_processing(data, p, d, q, P, D, Q, s):
    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()
    return model

@st.cache_data
def forecast_sarima(model, forecast_steps):
    forecast = model.forecast(steps=forecast_steps)
    return pd.DataFrame(forecast, columns=['mean'])

def apply_differencing(data, d, D=0, s=1):
    diff_data = data.diff(d).dropna()
    if D > 0:
        diff_data = diff_data.diff(s*D).dropna()
    return diff_data