import numpy as np
import pandas as pd
import altair as alt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st


def calculate_acf_pacf(data):
    acf_values = acf(data)
    pacf_values = pacf(data)
    acf_df = pd.DataFrame({"Lag": np.arange(len(acf_values)), "Value": acf_values})
    pacf_df = pd.DataFrame({"Lag": np.arange(len(pacf_values)), "Value": pacf_values})
    return acf_df, pacf_df


def plot_acf_pacf(acf_df, pacf_df):
    acf_chart = (
        alt.Chart(acf_df)
        .mark_bar()
        .encode(x="Lag", y="Value")
        .properties(title="Autocorrelation Function (ACF)")
    )
    pacf_chart = (
        alt.Chart(pacf_df)
        .mark_bar()
        .encode(x="Lag", y="Value")
        .properties(title="Partial Autocorrelation Function (PACF)")
    )
    return pacf_chart, acf_chart


@st.cache_data
def arma_processing(data, p, q):
    model = ARIMA(data, order=(p, 0, q)).fit()
    st.write("Model Summary:")
    st.text(model.summary())
    return model


def forecast_arma(model, steps):
    forecast = model.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame()
    return forecast_df
