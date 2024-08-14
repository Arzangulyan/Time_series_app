import pandas as pd
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from config import STATIONARITY_THRESHOLD


def smooth_time_series(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=1).mean()


def check_stationarity(data):
    stat_test_res = adfuller(data)[1]
    st.sidebar.write("Результаты теста на стационарность (p-value): ", stat_test_res)
    if stat_test_res < STATIONARITY_THRESHOLD:
        st.sidebar.write("Ряд стационарен")
    else:
        st.sidebar.write("Ряд НЕ стационарен")
    return stat_test_res


# TODO: Добавить сюда другие функции обработки данных
