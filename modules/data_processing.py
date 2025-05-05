import pandas as pd
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from config import STATIONARITY_THRESHOLD


def smooth_time_series(df: pd.DataFrame, window: int) -> pd.DataFrame:
    smoothed = df.rolling(window=window, min_periods=window).mean()
    return smoothed.dropna()




def check_stationarity(data):
    result = adfuller(data)
    stat_test_res = result[1]  # p-значение
    
    # Выводим результаты в основной интерфейс вместо сайдбара
    st.write("Результаты теста на стационарность (p-value): ", stat_test_res)
    if stat_test_res < STATIONARITY_THRESHOLD:
        st.success("Ряд стационарен")
    else:
        st.warning("Ряд НЕ стационарен")
        
    # Добавляем более детальную информацию о тесте
    st.expander("Подробные результаты теста Дики-Фуллера").write({
        "p-значение": stat_test_res,
        "Тестовая статистика": result[0],
        "Критические значения": result[4] if len(result) > 4 else "Недоступно"
    })
    
    return stat_test_res


# TODO: Добавить сюда другие функции обработки данных
