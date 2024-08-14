import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def perform_statistical_analysis(time_series):
    """
    Выполняет комплексный статистический анализ временного ряда.

    Args:
    time_series (pd.Series): Временной ряд для анализа.

    Returns:
    dict: Словарь с результатами различных статистических тестов и декомпозиции.
    """
    basic_stats = calculate_basic_stats(time_series)
    stationarity = check_stationarity(time_series)
    acf_values = calculate_autocorrelation(time_series)
    pacf_values = calculate_partial_autocorrelation(time_series)

    return {
        "basic_stats": basic_stats,
        "stationarity": stationarity,
        "acf": acf_values,
        "pacf": pacf_values,
    }


def calculate_basic_stats(time_series):
    """
    Рассчитывает базовые статистические характеристики временного ряда.

    Args:
    time_series (pd.Series): Временной ряд для анализа.

    Returns:
    pd.Series: Серия с базовыми статистическими показателями.
    """
    stats = {
        "Среднее": time_series.mean(),
        "Медиана": time_series.median(),
        "Стандартное отклонение": time_series.std(),
        "Минимум": time_series.min(),
        "Максимум": time_series.max(),
        "Первый квартиль": time_series.quantile(0.25),
        "Третий квартиль": time_series.quantile(0.75),
    }
    return pd.Series(stats)


def check_stationarity(time_series):
    """
    Проверяет стационарность временного ряда с помощью теста Дики-Фуллера.

    Args:
    time_series (pd.Series): Временной ряд для проверки.

    Returns:
    dict: Словарь с результатами теста Дики-Фуллера.
    """
    result = adfuller(time_series.dropna())
    return {
        "Тестовая статистика": result[0],
        "p-значение": result[1],
        "Критические значения": result[4],
    }


def calculate_autocorrelation(time_series, lags=40):
    """
    Рассчитывает автокорреляцию временного ряда.

    Args:
    time_series (pd.Series): Временной ряд для анализа.
    lags (int): Количество лагов для расчета автокорреляции.

    Returns:
    np.array: Массив значений автокорреляции.
    """
    return acf(time_series.dropna(), nlags=lags)


def calculate_partial_autocorrelation(time_series, lags=40):
    """
    Рассчитывает частичную автокорреляцию временного ряда.

    Args:
    time_series (pd.Series): Временной ряд для анализа.
    lags (int): Количество лагов для расчета частичной автокорреляции.

    Returns:
    np.array: Массив значений частичной автокорреляции.
    """
    return pacf(time_series.dropna(), nlags=lags)


def decompose_time_series(time_series, period):
    """
    Разлагает временной ряд на тренд, сезонность и остаток.

    Args:
        time_series (pd.Series): Временной ряд для анализа.
        period (int): Период сезонности, задаваемый пользователем.

    Returns:
        DecomposeResult: Объект с результатами декомпозиции.
    """
    result = seasonal_decompose(time_series, period=period)
    return result


def plot_decomposition(decomposition_result):
    """
    Строит график декомпозиции временного ряда.

    Args:
    decomposition_result (DecomposeResult): Результат декомпозиции временного ряда.

    Returns:
    matplotlib.figure.Figure: Объект фигуры с графиками декомпозиции.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    decomposition_result.observed.plot(ax=ax1)
    ax1.set_title("Исходный ряд")
    decomposition_result.trend.plot(ax=ax2)
    ax2.set_title("Тренд")
    decomposition_result.seasonal.plot(ax=ax3)
    ax3.set_title("Сезонность")
    decomposition_result.resid.plot(ax=ax4)
    ax4.set_title("Остаток")
    plt.tight_layout()
    return fig


def display_statistical_analysis(time_series, period=None):
    """
    Отображает результаты статистического анализа временного ряда в Streamlit.

    Args:
    time_series (pd.DataFrame): DataFrame с временным рядом. Предполагается наличие столбца 'value'.
    """
    st.subheader("Статистический анализ временного ряда")

    # Расчет и отображение базовой статистики
    basic_stats = calculate_basic_stats(time_series["value"])
    st.write("Базовая статистика:")
    st.write(basic_stats)

    # Проверка стационарности и отображение результатов
    stationarity = check_stationarity(time_series["value"])
    st.write("Результаты теста на стационарность (тест Дики-Фуллера):")
    st.write(stationarity)

    # Расчет и отображение автокорреляции и частичной автокорреляции
    st.subheader("Автокорреляция и частичная автокорреляция")
    acf = calculate_autocorrelation(time_series["value"])
    pacf = calculate_partial_autocorrelation(time_series["value"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(acf)
    ax1.set_title("Автокорреляция")
    ax2.plot(pacf)
    ax2.set_title("Частичная автокорреляция")
    st.pyplot(fig)
