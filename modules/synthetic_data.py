import numpy as np
import pandas as pd
from .utils import DF_wrapper

from streamlit import write


def generate_synthetic_time_series(
    date_range: pd.DatetimeIndex,
    S_1_coef: float = 0.0,
    S_1_freq: float = 0.0,
    S_2_coef: float = 0.0,
    S_2_freq: float = 0.0,
    S_3_coef: float = 0.0,
    S_3_freq: float = 0.0,
    C_1_coef: float = 0.0,
    C_1_freq: float = 0.0,
    C_2_coef: float = 0.0,
    C_2_freq: float = 0.0,
    C_3_coef: float = 0.0,
    C_3_freq: float = 0.0,
    NoiseCoef: float = 0.0,
    TrendSlope: float = 0.0,
) -> pd.DataFrame:
    """
    Генерирует синтетический временной ряд.

    Args:
        date_range (pd.DatetimeIndex): Диапазон дат для временного ряда.
        S_1_coef, S_2_coef, S_3_coef (float): Коэффициенты амплитуды для синусоид.
        S_1_freq, S_2_freq, S_3_freq (float): Частоты для синусоид.
        C_1_coef, C_2_coef, C_3_coef (float): Коэффициенты амплитуды для косинусоид.
        C_1_freq, C_2_freq, C_3_freq (float): Частоты для косинусоид.
        NoiseCoef (float): Коэффициент шума.
        TrendSlope (float): Наклон линейного тренда.

    Returns:
        pd.DataFrame: DataFrame с синтетическим временным рядом.
    """
    length = len(date_range)

    t = np.linspace(0, length - 1, length)

    sinusoids = (
        S_1_coef * np.sin(2 * np.pi * t * S_1_freq)
        + S_2_coef * np.sin(2 * np.pi * t * S_2_freq)
        + S_3_coef * np.sin(2 * np.pi * t * S_3_freq)
    )

    cosines = (
        C_1_coef * np.cos(2 * np.pi * t * C_1_freq)
        + C_2_coef * np.cos(2 * np.pi * t * C_2_freq)
        + C_3_coef * np.cos(2 * np.pi * t * C_3_freq)
    )

    trend = TrendSlope * t
    noise = np.random.normal(0, NoiseCoef, length)

    time_series = trend + sinusoids + cosines + noise
    print("date_range:", date_range)
    print("time_series shape:", time_series.shape)
    print("time_series first few values:", time_series[:5])
    # time_series = DF_wrapper(date_range, pd.Series(time_series))
    # return time_series
    return pd.DataFrame({"Время": date_range, "Ряд": time_series})
