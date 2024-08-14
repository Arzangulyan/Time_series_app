import numpy as np
import pandas as pd
from .utils import DF_wrapper

from streamlit import write

def generate_synthetic_time_series(
    date_range: pd.DatetimeIndex,
    S_1_coef: float = 0.0,
    S_1_period: float = 1.0,
    S_2_coef: float = 0.0,
    S_2_period: float = 1.0,
    S_3_coef: float = 0.0,
    S_3_period: float = 1.0,
    C_1_coef: float = 0.0,
    C_1_period: float = 1.0,
    C_2_coef: float = 0.0,
    C_2_period: float = 1.0,
    C_3_coef: float = 0.0,
    C_3_period: float = 1.0,
    NoiseCoef: float = 0.0,
    TrendSlope: float = 0.0,
) -> pd.DataFrame:
    """
    Генерирует синтетический временной ряд.

    Args:
        date_range (pd.DatetimeIndex): Диапазон дат для временного ряда.
        S_1_coef, S_2_coef, S_3_coef (float): Коэффициенты амплитуды для синусоид.
        S_1_period, S_2_period, S_3_period (float): Периоды для синусоид.
        C_1_coef, C_2_coef, C_3_coef (float): Коэффициенты амплитуды для косинусоид.
        C_1_period, C_2_period, C_3_period (float): Периоды для косинусоид.
        NoiseCoef (float): Коэффициент шума.
        TrendSlope (float): Наклон линейного тренда.

    Returns:
        pd.DataFrame: DataFrame с синтетическим временным рядом.
    """
    length = len(date_range)
    t = np.linspace(0, length - 1, length)

    sinusoids = (
        S_1_coef * np.sin(2 * np.pi * t / S_1_period) if S_1_coef != 0 else 0
        + S_2_coef * np.sin(2 * np.pi * t / S_2_period) if S_2_coef != 0 else 0
        + S_3_coef * np.sin(2 * np.pi * t / S_3_period) if S_3_coef != 0 else 0
    )

    cosines = (
        C_1_coef * np.cos(2 * np.pi * t / C_1_period) if C_1_coef != 0 else 0
        + C_2_coef * np.cos(2 * np.pi * t / C_2_period) if C_2_coef != 0 else 0
        + C_3_coef * np.cos(2 * np.pi * t / C_3_period) if C_3_coef != 0 else 0
    )

    trend = TrendSlope * t if TrendSlope != 0 else 0
    noise = np.random.normal(0, NoiseCoef, length) if NoiseCoef != 0 else 0

    time_series = trend + sinusoids + cosines + noise
    return pd.DataFrame({"Время": date_range, "Ряд": time_series})