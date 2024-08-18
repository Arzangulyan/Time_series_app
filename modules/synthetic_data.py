import numpy as np
import pandas as pd
from .utils import DF_wrapper

from streamlit import write


def generate_synthetic_time_series(
    date_range: pd.DatetimeIndex,
    SIN_1_coef: float = 0.0,
    SIN_1_period: float = 0.0,
    SIN_2_coef: float = 0.0,
    SIN_2_period: float = 0.0,
    SIN_3_coef: float = 0.0,
    SIN_3_period: float = 0.0,
    COS_1_coef: float = 0.0,
    COS_1_period: float = 0.0,
    COS_2_coef: float = 0.0,
    COS_2_period: float = 0.0,
    COS_3_coef: float = 0.0,
    COS_3_period: float = 0.0,
    Noise_Coef: float = 0.0,
    Trend_Slope: float = 0.0,
) -> pd.DataFrame:
    """
    Генерирует синтетический временной ряд.

    Args:
        date_range (pd.DatetimeIndex): Диапазон дат для временного ряда.
        SIN_1_coef, SIN_2_coef, SIN_3_coef (float): Коэффициенты амплитуды для синусоид.
        SIN_1_period, SIN_2_period, SIN_3_period (float): Периоды для синусоид.
        COS_1_coef, COS_2_coef, COS_3_coef (float): Коэффициенты амплитуды для косинусоид.
        COS_1_period, COS_2_period, COS_3_period (float): Периоды для косинусоид.
        NoiseCoef (float): Коэффициент шума.
        Trend_Slope (float): Наклон линейного тренда.

    Returns:
        pd.DataFrame: DataFrame с синтетическим временным рядом.
    """
    length = len(date_range)
    t = np.linspace(0, length - 1, length)

    sinusoids = np.zeros(length)
    if SIN_1_period != 0:
        sinusoids += SIN_1_coef * np.sin(2 * np.pi * t / SIN_1_period)
    if SIN_2_period != 0:
        sinusoids += SIN_2_coef * np.sin(2 * np.pi * t / SIN_2_period)
    if SIN_3_period != 0:
        sinusoids += SIN_3_coef * np.sin(2 * np.pi * t / SIN_3_period)

    cosines = np.zeros(length)
    if COS_1_period != 0:
        cosines += COS_1_coef * np.cos(2 * np.pi * t / COS_1_period)
    if COS_2_period != 0:
        cosines += COS_2_coef * np.cos(2 * np.pi * t / COS_2_period)
    if COS_3_period != 0:
        cosines += COS_3_coef * np.cos(2 * np.pi * t / COS_3_period)

    trend = Trend_Slope * t if Trend_Slope != 0 else np.zeros(length)
    noise = (
        np.random.normal(0, Noise_Coef, length) if Noise_Coef != 0 else np.zeros(length)
    )

    time_series = trend + sinusoids + cosines + noise

    return pd.DataFrame({"Время": date_range, "Ряд": time_series})
