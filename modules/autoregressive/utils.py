"""
Вспомогательные функции для работы с временными рядами и авторегрессионными моделями.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
import warnings
from datetime import datetime, timedelta


def check_input_series(series: Union[pd.Series, pd.DataFrame, np.ndarray, list]) -> pd.Series:
    """
    Проверяет входные данные и преобразует их в pandas.Series, если это возможно.
    
    Параметры:
    -----------
    series : Union[pd.Series, pd.DataFrame, np.ndarray, list]
        Входные данные временного ряда
        
    Возвращает:
    -----------
    pd.Series
        Временной ряд в формате pandas.Series
        
    Вызывает:
    -----------
    TypeError
        Если входные данные не могут быть преобразованы в pandas.Series
    ValueError
        Если входные данные содержат невалидные значения
    """
    if isinstance(series, pd.Series):
        return series
    
    elif isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            return series.iloc[:, 0]
        else:
            # Если DataFrame имеет несколько столбцов, используем первый
            warnings.warn("DataFrame содержит несколько столбцов. Будет использован только первый столбец.")
            return series.iloc[:, 0]
    
    elif isinstance(series, (np.ndarray, list)):
        # Преобразуем в Series с индексом по умолчанию
        return pd.Series(series)
    
    else:
        raise TypeError("Входные данные должны быть типа pandas.Series, pandas.DataFrame, numpy.ndarray или list")


def validate_time_series(series: pd.Series) -> Tuple[bool, str]:
    """
    Проверяет временной ряд на наличие проблем (пропущенные значения, нечисловые значения).
    
    Параметры:
    -----------
    series : pd.Series
        Проверяемый временной ряд
        
    Возвращает:
    -----------
    Tuple[bool, str]
        (True, "") если временной ряд валиден, иначе (False, сообщение об ошибке)
    """
    # Проверка на наличие пропущенных значений
    if series.isna().any():
        return False, "Временной ряд содержит пропущенные значения (NaN)"
    
    # Проверка на нечисловые значения
    if not pd.api.types.is_numeric_dtype(series):
        return False, "Временной ряд содержит нечисловые значения"
    
    # Проверка на длину ряда
    if len(series) < 10:
        return False, "Временной ряд слишком короткий для анализа (минимум 10 наблюдений)"
    
    # Проверка на наличие бесконечных значений
    if np.isinf(series).any():
        return False, "Временной ряд содержит бесконечные значения (Inf)"
    
    # Проверка на наличие слишком больших или маленьких значений
    if (series.abs() > 1e15).any():
        return False, "Временной ряд содержит чрезвычайно большие значения"
    
    return True, ""


def convert_to_datetime_index(series: pd.Series) -> pd.Series:
    """
    Преобразует индекс временного ряда в DatetimeIndex, если это возможно.
    
    Параметры:
    -----------
    series : pd.Series
        Входной временной ряд
        
    Возвращает:
    -----------
    pd.Series
        Временной ряд с индексом DatetimeIndex
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            # Пытаемся преобразовать индекс в DatetimeIndex
            series = series.copy()
            series.index = pd.to_datetime(series.index)
        except:
            warnings.warn("Не удалось преобразовать индекс в DatetimeIndex. Используется оригинальный индекс.")
    
    return series


def estimate_forecast_horizon(series: pd.Series, max_horizon_ratio: float = 0.5) -> int:
    """
    Оценивает разумный горизонт прогнозирования на основе длины временного ряда.
    
    Параметры:
    -----------
    series : pd.Series
        Временной ряд для анализа
    max_horizon_ratio : float, default=0.5
        Максимальное соотношение горизонта прогноза к длине ряда
        
    Возвращает:
    -----------
    int
        Рекомендуемый горизонт прогнозирования
    """
    n = len(series)
    
    # Определяем разумный горизонт прогноза в зависимости от длины ряда
    if n < 30:
        # Для коротких рядов прогноз на небольшой период
        forecast_horizon = max(1, int(n * 0.1))
    elif n < 100:
        # Для средних рядов
        forecast_horizon = max(3, int(n * 0.15))
    else:
        # Для длинных рядов
        forecast_horizon = max(5, int(n * 0.2))
    
    # Ограничиваем maximum_forecast_horizon на основе параметра max_horizon_ratio
    maximum_forecast_horizon = int(n * max_horizon_ratio)
    
    # Ограничиваем horizon, чтобы избежать экстраполяции слишком далеко
    return min(forecast_horizon, maximum_forecast_horizon, 100)


def generate_forecast_index(last_date: pd.Timestamp, steps: int, freq: Optional[str] = None) -> pd.DatetimeIndex:
    """
    Генерирует индекс для прогноза на основе последней даты в данных.
    
    Параметры:
    -----------
    last_date : pd.Timestamp
        Последняя дата во временном ряде
    steps : int
        Количество шагов для прогноза
    freq : str, optional
        Частота данных. Если None, частота будет определена на основе последней даты
        
    Возвращает:
    -----------
    pd.DatetimeIndex
        Индекс для прогноза
    """
    if freq is None or freq == "":
        # Пытаемся определить частоту на основе формата даты
        if last_date.hour != 0 or last_date.minute != 0:
            # Вероятно, почасовые или поминутные данные
            if last_date.minute != 0:
                freq = "T"  # минуты
            else:
                freq = "H"  # часы
        else:
            # Проверяем день месяца, чтобы понять, ежедневные или ежемесячные данные
            if last_date.day == 1:
                # Вероятно месячные данные
                freq = "MS"  # начало месяца
            else:
                # Предполагаем ежедневные данные
                freq = "D"
    
    # Генерируем новый индекс
    try:
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)
    except:
        # Если не удалось создать с указанной частотой, пробуем дневную
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
    
    return forecast_index


def format_model_params(model_type: str, params: Dict[str, int]) -> str:
    """
    Форматирует параметры модели в удобочитаемую строку.
    
    Параметры:
    -----------
    model_type : str
        Тип модели ('ARMA', 'ARIMA' или 'SARIMA')
    params : Dict[str, int]
        Словарь с параметрами модели
        
    Возвращает:
    -----------
    str
        Форматированная строка с параметрами
    """
    if model_type == "ARMA":
        return f"ARMA({params['p']}, {params['q']})"
    elif model_type == "ARIMA":
        return f"ARIMA({params['p']}, {params['d']}, {params['q']})"
    elif model_type == "SARIMA":
        return f"SARIMA({params['p']}, {params['d']}, {params['q']})({params['P']}, {params['D']}, {params['Q']}){params['s']}"
    else:
        return str(params)


def clean_time_series(series: pd.Series, handle_missing: str = 'interpolate') -> pd.Series:
    """
    Очищает временной ряд от проблемных значений (пропусков, выбросов).
    
    Параметры:
    -----------
    series : pd.Series
        Исходный временной ряд
    handle_missing : str, default='interpolate'
        Способ обработки пропущенных значений:
        - 'interpolate': линейная интерполяция
        - 'forward': заполнение предыдущим значением
        - 'backward': заполнение следующим значением
        - 'mean': заполнение средним значением
        - 'median': заполнение медианным значением
        - 'drop': удаление пропущенных значений
        
    Возвращает:
    -----------
    pd.Series
        Очищенный временной ряд
    """
    # Создаем копию для обработки
    clean_series = series.copy()
    
    # Обрабатываем пропущенные значения
    if clean_series.isna().any():
        if handle_missing == 'interpolate':
            clean_series = clean_series.interpolate(method='linear')
        elif handle_missing == 'forward':
            clean_series = clean_series.fillna(method='ffill')
        elif handle_missing == 'backward':
            clean_series = clean_series.fillna(method='bfill')
        elif handle_missing == 'mean':
            clean_series = clean_series.fillna(clean_series.mean())
        elif handle_missing == 'median':
            clean_series = clean_series.fillna(clean_series.median())
        elif handle_missing == 'drop':
            clean_series = clean_series.dropna()
        
        # Если остались пропуски в начале или конце ряда
        if clean_series.isna().any():
            clean_series = clean_series.fillna(method='ffill').fillna(method='bfill')
    
    # Обрабатываем бесконечные значения
    clean_series = clean_series.replace([np.inf, -np.inf], np.nan)
    
    # Если появились новые NaN, заполняем их медианой
    if clean_series.isna().any():
        clean_series = clean_series.fillna(clean_series.median())
    
    return clean_series 