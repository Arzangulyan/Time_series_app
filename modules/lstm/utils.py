"""
Вспомогательные функции для работы с LSTM моделями.

Включает функции для:
- Проверки и преобразования входных данных
- Масштабирования временных рядов
- Создания индексов для прогнозов
- Работы с файловой системой
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import warnings


def check_input_series(series: Union[pd.Series, pd.DataFrame, np.ndarray, List]) -> pd.Series:
    """
    Проверяет и преобразует входные данные в формат pd.Series.
    
    Параметры:
    ----------
    series : pd.Series, pd.DataFrame, np.ndarray или list
        Входной временной ряд в различных форматах
        
    Возвращает:
    -----------
    pd.Series
        Преобразованный временной ряд
    """
    if isinstance(series, pd.Series):
        return series
    
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            return series.iloc[:, 0]
        else:
            warnings.warn("Датафрейм содержит несколько столбцов. Будет использован только первый столбец.")
            return series.iloc[:, 0]
    
    if isinstance(series, np.ndarray):
        if series.ndim == 1:
            return pd.Series(series)
        elif series.ndim == 2 and series.shape[1] == 1:
            return pd.Series(series.flatten())
        else:
            warnings.warn("Массив имеет неподходящую размерность. Будет использована первая колонка.")
            return pd.Series(series[:, 0])
    
    if isinstance(series, list):
        return pd.Series(series)
    
    raise TypeError("Неподдерживаемый тип данных. Используйте pd.Series, pd.DataFrame, np.ndarray или list.")


def scale_time_series(series: pd.Series, feature_range: Tuple[int, int] = (0, 1)) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Масштабирует временной ряд в заданный диапазон.
    
    Параметры:
    ----------
    series : pd.Series
        Входной временной ряд
    feature_range : tuple, default=(0, 1)
        Диапазон масштабирования
        
    Возвращает:
    -----------
    tuple
        (scaled_data, scaler)
    """
    # Преобразуем в numpy массив подходящей формы
    data = np.array(series).reshape(-1, 1)
    
    # Создаем и применяем масштабировщик
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler


def create_future_index(last_date, periods: int, freq: Optional[str] = None) -> pd.DatetimeIndex:
    """
    Создает индекс для будущих прогнозов на основе последней даты или существующего индекса.
    
    Параметры:
    ----------
    last_date : pd.Timestamp, pd.DatetimeIndex или pd.Index
        Последняя дата во временном ряде или весь индекс ряда
    periods : int
        Количество периодов для прогноза
    freq : str, optional
        Частота данных. Если None, будет определена на основе индекса или использована дневная частота.
        
    Возвращает:
    -----------
    pd.DatetimeIndex
        Индекс для прогнозных значений
    """
    if isinstance(last_date, pd.DatetimeIndex) or isinstance(last_date, pd.Index):
        # Если передан индекс целиком, берем последнюю дату
        if len(last_date) == 0:
            raise ValueError("Индекс пуст, невозможно определить начальную дату для прогноза")
        
        # Берем последнюю дату индекса
        last_timestamp = last_date[-1]
        
        # Определяем частоту, если не указана
        if freq is None:
            inferred_freq = pd.infer_freq(last_date)
            if inferred_freq is not None:
                freq = inferred_freq
            else:
                # Если частоту нельзя определить, вычисляем среднюю разницу
                if len(last_date) > 1:
                    deltas = []
                    for i in range(1, min(30, len(last_date))):
                        deltas.append(last_date[i] - last_date[i-1])
                    if deltas:  # Проверка на пустой список
                        avg_delta = sum(deltas) / len(deltas)
                        # Создаем индекс с использованием средней разницы
                        next_date = last_timestamp + avg_delta
                        step = avg_delta
                        dates = [next_date + i * step for i in range(periods)]
                        return pd.DatetimeIndex(dates)
                # Если не удалось определить частоту или дельту, используем дневную
                freq = 'D'
    else:
        # Если передана одна дата
        last_timestamp = last_date
        if freq is None:
            freq = 'D'
    
    # Создаем индекс с использованием pd.date_range
    try:
        next_date = last_timestamp + pd.Timedelta(1, unit=freq[0] if isinstance(freq, str) else 'D')
        return pd.date_range(start=next_date, periods=periods, freq=freq)
    except Exception as e:
        # Если возникает ошибка, используем дневную частоту
        next_date = last_timestamp + pd.Timedelta(days=1)
        return pd.date_range(start=next_date, periods=periods, freq='D')


def generate_forecast_index(original_index: pd.Index, forecast_steps: int) -> pd.Index:
    """
    Генерирует индекс для прогнозных значений, продолжая исходный индекс.
    
    Параметры:
    ----------
    original_index : pd.Index
        Индекс исходного временного ряда
    forecast_steps : int
        Количество шагов прогноза
        
    Возвращает:
    -----------
    pd.Index
        Индекс для прогнозных значений
    """
    if forecast_steps <= 0:
        return pd.Index([])
    
    if isinstance(original_index, pd.DatetimeIndex):
        # Определяем частоту индекса
        freq = pd.infer_freq(original_index)
        
        if freq is None:
            # Если частота не определена, вычисляем среднюю разницу между соседними точками
            if len(original_index) > 1:
                deltas = []
                for i in range(1, min(30, len(original_index))):
                    deltas.append(original_index[i] - original_index[i-1])
                avg_delta = pd.Timedelta(sum(deltas) / len(deltas))
                
                # Создаем индекс с использованием средней разницы
                start_date = original_index[-1] + avg_delta
                return pd.date_range(start=start_date, periods=forecast_steps, freq=avg_delta)
            else:
                # Если только одна точка, используем дневную частоту
                start_date = original_index[-1] + pd.Timedelta(days=1)
                return pd.date_range(start=start_date, periods=forecast_steps, freq='D')
        else:
            # Если частота определена, используем ее
            start_date = original_index[-1] + pd.Timedelta(1, unit=freq)
            return pd.date_range(start=start_date, periods=forecast_steps, freq=freq)
    else:
        # Для не-временных индексов просто продолжаем нумерацию
        start_idx = original_index[-1] + 1 if len(original_index) > 0 else 0
        return pd.RangeIndex(start=start_idx, stop=start_idx + forecast_steps)


def save_results_to_csv(results_df: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Сохраняет результаты прогнозирования в CSV файл.
    
    Параметры:
    ----------
    results_df : pd.DataFrame
        Датафрейм с результатами
    filename : str, optional
        Имя файла для сохранения. Если None, будет создано имя на основе текущей даты.
        
    Возвращает:
    -----------
    str
        Путь к сохраненному файлу
    """
    if filename is None:
        # Генерируем имя файла на основе текущей даты и времени
        now = datetime.now()
        filename = f"lstm_forecast_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Создаем директорию results, если она не существует
    os.makedirs("results", exist_ok=True)
    
    # Полный путь к файлу
    filepath = os.path.join("results", filename)
    
    # Сохраняем в CSV
    results_df.to_csv(filepath)
    
    return filepath


def load_model_from_file(model_path: str) -> tf.keras.Model:
    """
    Загружает модель из файла.
    
    Параметры:
    ----------
    model_path : str
        Путь к файлу модели
        
    Возвращает:
    -----------
    tf.keras.Model
        Загруженная модель
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден")
    
    # Загружаем модель
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}") 