"""
Основные функции для анализа и прогнозирования временных рядов с помощью LSTM.

Включает функции для:
- Подготовки данных для LSTM моделей
- Автоматического подбора параметров
- Прогнозирования будущих значений
- Оценки качества моделей
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import warnings

# Подавляем предупреждения TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает последовательности для обучения LSTM модели.
    
    Преобразует временной ряд в формат "скользящего окна", где каждый элемент X
    содержит последовательность длины sequence_length, а соответствующий элемент y
    содержит следующее значение ряда.
    
    Параметры:
    ----------
    data : np.ndarray
        Входной временной ряд (масштабированный)
    sequence_length : int
        Длина входной последовательности (лаг)
        
    Возвращает:
    -----------
    tuple
        (X - входные последовательности, y - целевые значения)
    """
    if len(data) <= sequence_length:
        return np.array([]), np.array([])
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)


def train_test_split_ts(data: Union[pd.Series, np.ndarray], train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.
    
    В отличие от обычного train_test_split, сохраняет временную последовательность,
    разделяя ряд на последовательные части.
    
    Параметры:
    ----------
    data : pd.Series или np.ndarray
        Входной временной ряд
    train_size : float
        Доля данных для обучения (0 < train_size < 1)
        
    Возвращает:
    -----------
    tuple
        (train_data, test_data)
    """
    if not (0 < train_size < 1):
        raise ValueError("train_size должен быть в диапазоне (0, 1)")
    
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.values.reshape(-1, 1)
    elif not isinstance(data, np.ndarray):
        data = np.array(data).reshape(-1, 1)
    
    # Проверка размерности
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n = len(data)
    if n < 10:  # Минимальное количество точек для разделения
        raise ValueError("Недостаточно данных для разделения на обучающую и тестовую выборки")
    
    train_size_idx = int(n * train_size)
    
    # Разделение данных
    train_data = data[:train_size_idx]
    test_data = data[train_size_idx:]
    
    return train_data, test_data


def auto_tune_lstm_params(time_series: pd.Series, complexity_level: str = 'auto') -> Dict[str, Any]:
    """
    Автоматически подбирает параметры LSTM на основе характеристик временного ряда.
    
    Анализирует длину ряда, частоту и другие характеристики для определения
    оптимальных параметров LSTM модели.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Входной временной ряд
    complexity_level : str
        Уровень сложности модели ('simple', 'medium', 'complex', 'auto')
        
    Возвращает:
    -----------
    dict
        Словарь с подобранными параметрами
    """
    if complexity_level not in ['simple', 'medium', 'complex', 'auto']:
        raise ValueError("complexity_level должен быть одним из: 'simple', 'medium', 'complex', 'auto'")
    
    # Определяем длину ряда
    ts_length = len(time_series)
    
    # Определяем периодичность (если есть)
    freq = None
    if hasattr(time_series, 'index') and isinstance(time_series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(time_series.index)
    
    # Подбираем параметры в зависимости от длины ряда
    if complexity_level == 'auto':
        if ts_length < 500:
            complexity_level = 'simple'
        elif ts_length < 2000:
            complexity_level = 'medium'
        else:
            complexity_level = 'complex'
    
    # Настройки по уровням сложности
    params = {
        'simple': {
            'sequence_length': 5,
            'units': [64],
            'dropout_rate': 0.1,
            'batch_size': 16,
            'epochs': 50,
            'bidirectional': False
        },
        'medium': {
            'sequence_length': 7,
            'units': [128, 64],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'bidirectional': False
        },
        'complex': {
            'sequence_length': 10,
            'units': [256, 128, 64],
            'dropout_rate': 0.3,
            'batch_size': 64,
            'epochs': 150,
            'bidirectional': True
        }
    }
    
    selected_params = params[complexity_level]
    
    # Корректируем параметры в зависимости от частоты данных
    if freq:
        if 'D' in str(freq):  # Дневные данные
            selected_params['sequence_length'] = min(7, selected_params['sequence_length'])
        elif 'W' in str(freq):  # Недельные данные
            selected_params['sequence_length'] = min(8, selected_params['sequence_length'])
        elif 'M' in str(freq):  # Месячные данные
            selected_params['sequence_length'] = min(12, selected_params['sequence_length'])
    
    # Всегда включаем раннюю остановку
    selected_params['early_stopping'] = True
    selected_params['patience'] = 15
    selected_params['validation_split'] = 0.1
    
    return selected_params


def forecast_future(model: tf.keras.Model, last_sequence: np.ndarray, scaler: MinMaxScaler, steps: int = 10) -> np.ndarray:
    """
    Прогнозирует будущие значения на основе обученной модели.
    
    Генерирует прогноз на заданное количество шагов вперед, используя последнюю
    известную последовательность как входные данные.
    
    Параметры:
    ----------
    model : tf.keras.Model
        Обученная модель LSTM
    last_sequence : np.ndarray
        Последняя известная последовательность (масштабированная)
    scaler : sklearn.preprocessing.MinMaxScaler
        Масштабировщик для обратного преобразования
    steps : int
        Количество шагов для прогноза вперед
        
    Возвращает:
    -----------
    np.ndarray
        Массив с прогнозами
    """
    if steps <= 0:
        raise ValueError("Количество шагов должно быть положительным числом")
    
    # Копируем последнюю последовательность
    current_sequence = last_sequence.copy()
    future_predictions = []
    
    # Генерируем прогнозы по одному шагу за раз
    for _ in range(steps):
        # Получаем прогноз
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        # Обновляем последовательность для следующего шага прогноза
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Преобразуем прогнозы обратно в исходный масштаб
    future_preds_array = np.array(future_predictions).reshape(-1, 1)
    future_preds_rescaled = scaler.inverse_transform(future_preds_array)
    
    return future_preds_rescaled.flatten()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Рассчитывает метрики качества прогноза.
    
    Вычисляет различные метрики для оценки точности прогноза, включая
    RMSE (среднеквадратичная ошибка), MAE (средняя абсолютная ошибка),
    MAPE (средняя абсолютная процентная ошибка) и другие.
    
    Параметры:
    ----------
    y_true : array-like
        Фактические значения
    y_pred : array-like
        Предсказанные значения
        
    Возвращает:
    -----------
    dict
        Словарь с метриками
    """
    # Преобразуем в одномерные массивы
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Удаляем пропущенные значения
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:  # Нужно минимум 2 точки для расчета метрик
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'mase': np.nan,
            'r2': np.nan,
            'adj_r2': np.nan
        }
    
    # Базовые метрики
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE с защитой от деления на ноль
    if np.any(y_true == 0):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Расчет MASE (Mean Absolute Scaled Error)
    # Используем сдвиг на 1 как наивный прогноз для знаменателя
    naive_errors = np.abs(np.diff(y_true))
    if len(naive_errors) > 0 and np.mean(naive_errors) != 0:
        mase = np.mean(np.abs(y_pred - y_true)) / np.mean(naive_errors)
    else:
        mase = np.nan
    
    # Расчет R² (коэффициент детерминации)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot != 0:
        r2 = 1 - (ss_res / ss_tot)
        # Скорректированный R² учитывает количество предикторов
        n = len(y_true)
        p = 1  # количество предикторов (для временных рядов обычно 1)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        r2 = np.nan
        adj_r2 = np.nan
    
    # Ограничиваем R² и adj_r2 снизу значением -1 для более понятной интерпретации
    r2 = max(r2, -1) if not np.isnan(r2) else np.nan
    adj_r2 = max(adj_r2, -1) if not np.isnan(adj_r2) else np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mase': mase,
        'r2': r2,
        'adj_r2': adj_r2
    }


def prepare_data_for_forecast(series: pd.Series, sequence_length: int) -> Dict[str, Any]:
    """
    Подготавливает данные для прогнозирования с помощью LSTM.
    
    Выполняет необходимую предобработку данных, включая масштабирование
    и создание последовательностей для LSTM модели.
    
    Параметры:
    ----------
    series : pd.Series
        Исходный временной ряд
    sequence_length : int
        Длина последовательности для LSTM модели
        
    Возвращает:
    -----------
    dict
        Словарь с подготовленными данными
    """
    if len(series) <= sequence_length:
        raise ValueError(f"Длина ряда ({len(series)}) должна быть больше длины последовательности ({sequence_length})")
    
    # Преобразуем в numpy массив
    data = np.array(series).reshape(-1, 1)
    
    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Создание последовательностей
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Подготовка последней последовательности для будущего прогнозирования
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    
    return {
        'X': X,
        'y': y,
        'scaler': scaler,
        'last_sequence': last_sequence
    } 