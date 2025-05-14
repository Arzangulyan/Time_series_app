"""
Модуль для обнаружения аномалий во временных рядах.

Этот модуль содержит функции для генерации синтетических рядов с аномалиями
и методы обнаружения аномалий в временных рядах.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
import streamlit as st


@st.cache_data
def generate_base_series(
    n: int = 300, 
    seed: int = 42, 
    season_amp: float = 0.5, 
    noise_std: float = 0.2, 
    freq_scale: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация базового временного ряда.
    
    Args:
        n: Количество точек во временном ряду.
        seed: Случайное зерно для воспроизводимости.
        season_amp: Амплитуда сезонной составляющей.
        noise_std: Стандартное отклонение шума.
        freq_scale: Масштаб частоты для синусоиды.
        
    Returns:
        Кортеж (массив данных ряда, массив временных меток).
    """
    np.random.seed(seed)
    t = np.arange(n)
    season = season_amp * np.sin(freq_scale * t)
    noise = np.random.normal(0, noise_std, n)
    base_data = season + noise
    return base_data, t


@st.cache_data
def add_point_anomalies(
    data: np.ndarray, 
    indices: List[int], 
    amplitude_range: Tuple[float, float] = (1, 2), 
    increase: bool = True
) -> np.ndarray:
    """
    Добавление точечных аномалий.
    
    Args:
        data: Исходный ряд данных.
        indices: Список индексов, в которых будут добавлены аномалии.
        amplitude_range: Диапазон изменения для смещения.
        increase: Если True, то аномалия добавляется (подъём), иначе вычитается (падение).
        
    Returns:
        Модифицированный ряд.
    """
    data_modified = data.copy()
    for idx in indices:
        # Используем нижнюю и верхнюю границу из кортежа
        shift = np.random.uniform(amplitude_range[0], amplitude_range[1])
        if not increase:
            shift = -shift
        data_modified[idx] += shift
    return data_modified


@st.cache_data
def add_extended_anomaly(
    data: np.ndarray, 
    start_idx: int, 
    duration: int, 
    level_shift: float = 2.0
) -> np.ndarray:
    """
    Добавление протяжённой аномалии – изменение уровня ряда на заданном интервале.
    
    Args:
        data: Исходный ряд.
        start_idx: Начальный индекс аномалии.
        duration: Длительность аномалии (количество точек).
        level_shift: Величина смещения уровня.
        
    Returns:
        Модифицированный ряд.
    """
    data_modified = data.copy()
    data_modified[start_idx:start_idx+duration] += level_shift
    return data_modified


@st.cache_data
def add_sensor_fault(
    data: np.ndarray, 
    start_idx: int, 
    duration: int, 
    fault_value: float = np.nan
) -> np.ndarray:
    """
    Моделирование отказа датчика – замена значений на константу или NaN.
    
    Args:
        data: Исходный ряд.
        start_idx: Начальный индекс аномалии.
        duration: Длительность аномалии (количество точек).
        fault_value: Значение, которым заменяются данные при отказе.
        
    Returns:
        Модифицированный ряд.
    """
    data_modified = data.copy()
    data_modified[start_idx:start_idx+duration] = fault_value
    return data_modified


@st.cache_data
def generate_anomalous_series(
    n: int = 300,
    seed: int = 42,
    season_amp: float = 0.5,
    noise_std: float = 0.2,
    freq_scale: float = 0.1,
    point_anomalies: Optional[List[Dict[str, Any]]] = None,
    extended_anomalies: Optional[List[Dict[str, Any]]] = None,
    sensor_faults: Optional[List[Dict[str, Any]]] = None        
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Генерация временного ряда с различными типами аномалий.
    
    Args:
        n: Количество точек во временном ряду.
        seed: Случайное зерно для воспроизводимости.
        season_amp: Амплитуда сезонной составляющей.
        noise_std: Стандартное отклонение шума.
        freq_scale: Масштаб частоты для синусоиды.
        point_anomalies: Список словарей с параметрами точечных аномалий.
        extended_anomalies: Список словарей с параметрами протяженных аномалий.
        sensor_faults: Список словарей с параметрами отказов датчиков.
        
    Returns:
        Кортеж (данные с аномалиями, временные метки, информация об аномалиях).
    """
    # Базовый ряд
    base_data, t = generate_base_series(n, seed, season_amp, noise_std, freq_scale)
    data_with_anomalies = base_data.copy()
    
    # Информация об аномалиях для дальнейшего анализа
    anomaly_info = []
    
    # Добавление точечных аномалий
    if point_anomalies:
        for pa in point_anomalies:
            indices = pa.get('indices', [])
            amplitude_range = pa.get('amplitude_range', (1, 2))
            increase = pa.get('increase', True)
            
            data_with_anomalies = add_point_anomalies(
                data_with_anomalies, indices, amplitude_range, increase
            )
            
            for idx in indices:
                anomaly_info.append({
                    'type': 'point',
                    'index': idx,
                    'increase': increase
                })
    
    # Добавление протяженных аномалий
    if extended_anomalies:
        for ea in extended_anomalies:
            start_idx = ea.get('start_idx', 0)
            duration = ea.get('duration', 10)
            level_shift = ea.get('level_shift', 2.0)
            
            data_with_anomalies = add_extended_anomaly(
                data_with_anomalies, start_idx, duration, level_shift
            )
            
            anomaly_info.append({
                'type': 'extended',
                'start_idx': start_idx,
                'duration': duration,
                'level_shift': level_shift
            })
    
    # Добавление отказов датчиков
    if sensor_faults:
        for sf in sensor_faults:
            start_idx = sf.get('start_idx', 0)
            duration = sf.get('duration', 5)
            fault_value = sf.get('fault_value', np.nan)
            
            data_with_anomalies = add_sensor_fault(
                data_with_anomalies, start_idx, duration, fault_value
            )
            
            anomaly_info.append({
                'type': 'sensor_fault',
                'start_idx': start_idx,
                'duration': duration,
                'fault_value': str(fault_value)  # Преобразуем в строку для JSON-совместимости
            })
    
    return data_with_anomalies, t, anomaly_info


@st.cache_data
def z_score_detection(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Обнаружение аномалий с помощью Z-оценки.
    
    Args:
        data: Исходный ряд данных.
        threshold: Порог для определения аномалии.
        
    Returns:
        Булев массив с отметками аномалий.
    """
    # Конвертируем вход в NumPy массив, если это DataFrame или Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
        
    mean = np.nanmean(data)
    std = np.nanstd(data)
    z_scores = np.abs((data - mean) / std)
    
    # Убедимся, что возвращаем именно NumPy массив
    result = z_scores > threshold
    return np.asarray(result)


@st.cache_data
def iqr_detection(data: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Обнаружение аномалий с помощью метода межквартильного размаха (IQR).
    
    Args:
        data: Исходный ряд данных.
        multiplier: Множитель для IQR.
        
    Returns:
        Кортеж из булева массива с отметками аномалий и кортежа с нижней и верхней границами.
    """
    # Конвертируем вход в NumPy массив, если это DataFrame или Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
        
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Убедимся, что возвращаем именно NumPy массив
    result = (data < lower_bound) | (data > upper_bound)
    return np.asarray(result), (lower_bound, upper_bound)


@st.cache_data
def hampel_filter(data: np.ndarray, window: int = 5, sigma: float = 3.0, 
                  adaptive_window: bool = False, window_percent: float = 0.5) -> np.ndarray:
    """
    Фильтр Хампеля для обнаружения аномалий.
    
    Args:
        data: Исходный ряд данных.
        window: Размер окна для расчета медианы.
        sigma: Множитель для определения порога.
        adaptive_window: Если True, размер окна рассчитывается как процент от длины ряда.
        window_percent: Процент от длины ряда для размера окна (если adaptive_window=True).
        
    Returns:
        Булев массив с отметками аномалий.
    """
    # Конвертируем вход в NumPy массив, если это DataFrame или Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
        
    n = len(data)
    result = np.zeros(n, dtype=bool)
    
    # Если включен адаптивный размер окна, рассчитываем его
    if adaptive_window:
        # Преобразуем процент в количество точек (минимум 5, максимум 20% от длины ряда)
        window = max(5, min(int(n * window_percent / 100), n // 5))
        
    # Вычисляем медиану и MAD в скользящем окне
    for i in range(n):
        # Находим границы окна
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        # Вычисляем медиану и MAD
        window_data = data[start:end]
        window_data = window_data[~np.isnan(window_data)]  # Исключаем NaN
        
        if len(window_data) > 0:
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))
            
            # Проверяем на аномалию
            if mad == 0:  # Для предотвращения деления на ноль
                result[i] = False
            else:
                deviation = np.abs(data[i] - median) / (1.4826 * mad)
                result[i] = deviation > sigma
    
    return result


@st.cache_data
def detect_plateau(data: np.ndarray, threshold: float = 1e-3, min_duration: int = 5) -> List[Dict[str, int]]:
    """
    Обнаружение плато – участков, где значения почти не меняются.
    
    Args:
        data: Исходный ряд данных.
        threshold: Максимальное допустимое изменение для плато.
        min_duration: Минимальная длительность плато.
        
    Returns:
        Список словарей с информацией о найденных плато (начало и конец).
    """
    n = len(data)
    plateaus = []
    start = None
    
    for i in range(1, n):
        # Проверяем, является ли текущий участок потенциальным плато
        if np.abs(data[i] - data[i-1]) < threshold:
            if start is None:
                start = i - 1
        # Если разница больше порога или конец ряда, проверяем накопившееся плато
        elif start is not None:
            duration = i - start
            if duration >= min_duration:
                plateaus.append({'start': start, 'end': i-1})
            start = None
    
    # Проверяем последнее потенциальное плато
    if start is not None:
        duration = n - start
        if duration >= min_duration:
            plateaus.append({'start': start, 'end': n-1})
    
    return plateaus


@st.cache_data
def add_anomalies_to_existing_data(
    data: np.ndarray,
    _time: np.ndarray = None,
    point_anomalies: Optional[List[Dict[str, Any]]] = None,
    extended_anomalies: Optional[List[Dict[str, Any]]] = None,
    sensor_faults: Optional[List[Dict[str, Any]]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Добавление аномалий в существующий временной ряд.

    Args:
        data: Исходный ряд данных.
        _time: Временные метки (если None, используется последовательность индексов).
        point_anomalies: Список словарей с параметрами точечных аномалий.
        extended_anomalies: Список словарей с параметрами протяженных аномалий.
        sensor_faults: Список словарей с параметрами отказов датчиков.
        
    Returns:
        Кортеж (данные с аномалиями, информация об аномалиях).
    """
    # Создаем копию исходных данных
    data_with_anomalies = data.copy()
    
    # Если временные метки не предоставлены, создаем их
    if _time is None:
        _time = np.arange(len(data))
    
    # Информация об аномалиях для дальнейшего анализа
    anomaly_info = []
    
    # Добавление точечных аномалий
    if point_anomalies:
        for pa in point_anomalies:
            indices = pa.get('indices', [])
            amplitude_range = pa.get('amplitude_range', (1, 2))
            increase = pa.get('increase', True)
            
            data_with_anomalies = add_point_anomalies(
                data_with_anomalies, indices, amplitude_range, increase
            )
            
            for idx in indices:
                anomaly_info.append({
                    'type': 'point',
                    'index': idx,
                    'increase': increase
                })
    
    # Добавление протяженных аномалий
    if extended_anomalies:
        for ea in extended_anomalies:
            start_idx = ea.get('start_idx', 0)
            duration = ea.get('duration', 10)
            level_shift = ea.get('level_shift', 2.0)
            
            data_with_anomalies = add_extended_anomaly(
                data_with_anomalies, start_idx, duration, level_shift
            )
            
            anomaly_info.append({
                'type': 'extended',
                'start_idx': start_idx,
                'duration': duration,
                'level_shift': level_shift
            })
    
    # Добавление отказов датчиков
    if sensor_faults:
        for sf in sensor_faults:
            start_idx = sf.get('start_idx', 0)
            duration = sf.get('duration', 5)
            fault_value = sf.get('fault_value', np.nan)
            
            data_with_anomalies = add_sensor_fault(
                data_with_anomalies, start_idx, duration, fault_value
            )
            
            anomaly_info.append({
                'type': 'sensor_fault',
                'start_idx': start_idx,
                'duration': duration,
                'fault_value': str(fault_value)  # Преобразуем в строку для JSON-совместимости
            })
    
    return data_with_anomalies, anomaly_info


@st.cache_data
def create_true_anomaly_mask(anomaly_info: List[Dict[str, Any]], length: int) -> np.ndarray:
    """
    Создает бинарную маску для истинных аномалий на основе информации о внедренных аномалиях.
    
    Args:
        anomaly_info: Список словарей с информацией о внедренных аномалиях.
        length: Длина временного ряда.
        
    Returns:
        Бинарный массив, где True указывает на аномалию.
    """
    true_mask = np.zeros(length, dtype=bool)
    
    for anomaly in anomaly_info:
        anom_type = anomaly.get('type')
        
        if anom_type == 'point':
            idx = anomaly.get('index')
            if 0 <= idx < length:
                true_mask[idx] = True
                
        elif anom_type in ['extended', 'sensor_fault']:
            start_idx = anomaly.get('start_idx', 0)
            duration = anomaly.get('duration', 0)
            end_idx = min(start_idx + duration, length)
            if start_idx < length:
                true_mask[start_idx:end_idx] = True
    
    return true_mask


@st.cache_data
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики качества обнаружения аномалий.
    
    Args:
        y_true: Истинные метки аномалий (бинарный массив).
        y_pred: Предсказанные метки аномалий (бинарный массив).
        
    Returns:
        Словарь с метриками precision, recall и f1-score.
    """
    # Проверка на пустые массивы
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if np.sum(y_true) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
    if np.sum(y_pred) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Вычисление метрик
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    false_positives = np.sum(np.logical_and(~y_true, y_pred))
    false_negatives = np.sum(np.logical_and(y_true, ~y_pred))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    # F1-score - гармоническое среднее precision и recall
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


@st.cache_data
def evaluate_anomaly_detection(
    anomaly_info: List[Dict[str, Any]], 
    detected_anomalies: Dict[str, Union[np.ndarray, pd.Series]], 
    length: int
) -> Dict[str, Dict[str, float]]:
    """
    Оценивает качество обнаружения аномалий различными методами.
    
    Args:
        anomaly_info: Список словарей с информацией о внедренных аномалиях.
        detected_anomalies: Словарь с результатами различных методов обнаружения.
        length: Длина временного ряда.
        
    Returns:
        Словарь с метриками качества для каждого метода обнаружения.
    """
    true_mask = create_true_anomaly_mask(anomaly_info, length)
    
    results = {}
    for method_name, anomaly_mask in detected_anomalies.items():
        # Проверяем, является ли значение булевым массивом нужной длины
        if method_name == 'iqr_bounds':
            # Пропускаем iqr_bounds, так как это кортеж с границами, а не маска аномалий
            continue
            
        # Преобразуем pandas Series/DataFrame в numpy array если нужно
        if isinstance(anomaly_mask, (pd.Series, pd.DataFrame)):
            anomaly_mask = anomaly_mask.values
            
        # Убедимся, что это ndarray с правильной формой
        if not isinstance(anomaly_mask, np.ndarray):
            print(f"Пропускаю {method_name}: не numpy массив (тип: {type(anomaly_mask)})")
            continue
            
        if len(anomaly_mask) != length:
            print(f"Пропускаю {method_name}: неверная длина ({len(anomaly_mask)} вместо {length})")
            continue
            
        # Тип должен быть bool или конвертируемым в bool
        try:
            if anomaly_mask.dtype != bool:
                # Если массив не булевый, но это одномерный числовой массив,
                # конвертируем в булевый тип
                if anomaly_mask.ndim == 1 and np.issubdtype(anomaly_mask.dtype, np.number):
                    anomaly_mask = anomaly_mask.astype(bool)
                else:
                    print(f"Пропускаю {method_name}: не булев тип и не числовой (dtype: {anomaly_mask.dtype})")
                    continue
        except Exception as e:
            print(f"Ошибка преобразования типа для {method_name}: {str(e)}")
            continue
        
        try:
            # Вычисляем метрики
            metrics = calculate_metrics(true_mask, anomaly_mask)
            results[method_name] = metrics
        except Exception as e:
            print(f"Ошибка при вычислении метрик для метода {method_name}: {str(e)}")
            continue
    
    return results