"""
Модуль для обнаружения аномалий во временных рядах.

Этот модуль содержит функции для генерации синтетических рядов с аномалиями
и методы обнаружения аномалий в временных рядах.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
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
            
            # Проверяем, что все индексы числовые и конвертируем при необходимости
            numeric_indices = []
            for idx in indices:
                if isinstance(idx, (int, np.integer)):
                    numeric_indices.append(idx)
                else:
                    # Skip non-integer indices
                    continue
            
            if numeric_indices:
                data_with_anomalies = add_point_anomalies(
                    data_with_anomalies, numeric_indices, amplitude_range, increase
                )
                
                for idx in numeric_indices:
                    anomaly_info.append({
                        'type': 'point',
                        'index': int(idx),  # Ensure index is integer
                        'increase': increase
                    })
    
    # Добавление протяженных аномалий
    if extended_anomalies:
        for ea in extended_anomalies:
            start_idx = ea.get('start_idx', 0)
            duration = ea.get('duration', 10)
            level_shift = ea.get('level_shift', 2.0)
            
            # Ensure start_idx is numeric
            if not isinstance(start_idx, (int, np.integer)):
                try:
                    start_idx = int(start_idx)
                except (ValueError, TypeError):
                    continue
            
            data_with_anomalies = add_extended_anomaly(
                data_with_anomalies, start_idx, duration, level_shift
            )
            
            anomaly_info.append({
                'type': 'extended',
                'start_idx': int(start_idx),  # Ensure index is integer
                'duration': duration,
                'level_shift': level_shift
            })
    
    # Добавление отказов датчиков
    if sensor_faults:
        for sf in sensor_faults:
            start_idx = sf.get('start_idx', 0)
            duration = sf.get('duration', 5)
            fault_value = sf.get('fault_value', np.nan)
            
            # Ensure start_idx is numeric
            if not isinstance(start_idx, (int, np.integer)):
                try:
                    start_idx = int(start_idx)
                except (ValueError, TypeError):
                    continue
            
            data_with_anomalies = add_sensor_fault(
                data_with_anomalies, start_idx, duration, fault_value
            )
            
            anomaly_info.append({
                'type': 'sensor_fault',
                'start_idx': int(start_idx),  # Ensure index is integer
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
            # Convert idx to integer if it's not already (handles Timestamp objects)
            if not isinstance(idx, (int, np.integer)):
                try:
                    # Try to find the position in the original array
                    # This is a fallback and may not work in all cases
                    idx = int(idx)
                except (ValueError, TypeError):
                    # Skip this anomaly if we can't convert the index
                    continue
            
            # Ensure index is within valid range
            if 0 <= idx < length:
                true_mask[idx] = True
                
        elif anom_type in ['extended', 'sensor_fault']:
            start_idx = anomaly.get('start_idx', 0)
            duration = anomaly.get('duration', 0)
            
            # Convert start_idx to integer if it's not already
            if not isinstance(start_idx, (int, np.integer)):
                try:
                    start_idx = int(start_idx)
                except (ValueError, TypeError):
                    # Skip this anomaly if we can't convert the index
                    continue
                    
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
                # Если массив не булев, но это одномерный числовой массив,
                # конверируем в булевый тип
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


@st.cache_data
def suggest_parameters(data: np.ndarray) -> Dict[str, float]:
    """
    Предлагает оптимальные параметры для обнаружения аномалий на основе характеристик ряда.
    
    Args:
        data: Исходный ряд данных.
        
    Returns:
        Словарь с рекомендуемыми параметрами для разных методов.
    """
    # Конвертируем вход в NumPy массив, если это DataFrame или Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    
    # Очищаем данные от NaN для расчета статистик
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)
    
    # Базовые статистики
    mean = np.mean(clean_data)
    std = np.std(clean_data)
    median = np.median(clean_data)
    q1 = np.percentile(clean_data, 25)
    q3 = np.percentile(clean_data, 75)
    iqr_value = q3 - q1
    
    # Оценка коэффициента вариации
    cv = std / abs(mean) if mean != 0 else float('inf')
    
    # Оценка асимметрии распределения
    skewness = np.sum((clean_data - mean) ** 3) / (n * std ** 3) if std > 0 else 0
    
    # Оценка "тяжелохвостости" распределения
    kurtosis = np.sum((clean_data - mean) ** 4) / (n * std ** 4) - 3 if std > 0 else 0
    
    # Рекомендации по параметрам
    suggested_params = {}
    
    # Z-Score порог
    if kurtosis > 3:  # Тяжелые хвосты
        suggested_params['z_threshold'] = max(3.5, min(5.0, 3.0 + kurtosis / 5))
    else:  # Нормальное или легче нормального
        suggested_params['z_threshold'] = max(2.0, min(3.5, 3.0 - kurtosis / 10))
    
    # IQR множитель
    if abs(skewness) > 1:  # Сильная асимметрия
        suggested_params['iqr_multiplier'] = max(1.0, min(3.0, 1.5 + abs(skewness) / 2))
    else:  # Близко к симметричному
        suggested_params['iqr_multiplier'] = 1.5
    
    # Hampel параметры
    # Размер окна как функция от длины ряда и изменчивости
    suggested_window_percent = max(0.2, min(2.0, 0.5 * (1 + cv)))
    suggested_params['hampel_window_percent'] = suggested_window_percent
    
    # Размер окна в точках
    suggested_window = max(5, min(n // 10, int(n * suggested_window_percent / 100)))
    suggested_params['hampel_window'] = suggested_window
    
    # Коэффициент чувствительности
    suggested_params['hampel_sigma'] = max(2.0, min(4.0, 3.0 + kurtosis / 10))
    
    # Plateau параметры
    # Порог как функция от стандартного отклонения
    noise_estimate = min(std, iqr_value / 1.35)  # Робастная оценка шума
    suggested_params['plateau_threshold'] = max(0.0001, min(0.01, noise_estimate / 100))
    
    # Минимальная длительность как функция от длины ряда
    suggested_params['plateau_duration'] = max(3, min(30, n // 50))
    
    return suggested_params


@st.cache_data
def run_parameter_experiment(
    data: np.ndarray,
    true_anomalies: np.ndarray,
    method: str,
    param_ranges: Dict[str, List[float]],
    fixed_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Проводит численный эксперимент, перебирая различные значения параметров
    для выбранного метода обнаружения аномалий.
    
    Args:
        data: Временной ряд для анализа
        true_anomalies: Истинная маска аномалий для оценки качества
        method: Название метода ('z_score', 'iqr', 'hampel', 'plateau')
        param_ranges: Словарь с диапазонами параметров для перебора
        fixed_params: Словарь с фиксированными параметрами
        
    Returns:
        DataFrame с результатами экспериментов
    """
    results = []
    
    # Установка значений по умолчанию для фиксированных параметров
    if fixed_params is None:
        fixed_params = {}
    
    # Получаем все комбинации параметров для перебора
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Генерируем все комбинации параметров
    param_combinations = list(itertools.product(*param_values))
    
    # Для каждой комбинации параметров
    for combo in param_combinations:
        # Создаем словарь параметров для текущей комбинации
        params = fixed_params.copy()
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]
        
        # Запускаем детекцию аномалий с текущими параметрами
        if method == 'z_score':
            anomaly_mask = z_score_detection(data, threshold=params.get('threshold', 3.0))
        elif method == 'iqr':
            anomaly_mask, _ = iqr_detection(data, multiplier=params.get('multiplier', 1.5))
        elif method == 'hampel':
            anomaly_mask = hampel_filter(
                data, 
                window=params.get('window', 5),
                sigma=params.get('sigma', 3.0),
                adaptive_window=params.get('adaptive_window', False),
                window_percent=params.get('window_percent', 0.5)
            )
        elif method == 'plateau':
            plateau_results = detect_plateau(
                data, 
                threshold=params.get('threshold', 1e-3),
                min_duration=params.get('min_duration', 5)
            )
            # Преобразуем результаты плато в маску
            anomaly_mask = np.zeros(len(data), dtype=bool)
            for p in plateau_results:
                anomaly_mask[p['start']:p['end']+1] = True
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Оцениваем качество обнаружения
        metrics = calculate_metrics(true_anomalies, anomaly_mask)
        
        # Добавляем количество обнаруженных аномалий
        num_anomalies = np.sum(anomaly_mask)
        
        # Сохраняем результаты
        result = {**params, **metrics, 'num_anomalies': num_anomalies}
        results.append(result)
    
    # Создаем DataFrame с результатами
    return pd.DataFrame(results)


@st.cache_data
def get_default_parameter_ranges():
    """
    Возвращает словарь с диапазонами параметров по умолчанию для экспериментов.
    
    Returns:
        Словарь с диапазонами параметров для каждого метода
    """
    return {
        'z_score': {
            'threshold': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        },
        'iqr': {
            'multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        },
        'hampel': {
            'window': [5, 10, 15, 20, 25, 30],
            'sigma': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'window_percent': [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        },
        'plateau': {
            'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'min_duration': [2, 5, 10, 15, 20, 25, 30]
        }
    }


def suggest_optimal_parameters(experiment_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует результаты экспериментов и предлагает оптимальные параметры.
    
    Args:
        experiment_results: DataFrame с результатами экспериментов
        
    Returns:
        Словарь с оптимальными параметрами и метриками
    """
    # Находим строку с максимальным F1-score
    best_idx = experiment_results['f1'].idxmax()
    best_result = experiment_results.loc[best_idx]
    
    # Находим строку с лучшей точностью
    best_precision_idx = experiment_results['precision'].idxmax()
    best_precision = experiment_results.loc[best_precision_idx]
    
    # Находим строку с лучшей полнотой
    best_recall_idx = experiment_results['recall'].idxmax()
    best_recall = experiment_results.loc[best_recall_idx]
    
    return {
        'best_f1': best_result.to_dict(),
        'best_precision': best_precision.to_dict(),
        'best_recall': best_recall.to_dict()
    }


def prepare_anomaly_report_data(
    data: np.ndarray,
    time_index: np.ndarray,
    anomaly_info: List[Dict[str, Any]],
    detection_results: Dict[str, np.ndarray],
    detection_params: Dict[str, Any],
    metrics_results: Optional[Dict[str, Dict[str, float]]] = None,
    experiment_results: Optional[pd.DataFrame] = None,
    optimal_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Подготавливает данные для отчета по обнаружению аномалий.
    
    Args:
        data: Временной ряд данных
        time_index: Временные метки
        anomaly_info: Информация о внедренных аномалиях
        detection_results: Результаты различных методов обнаружения
        detection_params: Параметры методов обнаружения
        metrics_results: Метрики качества обнаружения аномалий
        experiment_results: Результаты численных экспериментов (если проводились)
        optimal_params: Оптимальные параметры из экспериментов
        
    Returns:
        Словарь с данными для отчета
    """
    # Базовая информация о данных
    report_data = {
        'data_length': len(data),
        'has_nan': np.isnan(data).any(),
        'data_stats': {
            'mean': float(np.nanmean(data)),
            'std': float(np.nanstd(data)),
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'median': float(np.nanmedian(data)),
            'iqr': float(np.nanpercentile(data, 75) - np.nanpercentile(data, 25))
        },
        'detection_params': detection_params,
        'anomaly_info': anomaly_info
    }
    
    # Информация об обнаруженных аномалиях
    report_data['detection_results'] = {}
    for method_name, result in detection_results.items():
        if method_name == 'iqr_bounds':
            continue
        if isinstance(result, np.ndarray) and result.dtype == bool:
            report_data['detection_results'][method_name] = {
                'count': int(np.sum(result)),
                'percentage': float(np.sum(result) / len(data) * 100)
            }
    
    # Добавляем метрики качества, если они есть
    if metrics_results:
        report_data['metrics'] = metrics_results
    
    # Добавляем результаты экспериментов, если они есть
    if experiment_results is not None:
        report_data['experiment'] = {
            'total_experiments': len(experiment_results),
            'parameters': list(experiment_results.columns),
            'best_f1': experiment_results.loc[experiment_results['f1'].idxmax()].to_dict(),
            'best_precision': experiment_results.loc[experiment_results['precision'].idxmax()].to_dict(),
            'best_recall': experiment_results.loc[experiment_results['recall'].idxmax()].to_dict()
        }
        
        # Добавляем оптимальные параметры, если они есть
        if optimal_params:
            report_data['optimal_params'] = optimal_params
    
    return report_data


def generate_anomaly_detection_yaml(report_data: Dict[str, Any]) -> str:
    """
    Генерирует YAML-секцию для отчета по обнаружению аномалий.
    
    Args:
        report_data: Данные отчета
        
    Returns:
        YAML-секция в виде строки
    """
    import datetime
    import uuid
    
    # Формируем YAML front matter
    experiment_id = f"ANOMALY_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid.uuid4())[:8]}"
    yaml_lines = [
        '---',
        f'experiment: "{experiment_id}"',
        f'date: "{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"',
        f'model: "anomaly_detection"',
        f'data_length: {report_data["data_length"]}',
        f'has_nan: {str(report_data["has_nan"]).lower()}'
    ]
    
    # Добавляем статистики данных
    yaml_lines.append('data_stats:')
    for key, value in report_data['data_stats'].items():
        yaml_lines.append(f'  {key}: {value}')
    
    # Добавляем параметры методов обнаружения
    yaml_lines.append('detection_params:')
    for method, params in report_data['detection_params'].items():
        yaml_lines.append(f'  {method}:')
        if isinstance(params, dict):
            for param_name, param_value in params.items():
                yaml_lines.append(f'    {param_name}: {param_value}')
        else:
            yaml_lines.append(f'    enabled: {params}')
    
    # Добавляем результаты обнаружения
    yaml_lines.append('detection_results:')
    for method, results in report_data['detection_results'].items():
        yaml_lines.append(f'  {method}:')
        for key, value in results.items():
            yaml_lines.append(f'    {key}: {value}')
    
    # Добавляем метрики, если они есть
    if 'metrics' in report_data:
        yaml_lines.append('metrics:')
        for method, metrics in report_data['metrics'].items():
            yaml_lines.append(f'  {method}:')
            for metric_name, metric_value in metrics.items():
                yaml_lines.append(f'    {metric_name}: {metric_value}')
    
    # Добавляем информацию об экспериментах, если она есть
    if 'experiment' in report_data:
        yaml_lines.append('experiment:')
        yaml_lines.append(f'  total_experiments: {report_data["experiment"]["total_experiments"]}')
        if 'best_f1' in report_data['experiment']:
            yaml_lines.append('  best_f1:')
            for key, value in report_data['experiment']['best_f1'].items():
                yaml_lines.append(f'    {key}: {value}')
    
    yaml_lines.append('---\n')
    return '\n'.join(yaml_lines)


def format_anomaly_info_for_report(anomaly_info: List[Dict[str, Any]]) -> str:
    """
    Форматирует информацию об аномалиях для отчета.
    
    Args:
        anomaly_info: Список словарей с информацией об аномалиях
        
    Returns:
        Отформатированная строка markdown
    """
    if not anomaly_info:
        return "Не было добавлено аномалий."
    
    md = "## Внедренные аномалии\n\n"
    
    # Группируем аномалии по типам
    point_anomalies = [a for a in anomaly_info if a.get('type') == 'point']
    extended_anomalies = [a for a in anomaly_info if a.get('type') == 'extended']
    sensor_faults = [a for a in anomaly_info if a.get('type') == 'sensor_fault']
    
    # Форматируем точечные аномалии
    if point_anomalies:
        md += "### Точечные аномалии\n\n"
        md += "| Индекс | Направление |\n"
        md += "|--------|------------|\n"
        for pa in point_anomalies:
            direction = "Вверх" if pa.get('increase', True) else "Вниз"
            md += f"| {pa.get('index')} | {direction} |\n"
        md += "\n"
    
    # Форматируем протяженные аномалии
    if extended_anomalies:
        md += "### Протяженные аномалии\n\n"
        md += "| Начальный индекс | Длительность | Смещение уровня |\n"
        md += "|-----------------|-------------|----------------|\n"
        for ea in extended_anomalies:
            md += f"| {ea.get('start_idx')} | {ea.get('duration')} | {ea.get('level_shift')} |\n"
        md += "\n"
    
    # Форматируем сбои датчиков
    if sensor_faults:
        md += "### Сбои датчиков\n\n"
        md += "| Начальный индекс | Длительность | Значение сбоя |\n"
        md += "|-----------------|-------------|---------------|\n"
        for sf in sensor_faults:
            md += f"| {sf.get('start_idx')} | {sf.get('duration')} | {sf.get('fault_value')} |\n"
        md += "\n"
    
    return md