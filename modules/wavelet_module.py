import numpy as np
import pandas as pd
import pywt
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import plotly.express as px

# --- Константы для единиц измерения --- 
TIME_UNITS = {
    "Измерения": "measurements",
    "Секунды": "seconds",
    "Минуты": "minutes",
    "Часы": "hours",
    "Дни": "days",
    "Недели": "weeks",
    "Месяцы": "months", # Приблизительно
    "Годы": "years"    # Приблизительно
}
DEFAULT_TIME_UNIT = "Дни"
MEASUREMENT_UNIT_KEY = "Измерения"
# -------------------------------------

# --- Новая функция: Определение временного шага --- 
def get_time_delta(index):
    """
    Определяет временной шаг (timedelta) для DatetimeIndex.

    Пытается определить частоту, если не удается, вычисляет медианную разницу.
    
    Параметры:
    ----------
    index : pd.Index
        Индекс временного ряда.
        
    Возвращает:
    -----------
    pd.Timedelta or None
        Временной шаг или None, если индекс не DatetimeIndex или шаг определить не удалось.
    """
    if isinstance(index, pd.DatetimeIndex):
        freq = pd.infer_freq(index)
        if freq:
            try:
                # Преобразуем частоту в Timedelta
                # pd.Timedelta(pd.tseries.frequencies.to_offset(freq)) может быть сложным,
                # пробуем проще:
                if len(index) > 1:
                     # Возьмем разницу между первыми двумя точками как оценку, 
                     # если частота стандартная (D, h, min, s)
                     # Для сложных частот (ME, QE) это может быть неточно, но как базовая оценка
                     delta = index[1] - index[0]
                     # Проверим, что это типичная частота
                     if freq in ['D', 'h', 'min', 's', 'ms', 'us', 'ns']: 
                         return delta
                     else: # Для месячной, недельной и т.д. вернем медиану
                          diffs = index.to_series().diff().dropna()
                          if not diffs.empty:
                             return diffs.median()
            except Exception as e:
                print(f"Не удалось определить Timedelta из частоты '{freq}': {e}")
                # Если не получилось из частоты, считаем медиану
                pass 
        
        # Если частоту определить не удалось или не получилось выше, считаем медиану
        if len(index) > 1:
            diffs = index.to_series().diff().dropna()
            if not diffs.empty:
                median_delta = diffs.median()
                # Проверяем, что результат - Timedelta
                if isinstance(median_delta, pd.Timedelta):
                     return median_delta
    return None

# Кодовые имена вейвлетов для PyWavelets
mother_switcher = {
    "Морле": 'morl',
    "Гаусс": 'gaus1',
    "Мексиканская шляпа": 'mexh',
    "Симлет": 'sym5',
    "Добеши": 'db8',
    "Койфлет": 'coif5'
}

# Функция для создания гауссова окна (замена для scipy.signal.gaussian)
def gaussian_window(M, std, sym=True):
    """
    Возвращает окно Гаусса.
    
    Параметры:
    ----------
    M : int
        Размер окна.
    std : float
        Стандартное отклонение окна Гаусса.
    sym : bool, optional
        Симметричное окно. Если False, возвращается M+1 точек окна 
        с M из них симметрично расположенных относительно центра.
    
    Возвращает:
    -----------
    window : ndarray
        Окно Гаусса с максимальной амплитудой 1.
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1)
    
    # Создаем индексы для окна
    if sym:
        n = np.arange(0, M) - (M - 1) / 2.0
    else:
        n = np.arange(0, M)
    
    # Вычисляем значения функции Гаусса
    sigma = 2 * std * std
    w = np.exp(-n**2 / sigma)
    
    # Нормализуем максимум к 1 (добавлено для соответствия описанию)
    # w = w / np.max(w) # Эта нормализация не соответствует scipy.signal.gaussian
    
    return w

@st.cache_data(ttl=3600)  # Кэширование результатов на час для ускорения повторных запусков
def wavelet_transform(time_series, mother_wavelet="Морле", num_scales=256, min_scale=None, max_scale=None, return_periods=False):
    """
    Выполняет вейвлет-преобразование временного ряда.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    mother_wavelet : str, optional
        Тип материнского вейвлета ("Морле", "Гаусс", "Мексиканская шляпа", "Симлет", "Добеши", "Койфлет")
    num_scales : int, optional
        Количество масштабов для анализа
    min_scale : float, optional
        Минимальный масштаб (если None, будет определен автоматически)
    max_scale : float, optional
        Максимальный масштаб (если None, будет определен автоматически)
    return_periods : bool, optional
        Возвращать ли периоды соответствующие масштабам
    
    Возвращает:
    -----------
    tuple
        Кортеж с результатами. Если return_periods=True, то (коэффициенты, частоты, периоды).
        Иначе (коэффициенты, частоты).
        В случае ошибки возвращает кортеж с пустыми массивами.
    """
    print(f"[wavelet_module.py | wavelet_transform] Начало. Вейвлет: {mother_wavelet}, макс.масштабы: {num_scales}, длина ряда: {len(time_series)}")
    # --- ИСПРАВЛЕНИЕ ТИПОВ: Явное преобразование к numpy array --- 
    if isinstance(time_series, (pd.Series, pd.DataFrame)):
        signal = np.array(time_series.iloc[:, 0].values if isinstance(time_series, pd.DataFrame) else time_series.values)
    else:
        signal = np.asarray(time_series) # Преобразуем к numpy array, если это возможно

    if signal.ndim > 1:
         signal = signal.flatten() # Убедимся, что сигнал одномерный
         
    if not np.issubdtype(signal.dtype, np.number):
         # Попытка преобразовать к числовому типу, если возможно
         try:
             signal = signal.astype(np.float64)
         except ValueError:
             if return_periods:
                 return np.array([]), np.array([]), np.array([])
             else:
                 return np.array([]), np.array([])

    if signal.size == 0:
        if return_periods: return np.array([]), np.array([]), np.array([])
        else: return np.array([]), np.array([])

    signal_length = len(signal)

    # --- ИСПРАВЛЕНИЕ: np.mean --- 
    try:
        # Убедимся, что работаем с float для mean
        signal_float = signal.astype(np.float64)
        signal_mean = np.mean(signal_float)
        signal = signal_float - signal_mean
    except ValueError:
        print("Предупреждение: Не удалось вычесть среднее значение.")
        pass 

    if signal_length > 100:
        window_size = min(5, signal_length // 20)
        if window_size > 1:
            weights = np.ones(window_size) / window_size
            # --- ИСПРАВЛЕНИЕ ТИПОВ: np.pad --- 
            try:
                 padded = np.pad(signal.astype(float), (window_size//2, window_size//2), mode='edge') # Указываем тип float
                 filtered_signal = np.convolve(padded, weights, mode='valid')
                 # --- ИСПРАВЛЕНИЕ ТИПОВ: Умножение --- 
                 signal = 0.8 * signal + 0.2 * filtered_signal # Numpy сам справится с типами здесь
            except Exception as e:
                 print(f"Предупреждение: Не удалось применить сглаживание: {e}")

    wavelet_name = mother_switcher.get(mother_wavelet, 'morl')
    max_auto_scale = signal_length // 3
    if min_scale is None: min_scale = 1
    if max_scale is None: max_scale = max_auto_scale
    actual_num_scales = num_scales
    if signal_length > 1000: actual_num_scales = max(64, num_scales // 2)
    
    # Убедимся что min_scale < max_scale
    if min_scale >= max_scale:
         max_scale = min_scale + 1 # Или другое значение по умолчанию
         print(f"Предупреждение: min_scale ({min_scale}) >= max_scale ({max_scale}). Устанавливаю max_scale = {max_scale}")

    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), actual_num_scales)

    try:
        coef, freqs = pywt.cwt(signal, scales, wavelet_name)
    except Exception as e:
        print(f"Ошибка при выполнении вейвлет-преобразования: {e}")
        # --- ИСПРАВЛЕНИЕ: Возврат кортежа правильной длины при ошибке --- 
        if return_periods:
            return np.array([]), np.array([]), np.array([]) # Возвращаем 3 пустых массива
        else:
            return np.array([]), np.array([]) # Возвращаем 2 пустых массива

    if return_periods:
        # --- ИЗМЕНЕНИЕ: Используем pywt.scale2frequency --- 
        # Вычисляем частоты для каждого масштаба
        # dt=1, так как работаем в "измерениях"
        try:
            # --- УДАЛЕНО: Блок проверки существования вейвлета через families/wavelist ---
            # Этот блок мог вызывать конфликты или ошибки из-за версий PyWavelets
            # или проблем с обнаружением атрибутов линтером.
            # Будем полагаться на то, что pywt.scale2frequency сам вызовет ошибку
            # при некорректном имени вейвлета.
            # ----------------------------------------------------------------

            # Применяем scale2frequency к каждому масштабу
            frequencies = np.array([pywt.scale2frequency(wavelet_name, s, precision=8) for s in scales])
            # Обработка случая, когда частота равна 0 или None
            periods = np.full_like(frequencies, np.inf)
            valid_freqs_mask = (frequencies != 0) & (~np.isnan(frequencies)) & (np.isfinite(frequencies))
            if np.any(valid_freqs_mask):
                 safe_frequencies = frequencies[valid_freqs_mask]
                 # Исправляем индексацию для безопасного присваивания
                 temp_periods = periods[valid_freqs_mask]
                 temp_periods[safe_frequencies == 0] = np.inf
                 temp_periods[safe_frequencies != 0] = 1.0 / safe_frequencies[safe_frequencies != 0]
                 periods[valid_freqs_mask] = temp_periods

        # except AttributeError as e_attr: # Если families или wavelist не найдены линтером/версией
        #     print(f"Предупреждение: Не удалось проверить вейвлет '{wavelet_name}' через списки PyWavelets ({e_attr}). Продолжение без строгой проверки.")
        #     # В этом случае, если wavelet_name некорректен, scale2frequency все равно вызовет ошибку ниже.
        #     try:
        #         frequencies = np.array([pywt.scale2frequency(wavelet_name, s, precision=8) for s in scales])
        #         periods = np.full_like(frequencies, np.inf)
        #         valid_freqs_mask = (frequencies != 0) & (~np.isnan(frequencies)) & (np.isfinite(frequencies))
        #         if np.any(valid_freqs_mask):
        #             safe_frequencies = frequencies[valid_freqs_mask]
        #             temp_periods = periods[valid_freqs_mask]
        #             temp_periods[safe_frequencies == 0] = np.inf
        #             temp_periods[safe_frequencies != 0] = 1.0 / safe_frequencies[safe_frequencies != 0]
        #             periods[valid_freqs_mask] = temp_periods
        #     except Exception as e_scale:
        #         print(f"Ошибка при вычислении частот/периодов (после предупреждения о проверке): {e_scale}")
        #         return np.array([]), np.array([]), np.array([])

        except Exception as e: # Общая ошибка при вычислениях
            print(f"Ошибка при вычислении частот/периодов через scale2frequency для {wavelet_name}: {e}")
            return np.array([]), np.array([]), np.array([])
        # -------------------------------------------------
            
        print("[wavelet_module.py | wavelet_transform] Завершение (с периодами).")
        return coef, frequencies, periods # Возвращаем новые частоты и периоды
    else:
        # Если периоды не нужны, возвращаем частоты, которые вернул CWT
        # (они могут немного отличаться от scale2frequency)
        print("[wavelet_module.py | wavelet_transform] Завершение (без периодов, только частоты).")
        return coef, freqs 

# --- ИЗМЕНЕНИЕ: get_scale_ticks теперь принимает time_delta и unit --- 
def get_scale_ticks(min_period_meas, max_period_meas, time_delta=None, target_unit_key="Измерения", num_ticks=6):
    """
    Генерирует значения тиков для оси периодов (логарифмическая шкала).

    Параметры:
    ----------
    min_period_meas : float
        Минимальный период в измерениях.
    max_period_meas : float
        Максимальный период в измерениях.
    time_delta : pd.Timedelta, optional
        Временной шаг одного измерения.
    target_unit_key : str, optional
        Ключ целевой единицы измерения из TIME_UNITS.
    num_ticks : int, optional
        Желаемое количество тиков.

    Возвращает:
    -----------
    tuple (np.ndarray, list)
        Кортеж: (значения тиков в единицах log2(период_в_измерениях), 
                 текстовые метки тиков в выбранных единицах)
    """
    log_min, log_max = np.log2(max(1, min_period_meas)), np.log2(max(1, max_period_meas))
    log_tickvals_meas = np.linspace(log_min, log_max, num_ticks)
    tickvals_meas = np.exp2(log_tickvals_meas)

    # Форматируем текстовые метки в выбранных единицах
    ticktext = [format_period(p, time_delta, target_unit_key) for p in tickvals_meas]

    # Возвращаем логарифмические значения тиков (для оси) и текстовые метки
    return log_tickvals_meas, ticktext

# --- ИЗМЕНЕНИЕ: format_period для работы с timedelta и единицами --- 
def format_period(period_meas, time_delta=None, target_unit_key="Измерения"):
    """
    Форматирует значение периода в выбранных единицах.

    Параметры:
    ----------
    period_meas : float
        Период в количестве измерений.
    time_delta : pd.Timedelta, optional
        Временной шаг одного измерения.
    target_unit_key : str, optional
        Ключ целевой единицы измерения из TIME_UNITS ('Дни', 'Секунды' и т.д.).

    Возвращает:
    -----------
    str
        Отформатированное значение периода.
    """
    # Если выбраны "Измерения" или нет временного шага
    if target_unit_key == "Измерения" or time_delta is None:
        # Округляем до целых, если > 10, иначе 1 знак
        if period_meas < 10:
             return f"{period_meas:.1f} изм."
        else:
             return f"{period_meas:.0f} изм."

    # Если есть временной шаг, конвертируем период в timedelta
    try:
        period_time = period_meas * time_delta
    except TypeError:
        # Если time_delta не является Timedelta (например, None)
         return f"{period_meas:.0f} изм. (?)" # Возвращаем измерения с пометкой

    # Конвертируем timedelta в нужные единицы
    total_seconds = period_time.total_seconds()

    unit = TIME_UNITS.get(target_unit_key, "seconds") # По умолчанию секунды, если ключ не найден

    if unit == "years":
        val = total_seconds / (365.25 * 24 * 3600)
        unit_str = "г."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit == "months":
        val = total_seconds / (30.44 * 24 * 3600) # Средняя длина месяца
        unit_str = "мес."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit == "weeks":
        val = total_seconds / (7 * 24 * 3600)
        unit_str = "нед."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit == "days":
        val = total_seconds / (24 * 3600)
        unit_str = "дн."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit == "hours":
        val = total_seconds / 3600
        unit_str = "ч."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit == "minutes":
        val = total_seconds / 60
        unit_str = "мин."
        fmt = ".1f" if val < 10 else ".0f"
    else: # seconds или measurements (уже обработан)
        val = total_seconds
        unit_str = "сек."
        fmt = ".1f" if val < 10 else ".0f"

    return f"{val:{fmt}} {unit_str}"

# --- ИЗМЕНЕНИЕ: plot_wavelet_transform принимает time_delta и unit --- 
def plot_wavelet_transform(time_series_pd, coef, freqs, periods_meas, tickvals_log, ticktext, selected_unit_key):
    """
    Строит тепловую карту вейвлет-преобразования.

    Параметры:
    ----------
    time_series_pd : pd.Series или pd.DataFrame
         Временной ряд (с индексом)
    coef : np.ndarray
        Вейвлет-коэффициенты.
    freqs : np.ndarray
        Частоты.
    periods_meas : np.ndarray
        Периоды в измерениях.
    tickvals_log : np.ndarray
        Позиции тиков на оси Y (в log2 от периода в измерениях).
    ticktext : list
        Текстовые метки для тиков на оси Y.
    selected_unit_key : str
        Выбранная единица измерения периода (ключ из TIME_UNITS).
    """
    print(f"[wavelet_module.py | plot_wavelet_transform] Начало. Coef shape: {coef.shape if coef is not None else 'N/A'}, Periods_meas len: {len(periods_meas) if periods_meas is not None else 'N/A'}")
    # Определяем временной шаг
    time_delta = get_time_delta(time_series_pd.index)
    
    # Создаем массив текстовых строк для hovertext
    hover_texts = None # Инициализируем hover_texts
    hover_info_setting = 'text' # По умолчанию показываем кастомный текст

    # Ограничение на размер данных для генерации hover_texts
    MAX_TIME_POINTS_FOR_HOVERTEXT = 10000 # Порог

    if coef is not None and coef.shape[1] > MAX_TIME_POINTS_FOR_HOVERTEXT:
        print(f"[wavelet_module.py | plot_wavelet_transform] Слишком много временных точек ({coef.shape[1]}), отключаю детальный hovertext.")
        hover_info_setting = 'z' # Показываем только значение z (мощность)
    elif coef is not None: # coef существует и количество точек в норме
        hover_texts_list = []
        for i in range(len(periods_meas)):
            row_texts = []
            period_val_meas = periods_meas[i]
            # Форматируем период для hover text
            period_formatted = format_period(period_val_meas, time_delta, selected_unit_key)
            period_meas_formatted = format_period(period_val_meas, None, "Измерения") # Всегда добавляем измерения
            
            for j in range(coef.shape[1]):
                time_val = time_series_pd.index[j] if hasattr(time_series_pd, 'index') else j
                power_val = round(np.abs(coef)[i, j], 2)
                
                # Собираем текст для hover
                text = f"Время: {time_val}<br>"
                text += f"Период: {period_formatted}<br>"
                if selected_unit_key != "Измерения": # Показываем измерения, если они не выбраны основной единицей
                     text += f"({period_meas_formatted})<br>"
                text += f"Мощность: {power_val}"
                row_texts.append(text)
            hover_texts_list.append(row_texts)
        hover_texts = np.array(hover_texts_list)
        print(f"[wavelet_module.py | plot_wavelet_transform] Hover_texts shape: {hover_texts.shape}") # Логируем размер hover_texts
    else:
        # Случай, если coef is None, хотя это должно быть обработано раньше
        print("[wavelet_module.py | plot_wavelet_transform] Коэффициенты (coef) отсутствуют, hovertext не будет сгенерирован.")
        hover_info_setting = 'skip' # Ничего не показываем

    # fig = go.Figure(data=go.Heatmap(
    #     z=np.abs(coef),
    #     x=time_series_pd.index,
    #     y=np.log2(periods_meas), # Ось Y всегда в log2 от измерений
    #     colorscale='Viridis',
    #     colorbar=dict(title='Мощность'),
    #     hoverinfo='text', # Управляется hover_info_setting
    #     text=hover_texts # Будет None, если hover_info_setting не 'text'
    # ))

    heatmap_data = {}
    if coef is not None:
        heatmap_data['z'] = np.abs(coef)
        heatmap_data['x'] = time_series_pd.index
        heatmap_data['y'] = np.log2(periods_meas)
        heatmap_data['colorscale'] = 'Viridis'
        heatmap_data['colorbar'] = dict(title='Мощность')
        heatmap_data['hoverinfo'] = hover_info_setting
        if hover_info_setting == 'text' and hover_texts is not None:
            heatmap_data['text'] = hover_texts
        elif hover_info_setting == 'z':
            # Для hoverinfo='z' можно добавить hovertemplate для лучшего отображения только z
            heatmap_data['hovertemplate'] = 'Мощность: %{z:.2f}<extra></extra>'
    
    fig = go.Figure()
    if heatmap_data: # Добавляем слой, только если есть данные
        fig.add_trace(go.Heatmap(**heatmap_data))
    else:
        # Можно добавить сообщение или просто оставить пустой график
        fig.update_layout(title="Вейвлет-преобразование (нет данных для отображения)")

    fig.update_layout(
        title="Вейвлет-преобразование",
        xaxis_title="Время",
        yaxis_title=f"Период ({selected_unit_key})", # Заголовок оси Y в выбранных единицах
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals_log, # Позиции тиков
            ticktext=ticktext,     # Текстовые метки в выбранных единицах
        ),
    )
    
    return fig

# --- ИЗМЕНЕНИЕ: find_significant_periods_wavelet принимает time_delta и unit --- 
def find_significant_periods_wavelet(time_series_pd, mother_wavelet="Морле", power_threshold=0.2, num_scales=256, min_scale=1, max_scale=None, max_periods=10, threshold_percent=95, coef=None, periods_meas=None):
    """
    Находит значимые периодичности во временном ряде с использованием вейвлет-преобразования.
    Возвращает DataFrame с периодами в измерениях и в выбранных единицах времени (если возможно).
    
    Параметры:
    ----------
    time_series_pd : pandas.DataFrame или pd.Series
        Временной ряд для анализа (должен иметь индекс)
    mother_wavelet : str, optional
        Тип вейвлета для анализа
    power_threshold : float, optional
        Относительный порог мощности для определения значимых периодов (доля от максимума)
    num_scales : int, optional
        Количество масштабов для анализа
    min_scale : int, optional
        Минимальный масштаб для анализа
    max_scale : int, optional
        Максимальный масштаб для анализа
    max_periods : int, optional
        Максимальное количество возвращаемых периодов
    threshold_percent : float, optional
        Процентиль для определения порога значимости
    coef : np.ndarray, optional
        Предварительно рассчитанные вейвлет-коэффициенты.
    periods_meas : np.ndarray, optional
        Предварительно рассчитанные периоды в измерениях.
        
    Возвращает:
    -----------
    pandas.DataFrame
        Датафрейм с колонками: 
        'Период (изм.)', 'Мощность', 
        'Период (формат.)' (отформатированный в единицах по умолчанию или измерениях)
    """
    print(f"[wavelet_module.py | find_significant_periods_wavelet] Начало. Вейвлет: {mother_wavelet}, макс.масштабы: {num_scales}")
    # --- Получение данных и индекса --- 
    if isinstance(time_series_pd, (pd.Series, pd.DataFrame)):
        time_series = np.array(time_series_pd.iloc[:, 0].values if isinstance(time_series_pd, pd.DataFrame) else time_series_pd.values)
        ts_index = time_series_pd.index
    else:
        time_series = np.asarray(time_series_pd)
        ts_index = pd.RangeIndex(start=0, stop=len(time_series), step=1) # Создаем RangeIndex если нет
        
    # --- Определяем временной шаг --- 
    time_delta = get_time_delta(ts_index)

    if time_series.ndim > 1: time_series = time_series.flatten()
    if not np.issubdtype(time_series.dtype, np.number):
        try: time_series = time_series.astype(np.float64)
        except ValueError: return pd.DataFrame()
    if time_series.size == 0: return pd.DataFrame()

    # --- Вычисление или использование переданных coef/periods --- 
    if coef is None or periods_meas is None:
        # Вычитание среднего
        try:
            ts_float = time_series.astype(np.float64)
            time_series_proc = ts_float - np.mean(ts_float)
        except ValueError: 
             time_series_proc = time_series # Используем исходный, если не вышло
             pass

        # Сглаживание
        window_length = min(51, max(1, len(time_series_proc) // 10 * 2 + 1))
        polyorder = min(3, max(0, window_length - 2))
        time_series_smoothed = time_series_proc
        if window_length > polyorder and window_length > 0:
            try: time_series_smoothed = savgol_filter(time_series_proc, window_length, polyorder)
            except ValueError: pass

        # Параметры CWT
        signal_length = len(time_series_smoothed)
        if max_scale is None: max_scale = signal_length // 3
        if max_scale <= min_scale: max_scale = min_scale + 1 
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
        wavelet = mother_switcher.get(mother_wavelet, 'morl')

        # CWT
        try: 
            transform_result = wavelet_transform(time_series_smoothed, mother_wavelet, num_scales, min_scale, max_scale, return_periods=True)
            if len(transform_result) != 3: raise ValueError("Неверный результат wavelet_transform")
            coef, _, periods_meas = transform_result # _ для freqs
        except Exception as e: 
             print(f"Ошибка CWT в find_significant_periods: {e}")
             return pd.DataFrame()
    # -----------------------------------------------------
    else:
         # Используем переданные coef и periods_meas
         # Проверяем согласованность размерностей
         if coef.shape[0] != periods_meas.shape[0]:
              st.error("Размерности переданных coef и periods_meas не совпадают!")
              return pd.DataFrame()
              
    # Расчет мощности (если coef был пересчитан или передан)
    if coef.size == 0 or periods_meas.size == 0:
         return pd.DataFrame()
         
    power = np.abs(coef) ** 2
    combined_power = 0.7 * np.mean(power, axis=1) + 0.3 * np.max(power, axis=1)
    
    if np.all(np.isinf(periods_meas)) or combined_power.size == 0: return pd.DataFrame()

    # Нормализация мощности
    max_combined_power = np.max(combined_power)
    normalized_power = combined_power / max_combined_power if max_combined_power > 0 else np.zeros_like(combined_power)

    # Поиск пиков
    height_threshold = power_threshold * (threshold_percent / 100)
    peaks, _ = find_peaks(normalized_power, height=height_threshold, prominence=max(0, height_threshold/2), distance=3)
    if len(peaks) < 2: peaks, _ = find_peaks(normalized_power, height=max(0,height_threshold/2), prominence=max(0,height_threshold/4), distance=2)
    if len(peaks) == 0:
        sorted_indices = np.argsort(-normalized_power)
        peaks = []
        for idx in sorted_indices:
            if not np.isinf(periods_meas[idx]): # Используем periods_meas
                is_far_enough = all(abs(periods_meas[idx] - periods_meas[p]) / periods_meas[p] > 0.1 for p in peaks if periods_meas[p] != 0) if peaks else True
                if not peaks or is_far_enough:
                    peaks.append(idx)
                    if len(peaks) >= 3: break
        peaks = np.array(peaks)
    if len(peaks) == 0: return pd.DataFrame()

    # --- Формирование результатов --- 
    results = []
    # default_unit = DEFAULT_TIME_UNIT if time_delta else MEASUREMENT_UNIT_KEY # Убрано, форматирование снаружи
    
    for peak_idx in peaks:
        period_meas_val = float(periods_meas[peak_idx]) # Используем periods_meas
        power_value = float(normalized_power[peak_idx])
        if np.isinf(period_meas_val): continue

        # --- ИЗМЕНЕНИЕ: Не форматируем здесь --- 
        # period_formatted = format_period(period_meas, time_delta, default_unit)
        # period_meas_formatted = format_period(period_meas, None, MEASUREMENT_UNIT_KEY)

        results.append({
            'Период (изм.)': period_meas_val, # Возвращаем числовое значение
            'Мощность': power_value,
            # 'Период (формат.)': period_formatted # Убрано
        })

    if not results: return pd.DataFrame()

    results_df = pd.DataFrame(results)
    # Используем числовой 'Период (изм.)' для удаления дубликатов, округляя его для группировки
    results_df['Период (окр.)'] = results_df['Период (изм.)'].round(1) # Округляем до 1 знака для группировки
    results_df = results_df.sort_values('Мощность', ascending=False)
    results_df = results_df.drop_duplicates(subset=['Период (окр.)'], keep='first')
    results_df = results_df.sort_values('Мощность', ascending=False)
    print(f"[wavelet_module.py | find_significant_periods_wavelet] Завершение. Возвращено {len(results_df)} значимых периодов.")
    return results_df[['Период (изм.)', 'Мощность']].head(max_periods) # Возвращаем нужные колонки

# --- ИЗМЕНЕНИЕ: plot_wavelet_periodicity_analysis принимает рассчитанные данные --- 
def plot_wavelet_periodicity_analysis(
    time_series_pd, 
    mother_wavelet="Морле", 
    max_scales=None, 
    selected_unit_key="Измерения",
    # Добавляем опциональные параметры
    coef=None,
    periods_meas=None,
    significant_periods_df=None):
    """
    Создает визуализацию для анализа периодичностей.
    Ось X теперь отображается в выбранных единицах.
    Может принимать pre-calculated coef, periods_meas, significant_periods_df.
    
    Параметры:
    ----------
    ...
    coef : np.ndarray, optional
        Предварительно рассчитанные вейвлет-коэффициенты.
    periods_meas : np.ndarray, optional
        Предварительно рассчитанные периоды в измерениях.
    significant_periods_df : pd.DataFrame, optional
        Предварительно рассчитанный DataFrame значимых периодов (с колонкой 'Период (изм.)').
        
    Возвращает:
    -----------
    fig : plotly.graph_objects.Figure
    # significant_periods : pd.DataFrame # Больше не возвращаем, т.к. можем получить снаружи
    """
    print(f"[wavelet_module.py | plot_wavelet_periodicity_analysis] Начало. Coef shape: {coef.shape if coef is not None else 'N/A'}, Periods_meas len: {len(periods_meas) if periods_meas is not None else 'N/A'}, Sig.Periods DF len: {len(significant_periods_df) if significant_periods_df is not None else 'N/A'}")
    # --- Определяем временной шаг --- 
    time_delta = get_time_delta(time_series_pd.index)
    
    # --- Вычисление или использование переданных данных --- 
    if coef is None or periods_meas is None:
         if max_scales is None: max_scales = min(100, max(1, len(time_series_pd) // 2))
         # Вейвлет-преобразование (если не передано)
         transform_result = wavelet_transform(
             time_series_pd,
             mother_wavelet=mother_wavelet,
             num_scales=max_scales,
             return_periods=True
         )
         if len(transform_result) != 3 or transform_result[0].size == 0:
             return go.Figure() # Возвращаем пустую фигуру
         coef, _, periods_meas = transform_result # _ для freqs

    if significant_periods_df is None:
         # Находим значимые периоды (если не переданы)
         # Передаем coef и periods_meas, чтобы избежать повторного CWT
         significant_periods_df = find_significant_periods_wavelet(
             time_series_pd, 
             mother_wavelet=mother_wavelet, 
             num_scales=max_scales if max_scales is not None else (min(100, max(1, len(time_series_pd) // 2))), 
             power_threshold=0.1,
             coef=coef, # Передаем рассчитанные/полученные coef
             periods_meas=periods_meas # Передаем рассчитанные/полученные periods_meas
         )
    # -----------------------------------------------------
    
    # Проверка наличия данных после вычислений/получения
    if coef is None or coef.size == 0 or periods_meas is None or periods_meas.size == 0:
        st.warning("Нет данных коэффициентов или периодов для построения спектра мощности.")
        return go.Figure()

    # Расчет мощности
    power = np.abs(coef) ** 2
    mean_power = np.mean(power, axis=1)
    max_power = np.max(power, axis=1)
    combined_power = 0.7 * mean_power + 0.3 * max_power
    
    # significant_periods больше не рассчитывается здесь, он либо передан, либо рассчитан выше
    
    fig = go.Figure()
    
    # Исходный ряд
    ts_index = time_series_pd.index
    ts_values = time_series_pd.values
    if isinstance(time_series_pd, pd.DataFrame): ts_values = ts_values[:, 0]
    fig.add_trace(go.Scatter(x=ts_index, y=ts_values, mode='lines', name='Исходный ряд', visible='legendonly'))
    
    # --- Подготовка данных для графика спектра --- 
    finite_periods_mask = np.isfinite(periods_meas) & (periods_meas > 0)
    if not np.any(finite_periods_mask):
        st.warning("Нет конечных положительных периодов для отображения спектра.")
        return fig # Возвращаем фигуру с исходным рядом
        
    periods_meas_finite = periods_meas[finite_periods_mask]
    # Убедимся, что combined_power соответствует periods_meas по размеру
    if combined_power.shape[0] != periods_meas.shape[0]:
         st.error("Размерности мощности и периодов не совпадают после фильтрации!")
         # Пытаемся отфильтровать combined_power так же
         if combined_power.shape[0] == finite_periods_mask.shape[0]:
              combined_power_finite = combined_power[finite_periods_mask]
         else:
              return fig # Не можем продолжить безопасно
    else:
         combined_power_finite = combined_power[finite_periods_mask]
    
    # Нормализация мощности
    max_comb_power = np.max(combined_power_finite)
    normalized_power_finite = combined_power_finite / max_comb_power if max_comb_power > 0 else np.zeros_like(combined_power_finite)

    # Определяем диапазон отображения
    max_period_meas_to_show = None
    # Используем переданный/рассчитанный DataFrame significant_periods_df
    if significant_periods_df is not None and not significant_periods_df.empty and 'Период (изм.)' in significant_periods_df.columns:
         try:
             # Используем числовую колонку 'Период (изм.)'
             max_detected_period_meas = significant_periods_df['Период (изм.)'].max()
             max_period_meas_to_show = max(10, min(max_detected_period_meas * 2, periods_meas_finite.max()))
         except Exception as e:
              print(f"Ошибка при определении max_period_to_show: {e}")
              max_period_meas_to_show = min(1000, periods_meas_finite.max())
    else:
        max_period_meas_to_show = min(1000, periods_meas_finite.max())
        
    min_period_meas_to_show = max(1, periods_meas_finite.min())

    display_mask = (periods_meas_finite >= min_period_meas_to_show) & (periods_meas_finite <= max_period_meas_to_show)
    display_periods_meas = periods_meas_finite[display_mask]
    display_power_norm = normalized_power_finite[display_mask]
    
    if display_periods_meas.size == 0 or display_power_norm.size == 0:
         st.warning("Нет данных для отображения спектра мощности в выбранном диапазоне периодов.")
    else:
         print(f"[wavelet_module.py | plot_wavelet_periodicity_analysis] Display_periods_meas size for spectrum: {display_periods_meas.size}") # Логируем размер
         # --- График спектра мощности --- 
         fig.add_trace(go.Scatter(
             x=display_periods_meas, 
             y=display_power_norm,
             mode='lines',
             name='Спектр мощности',
             line=dict(color='red', width=2),
             customdata=np.array([format_period(p, time_delta, selected_unit_key) for p in display_periods_meas]),
             hovertemplate = '<b>Период</b>: %{customdata}<br>' +
                             '(Измерения: %{x:.1f})<br>'+
                             'Мощность: %{y:.3f}<extra></extra>' 
         ))

    # --- Отметки для значимых периодов --- 
    # Используем переданный/рассчитанный DataFrame significant_periods_df
    if significant_periods_df is not None and not significant_periods_df.empty and 'Период (изм.)' in significant_periods_df.columns:
        period_points_meas = []
        power_points = []
        text_labels = [] 
        annotation_labels = []
        
        for i, row in significant_periods_df.iterrows():
            try:
                period_meas_val = float(row['Период (изм.)']) # Колонка уже числовая
                power_val = float(row['Мощность'])
                # Форматируем здесь для отображения
                period_formatted_val = format_period(period_meas_val, time_delta, selected_unit_key)
                period_meas_formatted = format_period(period_meas_val, None, MEASUREMENT_UNIT_KEY)
            except Exception as e:
                print(f"Ошибка обработки строки significant_periods: {row}, {e}")
                continue

            if not (min_period_meas_to_show <= period_meas_val <= max_period_meas_to_show) or not np.isfinite(period_meas_val):
                continue
            
            period_points_meas.append(period_meas_val)
            power_points.append(power_val) 
            text_labels.append(f"{period_formatted_val}<br>({period_meas_formatted})<br>Мощность: {power_val:.3f}")
            annotation_labels.append(period_formatted_val)
            
            fig.add_vline(
                x=period_meas_val,
                line=dict(color="green", width=1, dash="dash"),
                opacity=0.7,
                annotation_text=period_formatted_val, 
                annotation_position="top right",
                annotation_font=dict(size=12, color='green')
            )
        
        if period_points_meas:
            fig.add_trace(go.Scatter(
                x=period_points_meas,
                y=power_points,
                mode='markers', 
                marker=dict(size=10, color='green', symbol='circle-open', line=dict(width=2, color='green')),
                name='Значимые периоды',
                hovertemplate = '<b>ЗНАЧИМЫЙ ПЕРИОД</b><br>' + 
                                '%{text}<extra></extra>', 
                text = text_labels
            ))

    # --- Настройка оси X --- 
    x_tickvals_meas, x_ticktext = get_scale_ticks(
        min_period_meas_to_show,
        max_period_meas_to_show, 
        time_delta,
        selected_unit_key,
        num_ticks=8
    )
        
    fig.update_layout(
        title="Анализ периодичностей (Спектр мощности)",
        xaxis_title=f"Период ({selected_unit_key})",
        yaxis_title="Нормализованная мощность",
        xaxis_type="log",
        xaxis=dict(
            tickmode="array",
            tickvals=np.exp2(x_tickvals_meas), 
            ticktext=x_ticktext               
        ),
        legend_title="Легенда",
        hovermode="closest",
        yaxis=dict(range=[-0.05, 1.1])
    )
    print(f"[wavelet_module.py | plot_wavelet_periodicity_analysis] Завершение.")
    return fig # Больше не возвращаем significant_periods


def plot_periodicity_heatmap(time_series, significant_periods_df, title="Тепловая карта периодичностей"):
    """
    Создает тепловую карту для визуализации наиболее значимых периодичностей в данных.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    significant_periods_df : pd.DataFrame
        Датафрейм со значимыми периодами
    title : str, optional
        Заголовок графика
        
    Возвращает:
    -----------
    plotly.graph_objects.Figure
        Тепловая карта периодичностей
    """
    # Если нет периодов или датафрейм пустой, возвращаем пустой график
    if significant_periods_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Нет обнаруженных периодичностей")
        return fig
    
    # Получаем наиболее значимые периоды (топ-5)
    top_periods = significant_periods_df.head(5)['Период'].values
    
    # Создаем DataFrame с индексом времени и колонками периодов
    if isinstance(time_series, pd.DataFrame):
        data = time_series.iloc[:, 0].copy()
    elif isinstance(time_series, pd.Series):
        data = time_series.copy()
    else:
        data = pd.Series(time_series)
    
    # Создаем пустой DataFrame для хранения компонент периодичностей
    periodicity_df = pd.DataFrame(index=data.index)
    
    # Для каждого значимого периода создаем скользящее среднее
    for period in top_periods:
        window_size = int(np.round(period))
        if window_size < 2:
            window_size = 2
        
        # Используем скользящее среднее для выделения компоненты
        periodicity_df[f'Период {period:.1f}'] = data.rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
    
    # Нормализуем значения для лучшей визуализации
    normalized_df = (periodicity_df - periodicity_df.min()) / (periodicity_df.max() - periodicity_df.min())
    
    # Создаем тепловую карту
    fig = px.imshow(
        normalized_df.T,
        labels=dict(x="Время", y="Период", color="Нормализованное значение"),
        title=title,
        aspect="auto"
    )
    
    # Настраиваем макет
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_periods))),
            ticktext=[f"Период {p:.1f}" for p in top_periods]
        )
    )
    
    return fig