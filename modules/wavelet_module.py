import numpy as np
import pandas as pd
import pywt
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import plotly.express as px

# Соотношения между масштабом и периодом для различных вейвлетов
# Эти соотношения соответствуют теоретическим значениям для данных вейвлетов
wavelet_period_factors = {
    "Морле": 1.22,  # Скорректированное значение для вейвлета Морле (было: 4 * np.pi / (6 + np.sqrt(2 + 6**2)))
    "Гаусс": 2.0,  # Более точный коэффициент для первой производной гауссова вейвлета
    "Мексиканская шляпа": 2 * np.pi / np.sqrt(2.5),  # Для "мексиканской шляпы" (DOG вейвлет)
    "Симлет": 1.5,  # Для симлета 5-го порядка
    "Добеши": 1.4,  # Для Добеши 8-го порядка
    "Койфлет": 1.2   # Для Койфлета 5-го порядка
}

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
    w = np.exp(-0.5 * (n / std)**2)
    
    # Нормализуем максимум к 1
    w = w / np.max(w)
    
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
    np.ndarray или tuple
        Массив вейвлет-коэффициентов или кортеж (коэффициенты, масштабы, периоды)
    """
    # Получаем данные из временного ряда
    if isinstance(time_series, pd.DataFrame):
        # Если передан DataFrame, берем первый столбец
        signal = time_series.iloc[:, 0].values
    elif isinstance(time_series, pd.Series):
        # Если передана Series, берем значения
        signal = time_series.values
    else:
        # Иначе преобразуем в numpy массив
        signal = np.array(time_series)
    
    # Получаем длину сигнала
    signal_length = len(signal)
    
    # Предобработка сигнала - удаляем среднее значение
    signal = signal - np.mean(signal)
    
    # Упрощенная предобработка: применяем сглаживание только для сигналов среднего и большого размера
    if signal_length > 100:
        # Применяем скользящее среднее без использования сложных окон
        window_size = min(5, signal_length // 20)
        if window_size > 1:
            # Используем простой метод свертки для сглаживания
            weights = np.ones(window_size) / window_size
            # Дополняем сигнал по краям для избежания краевых эффектов
            padded = np.pad(signal, (window_size//2, window_size//2), mode='edge')
            filtered_signal = np.convolve(padded, weights, mode='valid')
            
            # Применяем слабое сглаживание для сохранения информации о периодах
            signal = 0.8 * signal + 0.2 * filtered_signal
    
    # Получаем материнский вейвлет из словаря
    wavelet_name = mother_switcher.get(mother_wavelet, 'morl')
    
    # Определяем максимальный масштаб (не более 1/3 длины сигнала)
    max_auto_scale = signal_length // 3
    
    # Используем переданные параметры или автоматические
    if min_scale is None:
        # Минимальный масштаб - не менее 1
        min_scale = 1
    
    if max_scale is None:
        max_scale = max_auto_scale
    
    # Оптимизация: уменьшаем количество масштабов для очень длинных сигналов
    actual_num_scales = num_scales
    if signal_length > 1000:
        # Для длинных сигналов уменьшаем количество масштабов для ускорения
        actual_num_scales = max(64, num_scales // 2)
    
    # Создаем логарифмическую шкалу масштабов для лучшего разрешения
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), actual_num_scales)
    
    # Выполняем непрерывное вейвлет-преобразование
    try:
        # Упрощаем логику вызова cwt - используем единый подход
        coef, freqs = pywt.cwt(signal, scales, wavelet_name)
    except Exception as e:
        print(f"Ошибка при выполнении вейвлет-преобразования: {e}")
        # В случае ошибки возвращаем пустые массивы
        if return_periods:
            return np.array([]), np.array([])
        else:
            return np.array([])
    
    # Если нужно возвращать периоды, конвертируем масштабы в периоды
    if return_periods:
        # Периоды обратно пропорциональны частотам
        if mother_wavelet in wavelet_period_factors:
            # Используем предопределенные коэффициенты для вейвлетов
            period_factor = wavelet_period_factors[mother_wavelet]
            periods = scales * period_factor
        else:
            # Для других вейвлетов используем обратные частоты
            periods = 1.0 / freqs
        
        return coef, freqs, periods
    else:
        return coef, freqs

def get_scale_ticks(min_period, max_period, num_ticks=6):
    log_min, log_max = np.log2(min_period), np.log2(max_period)
    log_ticks = np.linspace(log_min, log_max, num_ticks)
    return np.exp2(log_ticks)

def format_period(period, unit: str = 'days'):
    if unit == 'days':
        if period < 7:
            return f"{period:.0f} дней"
        elif period < 30:
            return f"{period/7:.1f} недель"
        elif period < 365:
            return f"{period/30:.1f} месяцев"
        else:
            return f"{period/365:.1f} лет"
    else:  # measurements
        return f"{period:.0f} измерений"

def plot_wavelet_transform(time_series, coef, freqs, periods, tickvals, ticktext, scale_unit):
    # Создаем массив текстовых строк для hovertext
    hover_texts = []
    for i in range(len(periods)):
        row_texts = []
        for j in range(coef.shape[1]):
            # Создаем текст для каждой ячейки тепловой карты
            time_val = time_series.index[j] if hasattr(time_series, 'index') else j
            period_val = round(periods[i], 1)
            power_val = round(np.abs(coef)[i, j], 2)
            text = f"Время: {time_val}<br>Период: {period_val} измерений<br>Мощность: {power_val}"
            row_texts.append(text)
        hover_texts.append(row_texts)
    
    hover_texts = np.array(hover_texts)
    
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(coef),
        x=time_series.index,
        y=np.log2(periods),
        colorscale='Viridis',
        colorbar=dict(title='Мощность'),
        # Используем готовый текст вместо шаблона
        hoverinfo='text',
        text=hover_texts
    ))
    
    fig.update_layout(
        title="Вейвлет-преобразование",
        xaxis_title="Время",
        yaxis_title=f"Период ({scale_unit})",
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
    )
    
    return fig

def find_significant_periods_wavelet(df, mother_wavelet="Морле", power_threshold=0.2, num_scales=256, min_scale=1, max_scale=None, max_periods=10, threshold_percent=95):
    """
    Находит значимые периодичности во временном ряде с использованием вейвлет-преобразования.
    Универсальный алгоритм поиска пиков в спектре мощности.
    
    Параметры:
    ----------
    df : pandas.DataFrame или pd.Series
        Временной ряд для анализа
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
        
    Возвращает:
    -----------
    pandas.DataFrame
        Датафрейм с периодами и их мощностью
    """
    # Проверяем тип входных данных
    if isinstance(df, pd.Series):
        time_series = df.values
    elif isinstance(df, pd.DataFrame):
        if len(df.columns) == 0:
            raise ValueError("DataFrame не содержит данных")
        time_series = df.iloc[:, 0].values
    else:
        raise ValueError("Входные данные должны быть pandas DataFrame или Series")
    
    # Удаляем среднее значение из ряда
    time_series = time_series - np.mean(time_series)
    
    # Сглаживаем ряд для уменьшения шума
    window_length = min(51, len(time_series) // 10 * 2 + 1)  # Адаптивный размер окна
    polyorder = min(3, window_length - 1)  # Порядок полинома
    try:
        time_series_smoothed = savgol_filter(time_series, window_length, polyorder)
    except Exception:
        time_series_smoothed = time_series
    
    # Определяем масштабы для вейвлет-преобразования
    if max_scale is None:
        max_scale = len(time_series) // 3
    
    # Создаем логарифмическую шкалу масштабов с оптимальным количеством точек для баланса
    # между точностью и производительностью
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    # Выбираем материнский вейвлет
    if mother_wavelet == "Морле":
        wavelet = 'morl'
    elif mother_wavelet == "Мексиканская шляпа":
        wavelet = 'mexh'
    else:
        wavelet = mother_switcher.get(mother_wavelet, 'morl')  # Используем словарь для получения кода вейвлета
    
    # Выполняем вейвлет-преобразование
    coefficients, frequencies = pywt.cwt(time_series_smoothed, scales, wavelet)
    
    # Вычисляем мощность вейвлет-преобразования
    power = np.abs(coefficients) ** 2
    
    # Для более точного анализа берем максимальную мощность по времени для каждого периода
    max_power = np.max(power, axis=1)
    # А также среднюю мощность
    mean_power = np.mean(power, axis=1)
    
    # Комбинируем максимальную и среднюю мощность для улучшения обнаружения пиков
    combined_power = 0.7 * mean_power + 0.3 * max_power
    
    # Вычисляем периоды, соответствующие частотам - используем тот же метод, что и в wavelet_transform
    if mother_wavelet in wavelet_period_factors:
        # Используем предопределенные коэффициенты для вейвлетов
        period_factor = wavelet_period_factors[mother_wavelet]
        periods = scales * period_factor
    else:
        # Для других вейвлетов используем обратные частоты
        periods = 1.0 / frequencies
    
    # Нормализуем мощность для определения порога
    normalized_power = combined_power / np.max(combined_power)
    
    # Адаптивно настраиваем порог высоты и prominence в зависимости от threshold_percent
    height_threshold = power_threshold * (threshold_percent / 100)
    
    # Находим пики с разными параметрами чувствительности
    # Сначала с более строгими для отбора наиболее ярких
    peaks, _ = find_peaks(
        normalized_power, 
        height=height_threshold,
        prominence=height_threshold/2,
        distance=3
    )
    
    # Если найдено менее 2 пиков, пробуем с более мягкими параметрами
    if len(peaks) < 2:
        peaks, _ = find_peaks(
            normalized_power, 
            height=height_threshold/2,
            prominence=height_threshold/4,
            distance=2
        )
    
    # Если все ещё нет пиков, берем точки с максимальной мощностью
    if len(peaks) == 0:
        # Берем топ-3 точки с наибольшей мощностью, исключая очень близкие
        sorted_indices = np.argsort(-normalized_power)
        peaks = []
        for idx in sorted_indices:
            # Проверяем, не слишком ли близко к уже найденным пикам
            if not peaks or all(abs(periods[idx] - periods[p]) / periods[p] > 0.1 for p in peaks):
                peaks.append(idx)
                if len(peaks) >= 3:  # Максимум 3 пика
                    break
        peaks = np.array(peaks)
    
    # Если все ещё нет пиков, возвращаем пустой датафрейм
    if len(peaks) == 0:
        return pd.DataFrame()
    
    # Создаем список для хранения результатов
    results = []
    
    # Для каждого найденного пика
    for peak_idx in peaks:
        period = periods[peak_idx]
        power_value = normalized_power[peak_idx]
        
        # Округляем период для отображения
        if period < 10:
            period_rounded = round(period, 1)
        else:
            period_rounded = round(period)
        
        # Добавляем строку результата
        results.append({
            'Период': period,
            'Период (округленно)': period_rounded,
            'Нормализованная мощность': power_value,
        })
    
    # Создаем датафрейм с результатами
    results_df = pd.DataFrame(results)
    
    # Удаляем дубликаты периодов (оставляем только с максимальной мощностью)
    results_df = results_df.sort_values('Нормализованная мощность', ascending=False)
    results_df = results_df.drop_duplicates(subset=['Период (округленно)'], keep='first')
    
    # Сортируем по убыванию мощности
    results_df = results_df.sort_values('Нормализованная мощность', ascending=False)
    
    # Выбираем не более max_periods результатов
    return results_df.head(max_periods)

def plot_wavelet_periodicity_analysis(time_series, mother_wavelet="Морле", max_scales=None):
    """
    Создает визуализацию для анализа периодичностей с помощью вейвлет-преобразования.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    mother_wavelet : str, optional
        Тип материнского вейвлета
    max_scales : int, optional
        Максимальное количество масштабов
        
    Возвращает:
    -----------
    fig : plotly.graph_objects.Figure
        Визуализация анализа периодичностей
    significant_periods : pd.DataFrame
        Данные о значимых периодичностях
    """
    # Определяем максимальное количество масштабов
    if max_scales is None:
        max_scales = min(100, len(time_series) // 2)
    
    # Получаем вейвлет-коэффициенты и масштабы с периодами
    coef, freqs, periods = wavelet_transform(
        time_series,
        mother_wavelet=mother_wavelet,
        num_scales=max_scales,
        return_periods=True
    )
    
    # Вычисляем мощность для каждого периода
    power = np.abs(coef) ** 2
    
    # Используем комбинацию средней и максимальной мощности для лучшей визуализации
    mean_power = np.mean(power, axis=1)
    max_power = np.max(power, axis=1)
    combined_power = 0.7 * mean_power + 0.3 * max_power
    
    # Находим значимые периоды используя точно те же параметры, что и в основном анализе
    significant_periods = find_significant_periods_wavelet(
        time_series, 
        mother_wavelet=mother_wavelet, 
        num_scales=max_scales,
        power_threshold=0.1
    )
    
    # Создаем графики спектра мощности
    fig = go.Figure()
    
    # Исходный ряд (в отдельный subplot для лучшей визуализации)
    fig.add_trace(go.Scatter(
        x=time_series.index if hasattr(time_series, 'index') else np.arange(len(time_series)),
        y=time_series if isinstance(time_series, pd.Series) else time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series,
        mode='lines',
        name='Исходный ряд',
        visible='legendonly'  # Скрываем по умолчанию для фокуса на спектре
    ))
    
    # Определяем максимальный период для отображения (не более 2x от максимального значимого периода)
    max_period_to_show = None
    if not significant_periods.empty:
        max_detected_period = significant_periods['Период'].max()
        # Ограничиваем максимальный отображаемый период, но не менее 1000
        max_period_to_show = max(1000, min(max_detected_period * 2, periods.max()))
    else:
        # Если нет значимых периодов, показываем все до разумного предела
        max_period_to_show = min(1000, periods.max())
    
    # Фильтруем периоды и соответствующие мощности для отображения
    display_mask = periods <= max_period_to_show
    display_periods = periods[display_mask]
    normalized_power = combined_power / np.max(combined_power)
    display_power = normalized_power[display_mask]
    
    # Основной спектр мощности (комбинированный)
    fig.add_trace(go.Scatter(
        x=display_periods,
        y=display_power,
        mode='lines',
        name='Спектр мощности',
        line=dict(color='red', width=2)
    ))
    
    # Отметки для значимых периодов - для каждого периода из датафрейма
    if not significant_periods.empty:
        # Создаем массивы для всех периодов
        period_points = []
        power_points = []
        text_labels = []
        
        for i, row in significant_periods.iterrows():
            # Используем точные значения из таблицы без округления для согласованности
            period = row['Период']
            # Пропускаем периоды вне диапазона отображения
            if period > max_period_to_show:
                continue
                
            # Находим ближайший индекс в массиве периодов
            idx = np.argmin(np.abs(periods - period))
            actual_power = normalized_power[idx]
            
            period_points.append(period)
            power_points.append(actual_power)
            text_labels.append(f"{row['Период (округленно)']}")
            
            # Добавляем вертикальную линию для каждого периода
            fig.add_vline(
                x=period,
                line=dict(color="green", width=2, dash="dash"),
                opacity=0.8,
                annotation_text=f"{row['Период (округленно)']}",
                annotation_position="top right",
                annotation_font=dict(size=14, color='black', family='Arial Black')
            )
        
        # Добавляем все значимые периоды как одну точечную серию с большими маркерами
        if period_points:  # Проверяем, что список не пустой
            fig.add_trace(go.Scatter(
                x=period_points,
                y=power_points,
                mode='markers+text',
                text=text_labels,
                textposition='top center',
                marker=dict(size=16, color='green', symbol='circle', line=dict(width=2, color='black')),
                name='Значимые периоды'
            ))
    
    # Создаем логарифмические метки для оси X в человекочитаемом формате
    tickvals = []
    ticktext = []
    
    # Определяем диапазон периодов для отображения на шкале
    min_period = max(1, min(display_periods))
    
    # Создаем разумные метки для логарифмической шкалы
    if max_period_to_show <= 10:
        # Для малых периодов используем линейную шкалу
        tickvals = list(range(1, int(max_period_to_show) + 1))
        ticktext = [str(x) for x in tickvals]
    elif max_period_to_show <= 100:
        # Для средних периодов используем шаг 10
        tickvals = [1, 2, 5, 10, 20, 50, 100]
        tickvals = [x for x in tickvals if x <= max_period_to_show]
        ticktext = [str(x) for x in tickvals]
    else:
        # Для больших периодов используем логарифмический шаг
        base = 10
        power = 0
        while base**power <= max_period_to_show:
            if base**power >= min_period:
                tickvals.append(base**power)
                if base**power >= 1000:
                    ticktext.append(f"{base**power/1000:.0f}K")
                else:
                    ticktext.append(str(base**power))
            power += 1
            
            # Добавляем промежуточные значения для лучшего восприятия
            if base**(power-0.5) <= max_period_to_show and base**(power-0.5) >= min_period and power > 1:
                mid_val = int(base**(power-0.5))
                tickvals.append(mid_val)
                if mid_val >= 1000:
                    ticktext.append(f"{mid_val/1000:.1f}K")
                else:
                    ticktext.append(str(mid_val))
    
    # Настройка макета для лучшей визуализации
    fig.update_layout(
        title="Анализ периодичностей с помощью вейвлет-преобразования",
        xaxis_title="Период",
        yaxis_title="Нормализованная мощность",
        xaxis_type="log",
        xaxis=dict(
            range=[np.log10(min_period), np.log10(max_period_to_show)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext
        ),
        legend_title="Легенда",
        hovermode="closest",
        # Ограничиваем диапазон y для лучшего просмотра пиков
        yaxis=dict(range=[-0.1, 1.1])
    )
    
    # Увеличиваем размер маркеров и шрифта подписей для лучшей видимости
    fig.update_traces(
        selector=dict(mode='markers+text'),
        marker=dict(size=16),
        textfont=dict(size=14, color='black', family='Arial Black')
    )
    
    return fig, significant_periods

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