import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal

def get_fft_values(y_values, fs):
    """
    Вычисляет значения спектра Фурье.
    
    Параметры:
    ----------
    y_values : array-like
        Значения временного ряда
    fs : float
        Частота дискретизации
        
    Возвращает:
    -----------
    tuple
        (частоты, амплитуды)
    """
    N = len(y_values)
    fft_values = np.fft.rfft(y_values)
    fft_freq = np.fft.rfftfreq(N, d=1/fs)
    fft_amplitude = np.abs(fft_values) / N  # Нормализация амплитуды
    return fft_freq, fft_amplitude

@st.cache_data(ttl=3600)  # Кэширование результатов на час для ускорения повторных запусков
def fft_transform(time_series, nfft=None):
    """
    Выполняет преобразование Фурье временного ряда.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    nfft : int, optional
        Количество точек FFT (если None, используется следующая степень 2)
        
    Возвращает:
    -----------
    tuple
        (частоты, амплитуды, периоды)
    """
    # Подготовка сигнала
    if isinstance(time_series, pd.DataFrame):
        signal = time_series.iloc[:, 0].values
    elif isinstance(time_series, pd.Series):
        signal = time_series.values
    elif not isinstance(signal, np.ndarray):
        signal = np.array(time_series)

    # Удаляем среднее значение для лучшего анализа периодичностей
    signal = signal - np.mean(signal)
    
    # Определение временных параметров
    time = np.arange(len(signal))
    dt = 1.0  # Шаг дискретизации (по умолчанию 1)
    fs = 1 / dt  # Частота дискретизации
    
    # Определение количества точек FFT
    if nfft is None:
        nfft = next_power_of_2(len(signal))
    
    # Применение окна Ханна для уменьшения спектральной утечки
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # Выполнение FFT
    freqs, amplitudes = get_fft_values(signal_windowed, fs)
    
    # Расчет периодов
    periods = 1.0 / np.maximum(freqs, 1e-10)  # Избегаем деления на ноль
    
    return freqs, amplitudes, periods

def next_power_of_2(n):
    """
    Вычисляет следующую степень 2 для заданного числа.
    
    Параметры:
    ----------
    n : int
        Исходное число
        
    Возвращает:
    -----------
    int
        Следующая степень 2
    """
    return 2**np.ceil(np.log2(n)).astype(int)

def find_significant_periods_fourier(time_series, power_threshold=0.2, num_points=None, max_periods=10):
    """
    Находит значимые периодичности во временном ряде с использованием преобразования Фурье.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    power_threshold : float, optional
        Относительный порог мощности для определения значимых периодов (доля от максимума)
    num_points : int, optional
        Количество точек для FFT (если None, используется оптимальное значение)
    max_periods : int, optional
        Максимальное количество возвращаемых периодов
        
    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм с периодами и их мощностью
    """
    # Выполняем преобразование Фурье
    freqs, amplitudes, periods = fft_transform(time_series, nfft=num_points)
    
    # Создаем датафрейм с результатами
    df = pd.DataFrame({
        'Частота': freqs,
        'Период': periods,
        'Амплитуда': amplitudes
    })
    
    # Отфильтровываем очень низкие частоты (DC компонент)
    df = df[df['Частота'] > 0.0001]
    
    # Нормализуем амплитуды для определения порога
    df['Нормализованная амплитуда'] = df['Амплитуда'] / df['Амплитуда'].max()
    
    # Находим пики в спектре амплитуд
    from scipy.signal import find_peaks
    
    # Сортируем по периоду для лучшего поиска пиков
    sorted_df = df.sort_values('Период')
    
    # Находим пики с адаптивными параметрами
    peaks, _ = find_peaks(
        sorted_df['Амплитуда'].values,
        height=sorted_df['Амплитуда'].max() * power_threshold,
        distance=3,  # Минимальная дистанция между пиками
        prominence=sorted_df['Амплитуда'].max() * (power_threshold / 2)  # Минимальная выраженность пика
    )
    
    # Если пики не найдены, используем прямую фильтрацию по амплитуде
    if len(peaks) == 0:
        significant_df = df[df['Нормализованная амплитуда'] > power_threshold].copy()
        significant_df = significant_df.sort_values('Амплитуда', ascending=False)
    else:
        # Выбираем индексы пиков из отсортированного датафрейма
        significant_df = sorted_df.iloc[peaks].copy()
    
    # Округляем период для отображения
    significant_df['Период (округленно)'] = significant_df['Период'].apply(
        lambda x: round(x, 1) if x < 10 else round(x)
    )
    
    # Сортируем по убыванию амплитуды и ограничиваем количество
    significant_df = significant_df.sort_values('Амплитуда', ascending=False).head(max_periods)
    
    return significant_df

def plot_fourier_periodicity_analysis(time_series, num_points=None, max_periods=10, power_threshold=0.2):
    """
    Создает визуализацию для анализа периодичностей с помощью преобразования Фурье.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Временной ряд для анализа
    num_points : int, optional
        Количество точек для FFT (если None, используется оптимальное значение)
    max_periods : int, optional
        Максимальное количество показываемых периодов
    power_threshold : float, optional
        Порог мощности для определения значимых периодов
        
    Возвращает:
    -----------
    tuple
        (график, значимые периоды)
    """
    import plotly.graph_objects as go
    
    # Выполняем преобразование Фурье
    freqs, amplitudes, periods = fft_transform(time_series, nfft=num_points)
    
    # Находим значимые периоды
    significant_periods = find_significant_periods_fourier(
        time_series, 
        power_threshold=power_threshold,
        num_points=num_points,
        max_periods=max_periods
    )
    
    # Создаем график спектра
    fig = go.Figure()
    
    # Нормализуем амплитуды для отображения
    normalized_amplitudes = amplitudes / np.max(amplitudes)
    
    # Добавляем линию спектра мощности
    # Отфильтровываем очень низкие частоты для лучшей визуализации
    mask = freqs > 0.0001
    filtered_periods = periods[mask]
    filtered_amplitudes = normalized_amplitudes[mask]
    
    # Ограничиваем максимальный период для отображения
    max_period_to_show = None
    if not significant_periods.empty:
        max_detected_period = significant_periods['Период'].max()
        # Ограничиваем максимальный отображаемый период, но не менее 1000
        max_period_to_show = max(1000, min(max_detected_period * 2, filtered_periods.max()))
    else:
        # Если нет значимых периодов, показываем все до разумного предела
        max_period_to_show = min(1000, filtered_periods.max())
    
    # Фильтруем периоды для отображения
    display_mask = filtered_periods <= max_period_to_show
    display_periods = filtered_periods[display_mask]
    display_amplitudes = filtered_amplitudes[display_mask]
    
    # Добавляем основной спектр
    fig.add_trace(go.Scatter(
        x=display_periods,
        y=display_amplitudes,
        mode='lines',
        name='Спектр мощности',
        line=dict(color='red', width=2)
    ))
    
    # Добавляем точки для значимых периодов
    if not significant_periods.empty:
        period_points = []
        power_points = []
        text_labels = []
        
        for i, row in significant_periods.iterrows():
            period = row['Период']
            # Пропускаем периоды вне диапазона отображения
            if period > max_period_to_show:
                continue
            
            # Находим ближайшее значение в спектре для этого периода
            idx = np.argmin(np.abs(filtered_periods - period))
            actual_power = filtered_amplitudes[idx]
            
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
        
        # Добавляем точки для значимых периодов
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
    
    # Создаем логарифмические метки для оси X
    tickvals = []
    ticktext = []
    
    # Определяем диапазон периодов для отображения
    min_period = max(1, min(display_periods))
    
    # Создаем логарифмическую шкалу меток
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
    
    # Настройка графика
    fig.update_layout(
        title="Анализ периодичностей с помощью преобразования Фурье",
        xaxis_title="Период",
        yaxis_title="Нормализованная амплитуда",
        xaxis_type="log",
        xaxis=dict(
            range=[np.log10(min_period), np.log10(max_period_to_show)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext
        ),
        legend_title="Легенда",
        hovermode="closest",
        yaxis=dict(range=[-0.1, 1.1])  # Ограничиваем диапазон для лучшей визуализации
    )
    
    # Увеличиваем размер маркеров
    fig.update_traces(
        selector=dict(mode='markers+text'),
        marker=dict(size=16),
        textfont=dict(size=14, color='black', family='Arial Black')
    )
    
    return fig, significant_periods