import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal

# --- Константы для единиц измерения (скопировано из wavelet_module) ---
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

# --- Функция для определения временного шага (скопировано из wavelet_module) ---
def get_time_delta(index):
    """
    Определяет временной шаг (timedelta) для DatetimeIndex.
    Пытается определить частоту, если не удается, вычисляет медианную разницу.
    """
    if isinstance(index, pd.DatetimeIndex):
        freq = pd.infer_freq(index)
        if freq:
            try:
                if len(index) > 1:
                     delta = index[1] - index[0]
                     if freq in ['D', 'h', 'min', 's', 'ms', 'us', 'ns']:
                         return delta
                     else:
                          diffs = index.to_series().diff().dropna()
                          if not diffs.empty:
                             return diffs.median()
            except Exception as e:
                print(f"Не удалось определить Timedelta из частоты '{freq}': {e}")
                pass
        if len(index) > 1:
            diffs = index.to_series().diff().dropna()
            if not diffs.empty:
                median_delta = diffs.median()
                if isinstance(median_delta, pd.Timedelta):
                     return median_delta
    return None

# --- Функция форматирования периода (адаптировано из wavelet_module) ---
def format_period(period_meas, time_delta=None, target_unit_key="Измерения"):
    """
    Форматирует значение периода в выбранных единицах.
    """
    if target_unit_key == MEASUREMENT_UNIT_KEY or time_delta is None:
        if period_meas < 1:
            return f"{period_meas:.2f} изм."
        elif period_meas < 10:
             return f"{period_meas:.1f} изм."
        else:
             return f"{period_meas:.0f} изм."

    try:
        period_time = period_meas * time_delta
    except TypeError:
         return f"{period_meas:.0f} изм. (?)"

    total_seconds = period_time.total_seconds()
    unit_name = TIME_UNITS.get(target_unit_key, "seconds")

    val, unit_str, fmt = 0, "", ""
    if unit_name == "years":
        val = total_seconds / (365.25 * 24 * 3600)
        unit_str = "г."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit_name == "months":
        val = total_seconds / (30.44 * 24 * 3600)
        unit_str = "мес."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit_name == "weeks":
        val = total_seconds / (7 * 24 * 3600)
        unit_str = "нед."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit_name == "days":
        val = total_seconds / (24 * 3600)
        unit_str = "дн."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit_name == "hours":
        val = total_seconds / 3600
        unit_str = "ч."
        fmt = ".1f" if val < 10 else ".0f"
    elif unit_name == "minutes":
        val = total_seconds / 60
        unit_str = "мин."
        fmt = ".1f" if val < 10 else ".0f"
    else: # seconds
        val = total_seconds
        unit_str = "сек."
        fmt = ".1f" if val < 10 else ".0f"
    return f"{val:{fmt}} {unit_str}"

# --- Функция для генерации тиков оси X (адаптировано из wavelet_module.get_scale_ticks) ---
def get_fourier_x_axis_ticks(min_period_meas, max_period_meas, time_delta=None, target_unit_key="Измерения", num_ticks=8):
    """
    Генерирует значения тиков для оси периодов Фурье (логарифмическая шкала).
    """
    if min_period_meas <= 0: min_period_meas = 1e-1 # Избегаем log(0)
    if max_period_meas <= min_period_meas: max_period_meas = min_period_meas * 10

    log_min, log_max = np.log10(min_period_meas), np.log10(max_period_meas) # Используем log10 для Фурье
    
    # Генерируем тики в логарифмическом пространстве
    tickvals_log_space = np.logspace(log_min, log_max, num_ticks)
    
    # Если нужно больше специфичных тиков (например, 1, 2, 5, 10...)
    # можно добавить логику как в wavelet_module или plot_fourier_periodicity_analysis ранее
    # Пока оставляем logspace для простоты
    
    tickvals_meas = tickvals_log_space
    ticktext = [format_period(p, time_delta, target_unit_key) for p in tickvals_meas]
    
    # Убедимся, что первый и последний тики соответствуют границам
    # Это может потребовать дополнительной логики для красивого округления
    
    return tickvals_meas, ticktext

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
        (частоты, амплитуды, периоды_в_измерениях, временной_шаг_dt)
    """
    # Подготовка сигнала
    ts_index = None
    signal_arr = None # Используем другое имя для ясности, что это будет numpy array

    if isinstance(time_series, pd.DataFrame):
        signal_arr = time_series.iloc[:, 0].values # .values обычно возвращает numpy array
        ts_index = time_series.index
    elif isinstance(time_series, pd.Series):
        signal_arr = time_series.values # .values обычно возвращает numpy array
        ts_index = time_series.index
    elif isinstance(time_series, np.ndarray):
        signal_arr = time_series
    else: # Для других типов (list, etc.)
        try:
            signal_arr = np.asarray(time_series, dtype=float) # Попытка преобразовать в float numpy array
        except Exception as e:
            st.error(f"Не удалось преобразовать входные данные в числовой массив: {e}")
            return np.array([]), np.array([]), np.array([]), None

    # Убедимся, что signal_arr это numpy array и не пустой
    if not isinstance(signal_arr, np.ndarray):
        try:
            signal_arr = np.asarray(signal_arr, dtype=float)
        except Exception as e:
            st.error(f"Данные не могут быть преобразованы в numpy массив: {e}")
            return np.array([]), np.array([]), np.array([]), None
            
    if signal_arr.ndim > 1:
        signal_arr = signal_arr.flatten() # Если вдруг многомерный
        
    if signal_arr.size == 0:
        st.warning("Входной временной ряд пуст после обработки.")
        return np.array([]), np.array([]), np.array([]), None
        
    # Проверка на числовой тип данных внутри массива
    if not np.issubdtype(signal_arr.dtype, np.number):
        try:
            signal_arr = signal_arr.astype(float) # Попытка приведения к float
        except ValueError:
            st.error("Временной ряд содержит нечисловые значения, которые не удалось преобразовать.")
            return np.array([]), np.array([]), np.array([]), None

    # Создаем RangeIndex если ts_index все еще None (например, если на вход был np.ndarray без индекса)
    if ts_index is None:
        ts_index = pd.RangeIndex(start=0, stop=len(signal_arr), step=1)

    # Удаляем среднее значение для лучшего анализа периодичностей
    try:
        # astype(float) здесь для гарантии, что mean вычисляется по float и вычитание корректно
        mean_val = np.mean(signal_arr.astype(float))
        signal_processed = signal_arr.astype(float) - mean_val
    except Exception as e:
        st.warning(f"Не удалось вычесть среднее из сигнала: {e}")
        signal_processed = signal_arr # Используем исходный (уже numpy) массив если вычитание не удалось

    # Определение временных параметров
    # time = np.arange(len(signal_processed)) # time не используется далее
    dt = 1.0  # Шаг дискретизации (по умолчанию 1 измерение)
    time_delta_actual = get_time_delta(ts_index) # Получаем реальный временной шаг

    # Если мы используем реальные единицы времени, fs должна быть основана на них.
    # Однако, для простоты и совместимости с "измерениями", оставим dt=1 для расчетов FFT,
    # а time_delta_actual будем использовать для форматирования периодов.
    fs = 1 / dt  # Частота дискретизации в "измерениях"

    # Определение количества точек FFT
    if nfft is None:
        nfft = next_power_of_2(len(signal_processed)) # Используем длину обработанного сигнала
    
    # Применение окна Ханна для уменьшения спектральной утечки
    if len(signal_processed) > 0: # Проверка перед созданием окна
        window = np.hanning(len(signal_processed))
        signal_windowed = signal_processed * window
    else: # Если сигнал пуст после обработки, не можем продолжить
        return np.array([]), np.array([]), np.array([]), time_delta_actual
    
    # Выполнение FFT
    freqs, amplitudes = get_fft_values(signal_windowed, fs)
    
    # Расчет периодов в "измерениях"
    periods_meas = 1.0 / np.maximum(freqs, 1e-10)  # Избегаем деления на ноль
    
    return freqs, amplitudes, periods_meas, time_delta_actual

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

def find_significant_periods_fourier(
    time_series, 
    power_threshold=0.2, 
    num_points=None, 
    max_periods=10,
    use_log_for_peaks=False, # Новый параметр
    log_peak_threshold_factor=1.5 # Фактор для порога в лог. шкале (медиана + N*std)
):
    """
    Находит значимые периодичности во временном ряде с использованием преобразования Фурье.
    Возвращает DataFrame с периодами в измерениях и их мощностью.
    """
    freqs, amplitudes, periods_meas, _ = fft_transform(time_series, nfft=num_points)
    
    df = pd.DataFrame({
        'Частота': freqs,
        'Период (изм.)': periods_meas,
        'Амплитуда': amplitudes
    })
    
    df = df[(df['Частота'] > 0.0001) & (df['Период (изм.)'] > 0)].copy() # Используем .copy() для избежания SettingWithCopyWarning
    
    if df.empty:
        st.warning("Спектр пуст после фильтрации начальных частот.")
        return pd.DataFrame(columns=['Период (изм.)', 'Мощность'])

    st.write("--- Логи поиска пиков ---")

    if use_log_for_peaks:
        st.write("Режим поиска пиков: Логарифмический")
        # Используем безопасный логарифм, добавляя небольшую константу
        df['LogАмплитуда'] = np.log10(df['Амплитуда'] + 1e-12) 
        
        median_log_amp = df['LogАмплитуда'].median()
        std_log_amp = df['LogАмплитуда'].std()
        # Динамический порог: медиана + N * стандартное отклонение
        peak_height_threshold = median_log_amp + log_peak_threshold_factor * std_log_amp 
        # Минимальная высота пика, чтобы отсечь совсем уж мелкие флуктуации фона
        # Увеличиваем строгость prominence по умолчанию
        min_peak_prominence = std_log_amp * 1.0 # Изменено с 0.5 на 1.0

        st.write(f"Лог. спектр: Медиана={median_log_amp:.3f}, Std={std_log_amp:.3f}")
        st.write(f"Порог высоты для пиков (в лог. шкале): {peak_height_threshold:.3f}")
        st.write(f"Минимальная prominence для пиков (в лог. шкале): {min_peak_prominence:.3f}")
        
        # Нормализация для столбца "Мощность" будет производиться позже, после выбора пиков
        # Для find_peaks используем непосредственно LogАмплитуда
        signal_for_peaks = df['LogАмплитуда'].values
        
        # Нормализация для отображения и конечного столбца "Мощность"
        # Отмасштабируем LogАмплитуда так, чтобы максимум был ~0 (для совместимости с "Мощность")
        # Это делается уже после нахождения пиков, чтобы не влиять на пороги
        
    else:
        st.write("Режим поиска пиков: Линейный (по нормализованной амплитуде)")
        max_amplitude_val = df['Амплитуда'].max()
        if max_amplitude_val > 0:
            df['Нормализованная амплитуда'] = df['Амплитуда'] / max_amplitude_val
        else:
            df['Нормализованная амплитуда'] = 0.0
        
        peak_height_threshold = power_threshold
        min_peak_prominence = power_threshold / 2.0
        signal_for_peaks = df['Нормализованная амплитуда'].values
        st.write(f"Максимальная амплитуда (для линейной норм.): {max_amplitude_val:.4f}")
        st.write(f"Порог высоты для пиков (линейный): {peak_height_threshold:.3f}")
        st.write(f"Минимальная prominence для пиков (линейный): {min_peak_prominence:.3f}")

    from scipy.signal import find_peaks
    
    df_cleaned = df.dropna(subset=['Период (изм.)', 'Амплитуда']) # Проверяем основные столбцы
    if use_log_for_peaks:
        df_cleaned = df_cleaned.dropna(subset=['LogАмплитуда'])
    else:
        df_cleaned = df_cleaned.dropna(subset=['Нормализованная амплитуда'])

    if df_cleaned.empty:
        st.warning("Данные спектра пусты после очистки NaN перед поиском пиков!")
        return pd.DataFrame(columns=['Период (изм.)', 'Мощность'])

    sorted_df = df_cleaned.sort_values('Период (изм.)')
    
    # Используем соответствующий сигнал для поиска пиков
    current_signal_for_peaks = None
    if use_log_for_peaks:
        current_signal_for_peaks = sorted_df['LogАмплитуда'].values
    else:
        current_signal_for_peaks = sorted_df['Нормализованная амплитуда'].values

    # Проверка, что current_signal_for_peaks не пустой
    if len(current_signal_for_peaks) == 0:
        st.warning("Сигнал для поиска пиков пуст после сортировки/очистки.")
        return pd.DataFrame(columns=['Период (изм.)', 'Мощность'])
        
    peaks_indices, properties = find_peaks(
        current_signal_for_peaks,
        height=peak_height_threshold,
        distance=5,  # Увеличим немного дистанцию между пиками
        prominence=min_peak_prominence 
    )
    
    st.write(f"Найдено пиков: {len(peaks_indices)}")
    if len(peaks_indices) > 0:
        st.write("Детали пиков (индексы из sorted_df):")
        for i, idx_in_sorted in enumerate(peaks_indices):
            period_val = sorted_df.iloc[idx_in_sorted]['Период (изм.)']
            amp_val = sorted_df.iloc[idx_in_sorted]['Амплитуда']
            
            log_amp_val_display = "N/A"
            if 'LogАмплитуда' in sorted_df.columns:
                 log_amp_val_display = f"{sorted_df.iloc[idx_in_sorted]['LogАмплитуда']:.3f}"
            
            norm_amp_val_display = "N/A"
            if 'Нормализованная амплитуда' in sorted_df.columns:
                 norm_amp_val_display = f"{sorted_df.iloc[idx_in_sorted]['Нормализованная амплитуда']:.3f}"

            st.write(
                f"  Пик {i+1}: Период={period_val:.2f} изм., Амплитуда={amp_val:.4f}, "
                f"Лог.Амплитуда={log_amp_val_display}, Норм.Амплитуда={norm_amp_val_display}, "
                f"Высота пика (из find_peaks): {properties['peak_heights'][i]:.3f}"
            )
    else:
        st.info("Автоматический поиск пиков не дал результатов с текущими параметрами.")


    if len(peaks_indices) == 0:
        # Фоллбек, если пики не найдены: просто берем топ N по исходной амплитуде (если не лог режим)
        # или по LogАмплитуде (если лог режим)
        # Это более простой вариант, чем сложная фильтрация по порогу без явных пиков
        st.warning("Пики не найдены find_peaks. Попытка выбрать топ по амплитуде/лог.амплитуде.")
        if use_log_for_peaks:
            significant_df_intermediate = sorted_df.sort_values('LogАмплитуда', ascending=False).head(max_periods).copy()
        else:
            significant_df_intermediate = sorted_df.sort_values('Амплитуда', ascending=False).head(max_periods).copy()
    else:
        significant_df_intermediate = sorted_df.iloc[peaks_indices].copy()
    
    # Теперь создаем колонку "Мощность" [0,1] для всех выбранных пиков
    if not significant_df_intermediate.empty:
        if use_log_for_peaks:
            # Нормализуем LogАмплитуду для пиков, чтобы получить "Мощность"
            min_log_peak = significant_df_intermediate['LogАмплитуда'].min()
            max_log_peak = significant_df_intermediate['LogАмплитуда'].max()
            if (max_log_peak - min_log_peak) > 1e-6: # если есть разброс
                significant_df_intermediate['Мощность'] = \
                    (significant_df_intermediate['LogАмплитуда'] - min_log_peak) / (max_log_peak - min_log_peak)
            else: # если все значения почти одинаковые
                significant_df_intermediate['Мощность'] = 1.0 
        else:
            # Для линейного режима, "Мощность" это 'Нормализованная амплитуда'
            # Убедимся, что колонка есть
            if 'Нормализованная амплитуда' in significant_df_intermediate:
                 significant_df_intermediate['Мощность'] = significant_df_intermediate['Нормализованная амплитуда']
            elif 'Амплитуда' in significant_df_intermediate: # Если вдруг нет нормализованной, но есть обычная
                 max_sig_amp = significant_df_intermediate['Амплитуда'].max()
                 if max_sig_amp > 0:
                      significant_df_intermediate['Мощность'] = significant_df_intermediate['Амплитуда'] / max_sig_amp
                 else:
                      significant_df_intermediate['Мощность'] = 0.0
            else: # Крайний случай
                 significant_df_intermediate['Мощность'] = 0.0
    else: # Если significant_df_intermediate пуст
        st.info("Значимых периодов для дальнейшей обработки не найдено.")
        return pd.DataFrame(columns=['Период (изм.)', 'Мощность'])

    # Выбираем нужные колонки и сортируем
    final_significant_df = significant_df_intermediate[['Период (изм.)', 'Мощность']].copy()
    final_significant_df = final_significant_df.sort_values('Мощность', ascending=False).head(max_periods)
    
    st.write("--- Конец логов поиска пиков ---")
    return final_significant_df

def interpolate_peak_parameters(freqs: np.ndarray, amplitudes: np.ndarray, peak_idx: int) -> tuple[float, float]:
    """
    Уточняет частоту и амплитуду пика через параболическую интерполяцию.
    freqs: массив частот (отсортированный или соответствующий amplitudes).
    amplitudes: массив амплитуд.
    peak_idx: индекс найденного пика в этих массивах.
    Возвращает (интерполированная_частота, интерполированная_амплитуда).
    """
    if peak_idx == 0 or peak_idx >= len(freqs) - 1 or peak_idx >= len(amplitudes) -1:
        # Невозможно выполнить интерполяцию на границах или если массивы слишком короткие
        return freqs[peak_idx], amplitudes[peak_idx]
        
    # Берем три точки вокруг пика
    # Убедимся, что y - это амплитуды, а не логарифмы для правильной интерполяции пика
    y = amplitudes[peak_idx-1:peak_idx+2]
    x = freqs[peak_idx-1:peak_idx+2]
    
    # Проверка на случай, если все y одинаковы (плоская вершина) или другие дегенеративные случаи
    if np.all(y == y[0]) or len(np.unique(x)) < 3: # если частоты совпадают
        return freqs[peak_idx], amplitudes[peak_idx]

    # Параболическая интерполяция y = ax^2 + bx + c не подходит напрямую для нахождения максимума амплитуды.
    # Лучше использовать формулу для нахождения вершины параболы, аппроксимирующей точки (x_i, y_i)
    # x_p = x_1 - ( (x_1-x_0)(y_1-y_2) - (x_1-x_2)(y_1-y_0) ) / ( 2 * ( (x_1-x_0)(y_1-y_2) - (x_1-x_2)(y_1-y_0) ) - ( (x_1-x_0)^2 * (y_1-y_2) - (x_1-x_2)^2 * (y_1-y_0) ) / (x_1-x_0) )
    # Это сложно. Проще использовать стандартные формулы для коэффициентов параболы ax^2+bx+c,
    # проходящей через (x_0,y_0), (x_1,y_1), (x_2,y_2), где (x_1,y_1) - пик.
    # x_coords = [-1, 0, 1] (относительно пика) и y_coords = [y_left, y_peak, y_right]
    # y = a*x^2 + b*x + c
    # y_left = a - b + c
    # y_peak = c
    # y_right = a + b + c
    # Решая систему: a = (y_left + y_right)/2 - y_peak; b = (y_right - y_left)/2; c = y_peak
    # Вершина параболы x_vertex = -b / (2a) (в относительных координатах [-1, 1])
    # freq_interp = freq_peak + x_vertex * (freq_step) # freq_step - шаг частоты
    # amp_interp = a*x_vertex^2 + b*x_vertex + c
    
    y0, y1, y2 = y[0], y[1], y[2] # y1 - это амплитуда в peak_idx
    x0, x1, x2 = x[0], x[1], x[2] # x1 - это частота в peak_idx

    # Проверка на коллинеарность или другие проблемы, которые могут привести к den=0
    # (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0) это 2*площадь треугольника. Если 0, точки коллинеарны.
    # Используем формулу Lagrange interpolation или直接coefficients
    # alpha = log(y1/y0), beta = log(y1/y2)
    # p = 0.5 * (alpha - beta) / (alpha + beta) if (alpha+beta) !=0 else 0
    # freq_interp = x1 + p * (x1-x0) # Приблизительно, если шаг частот постоянный
    # Более точная формула для вершины параболы, проходящей через 3 точки:
    # x_v = ( (x0^2-x1^2)*y2 + (x1^2-x2^2)*y0 + (x2^2-x0^2)*y1 ) / ( 2 * ( (x0-x1)*y2 + (x1-x2)*y0 + (x2-x0)*y1 ) )
    # y_v = ... сложнее

    # Используем формулу из scipy.signal.peak_widths -> _peak_prominences -> _peak_on_PchipEdge
    # или классическую 3-точечную параболическую интерполяцию для максимума/минимума:
    # x_peak = x1 + 0.5 * ( (x0-x1)*(y1-y2) - (x1-x2)*(y1-y0) ) / ( (y1-y0)*(x1-x2) - (y1-y2)*(x1-x0) ) если знаменатель не 0.
    # Это не совсем то. 
    # Формула вершины параболы (для равных интервалов по x, что у нас не всегда так для частот):
    # p = (y0 - y2) / (2 * (y0 - 2*y1 + y2))  -- это смещение от x1 в единицах шага
    # freq_interp = x1 + p * (x2-x1) # или (x1-x0)
    # amp_interp = y1 - 0.25 * (y0-y2) * p

    # Проверенный метод квадратичной интерполяции для нахождения вершины:
    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    # Предполагается, что y_i - это значения мощности или log-мощности.
    # Если y_i - это амплитуды, то для мощности y_i^2.
    # Будем интерполировать сами амплитуды, как предложено.

    alpha = y[0]
    beta = y[1]
    gamma = y[2]

    # p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma) # смещение от средней точки в долях интервала
    # Проверка знаменателя, чтобы избежать деления на ноль
    denominator = alpha - 2 * beta + gamma
    if abs(denominator) < 1e-9: # Если знаменатель очень мал (точки почти коллинеарны или плато)
        return freqs[peak_idx], amplitudes[peak_idx] # Возвращаем исходный пик
    
    p = 0.5 * (alpha - gamma) / denominator
    
    # Интерполированная частота (x-координата вершины)
    # x1 здесь это freqs[peak_idx]
    freq_interp = freqs[peak_idx] - p * (freqs[peak_idx+1] - freqs[peak_idx-1]) / 2.0
    # Выражение -p * (dx_right_half - dx_left_half) = -p * ( (x2-x1) - (x1-x0) ) / 2 - это не совсем то.
    # Если шаг частот d_f, то f_interp = f_peak - p * d_f
    # В нашем случае шаг частот может быть нерегулярным, если freqs не из rfftfreq напрямую.
    # Однако, freqs из fft_transform это результат rfftfreq, так что шаг почти постоянный (кроме краев).
    # Для простоты и если шаг частот примерно одинаков:
    # freq_interp = freqs[peak_idx] - p * (freqs[peak_idx+1] - freqs[peak_idx]) # Неверно, если p вне [-0.5, 0.5]
    # freq_interp = freqs[peak_idx] + p * (freqs[peak_idx+1] - freqs[peak_idx-1]) / 2.0 # Если p это смещение от peak_idx
    # Формула x_v = x_beta - p * delta_x, где delta_x - это шаг сетки, а p - смещение от x_beta в единицах delta_x.
    # p = (alpha - gamma) / (2 * (alpha - 2*beta + gamma))
    # x_interp = x_peak - p * (x_step) # x_step - шаг частоты
    # freq_interp = freqs[peak_idx] - p * (freqs[peak_idx+1] - freqs[peak_idx-1]) / 2.0 # Это если p из [-0.5,0.5] и x_coords [-1,0,1]
    # Если p как вычислено (относительно средней точки): freq_interp = (x[0]+x[2])/2 + p * (x[2]-x[0])/2 - неверно
    # freq_interp = freqs[peak_idx] - p * (freqs[peak_idx+1] - freqs[peak_idx-1])/2.0 # моя старая формула. p было [-1,1]
    # Верная формула для смещения p относительно x_beta (x1 в наших обозначениях) в долях интервала между точками:
    # delta = p (как рассчитано выше)
    # interpolated_x = x1 - delta * (x2 - x0) / 2.0  (если x0, x1, x2 симметричны вокруг x1)
    # Если x не симметричны, то сложнее.
    # Для нашего случая, freqs[peak_idx] это x1 (бета).
    # p - это смещение ОТНОСИТЕЛЬНО peak_idx, выраженное в единицах ПОЛОВИНЫ ИНТЕРВАЛА МЕЖДУ x[0] и x[2]
    # то есть, если p=0, вершина в x1. если p=1, вершина в (x1+x2)/2. если p=-1, вершина в (x0+x1)/2
    # Лучше использовать формулу из источника (jos): x_peak_offset = p. Тогда f_interp = f_peak + p * df (где df - шаг частотной сетки)
    # Но p там определяется по-другому. С нашим p: freq_interp = x1 - p * (x2-x0)/2
    # (x2-x0)/2 это примерный шаг, если точки вокруг пика относительно симметричны.    
    # freq_interp = freqs[peak_idx] - p * (freqs[peak_idx+1] - freqs[peak_idx-1]) * 0.5 # Используем эту, как аппроксимацию
    # Ограничим p, чтобы не выходить далеко за пределы среднего бина
    p = np.clip(p, -0.5, 0.5) 
    freq_interp = freqs[peak_idx] - p * ( (freqs[peak_idx+1]-freqs[peak_idx]) + (freqs[peak_idx]-freqs[peak_idx-1]) )/2.0

    # Интерполированная амплитуда (y-координата вершины)
    amp_interp = beta - 0.25 * (alpha - gamma) * p # или beta - p * (alpha - gamma) / 2
    # amp_interp = amplitudes[peak_idx] - 0.25 * (amplitudes[peak_idx-1] - amplitudes[peak_idx+1]) * p

    # Гарантируем, что интерполированная частота не выходит за пределы соседних бинов
    # (хотя с clip(p) это менее вероятно для частоты)
    if freq_interp < freqs[peak_idx-1] or freq_interp > freqs[peak_idx+1]:
        freq_interp = freqs[peak_idx] # Возврат к неинтерполированной, если вышли за пределы
        amp_interp = amplitudes[peak_idx]
    
    # Интерполированная амплитуда не должна быть больше максимальной из трех точек (для параболы с ветвями вниз)
    # или меньше минимальной (для ветвей вверх, но мы ищем пики)
    # На практике, она может быть чуть больше из-за интерполяции.
    # Если amp_interp значительно отличается или становится отрицательной, что-то не так.
    if amp_interp < 0: amp_interp = amplitudes[peak_idx] # Не должно быть отрицательной амплитуды

    return freq_interp, amp_interp

def find_significant_periods_fourier_iterative(
    time_series: pd.Series,
    num_iterations: int = 5, 
    nfft: int | None = None,  
    max_total_periods: int = 10, 
    loop_gain: float = 0.6 # Новый параметр loop_gain
) -> pd.DataFrame:
    """
    Итеративный поиск периодов с loop_gain и интерполяцией пика.
    """
    if not isinstance(time_series, pd.Series):
        try:
            ts_values = pd.Series(time_series, name="value").astype(float).values
        except Exception as e:
            st.error(f"Ошибка преобразования входных данных в pd.Series/numpy array: {e}")
            return pd.DataFrame()
    else:
        try:
            ts_values = time_series.astype(float).values
        except Exception as e:
            st.error(f"Ошибка преобразования данных Series в числовой numpy array: {e}")
            return pd.DataFrame()

    remaining_signal_values = ts_values.copy()
    original_length = len(remaining_signal_values)
    current_time_index = pd.RangeIndex(start=0, stop=len(remaining_signal_values), step=1)
    all_found_periods_info = []
    st.write(f"--- Начало итеративного поиска Фурье (итераций: {num_iterations}, loop_gain: {loop_gain}) ---")

    for iteration in range(num_iterations):
        st.write(f"Итерация {iteration + 1}")
        current_series_for_fft = pd.Series(remaining_signal_values, index=current_time_index)
        freqs, amplitudes, periods_meas, _ = fft_transform(current_series_for_fft, nfft=nfft)
        
        if freqs.size == 0 or amplitudes.size < 3 or np.all(amplitudes < 1e-9):
            st.warning(f"Спектр пуст, слишком мал или почти нулевой на итерации {iteration + 1}. Прерывание.")
            break
            
        valid_mask = freqs > 1e-9
        if not np.any(valid_mask):
            st.warning(f"Нет частот > 0 на итерации {iteration + 1}. Прерывание.")
            break

        masked_amplitudes_iter = amplitudes[valid_mask]
        masked_freqs_iter = freqs[valid_mask]
        masked_periods_meas_iter = periods_meas[valid_mask]
        
        if masked_amplitudes_iter.size < 3: # Нужно хотя бы 3 точки для интерполяции вокруг максимума
            st.warning(f"Недостаточно точек в спектре для анализа на итерации {iteration + 1}. Прерывание.")
            break
            
        # Находим индекс пика в текущих маскированных массивах
        max_amp_local_idx_in_masked = np.argmax(masked_amplitudes_iter)
        
        # Получаем индекс этого пика в немаскированных freqs и amplitudes (которые использовались для fft_transform)
        # Это нужно для передачи в interpolate_peak_parameters полных массивов, из которых был взят пик.
        original_indices_of_valid_mask = np.where(valid_mask)[0]
        idx_of_peak_in_full_spectrum = original_indices_of_valid_mask[max_amp_local_idx_in_masked]

        # Интерполяция параметров пика
        # Передаем полные (но с удаленным DC) freqs и amplitudes из fft_transform
        # peak_idx должен быть индексом в этих полных freqs/amplitudes
        interpolated_frequency, interpolated_amplitude = interpolate_peak_parameters(
            freqs, amplitudes, idx_of_peak_in_full_spectrum
        )

        if interpolated_amplitude < 1e-9: # Если интерполированная амплитуда очень мала
            st.warning(f"Интерполированная амплитуда пика очень мала ({interpolated_amplitude:.2e}) на итерации {iteration + 1}. Прерывание.")
            break

        # Расчет периода по интерполированной частоте
        interpolated_period_meas = 1.0 / interpolated_frequency if interpolated_frequency > 1e-9 else np.inf
        amplitude_component = interpolated_amplitude # Это A_k/2
        frequency_component = interpolated_frequency
        
        st.write(f"  Найден пик (до интерп.): Период={masked_periods_meas_iter[max_amp_local_idx_in_masked]:.2f} изм., "
                 f"Ампл.комп.={masked_amplitudes_iter[max_amp_local_idx_in_masked]:.4f}, Частота={masked_freqs_iter[max_amp_local_idx_in_masked]:.4e}")
        st.write(f"  Найден пик (ПОСЛЕ интерп.): Период={interpolated_period_meas:.2f} изм., "
                 f"Ампл.комп.={amplitude_component:.4f}, Частота={frequency_component:.4e}")

        all_found_periods_info.append({
            'Период (изм.)': interpolated_period_meas,
            'Мощность': amplitude_component, 
            'Итерация': iteration + 1,
            'Частота': frequency_component
        })
        
        sinusoid_amplitude_to_subtract = 2 * amplitude_component * loop_gain # Учитываем loop_gain
        
        n_points_for_phase_rfft = nfft if nfft is not None else next_power_of_2(len(remaining_signal_values))
        current_rfft_coeffs = np.fft.rfft(np.asarray(remaining_signal_values, dtype=float), n=n_points_for_phase_rfft)
        
        # Для фазы используем ближайший бин к interpolated_frequency
        # Это упрощение. Более точно было бы учитывать фазу на интерполированной частоте, но это сложнее.
        # Найдем индекс ближайшей ДИСКРЕТНОЙ частоты в freqs (из fft_transform) к interpolated_frequency
        # freqs - это результат rfftfreq(n_points_for_phase_rfft, d=1.0), если nfft совпадает
        # Если nfft не задан, то n_points_for_phase_rfft это next_power_of_2(len(signal))
        # А freqs из fft_transform(signal, nfft=None) использует next_power_of_2(len(signal))
        # Значит, freqs массив должен соответствовать current_rfft_coeffs по размеру и частотам.
        
        idx_for_phase = np.argmin(np.abs(freqs - frequency_component)) # Ищем ближайший бин
        phase_component = np.angle(current_rfft_coeffs[idx_for_phase])
                
        t_array = np.arange(original_length)
        synthetic_component_to_subtract = sinusoid_amplitude_to_subtract * np.cos(2 * np.pi * frequency_component * t_array + phase_component)
        
        signal_before_subtract = remaining_signal_values.copy()
        remaining_signal_values = remaining_signal_values - synthetic_component_to_subtract
        
        if iteration < num_iterations - 1:
            sbs_array = np.asarray(signal_before_subtract, dtype=float)
            scs_array = np.asarray(synthetic_component_to_subtract, dtype=float)
            rsv_array = np.asarray(remaining_signal_values, dtype=float)
            st.write(f"  Энергия сигнала до вычитания: {np.sum(np.square(sbs_array)):.2f}")
            st.write(f"  Энергия вычитаемого компонента (с loop_gain): {np.sum(np.square(scs_array)):.2f}")
            st.write(f"  Энергия сигнала после вычитания: {np.sum(np.square(rsv_array)):.2f}")

    st.write("--- Итеративный поиск Фурье завершен ---")
    if not all_found_periods_info:
        st.warning("Итеративный поиск не нашел ни одного периода.")
        return pd.DataFrame(columns=['Период (изм.)', 'Мощность', 'Итерация', 'Частота'])
    result_df = pd.DataFrame(all_found_periods_info)
    max_power_overall = result_df['Мощность'].max()
    if max_power_overall > 1e-9:
        result_df['Норм. мощность'] = result_df['Мощность'] / max_power_overall
    else:
        result_df['Норм. мощность'] = 0.0
    result_df = result_df.sort_values('Мощность', ascending=False).head(max_total_periods)
    return result_df

def plot_fourier_periodicity_analysis(
    time_series, 
    num_points=None, 
    max_periods=10, 
    power_threshold=0.2, # Остается для линейного режима
    selected_unit_key="Измерения",
    use_log_scale_y=False, # Новый параметр для оси Y графика
    use_log_for_peaks=False, # Новый параметр для поиска пиков
    log_peak_threshold_factor=1.5 # Передается в find_significant_periods_fourier
):
    """
    Создает визуализацию для анализа периодичностей с помощью преобразования Фурье.
    Ось X теперь отображается в выбранных единицах.
    Ось Y может быть линейной или логарифмической.
    """
    import plotly.graph_objects as go
    
    freqs, amplitudes, periods_meas, time_delta = fft_transform(time_series, nfft=num_points)

    if freqs.size == 0 or amplitudes.size == 0:
        st.warning("Преобразование Фурье не вернуло данных.")
        fig = go.Figure()
        fig.update_layout(title="Анализ Фурье (нет данных для спектра)")
        return fig, pd.DataFrame(columns=['Период (изм.)', 'Мощность'])

    significant_periods_df = find_significant_periods_fourier(
        time_series, # Передаем исходный time_series, т.к. fft_transform будет вызван внутри
        power_threshold=power_threshold,
        num_points=num_points,
        max_periods=max_periods,
        use_log_for_peaks=use_log_for_peaks, # Передаем новый параметр
        log_peak_threshold_factor=log_peak_threshold_factor # Передаем новый параметр
    )

    fig = go.Figure()
    
    # Подготовка данных для графика
    mask = (freqs > 0.0001) & (periods_meas > 0)
    filtered_periods_meas = periods_meas[mask]
    filtered_amplitudes = amplitudes[mask]
    
    if filtered_periods_meas.size == 0:
        fig.update_layout(title=f"Анализ Фурье (нет данных для спектра после фильтрации)")
        return fig, significant_periods_df

    # --- Определение диапазона отображаемых периодов (логика из предыдущего шага) ---
    min_val_from_filtered = np.min(filtered_periods_meas)
    max_val_from_filtered = np.max(filtered_periods_meas)
    min_period_actual = min_val_from_filtered if np.isfinite(min_val_from_filtered) and min_val_from_filtered > 0 else 1e-1
    max_period_actual = max_val_from_filtered if np.isfinite(max_val_from_filtered) and max_val_from_filtered > min_period_actual else min_period_actual * 100.0
    min_period_to_show_meas = min_period_actual
    if not significant_periods_df.empty:
        max_detected_period_meas_val = significant_periods_df['Период (изм.)'].max()
        if not np.isfinite(max_detected_period_meas_val) or max_detected_period_meas_val <= 0:
            max_period_to_show_meas = max_period_actual
        else:
            upper_bound_candidate = max(max_detected_period_meas_val * 3.0, 200.0)
            max_period_to_show_meas = min(max_period_actual, upper_bound_candidate)
    else:
        max_period_to_show_meas = max_period_actual
    if max_period_to_show_meas <= min_period_to_show_meas:
        # (логика корректировки диапазона опущена для краткости, она была в предыдущем ответе)
        if min_period_to_show_meas > 0: 
            max_period_to_show_meas = min_period_to_show_meas * 100.0 
        else: 
            min_period_to_show_meas = 1e-1 
            max_period_to_show_meas = 10.0  
        if max_period_to_show_meas > max_period_actual:
             max_period_to_show_meas = max_period_actual
        if max_period_to_show_meas <= min_period_to_show_meas:
            if min_period_to_show_meas > 0:
                 max_period_to_show_meas = min_period_to_show_meas + 10.0 
            else: 
                 min_period_to_show_meas = 0.1
                 max_period_to_show_meas = 1.0 
    # --- Конец определения диапазона ---

    display_mask = (filtered_periods_meas <= max_period_to_show_meas) & (filtered_periods_meas >= min_period_to_show_meas)
    display_periods_meas = filtered_periods_meas[display_mask]
    display_amplitudes_for_plot = filtered_amplitudes[display_mask]

    if display_periods_meas.size == 0:
        fig.update_layout(title=f"Анализ Фурье (нет данных в выбранном диапазоне периодов)")
        return fig, significant_periods_df

    # Подготовка амплитуд для оси Y графика
    y_axis_title = "Нормализованная амплитуда"
    y_axis_type = "linear"
    plot_y_values = display_amplitudes_for_plot
    
    if use_log_scale_y:
        plot_y_values = np.log10(display_amplitudes_for_plot + 1e-12) # Безопасный логарифм
        y_axis_title = "Log10(Амплитуда)"
        y_axis_type = "linear" # Plotly обрабатывает log display на основе данных, если y_axis_type="log", но здесь мы уже дали log(данные)
                               # Если бы мы давали исходные амплитуды и хотели log шкалу, то y_axis_type="log"
                               # Так как мы передаем уже np.log10(values), то сама ось остается линейной по типу.
                               # Либо можно передавать display_amplitudes_for_plot и y_axis_type = "log"
        # Для корректного отображения с plotly_chart и уже логарифмированными данными,
        # лучше оставить y_axis_type = "linear".
        # Если использовать y_axis_type="log", то plotly будет ожидать исходные значения, а не логарифмированные.
        
        # Для отображения пиков на лог. шкале, их значения "Мощности" (которые [0,1]) нужно будет
        # либо не отображать как y-координату пика, либо пересчитать их положение на лог. оси.
        # Проще всего - показывать пики на их фактической лог. амплитуде, если use_log_scale_y.

    else: # Линейная шкала Y
        max_amp_display = np.max(display_amplitudes_for_plot) if display_amplitudes_for_plot.size > 0 and np.max(display_amplitudes_for_plot) > 0 else 1
        plot_y_values = display_amplitudes_for_plot / max_amp_display # Нормализация для линейной шкалы


    hover_texts_spectrum = [
        f"Период: {format_period(p_m, time_delta, selected_unit_key)}<br>" +
        f"(Измерения: {p_m:.2f})<br>" +
        (f"Log10(Ампл): {y_val:.3f}" if use_log_scale_y else f"Норм. амплитуда: {y_val:.3f}") +
        f"<extra></extra>"
        for p_m, y_val in zip(display_periods_meas, plot_y_values)
    ]

    fig.add_trace(go.Scatter(
        x=display_periods_meas,
        y=plot_y_values,
        mode='lines',
        name='Спектр',
        line=dict(color='red', width=2),
        hoverinfo='text',
        text=hover_texts_spectrum
    ))
    
    if not significant_periods_df.empty:
        period_points_meas = []
        peak_y_values = [] # Значения Y для пиков на графике
        hover_texts_peaks = []
        
        for i, row in significant_periods_df.iterrows():
            period_meas_val = row['Период (изм.)']
            # Мощность здесь всегда в диапазоне [0,1] из find_significant_periods_fourier
            # Используем исходную амплитуду пика для его позиционирования на графике Y
            
            # Найдем исходную или лог. амплитуду этого пика в исходном спектре (df внутри find_significant_periods_fourier)
            # Это немного усложняется, так как find_significant_periods_fourier возвращает только Период и Мощность.
            # Нам нужно найти соответствующую амплитуду пика в display_periods_meas / display_amplitudes_for_plot
            
            # Ищем индекс этого периода в display_periods_meas, чтобы взять правильную амплитуду для графика
            idx_in_display_array = np.where(np.isclose(display_periods_meas, period_meas_val))[0]

            if idx_in_display_array.size > 0:
                actual_peak_amplitude_for_plot = display_amplitudes_for_plot[idx_in_display_array[0]]
                
                y_coord_for_peak = 0
                if use_log_scale_y:
                    y_coord_for_peak = np.log10(actual_peak_amplitude_for_plot + 1e-12)
                else:
                    # Нормализуем так же, как основной спектр для линейной шкалы
                    max_amp_display_for_norm = np.max(display_amplitudes_for_plot) if display_amplitudes_for_plot.size > 0 and np.max(display_amplitudes_for_plot) > 0 else 1
                    y_coord_for_peak = actual_peak_amplitude_for_plot / max_amp_display_for_norm
                
                # Пропускаем периоды вне диапазона отображения (уже сделано display_mask)
                # if not (min_period_to_show_meas <= period_meas_val <= max_period_to_show_meas):
                #    continue 
                # Эта проверка не нужна, так как significant_periods_df может содержать пики,
                # которые потом отфильтруются display_mask. Мы должны добавлять только те, что есть в display_periods_meas.

                period_points_meas.append(period_meas_val)
                peak_y_values.append(y_coord_for_peak)
            
                formatted_period = format_period(period_meas_val, time_delta, selected_unit_key)
                formatted_period_meas_only = format_period(period_meas_val, None, MEASUREMENT_UNIT_KEY)
            
                hover_texts_peaks.append(
                    f"<b>ЗНАЧИМЫЙ ПЕРИОД</b><br>" +
                    f"Период: {formatted_period} ({formatted_period_meas_only})<br>" +
                    (f"Log10(Ампл): {y_coord_for_peak:.3f}<br>" if use_log_scale_y else f"Норм.Ампл: {y_coord_for_peak:.3f}<br>") +
                    f"Мощность (отн.): {row['Мощность']:.3f}<extra></extra>" # row['Мощность'] это [0,1]
                )
            
                fig.add_vline(
                    x=period_meas_val,
                    line=dict(color="green", width=2, dash="dash"),
                    opacity=0.8,
                    annotation_text=formatted_period, 
                    annotation_position="top right",
                    annotation_font=dict(size=14, color='black', family='Arial Black')
                )
        
        if period_points_meas:
            fig.add_trace(go.Scatter(
                x=period_points_meas, 
                y=peak_y_values, # Y координата пика на текущей шкале графика
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle-open', line=dict(width=3, color='darkgreen')),
                name='Значимые периоды',
                hoverinfo='text',
                text=hover_texts_peaks
            ))

    x_tickvals_meas, x_ticktext_formatted = get_fourier_x_axis_ticks(
        min_period_to_show_meas,
        max_period_to_show_meas,
        time_delta,
        selected_unit_key,
        num_ticks=8
    )

    fig.update_layout(
        title=f"Анализ Фурье: Спектр мощности",
        xaxis_title=f"Период ({selected_unit_key})",
        yaxis_title=y_axis_title, # Динамический заголовок оси Y
        xaxis_type="log",
        xaxis=dict(
            tickmode="array",
            tickvals=x_tickvals_meas,
            ticktext=x_ticktext_formatted
        ),
        yaxis_type= "linear", # Ось Y всегда linear, так как мы сами преобразуем данные если нужен лог. масштаб
                              # Если бы мы хотели, чтобы plotly сам делал лог. масштаб, то здесь было бы y_axis_type="log"
                              # и мы бы передавали нелогарифмированные plot_y_values.
        legend_title="Легенда",
        hovermode="closest"
    )
    # Если линейная шкала, установим диапазон Y от -0.1 до 1.1 для нормализованных данных
    if not use_log_scale_y:
        fig.update_layout(yaxis_range=[-0.1, 1.1])
    # Для логарифмической шкалы диапазон Y установится автоматически, либо можно задать его динамически

    return fig, significant_periods_df