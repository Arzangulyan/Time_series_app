import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Анализ аномалий во временных рядах", layout="wide")

from modules.anomaly_module import (
    generate_anomalous_series,
    add_anomalies_to_existing_data,
    z_score_detection,
    iqr_detection,
    hampel_filter,
    detect_plateau,
    evaluate_anomaly_detection,  # Импортируем новую функцию
    create_true_anomaly_mask  # Добавляем импорт этой функции для визуализации
)

# Инициализация состояния сессии
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = {
        'point': [],
        'extended': [],
        'sensor': []
    }

# Проверка наличия загруженных данных согласно подходу в template
has_loaded_data = 'time_series' in st.session_state and st.session_state.time_series is not None and not st.session_state.time_series.empty

# ====================
# КОНФИГУРАЦИЯ ИНТЕРФЕЙСА
# ====================
st.title("Анализ временных рядов с аномалиями")

with st.sidebar:
    st.header("Настройки данных")
    
    # Выбор источника данных
    data_source = st.radio(
        "Источник данных",
        ["Синтетические данные", "Загруженные данные"],
        index=1 if has_loaded_data else 0,
        disabled=not has_loaded_data
    )
    
    if data_source == "Загруженные данные" and not has_loaded_data:
        st.warning("Нет загруженных данных. Сначала загрузите данные на главной странице.")
        data_source = "Синтетические данные"
    
    if data_source == "Загруженные данные":
        st.info("Используются данные, загруженные на главной странице")
        
        # Получаем информацию о выбранной колонке из session_state
        main_column = st.session_state.get("main_column", None)
        
        # Выбор столбца с данными, если main_column не определен или данные - DataFrame с несколькими столбцами
        if has_loaded_data:
            time_series = st.session_state.time_series
            
            if isinstance(time_series, pd.DataFrame):
                numeric_cols = time_series.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    # Если есть main_column, используем его как значение по умолчанию
                    default_idx = numeric_cols.index(main_column) if main_column in numeric_cols else 0
                    selected_column = st.selectbox("Выберите столбец с данными", numeric_cols, index=default_idx)
                elif len(numeric_cols) == 1:
                    selected_column = numeric_cols[0]
                    st.success(f"Выбран единственный числовой столбец: {selected_column}")
                else:
                    selected_column = None
                    st.error("Нет числовых столбцов в загруженных данных")
                    data_source = "Синтетические данные"
            elif isinstance(time_series, pd.Series):
                selected_column = time_series.name if time_series.name else "Value"
                st.success(f"Используется временной ряд: {selected_column}")
            else:
                st.error(f"Неподдерживаемый тип данных: {type(time_series)}")
                data_source = "Синтетические данные"
    
    # Основные параметры для синтетических данных
    if data_source == "Синтетические данные":
        st.header("Параметры генерации")
        n = st.slider("Длина ряда", 100, 1000, 300)
        season_amp = st.slider("Амплитуда сезонности", 0.0, 2.0, 0.5)
        noise_std = st.slider("Уровень шума", 0.0, 1.0, 0.2)
    
    # Ручная настройка аномалий
    st.subheader("Ручная настройка аномалий")
    
    # Точечные аномалии
    with st.expander("➕ Точечные аномалии"):
        point_indices = st.text_input("Индексы (через запятую)", key="point_indices")
        point_amp_min = st.number_input("Минимальная амплитуда", 0.1, 5.0, 1.0, key="point_amp_min")
        point_amp_max = st.number_input("Максимальная амплитуда", 0.1, 5.0, 2.0, key="point_amp_max")
        point_direction = st.radio("Направление", ["Вверх", "Вниз"], key="point_dir")
        
        if st.button("Добавить точечные аномалии"):
            try:
                indices = []
                for i in point_indices.split(","):
                    i_clean = i.strip()
                    if i_clean:
                        if not i_clean.isdigit():
                            raise ValueError(f"Некорректный индекс: {i_clean}")
                        idx = int(i_clean)
                        # Получаем максимальный допустимый индекс в зависимости от источника данных
                        max_idx = n - 1 if data_source == "Синтетические данные" else len(st.session_state.time_series) - 1
                        if idx < 0 or idx > max_idx:
                            raise ValueError(f"Индекс {idx} вне диапазона [0, {max_idx}]")
                        indices.append(idx)
                indices = list(set(indices))  # удаление дубликатов
                
                # Добавление точечных аномалий в session_state
                if indices:  # Убедимся, что список не пустой
                    new_anom = {
                        'indices': indices,
                        'amplitude_range': (point_amp_min, point_amp_max),
                        'increase': point_direction == "Вверх"
                    }
                    st.session_state.anomalies['point'].append(new_anom)
                    st.success(f"Добавлены точечные аномалии для индексов: {indices}")
            except Exception as e:
                st.error(f"Ошибка обработки индексов: {str(e)}")
    
    # Протяженные аномалии
    with st.expander("📏 Протяженные аномалии"):
        # Получаем максимальный допустимый индекс в зависимости от источника данных
        max_idx = n - 1 if data_source == "Синтетические данные" else len(st.session_state.time_series) - 1 if has_loaded_data else 299
        
        ext_start = st.number_input("Начальный индекс", 0, max_idx, min(80, max_idx), key="ext_start")
        ext_duration = st.number_input("Длительность", 1, min(100, max_idx - ext_start + 1), 
                                       min(25, max_idx - ext_start + 1), key="ext_dur")
        ext_shift = st.number_input("Смещение уровня", -5.0, 5.0, -2.5, key="ext_shift")
        
        if st.button("Добавить протяженную аномалию"):
            new_anom = {
                'start_idx': ext_start,
                'duration': ext_duration,
                'level_shift': ext_shift
            }
            st.session_state.anomalies['extended'].append(new_anom)
            st.success(f"Добавлена протяженная аномалия начиная с индекса {ext_start}")
    
    # Сбои датчиков
    with st.expander("⚠️ Сбои датчиков"):
        # Получаем максимальный допустимый индекс в зависимости от источника данных
        max_idx = n - 1 if data_source == "Синтетические данные" else len(st.session_state.time_series) - 1 if has_loaded_data else 299
        
        fault_start = st.number_input("Начало сбоя", 0, max_idx, min(220, max_idx), key="fault_start")
        fault_duration = st.number_input("Длительность сбоя", 1, min(100, max_idx - fault_start + 1), 
                                        min(35, max_idx - fault_start + 1), key="fault_dur")
        fault_value = st.selectbox("Значение", ["NaN", "0", "1", "Другое"], key="fault_val")
        custom_value = st.number_input("Свое значение", key="custom_fault") if fault_value == "Другое" else None
        
        if st.button("Добавить сбой датчика"):
            value = np.nan if fault_value == "NaN" else (
                custom_value if fault_value == "Другое" else float(fault_value))
            
            new_anom = {
                'start_idx': fault_start,
                'duration': fault_duration,
                'fault_value': value
            }
            st.session_state.anomalies['sensor'].append(new_anom)
            st.success(f"Добавлен сбой датчика начиная с индекса {fault_start}")
    
    # Кнопка сброса
    if st.button("Очистить все аномалии"):
        st.session_state.anomalies = {'point': [], 'extended': [], 'sensor': []}
        st.success("Все аномалии удалены")

# ====================
# ГЕНЕРАЦИЯ ИЛИ ЗАГРУЗКА ДАННЫХ
# ====================
@st.cache_data
def generate_data(params):
    return generate_anomalous_series(**params)

@st.cache_data
def add_anomalies_to_real_data(data, _time, params):
    data_with_anomalies, anomaly_info = add_anomalies_to_existing_data(
        data, _time, 
        point_anomalies=params['point_anomalies'],
        extended_anomalies=params['extended_anomalies'],
        sensor_faults=params['sensor_faults']
    )
    return data_with_anomalies, anomaly_info

try:
    if data_source == "Синтетические данные":
        # Генерация синтетических данных
        generation_params = {
            'n': n,
            'season_amp': season_amp,
            'noise_std': noise_std,
            'point_anomalies': st.session_state.anomalies['point'],
            'extended_anomalies': st.session_state.anomalies['extended'],
            'sensor_faults': st.session_state.anomalies['sensor']
        }
        
        # Получаем данные, время и метаданные
        data, time, metadata = generate_data(generation_params)
        df = pd.DataFrame({'data': data, 'time': time})
        original_data = None
    else:
        # Использование реальных данных из st.session_state.time_series
        time_series = st.session_state.time_series
        
        if isinstance(time_series, pd.Series):
            # Если Series, используем его напрямую
            original_data = time_series.values
            time_index = time_series.index
        elif isinstance(time_series, pd.DataFrame) and selected_column in time_series.columns:
            # Если DataFrame, выбираем нужную колонку
            original_data = time_series[selected_column].values
            time_index = time_series.index
        else:
            st.error("Не удалось получить данные из выбранного источника")
            st.stop()
        
        # Добавляем аномалии к реальным данным
        anomaly_params = {
            'point_anomalies': st.session_state.anomalies['point'],
            'extended_anomalies': st.session_state.anomalies['extended'],
            'sensor_faults': st.session_state.anomalies['sensor']
        }
        
        data_with_anomalies, metadata = add_anomalies_to_real_data(original_data, time_index, anomaly_params)
        
        # Создаем DataFrame с оригинальными и модифицированными данными
        df = pd.DataFrame({
            'original': original_data,
            'data': data_with_anomalies,
            'time': time_index
        })
        
        # Используем только модифицированные данные для анализа аномалий
        data = data_with_anomalies
except Exception as e:
    st.error(f"Ошибка обработки данных: {str(e)}")
    st.stop()

# ====================
# ВИЗУАЛИЗАЦИЯ НАСТРОЕК
# ====================
with st.expander("Текущие настройки аномалий"):
    if not any(st.session_state.anomalies.values()):
        st.write("Аномалии не добавлены")
    else:
        anom_list = []
        for anom_type, anoms in st.session_state.anomalies.items():
            for anom in anoms:
                record = {'Тип': anom_type}
                record.update(anom)
                anom_list.append(record)
        
        anom_df = pd.DataFrame(anom_list)
        st.dataframe(
            anom_df,
            column_order=['Тип', 'indices', 'amplitude_range', 
                         'start_idx', 'duration', 'level_shift', 'fault_value'],
            use_container_width=True
        )

# ====================
# НАСТРОЙКИ ДЕТЕКЦИИ
# ====================
with st.sidebar:
    st.subheader("⚙️ Параметры детекции аномалий")
    
    # Z-Score
    with st.expander("📈 Z-Score метод", expanded=True):
        use_zscore = st.checkbox("Использовать Z-Score", value=True, 
                                help="Метод на основе стандартных отклонений от среднего значения")
        z_threshold = st.slider("Порог Z-Score", 1.0, 5.0, 3.0, step=0.5,
                               help=(
                                   "Количество стандартных отклонений от среднего:\n"
                                   "• 1σ ≈ 68% данных\n"
                                   "• 2σ ≈ 95% данных\n"
                                   "• 3σ ≈ 99.7% данных\n"
                                   "Рекомендуемое значение: 3 (выявляет явные выбросы)"
                               ))
    
    # Hampel
    with st.expander("🔍 Фильтр Хампеля", expanded=True):
        use_hampel = st.checkbox("Использовать Хампель", value=True,
                                help="Устойчивый метод на основе медиан и MAD")
        
        # Добавляем опцию для адаптивного размера окна
        hampel_adaptive = st.checkbox("Адаптивный размер окна", value=True, 
                                     help="Размер окна рассчитывается как процент от длины ряда")
        
        if hampel_adaptive:
            # Если выбран адаптивный режим, показываем слайдер для процента
            hampel_window_percent = st.slider("Процент от длины ряда (%)", 0.1, 5.0, 0.5, step=0.1,
                                            help=(
                                                "Размер окна как процент от длины ряда:\n"
                                                "• Меньшие значения: выше чувствительность к локальным изменениям\n"
                                                "• Большие значения: рассматривается более глобальный контекст\n"
                                                "Рекомендуется: 0.5-1% для ежеминутных данных за недели"
                                            ))
            # Рассчитываем приблизительный размер окна для текущих данных
            if data_source == "Загруженные данные" and "time_series" in st.session_state:
                approx_window = max(5, min(int(len(st.session_state.time_series) * hampel_window_percent / 100), 
                                        len(st.session_state.time_series) // 5))
                st.info(f"Примерный размер окна для текущего ряда: {approx_window} точек")
            hampel_window = 0  # Устанавливаем в 0, чтобы использовалось значение процента
        else:
            # Если адаптивный режим выключен, используем обычный слайдер для окна
            hampel_window = st.slider("Размер окна", 5, 500, 20, step=5,
                                    help=(
                                        "Количество точек для расчета локальной медианы:\n"
                                        "• Меньшие значения: выше чувствительность к локальным изменениям\n"
                                        "• Большие значения: устойчивее к шумам\n"
                                        "Для больших рядов (>10K точек) рекомендуется 50-200"
                                    ))
            hampel_window_percent = 0.5  # Значение по умолчанию, не будет использоваться
        
        hampel_sigma = st.slider("Коэффициент чувствительности", 1.0, 5.0, 3.0, step=0.1,
                                help=(
                                    "Множитель для медианного абсолютного отклонения (MAD):\n"
                                    "• 3.0 соответствует ~3σ в нормальном распределении\n"
                                    "• Уменьшайте для большей чувствительности\n"
                                    "• Увеличивайте для снижения ложных срабатываний"
                                ))
    
    # IQR
    with st.expander("📏 Межквартильный размах (IQR)"):
        use_iqr = st.checkbox("Использовать IQR", value=True,
                             help="Метод на основе квартилей распределения")
        iqr_multiplier = st.slider("Множитель IQR", 0.5, 5.0, 1.5, step=0.1,
                                  help=(
                                      "Определяет границы выбросов:\n"
                                      "• Стандартное значение: 1.5 (выявляет умеренные выбросы)\n"
                                      "• 3.0 для экстремальных выбросов\n"
                                      "• Формула: Q1 - k*IQR и Q3 + k*IQR"
                                  ))
    
    # Plateau detection
    with st.expander("⏸ Детекция плато"):
        use_plateau = st.checkbox("Использовать детекцию плато", value=True,
                                 help="Обнаружение участков с постоянным значением")
        plateau_threshold = st.slider("Порог производной", 0.0, 0.1, 0.001, step=0.001)
        plateau_duration = st.slider("Минимальная длительность", 1, 50, 10)
        detect_nan = st.checkbox("Считать NaN как плато", value=True,
                                help="Помечать последовательности пропущенных значений как аномалии")



# ====================
# ОБНАРУЖЕНИЕ АНОМАЛИЙ
# ====================
def detect_all_anomalies(data):
    results = {}
    
    # Z-Score
    if use_zscore:
        z_outliers = z_score_detection(data, z_threshold)
        # Убедимся, что результат - именно numpy array
        results['z_outliers'] = np.asarray(z_outliers)
    
    # Hampel - обновляем вызов с новыми параметрами
    if use_hampel:
        hampel_result = hampel_filter(data, 
                                    window=hampel_window, 
                                    sigma=hampel_sigma,
                                    adaptive_window=hampel_adaptive,
                                    window_percent=hampel_window_percent)
        results['hampel_outliers'] = np.asarray(hampel_result)
    
    # IQR
    if use_iqr:
        iqr_outliers, iqr_bounds = iqr_detection(data, iqr_multiplier)
        # Убедимся, что результат - именно numpy array
        results['iqr_outliers'] = np.asarray(iqr_outliers)
        results['iqr_bounds'] = iqr_bounds
    
    # Plateau
    if use_plateau:
        plateau = detect_plateau(data.fillna(0), 
                                threshold=plateau_threshold,
                                min_duration=plateau_duration)
        # Преобразуем список словарей {start, end} в булевый массив для индикации точек плато
        plateau_mask = np.zeros(len(data), dtype=bool)
        for p in plateau:
            plateau_mask[p['start']:p['end']+1] = True
        results['plateau_outliers'] = plateau_mask
    
    return results


anomalies = detect_all_anomalies(df['data'])

# ====================
# ВИЗУАЛИЗАЦИЯ
# ====================
st.subheader("Визуализация ряда и аномалий")

# Словарь для перевода названий методов
method_names = {
    'z_outliers': 'Z-метод',
    'hampel_outliers': 'Фильтр Хампеля',
    'iqr_outliers': 'Межквартильный метод',
    'plateau_outliers': 'Обнаружение плато'
}

colors = {
    'z_outliers': ('red', 'circle'),
    'hampel_outliers': ('black', 'triangle-up'),
    'iqr_outliers': ('green', 'x'),
    'plateau_outliers': ('purple', 'square')
}

method_flags = {
    'z_outliers': use_zscore,
    'hampel_outliers': use_hampel, 
    'iqr_outliers': use_iqr,
    'plateau_outliers': use_plateau
}

fig = make_subplots()

# Если используются реальные данные, показываем оригинальные данные
if data_source == "Загруженные данные" and 'original' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['original'],
        mode='lines',
        name='Исходный ряд',
        line=dict(color='lightblue', width=1, dash='dot'),
        opacity=0.6
    ))

fig.add_trace(go.Scatter(
    x=df['time'], 
    y=df['data'],
    mode='lines',
    name='Ряд с аномалиями' if data_source == "Загруженные данные" else 'Исходный ряд',
    line=dict(color='blue', width=1.5),
    opacity=0.7
))

for method in colors.keys():
    if method in anomalies and len(anomalies[method]) > 0:
        idx = np.where(anomalies[method])[0]
        if len(idx) > 0:
            fig.add_trace(go.Scatter(
                x=df.iloc[idx]['time'],
                y=df.iloc[idx]['data'],
                mode='markers',
                name=method_names[method],  # Используем русское название
                marker=dict(
                    color=colors[method][0],
                    size=10,
                    symbol=colors[method][1],
                    line=dict(width=2, color='white')
                ),
                visible=True if method_flags[method] else 'legendonly',
                uid=method
            ))

if use_iqr:
    iqr_lower, iqr_upper = anomalies.get('iqr_bounds', (None, None))
    if iqr_lower is not None and iqr_upper is not None:
        fig.add_hline(y=iqr_lower, line_dash="dot", line_color="green", opacity=0.5, name="Нижняя граница IQR")
        fig.add_hline(y=iqr_upper, line_dash="dot", line_color="green", opacity=0.5, name="Верхняя граница IQR")

fig.update_layout(
    title="Обнаруженные аномалии",
    xaxis_title="Время",
    yaxis_title="Значение",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        itemclick=False,
        itemdoubleclick=False
    ),
    height=600,
    hovermode='x unified',
    uirevision='constant'
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


# ====================
# СТАТИСТИКА
# ====================
st.subheader("Статистика аномалий")

stats = []
if use_zscore: 
    z_outliers = anomalies.get('z_outliers', [])
    z_count = np.count_nonzero(z_outliers) if len(z_outliers) > 0 else 0
    stats.append(("Z-метод", z_count))
    
if use_hampel: 
    hampel_outliers = anomalies.get('hampel_outliers', [])
    hampel_count = np.count_nonzero(hampel_outliers) if len(hampel_outliers) > 0 else 0
    stats.append(("Фильтр Хампеля", hampel_count))
    
if use_iqr: 
    iqr_outliers = anomalies.get('iqr_outliers', [])
    iqr_count = np.count_nonzero(iqr_outliers) if len(iqr_outliers) > 0 else 0
    stats.append(("Межквартильный метод", iqr_count))
    
if use_plateau: 
    plateau_outliers = anomalies.get('plateau_outliers', [])
    plateau_count = np.count_nonzero(plateau_outliers) if len(plateau_outliers) > 0 else 0
    stats.append(("Обнаружение плато", plateau_count))

cols = st.columns(len(stats))
for i, (name, value) in enumerate(stats):
    cols[i].metric(name, value)

# ====================
# ОЦЕНКА КАЧЕСТВА ОБНАРУЖЕНИЯ АНОМАЛИЙ
# ====================
# Проверка наличия внедренных аномалий
has_injected_anomalies = (any(st.session_state.anomalies.values()) 
                          and (data_source == "Синтетические данные" or 
                              (data_source == "Загруженные данные" and "original" in df.columns)))

st.subheader("Оценка качества обнаружения аномалий")

if has_injected_anomalies:
    # Если есть внедренные аномалии, включаем опцию оценки
    enable_evaluation = st.checkbox("Включить оценку качества обнаружения", value=True,
                                   help="Вычисляет метрики Precision, Recall и F1-score для каждого метода")
    
    if enable_evaluation:
        # Получаем информацию о внедренных аномалиях
        if data_source == "Синтетические данные":
            # Для синтетических данных используем metadata
            injected_anomalies = metadata
        else:
            # Для реальных данных берем данные из ранее сохраненных метаданных
            injected_anomalies = metadata
        
        # Удаляем не-маски из словаря аномалий перед оценкой
        anomalies_for_evaluation = {}
        for method_name, detection_result in anomalies.items():
            # Пропускаем iqr_bounds и другие не-маски
            if method_name == 'iqr_bounds':
                continue
                
            # Преобразуем pandas Series/DataFrame в numpy array если нужно
            if isinstance(detection_result, (pd.Series, pd.DataFrame)):
                detection_result = detection_result.values
                
            # Проверяем, что результат имеет правильный формат
            if isinstance(detection_result, np.ndarray) and detection_result.shape == (len(df),):
                if detection_result.dtype == bool:
                    anomalies_for_evaluation[method_name] = detection_result
                elif np.issubdtype(detection_result.dtype, np.number):
                    # Преобразуем к булевому типу, если это числовой массив
                    anomalies_for_evaluation[method_name] = detection_result.astype(bool)
        
        # Для отладки: показать, какие методы включены в оценку
        with st.expander("Отладочная информация"):
            st.write("### Методы, включенные в оценку:")
            for method in anomalies_for_evaluation.keys():
                st.write(f"✅ {method_names.get(method, method)}")
                
            st.write("### Методы, исключенные из оценки:")
            excluded_methods = set(anomalies.keys()) - set(anomalies_for_evaluation.keys()) - {'iqr_bounds'}
            for method in excluded_methods:
                st.write(f"❌ {method_names.get(method, method)}")
                if method in anomalies:
                    st.write(f"   Причина: тип={type(anomalies[method])}, форма={getattr(anomalies[method], 'shape', 'Нет атрибута shape')}")
        
        # Вычисляем метрики для каждого метода
        try:
            # Создаем истинную маску аномалий для визуализации
            true_anomaly_mask = create_true_anomaly_mask(injected_anomalies, len(df))
            
            # Вычисляем метрики
            metrics_results = evaluate_anomaly_detection(injected_anomalies, anomalies_for_evaluation, len(df))
            
            # Визуализация метрик
            if metrics_results:
                # Создаем DataFrame для удобного отображения
                metrics_df = pd.DataFrame(columns=["Метод", "Precision", "Recall", "F1-Score"])
                
                # Заполняем данными
                for method, metrics in metrics_results.items():
                    # Получаем русское название метода
                    method_name = method_names.get(method, method)
                    
                    # Форматируем значения метрик
                    precision = f"{metrics['precision']:.3f}"
                    recall = f"{metrics['recall']:.3f}"
                    f1 = f"{metrics['f1']:.3f}"
                    
                    # Добавляем в DataFrame
                    new_row = pd.DataFrame({
                        "Метод": [method_name],
                        "Precision": [precision],
                        "Recall": [recall],
                        "F1-Score": [f1]
                    })
                    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
                
                # Отображаем таблицу с метриками
                st.write("### Метрики для методов обнаружения аномалий:")
                st.dataframe(metrics_df, use_container_width=True)
                
                # Создаем вкладки для разных типов визуализации
                tab1, tab2 = st.tabs(["📊 Сравнение метрик", "🔍 Визуализация детекций"])
                
                with tab1:
                    # Визуализация метрик с помощью графиков
                    fig_metrics = go.Figure()
                    
                    # Добавляем бары для каждой метрики
                    for idx, metric in enumerate(["precision", "recall", "f1"]):
                        y_values = [metrics[metric] for method, metrics in metrics_results.items()]
                        method_labels = [method_names.get(method, method) for method in metrics_results.keys()]
                        
                        fig_metrics.add_trace(go.Bar(
                            x=method_labels,
                            y=y_values,
                            name=metric.capitalize(),
                            text=[f"{val:.3f}" for val in y_values],
                            textposition='auto'
                        ))
                    
                    fig_metrics.update_layout(
                        title="Сравнение методов обнаружения аномалий",
                        xaxis_title="Метод",
                        yaxis_title="Значение метрики",
                        barmode='group',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Таблица с описанием метрик
                    st.markdown("""
                    ### Пояснение метрик обнаружения аномалий
                    
                    | Метрика | Описание | Формула | Интерпретация |
                    |---------|---------|---------|---------------|
                    | **Precision** (Точность) | Доля правильно обнаруженных аномалий среди всех обнаруженных точек | TP / (TP + FP) | Высокое значение означает низкий уровень ложных срабатываний |
                    | **Recall** (Полнота) | Доля правильно обнаруженных аномалий среди всех фактических аномалий | TP / (TP + FN) | Высокое значение означает, что метод обнаруживает большинство аномалий |
                    | **F1-Score** | Гармоническое среднее точности и полноты | 2 × (Precision × Recall) / (Precision + Recall) | Сбалансированная метрика для методов с компромиссом между точностью и полнотой |
                    
                    где:
                    - **TP** (True Positive) — верно обнаруженные аномалии
                    - **FP** (False Positive) — нормальные точки, ошибочно отмеченные как аномалии
                    - **FN** (False Negative) — пропущенные аномалии
                    """)
                
                with tab2:
                    # Интерактивная визуализация детекций
                    st.write("### Визуализация качества обнаружения аномалий")
                    selected_method = st.selectbox(
                        "Выберите метод для отображения:",
                        options=list(metrics_results.keys()),
                        format_func=lambda x: method_names.get(x, x)
                    )
                    
                    if selected_method:
                        # Строим визуализацию TP, FP, FN для выбранного метода
                        detection_mask = anomalies_for_evaluation[selected_method]
                        
                        # Вычисляем TP, FP, FN
                        tp_mask = true_anomaly_mask & detection_mask
                        fp_mask = (~true_anomaly_mask) & detection_mask
                        fn_mask = true_anomaly_mask & (~detection_mask)
                        
                        # Строим график
                        fig_detection = go.Figure()
                        
                        # Добавляем исходный ряд
                        fig_detection.add_trace(go.Scatter(
                            x=df['time'],
                            y=df['data'],
                            mode='lines',
                            name='Временной ряд',
                            line=dict(color='lightgray', width=1.5),
                            opacity=0.7
                        ))
                        
                        # Добавляем истинные аномалии (обозначение на фоне)
                        if np.any(true_anomaly_mask):
                            idx = np.where(true_anomaly_mask)[0]
                            true_x = df.iloc[idx]['time']
                            # Создаем зоны истинных аномалий
                            for i in range(len(true_x)):
                                fig_detection.add_vrect(
                                    x0=true_x.iloc[i] - 0.1,
                                    x1=true_x.iloc[i] + 0.1,
                                    fillcolor="rgba(220, 220, 220, 0.3)",
                                    layer="below",
                                    line_width=0
                                )
                        
                        # True Positives (верно обнаруженные)
                        if np.any(tp_mask):
                            idx = np.where(tp_mask)[0]
                            fig_detection.add_trace(go.Scatter(
                                x=df.iloc[idx]['time'],
                                y=df.iloc[idx]['data'],
                                mode='markers',
                                name='Верно обнаруженные (TP)',
                                marker=dict(
                                    color='green',
                                    size=10,
                                    symbol='circle',
                                    line=dict(width=2, color='white')
                                )
                            ))
                        
                        # False Positives (ложные срабатывания)
                        if np.any(fp_mask):
                            idx = np.where(fp_mask)[0]
                            fig_detection.add_trace(go.Scatter(
                                x=df.iloc[idx]['time'],
                                y=df.iloc[idx]['data'],
                                mode='markers',
                                name='Ложные срабатывания (FP)',
                                marker=dict(
                                    color='red',
                                    size=10,
                                    symbol='x',
                                    line=dict(width=2, color='white')
                                )
                            ))
                        
                        # False Negatives (пропущенные аномалии)
                        if np.any(fn_mask):
                            idx = np.where(fn_mask)[0]
                            fig_detection.add_trace(go.Scatter(
                                x=df.iloc[idx]['time'],
                                y=df.iloc[idx]['data'],
                                mode='markers',
                                name='Пропущенные аномалии (FN)',
                                marker=dict(
                                    color='orange',
                                    size=10,
                                    symbol='triangle-down',
                                    line=dict(width=2, color='white')
                                )
                            ))
                        
                        # Настраиваем внешний вид графика
                        fig_detection.update_layout(
                            title=f"Качество обнаружения аномалий: {method_names.get(selected_method, selected_method)}",
                            xaxis_title="Время",
                            yaxis_title="Значение",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=500,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_detection, use_container_width=True)
                        
                        # Показываем статистику для этого метода
                        metrics = metrics_results[selected_method]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("True Positives", np.sum(tp_mask))
                        col2.metric("False Positives", np.sum(fp_mask))
                        col3.metric("False Negatives", np.sum(fn_mask))
                        col4.metric("Precision/Recall", f"{metrics['precision']:.3f} / {metrics['recall']:.3f}")
                        
                        # Добавляем интерпретацию результатов
                        st.info(f"""
                        **Интерпретация результатов для метода {method_names.get(selected_method, selected_method)}:**
                        
                        - **Precision = {metrics['precision']:.3f}**: метод верно определяет {metrics['precision']*100:.1f}% случаев из всех обнаруженных аномалий.
                        - **Recall = {metrics['recall']:.3f}**: метод обнаруживает {metrics['recall']*100:.1f}% от всех фактических аномалий.
                        - **F1-Score = {metrics['f1']:.3f}**: сбалансированная оценка эффективности метода.
                        
                        {'Высокая точность, но низкая полнота: метод редко ошибается, но пропускает много аномалий.' 
                        if metrics['precision'] > 0.8 and metrics['recall'] < 0.5 else
                        'Высокая полнота, но низкая точность: метод находит большинство аномалий, но часто даёт ложные срабатывания.'
                        if metrics['recall'] > 0.8 and metrics['precision'] < 0.5 else
                        'Хороший баланс точности и полноты: метод эффективен для обнаружения аномалий.'
                        if metrics['precision'] > 0.7 and metrics['recall'] > 0.7 else
                        'Низкие показатели: требуется настройка параметров метода.'}
                        """)
            else:
                st.info("Ни один из методов обнаружения не дал результатов для оценки.")
        except Exception as e:
            st.error(f"Ошибка при оценке качества обнаружения: {str(e)}")
            st.error("Подробности: " + "\n".join(str(e).split("\n")[:5]))
            
            # Показываем дополнительную отладочную информацию
            with st.expander("Дополнительная отладочная информация"):
                st.write("### Состояние переменных:")
                st.write(f"Размерность DataFrame: {df.shape}")
                st.write(f"Количество инъецированных аномалий: {len(injected_anomalies)}")
                st.write(f"Методы для оценки: {list(anomalies_for_evaluation.keys())}")
                for method, arr in anomalies_for_evaluation.items():
                    st.write(f"{method}: тип={type(arr)}, форма={arr.shape}, dtype={arr.dtype}")
else:
    st.info("Для оценки качества обнаружения добавьте аномалии в данные.")

# ====================
# ЭКСПОРТ ДАННЫХ
# ====================
if data_source == "Загруженные данные" and 'original' in df.columns:
    # Для реальных данных предлагаем скачать данные с добавленными аномалиями
    export_df = pd.DataFrame({
        'original_data': df['original'],
        'data_with_anomalies': df['data']
    })
    
    # Если исходный DataFrame имел DatetimeIndex, сохраняем его
    if isinstance(st.session_state.data.index, pd.DatetimeIndex):
        export_df.index = st.session_state.data.index
    
    download_label = "Скачать данные с аномалиями"
    download_filename = 'time_series_with_anomalies.csv'
else:
    # Для синтетических данных экспортируем только сгенерированный ряд
    export_df = df
    download_label = "Скачать данные"
    download_filename = 'synthetic_time_series.csv'

st.download_button(
    label=download_label,
    data=export_df.to_csv(index=True).encode('utf-8'),
    file_name=download_filename,
    mime='text/csv'
)