import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

st.set_page_config(page_title="Анализ аномалий во временных рядах", layout="wide")

# Импортируем новые функции из модуля
from modules.anomaly_module import (
    generate_anomalous_series,
    add_anomalies_to_existing_data,
    z_score_detection,
    iqr_detection,
    hampel_filter,
    detect_plateau,
    evaluate_anomaly_detection,
    create_true_anomaly_mask,
    run_parameter_experiment,
    get_default_parameter_ranges,
    suggest_optimal_parameters,
    # Add these three functions for report generation
    prepare_anomaly_report_data,
    generate_anomaly_detection_yaml,
    format_anomaly_info_for_report
)

# Импортируем модуль отчетов
import modules.reporting as reporting

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
        point_amp_min = st.number_input("Минимальная амплитуда", 0.1, 100.0, 1.0, key="point_amp_min")
        point_amp_max = st.number_input("Максимальная амплитуда", 0.1, 100.0, 2.0, key="point_amp_max")
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
        ext_shift = st.number_input("Смещение уровня", -100.0, 100.0, -2.5, key="ext_shift")
        
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
    
    # Добавляем пресеты настроек
    preset_options = {
        "Сбалансированный": {
            "z_threshold": 3.0,
            "hampel_window_percent": 0.5,
            "hampel_sigma": 3.0,
            "iqr_multiplier": 1.5,
            "plateau_threshold": 0.001,
            "plateau_duration": 10
        },
        "Чувствительный": {
            "z_threshold": 2.0,
            "hampel_window_percent": 0.3,
            "hampel_sigma": 2.0,
            "iqr_multiplier": 1.0,
            "plateau_threshold": 0.0005,
            "plateau_duration": 5
        },
        "Строгий": {
            "z_threshold": 4.0,
            "hampel_window_percent": 1.0,
            "hampel_sigma": 4.0,
            "iqr_multiplier": 2.5,
            "plateau_threshold": 0.002,
            "plateau_duration": 15
        },
        "Пользовательский": {}  # Пустой пресет для пользовательских настроек
    }
    
    # Устанавливаем значения по умолчанию для флагов методов
    # Эти значения будут использоваться во всех режимах настройки
    use_zscore = True
    use_hampel = True
    use_iqr = True
    use_plateau = True
    
    preset = st.selectbox(
        "Выберите пресет настроек",
        options=list(preset_options.keys()),
        help="Предустановленные наборы параметров для разных сценариев"
    )
    
    # Выбор методов обнаружения (одинаков для всех режимов)
    method_selection = st.multiselect(
        "Выберите методы обнаружения",
        options=["Z-Score", "Фильтр Хампеля", "IQR", "Плато"],
        default=["Z-Score", "Фильтр Хампеля", "IQR", "Плато"],
        help="Выберите, какие методы использовать для обнаружения аномалий"
    )
    
    # Устанавливаем флаги на основе выбора пользователя
    use_zscore = "Z-Score" in method_selection
    use_hampel = "Фильтр Хампеля" in method_selection
    use_iqr = "IQR" in method_selection
    use_plateau = "Плато" in method_selection
    
    # Автоматическая настройка параметров
    auto_adjust = st.checkbox(
        "Адаптивная настройка параметров", 
        value=False,
        help="Автоматически настраивает параметры на основе характеристик ряда"
    )
    
    if auto_adjust:
        # Расчет адаптивных параметров на основе данных
        data_stats = {
            "mean": float(np.nanmean(df['data'])),
            "std": float(np.nanstd(df['data'])),
            "iqr": float(np.nanpercentile(df['data'], 75) - np.nanpercentile(df['data'], 25)),
            "length": len(df),
            "has_nan": np.isnan(df['data']).any()
        }
        
        # Адаптивно настраиваем параметры
        adapt_z_threshold = max(2.5, min(4.0, 3.0 * data_stats["std"] / (data_stats["iqr"] / 1.35)))
        adapt_hampel_window = max(5, min(50, int(np.sqrt(data_stats["length"]) / 2)))
        adapt_hampel_window_percent = adapt_hampel_window / data_stats["length"] * 100
        adapt_iqr_multiplier = max(1.0, min(3.0, 1.5 * data_stats["std"] / data_stats["iqr"]))
        
        # Показываем рассчитанные параметры
        st.info(f"""
        **Адаптивные параметры для текущего ряда:**
        - Z-Score порог: {adapt_z_threshold:.2f}
        - Размер окна Хампеля: {adapt_hampel_window} точек ({adapt_hampel_window_percent:.2f}%)
        - IQR множитель: {adapt_iqr_multiplier:.2f}
        """)
        
        # Используем адаптивные параметры
        z_threshold = adapt_z_threshold
        hampel_window_percent = adapt_hampel_window_percent
        hampel_sigma = 3.0
        iqr_multiplier = adapt_iqr_multiplier
        plateau_threshold = 0.001
        plateau_duration = 10
        
        # Настройка адаптивного размера окна для Хампеля
        hampel_adaptive = True
        hampel_window = 0  # Будет использоваться процентное значение
        
        # Опция для обнаружения плато NaN
        detect_nan = True
    else:
        # Используем параметры из пресета или пользовательские настройки
        use_preset_values = preset != "Пользовательский"
        
        # Показываем параметры в режиме "Простые" или "Расширенные"
        parameter_mode = st.radio(
            "Режим настройки",
            ["Простой", "Расширенный"],
            horizontal=True,
            help="Выберите режим отображения параметров"
        )
        
        # Z-Score
        with st.expander("📈 Z-Score метод", expanded=True):
            use_zscore = st.checkbox("Использовать Z-Score", value=True, 
                                    help="Метод на основе стандартных отклонений от среднего значения")
            
            # В расширенном режиме показываем дополнительную информацию
            if parameter_mode == "Расширенный":
                st.markdown("""
                **Z-Score** измеряет отклонение точки от среднего значения в единицах стандартного отклонения:
                - Z > 3: точка отстоит более чем на 3 стандартных отклонения (аномалия)
                - Чем больше порог, тем меньше аномалий будет найдено
                """)
            
            if use_preset_values and preset in preset_options:
                z_threshold = preset_options[preset]["z_threshold"]
                st.info(f"Порог Z-Score: {z_threshold} (из пресета '{preset}')")
            else:
                z_threshold = st.slider(
                    "Порог Z-Score", 
                    1.0, 5.0, 3.0, step=0.5 if parameter_mode == "Простой" else 0.1,
                    help=(
                        "Количество стандартных отклонений от среднего:\n"
                        "• 1σ ≈ 68% данных\n"
                        "• 2σ ≈ 95% данных\n"
                        "• 3σ ≈ 99.7% данных\n"
                        "Рекомендуемое значение: 3 (выявляет явные выбросы)"
                    )
                )
            
            # Интерактивная визуализация для расширенного режима
            if parameter_mode == "Расширенный" and not use_preset_values:
                # Показываем, как изменение порога влияет на количество обнаруженных аномалий
                test_thresholds = np.arange(1.0, 5.1, 0.5)
                anomaly_counts = []
                
                for thresh in test_thresholds:
                    anomaly_counts.append(np.sum(z_score_detection(df['data'], thresh)))
                
                # Строим маленький график
                threshold_fig = go.Figure()
                threshold_fig.add_trace(go.Scatter(
                    x=test_thresholds,
                    y=anomaly_counts,
                    mode='lines+markers',
                    name='Количество аномалий'
                ))
                threshold_fig.add_vline(
                    x=z_threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Текущий порог: {z_threshold}",
                    annotation_position="top right"
                )
                threshold_fig.update_layout(
                    title="Влияние порога на количество обнаруженных аномалий",
                    xaxis_title="Порог Z-Score",
                    yaxis_title="Количество аномалий",
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(threshold_fig, use_container_width=True)
        
        # Hampel
        with st.expander("🔍 Фильтр Хампеля", expanded=True):
            use_hampel = st.checkbox("Использовать Хампель", value=True,
                                   help="Устойчивый метод на основе медиан и MAD")
            
            # В расширенном режиме показываем дополнительную информацию
            if parameter_mode == "Расширенный":
                st.markdown("""
                **Фильтр Хампеля** использует локальную медиану и MAD для обнаружения выбросов:
                - Более устойчив к шумам по сравнению с Z-Score
                - Эффективен для рядов с трендами и сезонностью
                """)
            
            # Добавляем опцию для адаптивного размера окна
            hampel_adaptive = st.checkbox("Адаптивный размер окна", value=True, 
                                        help="Размер окна рассчитывается как процент от длины ряда")
            
            if hampel_adaptive:
                # Если выбран адаптивный режим, показываем слайдер для процента
                if use_preset_values and preset in preset_options:
                    hampel_window_percent = preset_options[preset]["hampel_window_percent"]
                    st.info(f"Процент от длины ряда: {hampel_window_percent}% (из пресета '{preset}')")
                else:
                    hampel_window_percent = st.slider(
                        "Процент от длины ряда (%)", 
                        0.1, 5.0, 0.5, 
                        step=0.1,
                        help=(
                            "Размер окна как процент от длины ряда:\n"
                            "• Меньшие значения: выше чувствительность к локальным изменениям\n"
                            "• Большие значения: рассматривается более глобальный контекст\n"
                            "Рекомендуется: 0.5-1% для ежеминутных данных за недели"
                        )
                    )
                
                # Рассчитываем приблизительный размер окна для текущих данных
                approx_window = max(5, min(int(len(df) * hampel_window_percent / 100), len(df) // 5))
                st.info(f"Примерный размер окна для текущего ряда: {approx_window} точек")
                hampel_window = 0  # Устанавливаем в 0, чтобы использовалось значение процента
            else:
                # Если адаптивный режим выключен, используем обычный слайдер для окна
                hampel_window = st.slider(
                    "Размер окна", 
                    5, 500, 20, 
                    step=5,
                    help=(
                        "Количество точек для расчета локальной медианы:\n"
                        "• Меньшие значения: выше чувствительность к локальным изменениям\n"
                        "• Большие значения: устойчивее к шумам\n"
                        "Для больших рядов (>10K точек) рекомендуется 50-200"
                    )
                )
                hampel_window_percent = 0.5  # Значение по умолчанию, не будет использоваться
            
            if use_preset_values and preset in preset_options:
                hampel_sigma = preset_options[preset]["hampel_sigma"]
                st.info(f"Коэффициент чувствительности: {hampel_sigma} (из пресета '{preset}')")
            else:
                hampel_sigma = st.slider(
                    "Коэффициент чувствительности", 
                    1.0, 5.0, 3.0, 
                    step=0.1,
                    help=(
                        "Множитель для медианного абсолютного отклонения (MAD):\n"
                        "• 3.0 соответствует ~3σ в нормальном распределении\n"
                        "• Уменьшайте для большей чувствительности\n"
                        "• Увеличивайте для снижения ложных срабатываний"
                    )
                )
            
            # Интерактивная визуализация для расширенного режима
            if parameter_mode == "Расширенный" and not use_preset_values:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Влияние размера окна**")
                    st.markdown("""
                    - Маленькое окно: чувствительность к локальным изменениям
                    - Большое окно: устойчивость к шумам, но может пропустить короткие аномалии
                    """)
                with col2:
                    st.markdown("**Влияние коэффициента**")
                    st.markdown("""
                    - Маленький коэффициент: больше аномалий, но и ложных тоже
                    - Большой коэффициент: меньше аномалий, но более уверенная детекция
                    """)
        
        # IQR
        with st.expander("📏 Межквартильный размах (IQR)"):
            use_iqr = st.checkbox("Использовать IQR", value=True,
                                help="Метод на основе квартилей распределения")
            
            # В расширенном режиме показываем дополнительную информацию
            if parameter_mode == "Расширенный":
                st.markdown("""
                **Межквартильный размах (IQR)** определяет выбросы на основе квартилей распределения:
                - Не зависит от среднего значения, более устойчив к выбросам
                - Хорошо работает для несимметричных распределений
                """)
            
            if use_preset_values and preset in preset_options:
                iqr_multiplier = preset_options[preset]["iqr_multiplier"]
                st.info(f"Множитель IQR: {iqr_multiplier} (из пресета '{preset}')")
            else:
                iqr_multiplier = st.slider(
                    "Множитель IQR", 
                    0.5, 5.0, 1.5, 
                    step=0.1,
                    help=(
                        "Определяет границы выбросов:\n"
                        "• Стандартное значение: 1.5 (выявляет умеренные выбросы)\n"
                        "• 3.0 для экстремальных выбросов\n"
                        "• Формула: Q1 - k*IQR и Q3 + k*IQR"
                    )
                )
                
            # Интерактивная визуализация для расширенного режима
            if parameter_mode == "Расширенный" and not use_preset_values:
                # Показываем, как выглядят границы IQR на боксплоте
                q1 = np.nanpercentile(df['data'], 25)
                q3 = np.nanpercentile(df['data'], 75)
                iqr_value = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr_value
                upper_bound = q3 + iqr_multiplier * iqr_value
                
                boxplot_fig = go.Figure()
                boxplot_fig.add_trace(go.Box(
                    y=df['data'].dropna(),
                    name='Распределение',
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    boxmean=True
                ))
                boxplot_fig.add_hline(
                    y=lower_bound,
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Нижняя граница",
                    annotation_position="left"
                )
                boxplot_fig.add_hline(
                    y=upper_bound,
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Верхняя граница",
                    annotation_position="right"
                )
                boxplot_fig.update_layout(
                    title=f"Границы IQR с множителем {iqr_multiplier}",
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                st.plotly_chart(boxplot_fig, use_container_width=True)
        
        # Plateau detection
        with st.expander("⏸ Детекция плато"):
            use_plateau = st.checkbox("Использовать детекцию плато", value=True,
                                    help="Обнаружение участков с постоянным значением")
            
            # В расширенном режиме показываем дополнительную информацию
            if parameter_mode == "Расширенный":
                st.markdown("""
                **Детекция плато** находит участки, где значения почти не меняются:
                - Полезно для обнаружения "застрявших" датчиков
                - Может выявлять периоды отсутствия изменений
                """)
            
            if use_preset_values and preset in preset_options:
                plateau_threshold = preset_options[preset]["plateau_threshold"]
                st.info(f"Порог производной: {plateau_threshold} (из пресета '{preset}')")
            else:
                plateau_threshold = st.slider(
                    "Порог производной", 
                    0.0, 0.1, 0.001, 
                    step=0.001 if parameter_mode == "Расширенный" else 0.005
                )
            
            if use_preset_values and preset in preset_options:
                plateau_duration = preset_options[preset]["plateau_duration"]
                st.info(f"Минимальная длительность: {plateau_duration} (из пресета '{preset}')")
            else:
                plateau_duration = st.slider(
                    "Минимальная длительность", 
                    1, 50, 10
                )
            
            detect_nan = st.checkbox(
                "Считать NaN как плато", 
                value=True,
                help="Помечать последовательности пропущенных значений как аномалии"
            )
            
            # Интерактивная визуализация для расширенного режима
            if parameter_mode == "Расширенный" and not use_preset_values:
                st.markdown("""
                **Рекомендации по настройке:**
                - **Порог производной**: чем меньше, тем строже определение "плато"
                - **Минимальная длительность**: зависит от частоты данных и желаемого минимального времени плато
                """)

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
    
    # Hampel
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
# ЧИСЛЕННЫЙ ЭКСПЕРИМЕНТ
# ====================

st.header("🧪 Численный эксперимент", help="Анализ влияния параметров на качество обнаружения аномалий")

if has_injected_anomalies:
    # Если есть аномалии, предлагаем провести эксперимент
    experiment_tab1, experiment_tab2 = st.tabs([
        "🔬 Настройка эксперимента", 
        "📊 Результаты"
    ])
    
    with experiment_tab1:
        st.subheader("Настройка численного эксперимента")
        
        # Выбор метода для эксперимента
        exp_method = st.selectbox(
            "Выберите метод для анализа",
            options=["Z-Score", "IQR", "Фильтр Хампеля", "Детекция плато"],
            format_func=lambda x: x,
            help="Метод обнаружения аномалий, параметры которого будут анализироваться"
        )
        
        # Преобразуем название метода для внутреннего использования
        method_mapping = {
            "Z-Score": "z_score",
            "IQR": "iqr",
            "Фильтр Хампеля": "hampel",
            "Детекция плато": "plateau"
        }
        internal_method = method_mapping[exp_method]
        
        # Получаем диапазоны параметров по умолчанию
        default_ranges = get_default_parameter_ranges()
        method_params = default_ranges.get(internal_method, {})
        
        # Настройка параметров для анализа в зависимости от выбранного метода
        st.write("### Выберите параметры для анализа")
        
        # Получаем список доступных параметров для выбранного метода
        available_params = list(method_params.keys())
        param_names_mapping = {
            "threshold": "Порог чувствительности",
            "multiplier": "Множитель IQR",
            "window": "Размер окна (точек)",
            "sigma": "Коэффициент чувствительности",
            "window_percent": "Размер окна (% от длины ряда)",
            "min_duration": "Минимальная длительность плато",
        }
        
        # Создаем удобные для пользователя названия параметров
        readable_params = [param_names_mapping.get(p, p) for p in available_params]
        
        # Пользователь выбирает, какие параметры включить в эксперимент
        selected_readable_params = st.multiselect(
            "Выберите параметры для анализа (один или несколько)",
            options=readable_params,
            default=[readable_params[0]] if readable_params else [],
            help="Выберите параметры, которые хотите изменять в эксперименте. Можно выбрать несколько параметров для многомерного анализа."
        )
        
        # Преобразование обратно в технические имена параметров
        reverse_mapping = {v: k for k, v in param_names_mapping.items()}
        selected_params = [reverse_mapping.get(p, p) for p in selected_readable_params]
        
        # Если не выбрано ни одного параметра, сообщаем об этом
        if not selected_params:
            st.warning("Выберите хотя бы один параметр для анализа")
            st.stop()
        
        param_ranges = {}
        fixed_params = {}
        
        # Создаем две колонки для визуального разделения настроек параметров
        col1, col2 = st.columns(2)
        
        # Для Z-Score
        if internal_method == "z_score":
            with col1:
                st.write("#### Настройка параметров:")
                for param in selected_params:
                    if param == "threshold":
                        threshold_min = st.number_input("Минимальное значение порога", 0.5, 10.0, 1.0, 0.5)
                        threshold_max = st.number_input("Максимальное значение порога", threshold_min, 10.0, 5.0, 0.5)
                        threshold_step = st.number_input("Шаг значения порога", 0.1, 1.0, 0.5, 0.5)
                        
                        # Создаем диапазон значений
                        threshold_range = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step).tolist()
                        param_ranges['threshold'] = threshold_range
        
        with col2:
            st.write("#### Описание параметров:")
            if "threshold" in selected_params:
                st.markdown("""
                **Порог чувствительности:**
                * Определяет чувствительность метода
                * Меньшие значения → больше аномалий
                * Большие значения → меньше аномалий, но более уверенная детекция
                * Стандартное значение: 3.0
                """)
    
            # Для IQR
            elif internal_method == "iqr":
                with col1:
                    st.write("#### Настройка параметров:")
                    for param in selected_params:
                        if param == "multiplier":
                            multiplier_min = st.number_input("Минимальное значение множителя", 0.5, 5.0, 0.5, 0.5)
                            multiplier_max = st.number_input("Максимальное значение множителя", multiplier_min, 5.0, 3.0, 0.5)
                            multiplier_step = st.number_input("Шаг значения множителя", 0.1, 1.0, 0.5, 0.1)
                            
                            # Создаем диапазон значений
                            multiplier_range = np.arange(multiplier_min, multiplier_max + multiplier_step/2, multiplier_step).tolist()
                            param_ranges['multiplier'] = multiplier_range
                
                with col2:
                    st.write("#### Описание параметров:")
                    if "multiplier" in selected_params:
                        st.markdown("""
                        **Множитель IQR:**
                        * Определяет границы выбросов: Q1 - k*IQR и Q3 + k*IQR
                        * Стандартное значение: 1.5 (умеренные выбросы)
                        * 3.0 для экстремальных выбросов
                        """)
            
            # Для Hampel
            elif internal_method == "hampel":
                with col1:
                    st.write("#### Настройка параметров:")
                    for param in selected_params:
                        if param == "window":
                            window_min = st.number_input("Минимальный размер окна", 3, 100, 5, 1)
                            window_max = st.number_input("Максимальный размер окна", window_min, 200, 30, 5)
                            window_step = st.number_input("Шаг размера окна", 1, 20, 5, 1)
                            
                            # Создаем диапазон значений
                            window_range = range(window_min, window_max + 1, window_step)
                            param_ranges['window'] = list(window_range)
                            
                            # Если окно выбрано как изменяемый параметр, устанавливаем adaptive_window в False
                            fixed_params['adaptive_window'] = False
                        
                        if param == "sigma":
                            sigma_min = st.number_input("Минимальное значение коэффициента", 0.5, 5.0, 1.0, 0.5)
                            sigma_max = st.number_input("Максимальное значение коэффициента", sigma_min, 10.0, 4.0, 0.5)
                            sigma_step = st.number_input("Шаг значения коэффициента", 0.1, 1.0, 0.5, 0.1)
                            
                            # Создаем диапазон значений
                            sigma_range = np.arange(sigma_min, sigma_max + sigma_step/2, sigma_step).tolist()
                            param_ranges['sigma'] = sigma_range
                        
                        if param == "window_percent":
                            wp_min = st.number_input("Минимальный процент", 0.1, 5.0, 0.1, 0.1)
                            wp_max = st.number_input("Максимальный процент", wp_min, 10.0, 2.0, 0.1)
                            wp_step = st.number_input("Шаг процента", 0.1, 1.0, 0.2, 0.1)
                            
                            # Создаем диапазон значений
                            wp_range = np.arange(wp_min, wp_max + wp_step/2, wp_step).tolist()
                            param_ranges['window_percent'] = wp_range
                            
                            # Если window_percent выбран как изменяемый параметр, устанавливаем adaptive_window в True
                            fixed_params['adaptive_window'] = True
            
            # Обработка конфликтов параметров
            if "window" in selected_params and "window_percent" in selected_params:
                st.warning("⚠️ Вы выбрали несовместимые параметры: 'Размер окна' и 'Размер окна (%)'. Метод может работать только с одним из них.")
            
            # Установка необходимых фиксированных параметров, если они не являются изменяемыми
            if "window" not in selected_params and "window_percent" not in selected_params:
                st.info("Для метода Хампеля нужно указать размер окна. Выберите один из параметров или укажите фиксированное значение:")
                if st.checkbox("Использовать адаптивное окно", value=True):
                    fixed_window_percent = st.number_input("Фиксированный процент от длины ряда (%)", 0.1, 5.0, 0.5, 0.1)
                    fixed_params['adaptive_window'] = True
                    fixed_params['window_percent'] = fixed_window_percent
                else:
                    fixed_window = st.number_input("Фиксированный размер окна (точек)", 3, 100, 20, 1)
                    fixed_params['adaptive_window'] = False
                    fixed_params['window'] = fixed_window
            
            if "sigma" not in selected_params:
                fixed_sigma = st.number_input("Фиксированный коэффициент чувствительности", 1.0, 5.0, 3.0, 0.5)
                fixed_params['sigma'] = fixed_sigma
        
        with col2:
            st.write("#### Описание параметров:")
            if "window" in selected_params:
                st.markdown("""
                **Размер окна (точек):**
                * Определяет контекст для расчета медианы
                * Меньшие значения → выше чувствительность к локальным изменениям
                * Большие значения → большая устойчивость к шумам
                * Несовместим с параметром "Размер окна (%)"
                """)
            
            if "sigma" in selected_params:
                st.markdown("""
                **Коэффициент чувствительности:**
                * Множитель для MAD (медианное абсолютное отклонение)
                * Меньшие значения → больше аномалий будет обнаружено
                * Большие значения → только явные аномалии
                * 3.0 примерно соответствует 3σ в нормальном распределении
                """)
            
            if "window_percent" in selected_params:
                st.markdown("""
                **Размер окна (% от длины ряда):**
                * Позволяет адаптировать размер окна к длине ряда
                * Рекомендуемые значения: 0.5-1% для данных с высокой частотой
                * 1-3% для данных с низкой частотой
                * Несовместим с параметром "Размер окна (точек)"
                """)
    
            # Для Plateau
            elif internal_method == "plateau":
                with col1:
                    st.write("#### Настройка параметров:")
                    for param in selected_params:
                        if param == "threshold":
                            # Используем логарифмическую шкалу для порога
                            threshold_min_exp = st.slider("Минимальное значение порога (10^x)", -6, -1, -4)
                            threshold_max_exp = st.slider("Максимальное значение порога (10^x)", threshold_min_exp, 0, -2)
                            num_steps = st.number_input("Количество точек", 3, 20, 10, 1)
                            
                            # Создаем логарифмический диапазон значений
                            threshold_range = np.logspace(threshold_min_exp, threshold_max_exp, num_steps).tolist()
                            param_ranges['threshold'] = threshold_range
                        
                        if param == "min_duration":
                            duration_min = st.number_input("Минимальная длительность", 1, 20, 2, 1)
                            duration_max = st.number_input("Максимальная длительность", duration_min, 100, 30, 1)
                            duration_step = st.number_input("Шаг длительности", 1, 10, 2, 1)
                            
                            # Создаем диапазон значений
                            duration_range = range(duration_min, duration_max + 1, duration_step)
                            param_ranges['min_duration'] = list(duration_range)
                
                with col2:
                    st.write("#### Описание параметров:")
                    if "threshold" in selected_params:
                        st.markdown("""
                        **Порог производной:**
                        * Определяет максимальное допустимое изменение для плато
                        * Меньшие значения → более строгое определение "плато"
                        * Большие значения → больше участков будет считаться плато
                        """)
                    
                    if "min_duration" in selected_params:
                        st.markdown("""
                        **Минимальная длительность плато:**
                        * Определяет, сколько последовательных точек должно быть в плато
                        * Меньшие значения → больше коротких плато будет обнаружено
                        * Большие значения → только длинные плато
                        """)

    # Расчет количества экспериментов и оценка сложности
    num_experiments = np.prod([len(values) for values in param_ranges.values()])
    
    st.subheader("Запуск эксперимента")
    st.info(f"""
    **Метод: {exp_method}**
    Выбранные параметры: {', '.join(selected_readable_params)}
    
    Будет выполнено {num_experiments} экспериментов.
    """)
    
    # Предупреждение о сложности эксперимента
    if num_experiments > 500:
        st.warning(f"⚠️ Внимание! Вы запускаете {num_experiments} экспериментов, что может занять значительное время. Рекомендуется уменьшить диапазоны параметров или их количество.")
    
    # Кнопка запуска эксперимента
    run_experiment = st.button("Запустить эксперимент", type="primary")
    
    if run_experiment:
        with st.spinner("Выполнение численного эксперимента..."):
            # Получаем истинную маску аномалий для оценки качества
            if data_source == "Синтетические данные":
                injected_anomalies = metadata
            else:
                injected_anomalies = metadata
            
            true_anomaly_mask = create_true_anomaly_mask(injected_anomalies, len(df))
            
            # Запускаем эксперимент
            experiment_results = run_parameter_experiment(
                data=df['data'].values,
                true_anomalies=true_anomaly_mask,
                method=internal_method,
                param_ranges=param_ranges,
                fixed_params=fixed_params
            )
            
            # Сохраняем результаты в session_state
            st.session_state.experiment_results = experiment_results
            st.session_state.experiment_method = internal_method
            st.session_state.experiment_params = list(param_ranges.keys())
            
            # Переключаемся на вкладку результатов
            st.success("Эксперимент завершен! Перейдите на вкладку 'Результаты' для просмотра.")

    # Вкладка с результатами
    with experiment_tab2:
        st.subheader("Результаты численного эксперимента")
        
        if 'experiment_results' in st.session_state:
            results_df = st.session_state.experiment_results
            method = st.session_state.experiment_method
            params = st.session_state.experiment_params
            
            # Показываем общую статистику
            st.write(f"### 📊 Всего экспериментов: {len(results_df)}")
            
            # Находим оптимальные параметры
            optimal_params = suggest_optimal_parameters(results_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("#### 🥇 Лучший F1-Score")
                best_f1 = optimal_params['best_f1']
                st.info(f"""
                **F1-Score: {best_f1['f1']:.3f}**
                Precision: {best_f1['precision']:.3f}
                Recall: {best_f1['recall']:.3f}
                
                Параметры:
                {', '.join([f'{key}: {value}' for key, value in best_f1.items() if key in params])}
                """)
            
            with col2:
                st.write("#### 🎯 Лучшая точность (Precision)")
                best_precision = optimal_params['best_precision']
                st.info(f"""
                **Precision: {best_precision['precision']:.3f}**
                Recall: {best_precision['recall']:.3f}
                F1-Score: {best_precision['f1']:.3f}
                
                Параметры:
                {', '.join([f'{key}: {value}' for key, value in best_precision.items() if key in params])}
                """)
            
            with col3:
                st.write("#### 🔍 Лучшая полнота (Recall)")
                best_recall = optimal_params['best_recall']
                st.info(f"""
                **Recall: {best_recall['recall']:.3f}**
                Precision: {best_recall['precision']:.3f}
                F1-Score: {best_recall['f1']:.3f}
                
                Параметры:
                {', '.join([f'{key}: {value}' for key, value in best_recall.items() if key in params])}
                """)
            
            # Выбор визуализации результатов
            st.write("### Визуализация результатов эксперимента")
            
            # Определяем размерность эксперимента
            experiment_dim = len(params)
            
            # Выбор метрики для визуализации
            selected_metric = st.selectbox(
                "Выберите метрику для визуализации:",
                options=["f1", "precision", "recall", "num_anomalies"],
                format_func=lambda x: {"f1": "F1-Score", "precision": "Precision", "recall": "Recall", "num_anomalies": "Количество аномалий"}[x]
            )
            
            # Визуализация в зависимости от размерности эксперимента
            if experiment_dim == 1:
                # Одномерный эксперимент - показываем график зависимости
                param = params[0]
                
                # Строим график зависимости метрик от параметра
                fig_metrics = go.Figure()
                
                # Сортируем DataFrame по значению параметра
                sorted_df = results_df.sort_values(param)
                
                # Добавляем линии для каждой метрики
                fig_metrics.add_trace(go.Scatter(
                    x=sorted_df[param], 
                    y=sorted_df['precision'],
                    mode='lines+markers',
                    name='Precision',
                    line=dict(color='green', width=2)
                ))
                
                fig_metrics.add_trace(go.Scatter(
                    x=sorted_df[param], 
                    y=sorted_df['recall'],
                    mode='lines+markers',
                    name='Recall',
                    line=dict(color='blue', width=2)
                ))
                
                fig_metrics.add_trace(go.Scatter(
                    x=sorted_df[param], 
                    y=sorted_df['f1'],
                    mode='lines+markers',
                    name='F1-Score',
                    line=dict(color='red', width=2)
                ))
                
                # Настраиваем внешний вид графика
                fig_metrics.update_layout(
                    title=f"Влияние параметра {param} на метрики качества",
                    xaxis_title=param,
                    yaxis_title="Значение метрики",
                    yaxis=dict(range=[0, 1.05]),
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
                
            elif experiment_dim == 2:
                # Двумерный эксперимент - показываем тепловую карту
                param1, param2 = params
                
                # Можно визуализировать тепловую карту, если результаты могут быть представлены в виде таблицы
                try:
                    # Создаем сводную таблицу для выбранной метрики
                    pivot_df = results_df.pivot(index=param1, columns=param2, values=selected_metric)
                    
                    # Визуализация в виде тепловой карты
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis',
                        colorbar=dict(title=selected_metric.capitalize()),
                        hoverongaps=False,
                        hovertemplate=f"{param1}: %{{y}}<br>{param2}: %{{x}}<br>{selected_metric}: %{{z}}<extra></extra>"
                    ))
                    
                    fig_heatmap.update_layout(
                        title=f"Тепловая карта {selected_metric} в зависимости от {param1} и {param2}",
                        xaxis_title=param2,
                        yaxis_title=param1,
                        height=600
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Визуализация в виде 3D поверхности
                    fig_3d = go.Figure(data=[go.Surface(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis',
                        colorbar=dict(title=selected_metric.capitalize())
                    )])
                    
                    fig_3d.update_layout(
                        title=f"3D-поверхность {selected_metric} в зависимости от {param1} и {param2}",
                        scene=dict(
                            xaxis_title=param2,
                            yaxis_title=param1,
                            zaxis_title=selected_metric
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Не удалось создать визуализацию: {str(e)}")
                    st.info("Возможно, некоторые параметры имеют нечисловые значения или результаты не могут быть представлены в виде таблицы.")
                
            elif experiment_dim >= 3:
                # Многомерный эксперимент (3+ параметров)
                st.write("#### Многомерная визуализация")
                
                # Выбор двух параметров для основной визуализации
                viz_params = st.multiselect(
                    "Выберите два параметра для визуализации:",
                    options=params,
                    default=params[:2] if len(params) >= 2 else params,
                    max_selections=2,
                    help="Выберите два параметра, которые будут отображаться на осях X и Y"
                )
                
                if len(viz_params) == 2:
                    param1, param2 = viz_params
                    
                    # Если выбрано больше двух параметров, позволяем фильтровать по оставшимся
                    other_params = [p for p in params if p not in viz_params]
                    
                    if other_params:
                        st.write("#### Фильтрация по другим параметрам")
                        
                        # Создаем фильтры для остальных параметров
                        filter_values = {}
                        
                        for param in other_params:
                            unique_values = sorted(results_df[param].unique())
                            if len(unique_values) <= 10:  # Если не слишком много уникальных значений
                                filter_value = st.selectbox(
                                    f"Значение для {param}:",
                                    options=unique_values,
                                    index=len(unique_values)//2  # Выбираем среднее значение по умолчанию
                               
                                )
                                filter_values[param] = filter_value
                            else:
                                # Если слишком много уникальных значений, используем слайдер
                                min_val = min(unique_values)
                                max_val = max(unique_values)
                                filter_value = st.slider(
                                    f"Значение для {param}:",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val + max_val)/2,
                                    step=(max_val - min_val)/20
                                )
                                # Находим ближайшее значение в данных
                                filter_values[param] = min(unique_values, key=lambda x: abs(x - filter_value))
                        
                        # Фильтруем DataFrame по выбранным значениям
                        filtered_df = results_df.copy()
                        for param, value in filter_values.items():
                            filtered_df = filtered_df[filtered_df[param] == value]
                        
                        st.write(f"#### Визуализация для отфильтрованных данных ({len(filtered_df)} экспериментов)")
                        
                        if not filtered_df.empty:
                            # Создаем сводную таблицу для визуализации
                            try:
                                # Создаем сводную таблицу для выбранной метрики
                                pivot_df = filtered_df.pivot(index=param1, columns=param2, values=selected_metric)
                                
                                # Визуализация в виде тепловой карты
                                fig_heatmap = go.Figure(data=go.Heatmap(
                                    z=pivot_df.values,
                                    x=pivot_df.columns,
                                    y=pivot_df.index,
                                    colorscale='Viridis',
                                    colorbar=dict(title=selected_metric.capitalize()),
                                    hoverongaps=False,
                                    hovertemplate=f"{param1}: %{{y}}<br>{param2}: %{{x}}<br>{selected_metric}: %{{z}}<extra></extra>"
                                ))
                                
                                # Добавляем информацию о фильтрах в заголовок
                                filter_info = ", ".join([f"{p}={v}" for p, v in filter_values.items()])
                                
                                fig_heatmap.update_layout(
                                    title=f"Тепловая карта {selected_metric} для {param1} и {param2} (при {filter_info})",
                                    xaxis_title=param2,
                                    yaxis_title=param1,
                                    height=600
                                )
                                
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Не удалось создать визуализацию: {str(e)}")
                                st.info("Возможно, отфильтрованных данных недостаточно для построения тепловой карты.")
                        else:
                            st.warning("Нет данных, соответствующих выбранным фильтрам.")
                    else:
                        # Если других параметров нет, просто показываем тепловую карту
                        try:
                            # Создаем сводную таблицу для выбранной метрики
                            pivot_df = results_df.pivot(index=param1, columns=param2, values=selected_metric)
                            
                            # Визуализация в виде тепловой карты
                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=pivot_df.values,
                                x=pivot_df.columns,
                                y=pivot_df.index,
                                colorscale='Viridis',
                                colorbar=dict(title=selected_metric.capitalize()),
                                hoverongaps=False,
                                hovertemplate=f"{param1}: %{{y}}<br>{param2}: %{{x}}<br>{selected_metric}: %{{z}}<extra></extra>"
                            ))
                            
                            fig_heatmap.update_layout(
                                title=f"Тепловая карта {selected_metric} в зависимости от {param1} и {param2}",
                                xaxis_title=param2,
                                yaxis_title=param1,
                                height=600
                            )
                            
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Не удалось создать визуализацию: {str(e)}")
                            st.info("Возможно, результаты не могут быть представлены в виде таблицы.")
            else:
                st.warning("Выберите ровно два параметра для визуализации.")
        
            # Таблица с результатами для всех случаев
            st.write("### Таблица результатов")
            st.caption("Лучшие значения для каждой метрики выделены цветом: Precision - зеленый, Recall - синий, F1-score - голубой")
            
            # Используем встроенные методы highlight_max
            st.dataframe(
                results_df.sort_values('f1', ascending=False).style.highlight_max(subset=['precision'], color='#a8d08d')
                .highlight_max(subset=['recall'], color='#8db3e2')
                .highlight_max(subset=['f1'], color='#c6e0b4'),
                use_container_width=True
            )
        else:
            st.info("Нет результатов экспериментов. Запустите эксперимент на вкладке 'Настройка эксперимента'.")

# ====================
# ОТЧЕТЫ
# ====================
st.subheader("📊 Отчеты и экспорт результатов")

# Define this function outside any try block
def create_matplotlib_figure_from_plotly(plotly_fig, width=10, height=6):
    """
    Преобразует график Plotly в график Matplotlib для более надежного экспорта
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Создаем новую фигуру Matplotlib
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Перебираем все трейсы из plotly и добавляем их в matplotlib
    for trace in plotly_fig.data:
        # Получаем данные и свойства
        x_data = trace.x if hasattr(trace, 'x') and trace.x is not None else []
        y_data = trace.y if hasattr(trace, 'y') and trace.y is not None else []
        
        # Пропускаем пустые трейсы
        if len(x_data) == 0 or len(y_data) == 0:
            continue
            
        name = trace.name if hasattr(trace, 'name') else 'Series'
        mode = trace.mode if hasattr(trace, 'mode') else 'lines'
        
        # Определяем стиль линии
        linestyle = '-'
        if hasattr(trace, 'line') and hasattr(trace.line, 'dash'):
            if trace.line.dash == 'dash':
                linestyle = '--'
            elif trace.line.dash == 'dot':
                linestyle = ':'
        
        # Определяем цвет
        color = 'blue'
        if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
            color = trace.marker.color
        elif hasattr(trace, 'line') and hasattr(trace.line, 'color'):
            color = trace.line.color
            
        # Рисуем в зависимости от режима
        if 'lines' in mode and 'markers' in mode:
            ax.plot(x_data, y_data, label=name, linestyle=linestyle, marker='o', color=color)
        elif 'lines' in mode:
            ax.plot(x_data, y_data, label=name, linestyle=linestyle, color=color)
        elif 'markers' in mode:
            ax.scatter(x_data, y_data, label=name, color=color)
    
    # Настраиваем заголовок и метки осей
    if hasattr(plotly_fig.layout, 'title') and hasattr(plotly_fig.layout.title, 'text'):
        ax.set_title(plotly_fig.layout.title.text)
    
    if hasattr(plotly_fig.layout, 'xaxis') and hasattr(plotly_fig.layout.xaxis, 'title') and hasattr(plotly_fig.layout.xaxis.title, 'text'):
        ax.set_xlabel(plotly_fig.layout.xaxis.title.text)
        
    if hasattr(plotly_fig.layout, 'yaxis') and hasattr(plotly_fig.layout.yaxis, 'title') and hasattr(plotly_fig.layout.yaxis.title, 'text'):
        ax.set_ylabel(plotly_fig.layout.yaxis.title.text)
    
    # Добавляем горизонтальные и вертикальные линии, если они есть
    if hasattr(plotly_fig, 'layout') and hasattr(plotly_fig.layout, 'shapes'):
        for shape in plotly_fig.layout.shapes:
            if shape.type == 'line':
                if shape.y0 == shape.y1:  # горизонтальная линия
                    ax.axhline(y=shape.y0, color='green', linestyle='--', alpha=0.5)
                elif shape.x0 == shape.x1:  # вертикальная линия
                    ax.axvline(x=shape.x0, color='red', linestyle='--', alpha=0.5)
    
    # Добавляем легенду
    if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
        ax.legend(loc='best')
    
    # Настраиваем сетку для лучшей читаемости
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Улучшаем внешний вид
    plt.tight_layout()
    
    return fig

with st.expander("Сформировать отчет об аномалиях", expanded=False):
    st.write("### Отчет об обнаружении аномалий")
    
    # Выбор типа отчета
    report_type = st.radio(
        "Тип отчета",
        ["Базовый отчет", "Подробный отчет с метриками", "Отчет по экспериментам"],
        horizontal=True,
        help="Выберите тип отчета для генерации"
    )
    
    if st.button("Сформировать отчет", type="primary"):
        try:
            with st.spinner("Формирование отчета..."):
                # Подготовка основных данных для отчета
                detection_params = {
                    "z_score": {
                        "enabled": use_zscore,
                        "threshold": z_threshold if 'z_threshold' in locals() else 3.0
                    },
                    "hampel": {
                        "enabled": use_hampel,
                        "window": hampel_window if 'hampel_window' in locals() else 0,
                        "sigma": hampel_sigma if 'hampel_sigma' in locals() else 3.0,
                        "adaptive_window": hampel_adaptive if 'hampel_adaptive' in locals() else True,
                        "window_percent": hampel_window_percent if 'hampel_window_percent' in locals() else 0.5
                    },
                    "iqr": {
                        "enabled": use_iqr,
                        "multiplier": iqr_multiplier if 'iqr_multiplier' in locals() else 1.5
                    },
                    "plateau": {
                        "enabled": use_plateau,
                        "threshold": plateau_threshold if 'plateau_threshold' in locals() else 0.001,
                        "min_duration": plateau_duration if 'plateau_duration' in locals() else 10,
                        "detect_nan": detect_nan if 'detect_nan' in locals() else True
                    }
                }
                
                # Преобразуем информацию об аномалиях в подходящий формат
                if data_source == "Синтетические данные":
                    anomaly_information = metadata
                else:
                    anomaly_information = metadata if 'metadata' in locals() else []
                
                # Получаем метрики, если есть информация о внедренных аномалиях
                metrics_data = None
                if has_injected_anomalies and 'metrics_results' in locals():
                    metrics_data = metrics_results
                
                # Данные экспериментов, если проводились
                experiment_data = None
                optimal_params_data = None
                if 'experiment_results' in st.session_state:
                    experiment_data = st.session_state.experiment_results
                    if 'optimal_params' in locals():
                        optimal_params_data = optimal_params
                
                # Подготавливаем данные для отчета
                report_data = prepare_anomaly_report_data(
                    data=df['data'].values,
                    time_index=df['time'].values,
                    anomaly_info=anomaly_information,
                    detection_results=anomalies,
                    detection_params=detection_params,
                    metrics_results=metrics_data,
                    experiment_results=experiment_data,
                    optimal_params=optimal_params_data
                )
                
                # Генерируем YAML
                yaml_section = generate_anomaly_detection_yaml(report_data)
                
                # Генерируем CSS
                css = """<style>
                img { 
                    display: block; 
                    margin: 20px auto; 
                    max-width: 100%; 
                    height: auto; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    border-radius: 4px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                @media print {
                    img {
                        max-width: 100%;
                        page-break-inside: avoid;
                    }
                    h2 { 
                        page-break-before: always; 
                    }
                    h2:first-of-type { 
                        page-break-before: avoid; 
                    }
                }
                </style>"""
                
                # Начало формирования отчета
                md_content = f"{yaml_section}{css}\n"
                md_content += "# Отчет по обнаружению аномалий во временных рядах\n\n"
                
                # Описательная часть
                md_content += "## Общая информация\n\n"
                md_content += f"* **Источник данных**: {data_source}\n"
                md_content += f"* **Длина ряда**: {len(df)}\n"
                md_content += f"* **Наличие пропусков**: {'Да' if report_data['has_nan'] else 'Нет'}\n\n"
                
                md_content += "## Статистики данных\n\n"
                md_content += "| Показатель | Значение |\n"
                md_content += "|------------|--------|\n"
                for key, value in report_data['data_stats'].items():
                    md_content += f"| {key} | {value:.4f} |\n"
                
                # Информация о параметрах обнаружения
                md_content += "## Параметры обнаружения аномалий\n\n"
                
                # Z-Score
                md_content += "### Z-Score\n\n"
                z_params = report_data['detection_params']['z_score']
                md_content += f"* **Включен**: {'Да' if z_params['enabled'] else 'Нет'}\n"
                if z_params['enabled']:
                    md_content += f"* **Порог**: {z_params['threshold']}\n\n"
                
                # Hampel
                md_content += "### Фильтр Хампеля\n\n"
                h_params = report_data['detection_params']['hampel']
                md_content += f"* **Включен**: {'Да' if h_params['enabled'] else 'Нет'}\n"
                if h_params['enabled']:
                    md_content += f"* **Адаптивное окно**: {'Да' if h_params['adaptive_window'] else 'Нет'}\n"
                    if h_params['adaptive_window']:
                        md_content += f"* **Процент от длины ряда**: {h_params['window_percent']}%\n"
                    else:
                        md_content += f"* **Размер окна**: {h_params['window']} точек\n"
                    md_content += f"* **Коэффициент чувствительности**: {h_params['sigma']}\n\n"
                
                # IQR
                md_content += "### Межквартильный размах (IQR)\n\n"
                iqr_params = report_data['detection_params']['iqr']
                md_content += f"* **Включен**: {'Да' if iqr_params['enabled'] else 'Нет'}\n"
                if iqr_params['enabled']:
                    md_content += f"* **Множитель**: {iqr_params['multiplier']}\n\n"
                
                # Plateau
                md_content += "### Обнаружение плато\n\n"
                p_params = report_data['detection_params']['plateau']
                md_content += f"* **Включен**: {'Да' if p_params['enabled'] else 'Нет'}\n"
                if p_params['enabled']:
                    md_content += f"* **Порог производной**: {p_params['threshold']}\n"
                    md_content += f"* **Минимальная длительность**: {p_params['min_duration']} точек\n"
                    md_content += f"* **Обнаружение NaN**: {'Да' if p_params['detect_nan'] else 'Нет'}\n\n"
                
                # Информация о внедренных аномалиях
                if anomaly_information:
                    md_content += format_anomaly_info_for_report(anomaly_information)
                
                # Результаты обнаружения
                md_content += "## Результаты обнаружения аномалий\n\n"
                
                md_content += "| Метод | Количество аномалий | Процент от ряда |\n"
                md_content += "|-------|-------------------|---------------|\n"
                
                for method, results in report_data['detection_results'].items():
                    method_name = method_names.get(method, method)
                    md_content += f"| {method_name} | {results['count']} | {results['percentage']:.2f}% |\n"
                
                md_content += "\n"
                
                # График аномалий
                # Сохраняем текущий график в base64 только если есть обнаруженные аномалии
                has_detected_anomalies = any(np.any(result) for name, result in anomalies.items() if name != 'iqr_bounds' and isinstance(result, np.ndarray))

                if 'fig' in locals() and has_detected_anomalies:
                    try:
                        # Пробуем сначала сохранить напрямую
                        anomaly_img_base64 = reporting.save_plot_to_base64(fig, backend='plotly')
                        md_content += "## График обнаруженных аномалий\n\n"
                        md_content += f"<img src=\"data:image/png;base64,{anomaly_img_base64}\" alt=\"Обнаруженные аномалии\">\n\n"
                    except Exception as e:
                        st.warning(f"Прямой экспорт графика не удался: {str(e)}. Пробуем альтернативный метод.")
                        try:
                            # Создаем эквивалентный график в matplotlib
                            mpl_fig = create_matplotlib_figure_from_plotly(fig)
                            anomaly_img_base64 = reporting.save_plot_to_base64(mpl_fig, backend='matplotlib')
                            md_content += "## График обнаруженных аномалий\n\n"
                            md_content += f"<img src=\"data:image/png;base64,{anomaly_img_base64}\" alt=\"Обнаруженные аномалии\">\n\n"
                        except Exception as e2:
                            st.error(f"Оба метода экспорта графика не удались: {str(e2)}")
                            md_content += "## График обнаруженных аномалий\n\n"
                            md_content += "График недоступен из-за технических проблем с экспортом.\n\n"
                else:
                    md_content += "## График обнаруженных аномалий\n\n"
                    md_content += "Не было обнаружено значимых аномалий для отображения на графике.\n\n"

                # Если выбран подробный отчет с метриками
                if report_type == "Подробный отчет с метриками" and metrics_data:
                    md_content += "## Метрики качества обнаружения аномалий\n\n"
                    
                    md_content += "| Метод | Precision | Recall | F1-Score |\n"
                    md_content += "|-------|-----------|--------|----------|\n"
                    
                    for method, metrics in metrics_data.items():
                        method_name = method_names.get(method, method)
                        precision = metrics.get('precision', 0)
                        recall = metrics.get('recall', 0)
                        f1 = metrics.get('f1', 0)
                        md_content += f"| {method_name} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n"
                    
                    md_content += "\n"
                    
                    # Добавляем график сравнения метрик, если есть
                    if 'fig_metrics' in locals() and len(metrics_data) > 0:
                        try:
                            metrics_img_base64 = reporting.save_plot_to_base64(fig_metrics, backend='plotly')
                            md_content += "## График сравнения метрик\n\n"
                            md_content += f"<img src=\"data:image/png;base64,{metrics_img_base64}\" alt=\"Сравнение метрик\">\n\n"
                        except Exception as e:
                            st.warning(f"Не удалось добавить график метрик в отчет: {str(e)}")
                            md_content += "## График сравнения метрик\n\n"
                            md_content += "График недоступен. Установите пакет kaleido для экспорта графиков в PDF.\n\n"

                # Если выбран отчет по экспериментам
                if report_type == "Отчет по экспериментам" and experiment_data is not None:
                    md_content += "## Результаты численного эксперимента\n\n"
                    
                    # Информация о методе и параметрах эксперимента
                    method_display_name = {
                        "z_score": "Z-Score",
                        "iqr": "IQR (Межквартильный размах)",
                        "hampel": "Фильтр Хампеля",
                        "plateau": "Детекция плато"
                    }.get(st.session_state.experiment_method, st.session_state.experiment_method)
                    
                    md_content += f"### Метод: {method_display_name}\n\n"
                    md_content += f"#### Всего экспериментов: {len(experiment_data)}\n\n"
                    
                    # Добавляем информацию о диапазонах параметров
                    md_content += "### Диапазоны исследуемых параметров\n\n"
                    md_content += "| Параметр | Минимум | Максимум | Шаг | Количество значений |\n"
                    md_content += "|----------|---------|---------|-----|-------------------|\n"
                    
                    param_ranges = {}
                    for param in st.session_state.experiment_params:
                        values = experiment_data[param].unique()
                        values.sort()
                        if len(values) > 1:
                            min_val = values[0]
                            max_val = values[-1]
                            
                            # Пытаемся определить шаг для числовых параметров
                            if len(values) > 2 and all(isinstance(x, (int, float)) for x in values):
                                step = (values[1] - values[0])
                                is_uniform = all(abs((values[i] - values[i-1]) - step) < 1e-6 for i in range(1, len(values)))
                                if is_uniform:
                                    step_display = f"{step}"
                                else:
                                    step_display = "неравномерный"
                            else:
                                step_display = "-"
                                
                            md_content += f"| {param} | {min_val} | {max_val} | {step_display} | {len(values)} |\n"
                            param_ranges[param] = values
                    
                    md_content += "\n"
                    
                    # Добавляем информацию о фиксированных параметрах, если они были
                    fixed_params = {}
                    for column in experiment_data.columns:
                        if column not in st.session_state.experiment_params and column not in ['precision', 'recall', 'f1', 'num_anomalies']:
                            values = experiment_data[column].unique()
                            if len(values) == 1:
                                fixed_params[column] = values[0]
                    
                    if fixed_params:
                        md_content += "### Фиксированные параметры\n\n"
                        md_content += "| Параметр | Значение |\n"
                        md_content += "|----------|----------|\n"
                        for param, value in fixed_params.items():
                            md_content += f"| {param} | {value} |\n"
                        md_content += "\n"
                    
                    # Лучшие результаты для каждой метрики
                    md_content += "### Лучшие результаты по метрикам\n\n"
                    
                    # Для F1-Score
                    best_f1_idx = experiment_data['f1'].idxmax()
                    best_f1 = experiment_data.loc[best_f1_idx]
                    
                    md_content += "#### Лучший результат по F1-Score\n\n"
                    md_content += f"**F1-Score: {best_f1['f1']:.4f}**\n\n"
                    md_content += "| Параметр | Значение |\n"
                    md_content += "|----------|----------|\n"
                    
                    for param in st.session_state.experiment_params:
                        md_content += f"| {param} | {best_f1[param]} |\n"
                    
                    md_content += f"| Precision | {best_f1['precision']:.4f} |\n"
                    md_content += f"| Recall | {best_f1['recall']:.4f} |\n"
                    num_anomalies_f1 = best_f1['num_anomalies']
                    if isinstance(num_anomalies_f1, (pd.Series, np.ndarray)):
                        num_anomalies_f1 = num_anomalies_f1.item() if hasattr(num_anomalies_f1, 'item') else int(num_anomalies_f1[0])
                    md_content += f"| Количество аномалий | {int(num_anomalies_f1)} |\n\n"
                    
                    # Для Precision
                    best_precision_idx = experiment_data['precision'].idxmax()
                    best_precision = experiment_data.loc[best_precision_idx]
                    
                    md_content += "#### Лучший результат по Precision\n\n"
                    md_content += f"**Precision: {best_precision['precision']:.4f}**\n\n"
                    md_content += "| Параметр | Значение |\n"
                    md_content += "|----------|----------|\n"
                    
                    for param in st.session_state.experiment_params:
                        md_content += f"| {param} | {best_precision[param]} |\n"
                    
                    md_content += f"| F1-Score | {best_precision['f1']:.4f} |\n"
                    md_content += f"| Recall | {best_precision['recall']:.4f} |\n"
                    num_anomalies_precision = best_precision['num_anomalies']
                    if isinstance(num_anomalies_precision, (pd.Series, np.ndarray)):
                        num_anomalies_precision = num_anomalies_precision.item() if hasattr(num_anomalies_precision, 'item') else int(num_anomalies_precision[0])
                    md_content += f"| Количество аномалий | {int(num_anomalies_precision)} |\n\n"
                    
                    # Для Recall
                    best_recall_idx = experiment_data['recall'].idxmax()
                    best_recall = experiment_data.loc[best_recall_idx]
                    
                    md_content += "#### Лучший результат по Recall\n\n"
                    md_content += f"**Recall: {best_recall['recall']:.4f}**\n\n"
                    md_content += "| Параметр | Значение |\n"
                    md_content += "|----------|----------|\n"
                    
                    for param in st.session_state.experiment_params:
                        md_content += f"| {param} | {best_recall[param]} |\n"
                    
                    md_content += f"| F1-Score | {best_recall['f1']:.4f} |\n"
                    md_content += f"| Precision | {best_recall['precision']:.4f} |\n"
                    num_anomalies_recall = best_recall['num_anomalies']
                    if isinstance(num_anomalies_recall, (pd.Series, np.ndarray)):
                        num_anomalies_recall = num_anomalies_recall.item() if hasattr(num_anomalies_recall, 'item') else int(num_anomalies_recall[0])
                    md_content += f"| Количество аномалий | {int(num_anomalies_recall)} |\n\n"

                    # Создание и добавление графиков для всех метрик
                    metrics_to_visualize = ['f1', 'precision', 'recall', 'num_anomalies']
                    metric_titles = {
                        'f1': 'F1-Score',
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'num_anomalies': 'Количество аномалий'
                    }
                    
                    md_content += "### Визуализация результатов\n\n"
                    
                    # Если у нас один параметр - создаем линейные графики для каждой метрики
                    if len(st.session_state.experiment_params) == 1:
                        param = st.session_state.experiment_params[0]
                        
                        for metric in metrics_to_visualize:
                            try:
                                # Создаем matplotlib график для линейной зависимости
                                plt.figure(figsize=(10, 6))
                                
                                # Сортируем данные по параметру
                                sorted_data = experiment_data.sort_values(param)
                                
                                # Строим график
                                plt.plot(sorted_data[param], sorted_data[metric], 'o-', linewidth=2)
                                
                                # Добавляем подписи и сетку
                                plt.title(f'Зависимость {metric_titles[metric]} от {param}')
                                plt.xlabel(param)
                                plt.ylabel(metric_titles[metric])
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                                
                                # Сохраняем в base64
                                img_base64 = reporting.save_plot_to_base64(plt.gcf(), backend='matplotlib')
                                plt.close()
                                
                                # Добавляем в отчет
                                md_content += f"#### Зависимость {metric_titles[metric]} от {param}\n\n"
                                md_content += f"<img src=\"data:image/png;base64,{img_base64}\" alt=\"{metric_titles[metric]} vs {param}\">\n\n"
                            except Exception as e:
                                md_content += f"#### Зависимость {metric_titles[metric]} от {param}\n\n"
                                md_content += f"Не удалось создать график: {str(e)}\n\n"
                    
                    # Если у нас два параметра - создаем тепловые карты для каждой метрики
                    elif len(st.session_state.experiment_params) == 2:
                        param1, param2 = st.session_state.experiment_params
                        
                        for metric in metrics_to_visualize:
                            try:
                                # Создаем сводную таблицу для тепловой карты
                                pivot_df = experiment_data.pivot(index=param1, columns=param2, values=metric)
                                
                                # Создаем matplotlib тепловую карту
                                plt.figure(figsize=(12, 8))
                                im = plt.imshow(pivot_df.values, cmap='viridis', aspect='auto')
                                plt.colorbar(im, label=f"{metric_titles[metric]} (среднее)")
                                
                                # Настраиваем оси
                                plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
                                plt.yticks(range(len(pivot_df.index)), pivot_df.index)
                                
                                plt.xlabel(param2)
                                plt.ylabel(param1)
                                plt.title(f'Тепловая карта {metric_titles[metric]} (усреднено)')
                                plt.tight_layout()
                                
                                # Сохраняем в base64
                                img_base64 = reporting.save_plot_to_base64(plt.gcf(), backend='matplotlib')
                                plt.close()
                                
                                # Добавляем в отчет
                                md_content += f"#### Тепловая карта {metric_titles[metric]} (по {param1} и {param2})\n\n"
                                md_content += f"<img src=\"data:image/png;base64,{img_base64}\" alt=\"Heatmap {metric}\">\n\n"
                            except Exception as e:
                                md_content += f"#### Тепловая карта {metric_titles[metric]}\n\n"
                                md_content += f"Не удалось создать тепловую карту: {str(e)}\n\n"
                    
                    # Для многомерных экспериментов (3+ параметра)
                    else:
                        # Выбираем два параметра для визуализации (первые два)
                        viz_params = st.session_state.experiment_params[:2]
                        param1, param2 = viz_params
                        
                        # Показываем предупреждение
                        md_content += "⚠️ *Эксперимент содержит больше двух параметров. Показаны две первые переменные, остальные параметры усреднены.*\n\n"
                        
                        for metric in metrics_to_visualize:
                            try:
                                # Группируем данные по двум параметрам и усредняем метрику
                                grouped_data = experiment_data.groupby([param1, param2])[metric].mean().reset_index()
                                pivot_df = grouped_data.pivot(index=param1, columns=param2, values=metric)
                                
                                # Создаем matplotlib тепловую карту
                                plt.figure(figsize=(12, 8))
                                im = plt.imshow(pivot_df.values, cmap='viridis', aspect='auto')
                                plt.colorbar(im, label=f"{metric_titles[metric]} (среднее)")
                                
                                # Настраиваем оси
                                plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
                                plt.yticks(range(len(pivot_df.index)), pivot_df.index)
                                
                                plt.xlabel(param2)
                                plt.ylabel(param1)
                                plt.title(f'Тепловая карта {metric_titles[metric]} (усреднено)')
                                plt.tight_layout()
                                
                                # Сохраняем в base64
                                img_base64 = reporting.save_plot_to_base64(plt.gcf(), backend='matplotlib')
                                plt.close()
                                
                                # Добавляем в отчет
                                md_content += f"#### Тепловая карта {metric_titles[metric]} (по {param1} и {param2})\n\n"
                                md_content += f"<img src=\"data:image/png;base64,{img_base64}\" alt=\"Heatmap {metric}\">\n\n"
                            except Exception as e:
                                md_content += f"#### Тепловая карта {metric_titles[metric]}\n\n"
                                md_content += f"Не удалось создать тепловую карту: {str(e)}\n\n"
                    
                    # Добавляем полную таблицу результатов
                    md_content += "### Полная таблица результатов (10 лучших по F1-Score)\n\n"
                    
                    # Сортируем по F1-Score и берем 10 лучших результатов
                    top_results = experiment_data.sort_values('f1', ascending=False).head(10)
                    
                    # Создаем заголовок таблицы
                    md_content += "| № |"
                    for param in st.session_state.experiment_params:
                        md_content += f" {param} |"
                    md_content += " F1-Score | Precision | Recall | Кол-во аномалий |\n"
                    
                    # Добавляем разделительную строку
                    md_content += "|---|"
                    for _ in st.session_state.experiment_params:
                        md_content += "---|"
                    md_content += "---|---|---|---|\n"
                    
                    # Добавляем данные
                    for i, (_, row) in enumerate(top_results.iterrows(), 1):
                        md_content += f"| {i} |"
                        for param in st.session_state.experiment_params:
                            md_content += f" {row[param]} |"
                        md_content += f" {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {int(row['num_anomalies'])} |\n"
                                    
                    # Генерируем отчет
                    file_name = f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
                    reporting.download_button_for_text(md_content, file_name, "⬇️ Скачать отчет (Markdown)")
                    
                    # Пытаемся конвертировать в PDF если доступен pdfkit
                    try:
                        pdf_file_name = file_name.replace('.md', '.pdf')
                        pdf_content = reporting.convert_markdown_to_pdf(md_content)
                        reporting.download_button_for_binary(pdf_content, pdf_file_name, "⬇️ Скачать отчет (PDF)")
                    except Exception as pdf_error:
                        st.warning(f"Не удалось создать PDF: {str(pdf_error)}. Скачайте версию в формате Markdown.")
        except Exception as e:
            st.error(f"Ошибка при создании отчета: {str(e)}")
            st.error("Подробности ошибки: " + "\n".join(str(e).split("\n")[:5]))
    else:
        st.info("Нет результатов экспериментов. Запустите эксперимент на вкладке 'Настройка эксперимента'.")
