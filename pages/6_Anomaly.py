import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Анализ аномалий во временных рядах", layout="wide")

from modules.anomaly_module import (
    generate_anomalous_series,
    z_score_detection,
    iqr_detection,
    hampel_filter,
    detect_plateau
)

# Инициализация состояния сессии
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = {
        'point': [],
        'extended': [],
        'sensor': []
    }

# ====================
# КОНФИГУРАЦИЯ ИНТЕРФЕЙСА
# ====================
st.title("Генератор временных рядов с аномалиями")

with st.sidebar:
    st.header("Параметры генерации")
    
    # Основные параметры
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
                        if idx < 0 or idx >= n:
                            raise ValueError(f"Индекс {idx} вне диапазона [0, {n-1}]")
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
        ext_start = st.number_input("Начальный индекс", 0, n-1, 80, key="ext_start")
        ext_duration = st.number_input("Длительность", 1, 100, 25, key="ext_dur")
        ext_shift = st.number_input("Смещение уровня", -5.0, 5.0, -2.5, key="ext_shift")
        
        if st.button("Добавить протяженную аномалию"):
            new_anom = {
                'start_idx': ext_start,
                'duration': ext_duration,
                'level_shift': ext_shift
            }
            st.session_state.anomalies['extended'].append(new_anom)
    
    # Сбои датчиков
    with st.expander("⚠️ Сбои датчиков"):
        fault_start = st.number_input("Начало сбоя", 0, n-1, 220, key="fault_start")
        fault_duration = st.number_input("Длительность сбоя", 1, 100, 35, key="fault_dur")
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
    
    # Кнопка сброса
    if st.button("Очистить все аномалии"):
        st.session_state.anomalies = {'point': [], 'extended': [], 'sensor': []}

# ====================
# ГЕНЕРАЦИЯ ДАННЫХ
# ====================
@st.cache_data
def generate_data(params):
    return generate_anomalous_series(**params)

generation_params = {
    'n': n,
    'season_amp': season_amp,
    'noise_std': noise_std,
    'point_anomalies': st.session_state.anomalies['point'],
    'extended_anomalies': st.session_state.anomalies['extended'],
    'sensor_faults': st.session_state.anomalies['sensor']
}

try:
    # Получаем данные, время и метаданные
    data, time, metadata = generate_data(generation_params)
    df = pd.DataFrame({'data': data, 'time': time})
except Exception as e:
    st.error(f"Ошибка генерации данных: {str(e)}")
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
        hampel_window = st.slider("Размер окна", 5, 50, 20, step=5,
                                 help=(
                                     "Количество точек для расчета локальной медианы:\n"
                                     "• Меньшие значения: выше чувствительность к локальным изменениям\n"
                                     "• Большие значения: устойчивее к шумам\n"
                                     "Рекомендуется: 10-30 для рядов длиной 300-1000 точек"
                                 ))
        hampel_sigma = st.slider("Коэффициент чувствительности", 1.0, 5.0, 3.0, step=0.5,
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
        results['z_outliers'] = z_outliers
    
    # Hampel
    if use_hampel:
        results['hampel_outliers'] = hampel_filter(data, 
                                                  window=hampel_window, 
                                                  sigma=hampel_sigma)
    
    # IQR
    if use_iqr:
        iqr_outliers, iqr_bounds = iqr_detection(data, iqr_multiplier)
        results['iqr_outliers'] = iqr_outliers
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
fig.add_trace(go.Scatter(
    x=df['time'], 
    y=df['data'],
    mode='lines',
    name='Исходный ряд',
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
# ЭКСПОРТ ДАННЫХ
# ====================
st.download_button(
    label="Скачать данные",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='time_series.csv',
    mime='text/csv'
)