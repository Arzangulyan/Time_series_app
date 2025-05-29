import streamlit as st
import pandas as pd
import numpy as np
import logging
from io import StringIO
import datetime
import plotly.express as px

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Кэшированные функции ---

@st.cache_data
def generate_synthetic_data(size, trend_coeff, noise_std, cycles):
    """Генерирует синтетический временной ряд."""
    logging.info("Генерация синтетических данных...")
    time_index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=size, freq='D'))
    time_numeric = np.arange(size)
    trend = trend_coeff * time_numeric
    
    # --- ИЗМЕНЕНИЕ: Добавляем несколько циклов --- 
    total_seasonality = np.zeros(size)
    if cycles: # Проверяем, что список циклов не пустой
        for period_days, amplitude in cycles:
            if period_days > 0 and amplitude != 0: # Игнорируем нулевые периоды/амплитуды
               # Частота цикла (в циклах за 1 день)
               frequency = 1.0 / period_days 
               # Добавляем компонент синусоиды
               total_seasonality += amplitude * np.sin(2 * np.pi * frequency * time_numeric)
    # -----------------------------------------------
    
    noise = np.random.normal(0, noise_std, size)
    # --- ИЗМЕНЕНИЕ: Используем total_seasonality --- 
    values = trend + total_seasonality + noise + 100 
    df = pd.DataFrame({'Timestamp': time_index, 'Value': values})
    logging.info(f"Сгенерированы синтетические данные размером: {df.shape}")
    return df

@st.cache_data
def load_csv_data(uploaded_file):
    """Загружает данные из CSV файла."""
    logging.info("Загрузка данных из CSV...")
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = None
        # Попытка угадать разделитель
        try:
            stringio.seek(0) # Убедимся, что читаем с начала
            df = pd.read_csv(stringio, sep=None, engine='python') # sep=None позволяет pandas угадать
            logging.info(f"CSV прочитан успешно (разделитель угадан).")
        except Exception as e_guess:
             logging.warning(f"Не удалось угадать разделитель CSV: {e_guess}. Попытка с ';'...")
             stringio.seek(0) # Перемотка StringIO в начало для повторного чтения
             df = pd.read_csv(stringio, sep=';')
             logging.info(f"CSV прочитан успешно (разделитель ';').")

        if df is not None:
            logging.info(f"Исходные загруженные колонки: {df.columns.tolist()}")
            logging.info(f"Типы данных исходных колонок:\\n{df.dtypes}")
            logging.info(f"Размер до удаления пустых колонок: {df.shape}")
            # Попытка удалить пустые колонки, если есть
            df_cleaned = df.dropna(axis=1, how='all')
            if df_cleaned.shape[1] < df.shape[1]:
                 logging.info(f"Удалены пустые колонки. Оставшиеся: {df_cleaned.columns.tolist()}")
                 logging.info(f"Типы данных после удаления пустых колонок:\\n{df_cleaned.dtypes}")
            else:
                 logging.info("Пустых колонок для удаления не найдено.")
            logging.info(f"Размер после удаления пустых колонок: {df_cleaned.shape}")
            return df_cleaned
        else:
            logging.error("Не удалось прочитать DataFrame из CSV.")
            return None

    except Exception as e:
        st.error(f"Критическая ошибка при чтении CSV файла: {e}")
        logging.exception("Критическая ошибка чтения CSV:") # Используем exception для полного трейсбека
        return None

@st.cache_data
def generate_predefined_synthetic_data(dataset_name):
    """Генерирует заранее определенные синтетические временные ряды."""
    logging.info(f"Генерация готового синтетического ряда: {dataset_name}")
    
    # Базовые параметры для всех рядов
    base_date = '2023-01-01'
    base_value = 100
    
    if dataset_name == "Линейный тренд с сезонностью":
        size = 365
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        trend = 0.2 * time_numeric
        seasonality = 15 * np.sin(2 * np.pi * time_numeric / 365) + 5 * np.sin(2 * np.pi * time_numeric / 30)
        noise = np.random.normal(0, 3, size)
        values = base_value + trend + seasonality + noise
        
    elif dataset_name == "Экспоненциальный рост":
        size = 200
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        exponential = base_value * np.exp(0.01 * time_numeric)
        seasonality = 10 * np.sin(2 * np.pi * time_numeric / 50)
        noise = np.random.normal(0, exponential * 0.05)
        values = exponential + seasonality + noise
        
    elif dataset_name == "Ряд с аномалиями":
        size = 300
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        trend = 0.1 * time_numeric
        seasonality = 20 * np.sin(2 * np.pi * time_numeric / 60)
        noise = np.random.normal(0, 5, size)
        values = base_value + trend + seasonality + noise
        
        # Добавляем аномалии
        anomaly_indices = np.random.choice(size, size//20, replace=False)
        values[anomaly_indices] += np.random.normal(0, 50, len(anomaly_indices))
        
    elif dataset_name == "Ступенчатая функция":
        size = 250
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        # Создаем ступенчатую функцию
        steps = np.where(time_numeric < 100, base_value, 
                np.where(time_numeric < 150, base_value + 30,
                np.where(time_numeric < 200, base_value + 50, base_value + 20)))
        seasonality = 8 * np.sin(2 * np.pi * time_numeric / 40)
        noise = np.random.normal(0, 4, size)
        values = steps + seasonality + noise
        
    elif dataset_name == "Недельная сезонность":
        size = 168  # 24 недели по 7 дней
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        trend = 0.05 * time_numeric
        weekly_pattern = 25 * np.sin(2 * np.pi * time_numeric / 7)  # Недельная сезонность
        monthly_pattern = 10 * np.sin(2 * np.pi * time_numeric / 30)  # Месячная сезонность
        noise = np.random.normal(0, 6, size)
        values = base_value + trend + weekly_pattern + monthly_pattern + noise
        
    elif dataset_name == "Случайное блуждание":
        size = 400
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        
        # Случайное блуждание
        changes = np.random.normal(0, 5, size)
        values = base_value + np.cumsum(changes)
        
    else:  # Дефолтный случай
        size = 365
        time_index = pd.to_datetime(pd.date_range(start=base_date, periods=size, freq='D'))
        time_numeric = np.arange(size)
        
        trend = 0.1 * time_numeric
        seasonality = 10 * np.sin(2 * np.pi * time_numeric / 365)
        noise = np.random.normal(0, 5, size)
        values = base_value + trend + seasonality + noise
    
    df = pd.DataFrame({'Timestamp': time_index, 'Value': values})
    logging.info(f"Сгенерирован готовый синтетический ряд '{dataset_name}' размером: {df.shape}")
    return df

# --- Инициализация Session State ---
# Хранит данные МЕЖДУ перезапусками скрипта
if 'data' not in st.session_state:
    st.session_state.data = None # Исходные загруженные/сгенерированные данные
if 'time_series' not in st.session_state:
    st.session_state.time_series = None # Данные после всех шагов обработки
if 'time_col' not in st.session_state:
    st.session_state.time_col = None # Имя выбранной колонки времени
if 'value_col' not in st.session_state:
    st.session_state.value_col = None # Имя выбранной колонки значений
if 'main_column' not in st.session_state:
    st.session_state.main_column = None # Имя колонки для других страниц
if 'resample_rule' not in st.session_state:
    st.session_state.resample_rule = None # Правило ресемплинга
if 'imputation_method' not in st.session_state:
    st.session_state.imputation_method = None # Метод заполнения пропусков


# --- Глобальные переменные и константы ---
# Доступные правила ресемплинга
resampling_options = {
    "Без ресемплинга": None, "По дням ('D')": 'D', "По неделям ('W')": 'W',
    "По месяцам ('ME')": 'ME', "По кварталам ('QE')": 'QE', "По годам ('YE')": 'YE',
    "По часам ('h')": 'h', "По минутам ('min')": 'min', "По секундам ('s')": 's'
}
# Агрегирующие функции
agg_funcs = {
    "Среднее": "mean", "Сумма": "sum", "Медиана": "median", "Максимум": "max",
    "Минимум": "min", "Первое значение": "first", "Последнее значение": "last"
}
# Методы обработки пропусков
imputation_options = {
    "Не обрабатывать": None, "Заполнить предыдущим значением (Forward fill)": 'ffill',
    "Заполнить следующим значением (Backward fill)": 'bfill', "Заполнить линейной интерполяцией": 'linear',
    "Заполнить средним значением": 'mean', "Заполнить медианой": 'median',
    "Заполнить нулем": 0, "Заполнить заданным значением": 'custom'
}


# --- Интерфейс Streamlit ---

st.set_page_config(layout="wide") # Должно быть первым вызовом Streamlit
st.title("Анализ и обработка временных рядов")

# --- Кнопка для сброса состояния и кэша ---
if st.button("⚠️ Очистить состояние и кэш", key='clear_state_cache_button'):
    # Список ключей для сброса в session_state
    keys_to_reset = [
        'data', 'time_series', 'time_col', 'value_col', 'main_column',
        'uploaded_filename', 'last_gen_params',
        'filter_start_date', 'filter_start_time', 'filter_end_date', 'filter_end_time', # Старые ключи фильтра
        'filter_applied',
        'widget_start_date', 'widget_start_time', 'widget_end_date', 'widget_end_time', # Новые ключи виджетов фильтра
        'applied_start_datetime', 'applied_end_datetime', # Ключи примененного фильтра
        'data_after_filtering',
        'resample_rule', 'agg_func_label', 'resample_applied',
        'imputation_method', 'imputation_applied_label',
        'fill_grid_check'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Очистка кэша функций
    st.cache_data.clear()

    st.success("Состояние приложения и кэш функций очищены! Перезагрузка...")
    logging.info("Состояние и кэш очищены пользователем.")
    st.rerun()

# --- Шаг 1: Загрузка или генерация данных ---
st.header("1. Загрузка или генерация данных")

data_source = st.radio(
    "Выберите источник данных:",
    ('Загрузить CSV', 'Сгенерировать синтетические данные', 'Выбрать готовый синтетический ряд'),
    key='data_source_radio',
    # При смене источника сбрасываем данные
    on_change=lambda: st.session_state.update(time_series=None, data=None, time_col=None, value_col=None, main_column=None)
)

uploaded_file = None
# df = None # Не используем глобальный df, будем работать с current_df

if data_source == 'Загрузить CSV':
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key='file_uploader')
    if uploaded_file is not None:
        # Загружаем только если данные еще не загружены или файл изменился
        if st.session_state.data is None or st.session_state.get('uploaded_filename') != uploaded_file.name:
            df_loaded = load_csv_data(uploaded_file)
            if df_loaded is not None:
                 st.session_state.data = df_loaded.copy()
                 st.session_state.time_series = df_loaded.copy() # Вместо processed_data
                 st.session_state.uploaded_filename = uploaded_file.name # Сохраняем имя файла
                 logging.info("Данные из CSV сохранены в session_state.")
                 st.rerun() # Перезапускаем, чтобы начать обработку с Шага 2

elif data_source == 'Сгенерировать синтетические данные':
    st.subheader("Параметры генерации:")
    # --- Основные параметры --- 
    col_main1, col_main2, col_main3 = st.columns(3)
    with col_main1: size = st.number_input("Размер ряда (дни)", 50, 5000, 365, 10, key='gen_size')
    with col_main2: trend_coeff = st.number_input("Коэфф. тренда", value=0.1, step=0.05, format="%.2f", key='gen_trend')
    with col_main3: noise_std = st.number_input("Ст. откл. шума", value=5.0, step=0.5, format="%.1f", key='gen_noise')

    # --- Параметры циклов (добавим 2 необязательных цикла) --- 
    st.markdown("**Циклические компоненты (необязательно):**")
    col_cycle1, col_cycle2 = st.columns(2)
    cycles_input = []
    
    with col_cycle1:
         st.markdown("Цикл 1")
         period1 = st.number_input("Период 1 (дни)", min_value=1.0, value=365.0, step=1.0, format="%.1f", key='gen_p1')
         ampl1 = st.number_input("Амплитуда 1", value=10.0, step=1.0, format="%.1f", key='gen_a1')
         if ampl1 != 0: cycles_input.append((period1, ampl1))
         
    with col_cycle2:
         st.markdown("Цикл 2")
         period2 = st.number_input("Период 2 (дни)", min_value=1.0, value=30.0, step=1.0, format="%.1f", key='gen_p2')
         ampl2 = st.number_input("Амплитуда 2", value=5.0, step=1.0, format="%.1f", key='gen_a2')
         if ampl2 != 0: cycles_input.append((period2, ampl2))
         
    # Добавить еще циклы можно аналогично...

    # --- Кнопка генерации и кэширование --- 
    # Собираем параметры для ключа кэша
    # Важно! Для списка кортежей cycles_input нужно преобразовать его в кортеж кортежей для кэширования
    cycles_tuple = tuple(sorted(cycles_input)) # Сортируем для независимости от порядка ввода
    gen_params = (size, trend_coeff, noise_std, cycles_tuple)
    
    # Проверяем кнопку или изменение параметров
    if st.button("Сгенерировать данные", key='generate_button') or st.session_state.get('last_gen_params') != gen_params:
        # Вызываем генератор с новым списком циклов
        df_generated = generate_synthetic_data(size, trend_coeff, noise_std, cycles_input)
        st.session_state.data = df_generated.copy()
        st.session_state.time_series = df_generated.copy() 
        st.session_state.last_gen_params = gen_params 
        logging.info("Сгенерированные данные сохранены в session_state.")
        st.session_state.uploaded_filename = None
        st.rerun()

elif data_source == 'Выбрать готовый синтетический ряд':
    st.subheader("Выберите готовый синтетический ряд:")
    
    predefined_datasets = {
        "Линейный тренд с сезонностью": "Годовая и месячная сезонность с восходящим трендом",
        "Экспоненциальный рост": "Экспоненциальный рост с недельной сезонностью",
        "Ряд с аномалиями": "Тренд с сезонностью и случайными выбросами",
        "Ступенчатая функция": "Скачкообразные изменения уровня",
        "Недельная сезонность": "Ярко выраженная недельная периодичность",
        "Случайное блуждание": "Случайное блуждание без тренда"
    }
    
    selected_dataset = st.selectbox(
        "Выберите тип ряда:",
        options=list(predefined_datasets.keys()),
        key='predefined_dataset_select',
        help="Каждый ряд имеет уникальные характеристики для анализа"
    )
    
    # Показываем описание выбранного ряда
    if selected_dataset in predefined_datasets:
        st.info(f"**{selected_dataset}**: {predefined_datasets[selected_dataset]}")
    
    # Генерируем данные при изменении выбора или нажатии кнопки
    current_predefined_choice = st.session_state.get('last_predefined_dataset')
    
    if st.button("Загрузить выбранный ряд", key='load_predefined_button') or current_predefined_choice != selected_dataset:
        df_predefined = generate_predefined_synthetic_data(selected_dataset)
        st.session_state.data = df_predefined.copy()
        st.session_state.time_series = df_predefined.copy()
        st.session_state.last_predefined_dataset = selected_dataset
        st.session_state.uploaded_filename = None
        logging.info(f"Загружен готовый синтетический ряд '{selected_dataset}' в session_state.")
        st.rerun()

# --- Проверка наличия данных для обработки ---
if 'time_series' not in st.session_state or st.session_state.time_series is None:
    st.info("Пожалуйста, загрузите CSV файл или сгенерируйте данные на Шаге 1.")
    st.stop()
else:
    # --- ДОБАВЛЕНИЕ: Визуализация исходных данных --- 
    st.subheader("Исходные данные (график)")
    try:
        # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка на None --- 
        if st.session_state.data is not None:
             plot_df = st.session_state.data.copy() # Используем исходные данные до обработки
             time_col_plot = None
             value_col_plot = None

             # Если данные сгенерированы, колонки известны
             if 'Timestamp' in plot_df.columns and 'Value' in plot_df.columns:
                  time_col_plot = 'Timestamp'
                  value_col_plot = 'Value'
             else:
                 # Пытаемся угадать для CSV
                 potential_time_cols = [col for col in plot_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                 if potential_time_cols:
                      try:
                           plot_df[potential_time_cols[0]] = pd.to_datetime(plot_df[potential_time_cols[0]], errors='coerce')
                           if pd.api.types.is_datetime64_any_dtype(plot_df[potential_time_cols[0]]):
                                time_col_plot = potential_time_cols[0]
                      except Exception:
                           pass 
                 
                 potential_value_cols = [col for col in plot_df.columns if col != time_col_plot and pd.api.types.is_numeric_dtype(plot_df[col])]
                 if potential_value_cols:
                      value_col_plot = potential_value_cols[0]
             
             if time_col_plot and value_col_plot:
                  fig = px.line(plot_df, x=time_col_plot, y=value_col_plot, 
                                title="Исходный временной ряд", 
                                labels={time_col_plot: "Время", value_col_plot: "Значение"})
                  fig.update_layout(hovermode="x unified")
                  st.plotly_chart(fig, use_container_width=True)
                  logging.info(f"Шаг 1: Отображен график исходных данных ({time_col_plot} vs {value_col_plot}).")
             elif value_col_plot: 
                  fig = px.line(plot_df, y=value_col_plot, 
                                title=f"Исходные данные (столбец '{value_col_plot}')",
                                labels={"index": "Индекс", value_col_plot: "Значение"})
                  fig.update_layout(hovermode="x unified")
                  st.plotly_chart(fig, use_container_width=True)
                  logging.info(f"Шаг 1: Отображен график исходных данных (только {value_col_plot}, ось X - индекс).")
             else:
                  st.warning("Не удалось автоматически определить колонки для построения графика исходных данных.")
                  logging.warning("Шаг 1: Не удалось построить график исходных данных - колонки не определены.")
        else:
            st.info("Данные еще не загружены/сгенерированы для отображения графика.")
            # --------------------------------------------

    except Exception as e:
        st.error(f"Ошибка при построении графика исходных данных: {e}")
        logging.exception("Ошибка визуализации исходных данных:")
    # ----------------------------------------------

    # Инициализируем current_df для текущего прогона скрипта из сохраненного состояния
    current_df = st.session_state.time_series.copy() 
    initial_shape = current_df.shape
    logging.info(f"--- Начало обработки --- Используются данные из session_state. Размер: {initial_shape}")

# --- Шаг 2: Первичный просмотр данных ---
st.header("2. Предварительный просмотр данных")
st.write(f"Текущие данные (размер: {current_df.shape}):")
st.dataframe(current_df.head())
logging.info(f"Шаг 2: Предварительный просмотр. Размер данных: {current_df.shape}")


# --- Шаг 3: Выбор столбцов (Время и Значение) ---
st.header("3. Выбор столбцов для анализа")

# Эти переменные будут определены в блоках if/else ниже
time_col = st.session_state.get('time_col')
value_col = st.session_state.get('value_col')

# Проверяем, является ли индекс уже DatetimeIndex
is_datetime_index = pd.api.types.is_datetime64_any_dtype(current_df.index)

if is_datetime_index:
    # --- Случай, когда индекс уже datetime ---
    st.info(f"Временной индекс '{current_df.index.name or 'index'}' уже установлен.")
    # Убедимся, что time_col в state соответствует имени индекса
    st.session_state.time_col = current_df.index.name

    value_col_options_step3 = [col for col in current_df.columns if pd.api.types.is_numeric_dtype(current_df[col])]
    if not value_col_options_step3:
        st.error("В данных не найдено числовых столбцов для анализа.")
        st.stop()

    # Выбираем столбец значений
    current_value_col_step3 = st.session_state.get('value_col', value_col_options_step3[0])
    if current_value_col_step3 not in value_col_options_step3: current_value_col_step3 = value_col_options_step3[0]

    value_col = st.selectbox(
        "Выберите столбец со значениями для анализа:", value_col_options_step3,
        index=value_col_options_step3.index(current_value_col_step3),
        key='value_col_select_indexed'
    )

    # Если выбранный столбец изменился, обновляем state и current_df
    if st.session_state.value_col != value_col:
        st.session_state.value_col = value_col
        # Обновляем current_df, оставляя только индекс и новый выбранный столбец
        # Читаем из session_state, так как current_df мог содержать старый столбец
        current_df = st.session_state.time_series[[value_col]].copy() # Вместо processed_data
        logging.info(f"Шаг 3: Выбран новый столбец значений '{value_col}'. Размер данных обновлен: {current_df.shape}")
        st.rerun() # Перезапускаем, чтобы остальные шаги использовали новый столбец

    logging.info(f"Шаг 3: Индекс уже datetime. Используется столбец значений '{value_col}'. Размер данных: {current_df.shape}")

else:
    # --- Случай, когда индекс НЕ datetime ---
    st.info("Временной индекс не установлен. Выберите столбцы.")
    available_columns_step3 = current_df.columns.tolist()
    if not available_columns_step3:
        st.error("В данных нет столбцов для выбора.")
        st.stop()

    # Выбор временного столбца
    potential_time_cols_step3 = [col for col in available_columns_step3 if 'time' in col.lower() or 'date' in col.lower()]
    default_time_col_step3 = potential_time_cols_step3[0] if potential_time_cols_step3 else available_columns_step3[0]
    current_time_col_step3 = st.session_state.get('time_col', default_time_col_step3)
    if current_time_col_step3 not in available_columns_step3: current_time_col_step3 = default_time_col_step3

    time_col = st.selectbox(
        "Выберите столбец с временем (для индекса):", available_columns_step3,
        index=available_columns_step3.index(current_time_col_step3),
        key='time_col_select'
    )

    # Выбор столбца со значениями
    remaining_cols = [col for col in available_columns_step3 if col != time_col]
    if not remaining_cols:
        st.error("Нужен хотя бы один столбец для значений, помимо временного.")
        st.stop()
    value_col_options_step3 = [col for col in remaining_cols if pd.api.types.is_numeric_dtype(current_df[col])]
    if not value_col_options_step3:
        st.error(f"Не найдено числовых столбцов для анализа (кроме '{time_col}').")
        st.stop()

    current_value_col_step3 = st.session_state.get('value_col', value_col_options_step3[0])
    if current_value_col_step3 not in value_col_options_step3: current_value_col_step3 = value_col_options_step3[0]

    value_col = st.selectbox(
        "Выберите столбец со значениями для анализа:", value_col_options_step3,
        index=value_col_options_step3.index(current_value_col_step3),
        key='value_col_select_non_indexed'
    )

    # Если выбор изменился, обновляем state и перезапускаем для применения
    if st.button("Применить выбор столбцов", key='apply_cols_button'):
        st.session_state.time_col = time_col
        st.session_state.value_col = value_col
        logging.info(f"Шаг 3: Нажата кнопка 'Применить'. Выбраны: время='{time_col}', значение='{value_col}'. Применяем изменения...")

        try:
            # Применяем выбор к исходным данным из state
            # Проверяем, что st.session_state.data не None перед копированием
            if st.session_state.data is None:
                st.error("Ошибка: Исходные данные (st.session_state.data) отсутствуют.")
                logging.error("Попытка обработки на Шаге 3, но st.session_state.data is None")
                st.stop()

            df_to_process = st.session_state.data.copy() # Начинаем с самых исходных данных
            logging.info(f"Шаг 3: Исходные колонки для обработки: {df_to_process.columns.tolist()}")

            # Проверка наличия колонок
            if time_col not in df_to_process.columns or value_col not in df_to_process.columns:
                 missing = [c for c in [time_col, value_col] if c not in df_to_process.columns]
                 st.error(f"Ошибка: Колонки {missing} не найдены в исходных данных!")
                 st.stop()

            df_processed = df_to_process[[time_col, value_col]].copy()

            # 1. Устанавливаем индекс
            df_processed = df_processed.set_index(time_col, drop=True)
            logging.info(f"Шаг 3: Индекс '{time_col}' установлен.")

            # 2. Преобразуем индекс в datetime
            original_index_name_step3 = df_processed.index.name
            df_processed.index = pd.to_datetime(df_processed.index, errors='coerce')
            df_processed.index.name = original_index_name_step3 # Восстанавливаем имя

            # 3. Проверяем и удаляем NaT
            nat_count = df_processed.index.isnull().sum()
            if nat_count > 0:
                st.warning(f"При преобразовании индекса из '{time_col}' в datetime возникло {nat_count} ошибок. Эти строки удалены.")
                df_processed = df_processed.dropna(axis=0, subset=[df_processed.index.name])
                logging.warning(f"Удалено {nat_count} строк из-за NaT в индексе.")

            # 4. Сортируем
            df_processed = df_processed.sort_index()

            # 5. Сохраняем результат в processed_data и перезапускаем
            st.session_state.time_series = df_processed.copy()
            st.session_state.main_column = value_col # Сохраняем имя выбранной колонки
            logging.info(f"Шаг 3: Успешно установлен индекс '{time_col}' и выбран столбец '{value_col}'. Новые данные сохранены в state. Размер: {df_processed.shape}")
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка на Шаге 3 при установке индекса/выборе значения: {e}")
            logging.exception("Детальная ошибка на Шаге 3:")
            # Сбрасываем состояние, чтобы избежать зацикливания
            # Проверяем, что st.session_state.data не None перед копированием
            if st.session_state.data is not None:
                 st.session_state.time_series = st.session_state.data.copy() # Вместо processed_data
            else:
                 st.session_state.time_series = None # Вместо processed_data
            st.session_state.main_column = None # Сбрасываем и main_column
            st.session_state.time_col = None
            st.session_state.value_col = None
            st.stop()

# --- Проверка после Шага 3: Убедимся, что value_col определен ---
if not value_col:
     st.warning("Столбец значений не выбран. Пожалуйста, выберите его на Шаге 3.")
     st.stop()

# --- Шаг 4: Фильтрация по временному интервалу ---
st.header("4. Фильтрация по временному интервалу")

df_before_filter_step4 = current_df.copy() # Копия данных до этого шага

if not pd.api.types.is_datetime64_any_dtype(df_before_filter_step4.index):
    st.info("Индекс не является datetime. Фильтрация по временному интервалу недоступна.")
    logging.info("Шаг 4: Пропущен, т.к. индекс не datetime.")
    # Сбрасываем состояние фильтра, если он был применен ранее к другому датасету
    st.session_state.filter_applied = False
    st.session_state.pop('applied_start_datetime', None)
    st.session_state.pop('applied_end_datetime', None)
else:
    min_date = df_before_filter_step4.index.min()
    max_date = df_before_filter_step4.index.max()

    st.write(f"Полный диапазон данных: от {min_date} до {max_date}")

    col_start, col_end = st.columns(2)

    # --- Валидация и получение значений по умолчанию для виджетов ---
    # Получаем сохраненные значения ИЛИ None
    widget_start_date_state = st.session_state.get('widget_start_date')
    widget_end_date_state = st.session_state.get('widget_end_date')

    # Проверяем и корректируем начальную дату
    if widget_start_date_state and min_date.date() <= widget_start_date_state <= max_date.date():
        actual_default_start_date = widget_start_date_state
    else:
        actual_default_start_date = min_date.date()
        if 'widget_start_date' in st.session_state: # Удаляем невалидное значение из state
             del st.session_state['widget_start_date']

    # Проверяем и корректируем конечную дату
    if widget_end_date_state and min_date.date() <= widget_end_date_state <= max_date.date():
         # Дополнительно проверим, что конечная дата не раньше начальной
        if widget_end_date_state >= actual_default_start_date:
             actual_default_end_date = widget_end_date_state
        else:
             actual_default_end_date = max_date.date() # Или можно actual_default_start_date, но max_date безопаснее
             if 'widget_end_date' in st.session_state:
                 del st.session_state['widget_end_date']
    else:
        actual_default_end_date = max_date.date()
        if 'widget_end_date' in st.session_state:
             del st.session_state['widget_end_date']

    # Для времени просто берем сохраненное или min/max время из данных
    # (st.time_input не имеет min/max_value, проблемы с диапазоном реже)
    actual_default_start_time = st.session_state.get('widget_start_time', min_date.time())
    actual_default_end_time = st.session_state.get('widget_end_time', max_date.time())

    # --- Отображение виджетов с валидными значениями по умолчанию ---
    with col_start:
        selected_start_date = st.date_input(
            "Начальная дата",
            value=actual_default_start_date, # Используем проверенное значение
            min_value=min_date.date(),
            max_value=max_date.date(),
            key='start_date_input'
        )
        selected_start_time = st.time_input(
            "Начальное время",
            value=actual_default_start_time, # Используем проверенное значение
            key='start_time_input'
        )

    with col_end:
        selected_end_date = st.date_input(
            "Конечная дата",
            value=actual_default_end_date, # Используем проверенное значение
            min_value=min_date.date(), # Важно оставить min_value от данных
            max_value=max_date.date(),
            key='end_date_input'
        )
        selected_end_time = st.time_input(
            "Конечное время",
            value=actual_default_end_time, # Используем проверенное значение
            key='end_time_input'
        )

    # Обновляем состояние виджетов в session_state немедленно
    st.session_state.widget_start_date = selected_start_date
    st.session_state.widget_start_time = selected_start_time
    st.session_state.widget_end_date = selected_end_date
    st.session_state.widget_end_time = selected_end_time

    # --- Кнопка для применения фильтра ---
    apply_filter_button = st.button("Применить фильтр", key='apply_filter_button')

    # --- Логика применения фильтра ---
    start_datetime, end_datetime = None, None
    valid_range_selected = False
    try:
        if isinstance(selected_start_date, datetime.date) and isinstance(selected_start_time, datetime.time) and \
           isinstance(selected_end_date, datetime.date) and isinstance(selected_end_time, datetime.time):

            start_datetime = pd.Timestamp.combine(selected_start_date, selected_start_time)
            end_datetime = pd.Timestamp.combine(selected_end_date, selected_end_time)

            if start_datetime >= end_datetime:
                st.warning("Начальная дата/время должна быть раньше конечной.")
                valid_range_selected = False
            else:
                 valid_range_selected = True
        else:
            # Это состояние не должно возникать с date/time input, но на всякий случай
            st.warning("Пожалуйста, выберите корректные начальные и конечные дату и время.")
            valid_range_selected = False

    except Exception as e:
        st.error(f"Ошибка при формировании выбранного диапазона дат: {e}")
        valid_range_selected = False


    if apply_filter_button:
        if valid_range_selected:
            # Сохраняем ВЫБРАННЫЙ и ВАЛИДНЫЙ диапазон как ПРИМЕНЕННЫЙ
            st.session_state.applied_start_datetime = start_datetime
            st.session_state.applied_end_datetime = end_datetime
            st.session_state.filter_applied = True
            logging.info(f"Шаг 4: Кнопка 'Применить фильтр' нажата. Применен диапазон: {start_datetime} - {end_datetime}.")
            st.rerun() # Перезапускаем, чтобы применить фильтр ниже и обновить интерфейс
        else:
            # Если кнопка нажата, но диапазон невалиден, сбрасываем фильтр
            st.session_state.filter_applied = False
            st.session_state.pop('applied_start_datetime', None)
            st.session_state.pop('applied_end_datetime', None)
            logging.warning("Шаг 4: Кнопка 'Применить фильтр' нажата, но выбранный диапазон невалиден. Фильтр сброшен.")
            st.rerun() # Перезапускаем, чтобы отобразить данные без фильтра

    # --- Применение сохраненного фильтра к current_df ---
    if st.session_state.get('filter_applied', False) and \
       'applied_start_datetime' in st.session_state and \
       'applied_end_datetime' in st.session_state:

        applied_start = st.session_state.applied_start_datetime
        applied_end = st.session_state.applied_end_datetime

        # Доп. проверка валидности сохраненного диапазона
        if isinstance(applied_start, pd.Timestamp) and isinstance(applied_end, pd.Timestamp) and applied_start < applied_end:
             try:
                 # Применяем сохраненный фильтр
                 current_df = df_before_filter_step4.loc[applied_start:applied_end].copy()
                 logging.info(f"Шаг 4: Применен сохраненный фильтр ({applied_start} - {applied_end}). Размер current_df: {current_df.shape}")
             except Exception as e:
                 st.error(f"Ошибка при применении сохраненного фильтра: {e}")
                 logging.exception("Ошибка loc при применении сохраненного фильтра:")
                 # Сбрасываем фильтр в случае ошибки применения
                 st.session_state.filter_applied = False
                 st.session_state.pop('applied_start_datetime', None)
                 st.session_state.pop('applied_end_datetime', None)
                 current_df = df_before_filter_step4 # Возвращаем неотфильтрованные данные для этого шага
        else:
             logging.warning("Шаг 4: Сохраненный диапазон фильтра невалиден. Фильтр не применяется.")
             st.session_state.filter_applied = False # Считаем фильтр не примененным
             current_df = df_before_filter_step4
    else:
         # Если фильтр не должен быть применен
         current_df = df_before_filter_step4
         logging.info("Шаг 4: Фильтр по времени не применен.")


    # --- Отображение результата ---
    st.markdown("---") # Разделитель
    if valid_range_selected:
        st.write(f"Выбранный диапазон (еще не применен): {start_datetime} по {end_datetime}")

    if st.session_state.get('filter_applied'):
        applied_start = st.session_state.get('applied_start_datetime')
        applied_end = st.session_state.get('applied_end_datetime')
        if applied_start and applied_end:
             st.success(f"Применен фильтр: от {applied_start} до {applied_end}")
             if not current_df.empty:
                 st.line_chart(current_df[value_col])
             else:
                 st.warning("Нет данных в примененном интервале.")
        else: # Такого не должно быть, если filter_applied=True, но на всякий случай
             st.warning("Фильтр помечен как примененный, но диапазон не найден в состоянии.")
             st.line_chart(current_df[value_col]) # Показываем текущий current_df (вероятно, нефильтрованный)

    else:
        st.info("Фильтр по времени не применен. Отображаются все данные после Шага 3.")
        st.line_chart(current_df[value_col]) # Показываем current_df без фильтра

# ---> ДОБАВЛЯЕМ СОХРАНЕНИЕ СОСТОЯНИЯ ПОСЛЕ ФИЛЬТРАЦИИ <---
st.session_state['data_after_filtering'] = current_df.copy()
logging.info(f"Шаг 4 Завершен: Состояние данных после фильтрации сохранено. Размер: {st.session_state['data_after_filtering'].shape}")

# --- Шаг 5: Ресемплинг ---
st.header("5. Ресемплинг данных")

if not pd.api.types.is_datetime64_any_dtype(current_df.index):
     st.info("Индекс не является datetime. Ресемплинг недоступен.")
     logging.info("Шаг 5: Пропущен, т.к. индекс не datetime.")
     st.session_state.resample_rule = None # Сбрасываем состояние
else:
    # Используем значение из state или дефолтное
    current_resample_label = next((k for k, v in resampling_options.items() if v == st.session_state.get('resample_rule')), "Без ресемплинга")

    selected_resample_label = st.selectbox(
        "Выберите частоту ресемплинга:", options=list(resampling_options.keys()),
        index=list(resampling_options.keys()).index(current_resample_label),
        key='resample_select'
    )
    resample_rule = None
    # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка ключа перед доступом через [] ---
    if selected_resample_label in resampling_options:
        resample_rule = resampling_options[selected_resample_label]
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    # Переменные для хранения выбора агрегации (если нужен ресемплинг)
    selected_agg_label = None
    agg_func = None

    if resample_rule:
        # Используем значение из state или дефолтное
        current_agg_label = st.session_state.get('agg_func_label', "Среднее")
        if current_agg_label not in agg_funcs: current_agg_label = "Среднее" # Защита

        selected_agg_label = st.selectbox(
            "Выберите функцию агрегации:", options=list(agg_funcs.keys()),
            index=list(agg_funcs.keys()).index(current_agg_label),
            key='agg_func_select'
        )
        agg_func = None
        # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка ключа перед доступом через [] ---
        if selected_agg_label in agg_funcs:
             agg_func = agg_funcs[selected_agg_label]
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Применяем ресемплинг, если выбор изменился или его еще не было
        if st.session_state.get('resample_rule') != resample_rule or st.session_state.get('agg_func_label') != selected_agg_label:
            st.session_state.resample_rule = resample_rule
            st.session_state.agg_func_label = selected_agg_label

            if resample_rule and agg_func:
                logging.info(f"Шаг 5: Применение ресемплинга (правило='{resample_rule}', агрегация='{agg_func}').")
                try:
                    # Ресемплируем current_df
                    resampled_series = current_df[value_col].resample(resample_rule).agg(agg_func)
                    current_df = resampled_series.to_frame() # Обновляем current_df
                    current_df.columns = [value_col] # Восстанавливаем имя столбца
                    logging.info(f"Шаг 5: Ресемплинг применен. Новый размер current_df: {current_df.shape}")
                    st.session_state.resample_applied = True
                except Exception as e:
                    st.error(f"Ошибка при ресемплинге: {e}")
                    logging.error(f"Ошибка ресемплинга: {e}")
                    # Возвращаем current_df к состоянию ДО ресемплинга
                    if 'data_after_filtering' in st.session_state:
                        current_df = st.session_state['data_after_filtering'].copy()
                        logging.info("Шаг 5 (Ошибка): Восстановлены данные из state 'data_after_filtering'.")
                    else:
                         # Аварийный вариант (не должно происходить)
                         logging.warning("Шаг 5 (Ошибка): Состояние 'data_after_filtering' не найдено! Попытка восстановить из time_series.")
                         current_df = st.session_state.time_series.copy() # Попытка отката к последнему сохраненному состоянию
                    st.session_state.resample_rule = None # Сбрасываем правило
                    st.session_state.agg_func_label = None
                    st.session_state.resample_applied = False

            else: # Если выбран "Без ресемплинга"
                 logging.info("Шаг 5: Ресемплинг не применяется (выбрано 'Без ресемплинга').")
                 # Возвращаем current_df к состоянию ДО ресемплинга
                 if 'data_after_filtering' in st.session_state:
                     current_df = st.session_state['data_after_filtering'].copy()
                     logging.info("Шаг 5 (Отмена): Восстановлены данные из state 'data_after_filtering'.")
                 else:
                     # Аварийный вариант
                     logging.warning("Шаг 5 (Отмена): Состояние 'data_after_filtering' не найдено! Попытка восстановить из time_series.")
                     current_df = st.session_state.time_series.copy()
                 st.session_state.resample_applied = False
                
    # Отображаем результат (если ресемплинг был применен)
    if st.session_state.get('resample_applied'):
        st.write(f"Данные ресемплированы с частотой '{st.session_state.resample_rule}' используя '{st.session_state.agg_func_label}'")
        if not current_df.empty:
             st.line_chart(current_df[value_col])
        else:
             st.warning("Нет данных после ресемплинга.")
    else:
         st.write("Ресемплинг не применен.")


# --- Шаг 6: Статистическая информация и проверка пропусков ---
st.header("6. Статистика и проверка пропусков")

if current_df.empty:
    st.warning("Нет данных для анализа после предыдущих шагов.")
else:
    st.subheader(f"Статистика для столбца '{value_col}'")
    st.dataframe(current_df[value_col].describe().to_frame())

    missing_values_count = current_df[value_col].isnull().sum()
    st.subheader("Проверка пропусков (NaN/None)")
    if missing_values_count > 0:
        st.warning(f"Обнаружено {missing_values_count} пропущенных значений (NaN/None).")
        st.dataframe(current_df[current_df[value_col].isnull()])
    else:
        st.success("Пропущенных значений (NaN/None) не обнаружено.")

    # Проверка на равномерность временной сетки
    st.subheader("Проверка равномерности временной сетки")
    inferred_freq = None
    missing_timestamps = pd.Index([])
    expected_index = None
    if pd.api.types.is_datetime64_any_dtype(current_df.index) and not current_df.empty:
        inferred_freq = pd.infer_freq(current_df.index)
        if inferred_freq:
            st.info(f"Определенная частота временного ряда: {inferred_freq}")
            expected_index = pd.date_range(start=current_df.index.min(), end=current_df.index.max(), freq=inferred_freq)
            missing_timestamps = expected_index.difference(current_df.index)
            if not missing_timestamps.empty:
                st.warning(f"Обнаружены пропуски в последовательности временных меток ({len(missing_timestamps)}).")
                st.dataframe(missing_timestamps[:min(10, len(missing_timestamps))].to_frame(index=False, name="Отсутствующая метка"))
            else:
                st.success("Временная сетка является равномерной.")
        else:
            st.info("Не удалось автоматически определить частоту ряда. Проверка на пропуски в сетке не выполнена.")
    elif not pd.api.types.is_datetime64_any_dtype(current_df.index):
         st.info("Индекс не datetime. Проверка равномерности сетки не выполнена.")
    # else: # current_df is empty
         # st.info("Нет данных для проверки сетки.")

logging.info(f"Шаг 6: Расчет статистики и проверка пропусков. Размер current_df: {current_df.shape}")


# --- Шаг 7: Обработка пропущенных значений ---
st.header("7. Обработка пропущенных значений")

fill_grid_option_checked = False # Флаг, что сетка была заполнена в этом прогоне
imputation_applied_this_run = False # Флаг, что была применена обработка NaN

if current_df.empty:
     st.warning("Нет данных для обработки пропусков.")
else:
    # Предложение заполнить пропуски сетки
    if inferred_freq and not missing_timestamps.empty:
        fill_grid_option = st.checkbox("Заполнить пропуски временной сетки значениями NaN?", value=True, key='fill_grid_check')
        if fill_grid_option:
            try:
                # Заполняем только если еще не заполнено (или параметры изменились)
                # Сравниваем индексы current_df и expected_index
                if not current_df.index.equals(expected_index):
                     current_df = current_df.reindex(expected_index) # Обновляем current_df
                     fill_grid_option_checked = True
                     st.success(f"Пропуски временной сетки ({len(missing_timestamps)}) заполнены NaN.")
                     logging.info(f"Шаг 7: Пропуски временной сетки заполнены NaN. Новый размер current_df: {current_df.shape}")
                     # Не перезапускаем
            except Exception as e:
                st.error(f"Ошибка при заполнении пропусков сетки: {e}")
                logging.error(f"Ошибка reindex для сетки: {e}")


    # Обработка NaN (исходных + добавленных при заполнении сетки)
    current_nan_count = current_df[value_col].isnull().sum()
    if current_nan_count > 0:
        st.write(f"Текущее количество пропусков (NaN/None): {current_nan_count}")

        # Выбор метода обработки
        current_imputation_label = next((k for k, v in imputation_options.items() if v == st.session_state.get('imputation_method')), "Не обрабатывать")

        selected_imputation_label = st.selectbox(
            "Выберите метод обработки пропусков:", options=list(imputation_options.keys()),
            index=list(imputation_options.keys()).index(current_imputation_label),
            key='impute_select'
        )
        imputation_method = None
        # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка ключа перед доступом через [] ---
        if selected_imputation_label in imputation_options:
             imputation_method = imputation_options[selected_imputation_label]
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        custom_value = None
        if imputation_method == 'custom':
            custom_value = st.number_input("Введите значение для заполнения:", value=0.0, format="%.3f", key='impute_custom_val')
            imputation_method = custom_value

        # Применяем, если метод выбран и он отличается от сохраненного
        if imputation_method is not None and st.session_state.get('imputation_method') != imputation_method:
            st.session_state.imputation_method = imputation_method # Сохраняем выбор

            logging.info(f"Шаг 7: Применение обработки пропусков методом '{selected_imputation_label}'.")
            try:
                original_nan_count = current_df[value_col].isnull().sum()
                # --- Применение методов ---
                if imputation_method == 'mean': fill_value = current_df[value_col].mean(); current_df[value_col] = current_df[value_col].fillna(fill_value)
                elif imputation_method == 'median': fill_value = current_df[value_col].median(); current_df[value_col] = current_df[value_col].fillna(fill_value)
                elif imputation_method == 'linear': current_df[value_col] = current_df[value_col].interpolate(method=imputation_method)
                elif isinstance(imputation_method, (int, float)): current_df[value_col] = current_df[value_col].fillna(imputation_method)
                else: current_df[value_col] = current_df[value_col].fillna(method=imputation_method) # ffill, bfill
                # --- Конец применения ---

                filled_count = original_nan_count - current_df[value_col].isnull().sum()
                st.success(f"Обработано {filled_count} пропущенных значений методом '{selected_imputation_label}'.")
                logging.info(f"Шаг 7: Пропуски обработаны. Новый размер current_df: {current_df.shape}")
                imputation_applied_this_run = True
                st.session_state.imputation_applied_label = selected_imputation_label # Сохраняем примененный метод
                # Не перезапускаем
            except Exception as e:
                st.error(f"Ошибка при обработке пропусков: {e}")
                logging.error(f"Ошибка fillna/interpolate: {e}")
                st.session_state.imputation_method = None # Сбросить состояние в случае ошибки
                st.session_state.imputation_applied_label = None
                # Не изменяем current_df при ошибке

        elif imputation_method is None and st.session_state.get('imputation_method') is not None:
             # Если выбрали "Не обрабатывать" после другого метода
             st.session_state.imputation_method = None
             st.session_state.imputation_applied_label = None
             logging.info("Шаг 7: Обработка пропусков отменена.")
             # Нужно вернуть current_df к состоянию до обработки пропусков
             # Это сложно, т.к. предыдущее состояние current_df не хранится явно.
             # Проще всего - перезапустить обработку с шага после ресемплинга.
             # Или требовать rerun после выбора "Не обрабатывать".
             st.warning("Чтобы отменить обработку пропусков, может потребоваться перезапуск страницы или повторный выбор параметров на предыдущих шагах.")


        # Отображаем результат, если обработка была применена
        applied_method_label = st.session_state.get('imputation_applied_label')
        if applied_method_label and applied_method_label != "Не обрабатывать":
            st.write(f"Пропуски обработаны методом '{applied_method_label}'.")
            st.line_chart(current_df[value_col])
        elif imputation_method is None:
            st.info("Пропуски не обрабатываются.")


    elif not fill_grid_option_checked: # Если NaN не было и сетку не заполняли
         st.info("Пропущенных значений для обработки нет.")
         logging.info("Шаг 7: Нет пропусков для обработки.")
         st.session_state.imputation_method = None # Убедимся, что state сброшен
         st.session_state.imputation_applied_label = None


# --- Шаг 8: Сохранение данных в Session State ---
st.header("8. Сохранение обработанных данных")

# Сохраняем финальное состояние current_df в session_state
# Это перезапишет данные от предыдущих прогонов финальным результатом текущего прогона
st.session_state.time_series = current_df.copy()
# Сохраняем имя основной колонки
if value_col in current_df.columns: # Доп. проверка, что колонка существует
    st.session_state.main_column = value_col
else:
     # Если вдруг колонки нет (не должно быть, но для надежности)
     if not current_df.empty:
          st.session_state.main_column = current_df.columns[0] # Берем первую доступную
          logging.warning(f"Шаг 8: Выбранная колонка '{value_col}' не найдена в финальном df. Установлена первая колонка '{st.session_state.main_column}' как main_column.")
     else:
          st.session_state.main_column = None
          logging.warning(f"Шаг 8: Финальный df пуст, main_column не установлен.")

st.success("Обработанные данные сохранены для использования на других страницах.")
st.write("Итоговый вид данных (первые 5 строк):")
if not current_df.empty:
    st.dataframe(current_df.head())
else:
    st.write("Нет данных для отображения.")

final_shape = st.session_state.time_series.shape
logging.info(f"Шаг 8: Финальные данные сохранены в session_state. Итоговый размер: {final_shape}")
logging.info(f"--- Пайплайн new_main.py завершен для текущего прогона ---")

st.markdown("---")
st.write("Данные доступны в `st.session_state.time_series`.")