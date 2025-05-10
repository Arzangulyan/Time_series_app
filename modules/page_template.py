import streamlit as st
import pandas as pd
import markdown2
from pathlib import Path


def setup_page(title, sidebar_header):
    st.set_page_config(page_title=title)
    st.title(title)
    st.sidebar.header(sidebar_header)


def load_time_series():
    if "time_series" in st.session_state and st.session_state.time_series is not None:
        time_series = st.session_state.time_series

        # Check if time_series is a DataFrame with 'Timestamp' and 'Value' columns
        if isinstance(time_series, pd.DataFrame) and 'Timestamp' in time_series.columns and 'Value' in time_series.columns:
            try:
                # Convert 'Timestamp' to datetime and set as index, then select 'Value' column
                time_series['Timestamp'] = pd.to_datetime(time_series['Timestamp'])
                time_series = time_series.set_index('Timestamp')['Value']
                st.session_state.time_series = time_series # Update session state
                st.success("DataFrame converted to Series with 'Timestamp' as index and 'Value' as data.")
            except Exception as e:
                st.error(f"Error converting DataFrame to Series: {e}")
                st.stop()
        
        # Получаем информацию о выбранной колонке
        main_column = st.session_state.get("main_column", None)

        if isinstance(time_series, pd.Series):
            main_column = time_series.name if time_series.name else "Value"
            st.success(f"Загружен одномерный временной ряд: '{main_column}'")
        elif isinstance(time_series, pd.DataFrame):
            if main_column and main_column in time_series.columns:
                st.success(f"Загружен временной ряд с основной колонкой: '{main_column}'")
            else:
                if time_series.shape[1] == 1:
                    main_column = time_series.columns[0]
                    st.success(f"Загружен одномерный временной ряд: '{main_column}'")
                else:
                    st.warning(f"Загружен временной ряд с {time_series.shape[1]} колонками, основная колонка не указана.")
        else:
            st.error("Неподдерживаемый тип данных для временного ряда.")
            st.stop()
        
        # Информация о размере временного ряда
        st.info(f"Количество записей: {len(time_series)}")
        
        # Проверяем тип индекса
        if isinstance(time_series.index, pd.DatetimeIndex):
            st.info(f"Временной диапазон: с {time_series.index.min().strftime('%d.%m.%Y %H:%M:%S')} по {time_series.index.max().strftime('%d.%m.%Y %H:%M:%S')}")
    else:
        st.warning(
            "Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App» и сохраните ряд для дальнейшего анализа."
        )
        st.stop()
    
    st.subheader("Загруженный ряд")
    display_data(time_series)
    return time_series


def display_data(df):
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.write(df)
    with col2:
        # Проверяем тип данных и соответствующим образом отображаем их
        if isinstance(df, pd.DataFrame):
            if df.shape[1] > 0:  # Убедимся, что есть хотя бы один столбец
                st.line_chart(df.iloc[:, 0])
            else:
                st.warning("DataFrame не содержит столбцов для отображения")
        elif isinstance(df, pd.Series):
            st.line_chart(df)
        else:
            st.warning(f"Неподдерживаемый тип данных: {type(df)}")


def run_calculations_on_button_click(calculation_function, *args, **kwargs):
    """
    Запускает вычисления при нажатии кнопки в сайдбаре и сохраняет состояние между обновлениями.

    Args:
        calculation_function (callable): Функция, выполняющая вычисления.
        *args: Позиционные аргументы для функции вычислений.
        **kwargs: Именованные аргументы для функции вычислений.
    """
    # Уникальный ключ для состояния кнопки
    button_state_key = f"run_calc_state_{calculation_function.__name__}"
    
    # Инициализация состояния, если оно еще не существует
    if button_state_key not in st.session_state:
        st.session_state[button_state_key] = False
    
    # Кнопка для запуска вычислений
    if st.sidebar.button("Запустить вычисления"):
        st.session_state[button_state_key] = not st.session_state[button_state_key]
    
    # Отображение текущего состояния
    st.sidebar.write("Статус вычислений: " + ("Запущено" if st.session_state[button_state_key] else "Остановлено"))
    
    # Выполнение вычислений, если состояние True
    if st.session_state[button_state_key]:
        calculation_function(*args, **kwargs)