import streamlit as st
import pandas as pd


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
    else:
        st.warning(
            "Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App»"
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