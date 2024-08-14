import streamlit as st
import pandas as pd


def setup_page(title, sidebar_header):
    st.set_page_config(page_title=title)
    st.title(f"Прогнозирование временных рядов с использованием {title}")
    st.sidebar.header(sidebar_header)


def display_data(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 0])


def load_time_series():
    if "time_series" in st.session_state and not st.session_state.time_series.empty:
        time_series = st.session_state.time_series
    else:
        st.warning(
            "Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App»"
        )
        st.stop()
    st.subheader("Загруженный ряд")
    display_data(time_series)
    return time_series


def run_calculations_on_button_click(calculation_function, *args, **kwargs):
    """
    Запускает вычисления только при нажатии кнопки в сайдбаре.

    Args:
        calculation_function (callable): Функция, выполняющая вычисления.
        *args: Позиционные аргументы для функции вычислений.
        **kwargs: Именованные аргументы для функции вычислений.
    """
    if st.sidebar.button("Запустить вычисления"):
        calculation_function(*args, **kwargs)
