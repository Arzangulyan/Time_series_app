import streamlit as st
import pandas as pd
from typing import Any, List, Union


def nothing_selected_sidebar(widget: Any) -> None:
    """
    Останавливает выполнение, если виджет не выбран.

    Args:
        widget: Виджет Streamlit для проверки.
    """
    if widget == "":
        st.sidebar.write("Ожидается ответ от пользователя...")
        st.stop()


def DF_wrapper(
    date_column: Union[pd.DatetimeIndex, pd.Series], series_column: pd.Series
) -> pd.DataFrame:
    """
    Создает DataFrame из временного ряда.

    Args:
        date_column (Union[pd.DatetimeIndex, pd.Series]): Индекс дат или серия с датами.
        series_column (pd.Series): Серия со значениями ряда.

    Returns:
        pd.DataFrame: DataFrame с колонками "Время" и "Ряд".
    """
    if isinstance(date_column, pd.DatetimeIndex):
        date_column = date_column.to_series()
    return pd.DataFrame({"Время": date_column, "Ряд": series_column})


def initialize_session_state(**kwargs):
    """
    Инициализирует значения в st.session_state, если они еще не существуют.

    Args:
        **kwargs: Именованные аргументы, где ключ — это имя переменной, а значение — значение по умолчанию.
    """
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value
