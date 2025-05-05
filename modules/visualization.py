import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def df_chart_display_loc(df: pd.DataFrame, data_col_loc: str):
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.write(df.loc[:])
    with col2:
        st.line_chart(df.loc[:, data_col_loc])


def df_chart_display_iloc(df: pd.DataFrame, data_col_iloc: int):
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.write(df.iloc[:])
    with col2:
        st.line_chart(df.iloc[:, data_col_iloc])


def plot_statistical_analysis(analysis_results):
    fig_acf_pacf, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(analysis_results["acf"])
    ax1.set_title("Автокорреляция")
    ax2.plot(analysis_results["pacf"])
    ax2.set_title("Частичная автокорреляция")

    return fig_acf_pacf


def plot_decomposition(decomposition):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title("Исходный ряд")
    decomposition.trend.plot(ax=ax2)
    ax2.set_title("Тренд")
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title("Сезонность")
    decomposition.resid.plot(ax=ax4)
    ax4.set_title("Остаток")
    plt.tight_layout()
    return fig


def plot_decomposition_plotly(decomposition):
    """
    Строит график декомпозиции временного ряда с использованием Plotly.
    
    Args:
    decomposition: Результат декомпозиции временного ряда.
    
    Returns:
    plotly.graph_objects.Figure: Объект фигуры с графиками декомпозиции.
    """
    # Создаем подграфики 4x1
    fig = make_subplots(
        rows=4, 
        cols=1,
        subplot_titles=("Исходный ряд", "Тренд", "Сезонность", "Остаток"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Добавляем данные на каждый подграфик
    fig.add_trace(
        go.Scatter(
            x=decomposition.observed.index, 
            y=decomposition.observed.values,
            mode='lines',
            name='Исходный ряд',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=decomposition.trend.index, 
            y=decomposition.trend.values,
            mode='lines',
            name='Тренд',
            line=dict(color='red', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index, 
            y=decomposition.seasonal.values,
            mode='lines',
            name='Сезонность',
            line=dict(color='green', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=decomposition.resid.index, 
            y=decomposition.resid.values,
            mode='lines',
            name='Остаток',
            line=dict(color='purple', width=1.5)
        ),
        row=4, col=1
    )
    
    # Обновляем макет
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=False,
        title_text="Декомпозиция временного ряда"
    )
    
    # Обеспечиваем, чтобы даты правильно отображались
    if isinstance(decomposition.observed.index, pd.DatetimeIndex):
        fig.update_xaxes(type='date')
    
    return fig
