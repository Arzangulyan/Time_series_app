"""
Унифицированные функции для построения графиков временных рядов
с гибкой временной сеткой и единообразным форматированием.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Optional, Union, List, Tuple


def detect_time_frequency(index: pd.Index) -> Tuple[str, str]:
    """
    Автоматически определяет частоту временного ряда и подходящий формат отображения.
    
    Returns:
        Tuple[str, str]: (частота для plotly, формат для matplotlib)
    """
    if len(index) < 2:
        return "D1", "%Y-%m-%d"
    
    # Вычисляем среднюю разность между соседними точками
    if isinstance(index, pd.DatetimeIndex):
        time_diff = (index[-1] - index[0]) / (len(index) - 1)
        
        if time_diff <= timedelta(minutes=1):
            return "M1", "%H:%M:%S"  # Секунды/минуты
        elif time_diff <= timedelta(hours=1):
            return "M15", "%H:%M"  # Минуты
        elif time_diff <= timedelta(days=1):
            return "H1", "%m-%d %H:%M"  # Часы
        elif time_diff <= timedelta(days=7):
            return "D1", "%m-%d"  # Дни
        elif time_diff <= timedelta(days=30):
            return "D7", "%Y-%m-%d"  # Недели
        elif time_diff <= timedelta(days=365):
            return "M1", "%Y-%m"  # Месяцы
        else:
            return "M3", "%Y"  # Кварталы/годы
    else:
        # Для числовых индексов
        return "D1", "%Y-%m-%d"


def _safe_convert_timestamp_for_plotly(timestamp):
    """
    Безопасно конвертирует pandas Timestamp в формат, понятный Plotly.
    """
    if isinstance(timestamp, pd.Timestamp):
        # Конвертируем в datetime объект Python
        return timestamp.to_pydatetime()
    elif hasattr(timestamp, 'to_pydatetime'):
        return timestamp.to_pydatetime()
    else:
        return timestamp


def create_unified_forecast_plot_plotly(
    original_series: pd.Series,
    train_predictions: Optional[pd.Series] = None,
    test_predictions: Optional[pd.Series] = None,
    future_predictions: Optional[pd.Series] = None,
    train_data: Optional[pd.Series] = None,
    test_data: Optional[pd.Series] = None,
    title: str = "Прогнозирование временного ряда",
    height: int = 600,
    show_confidence_interval: bool = False,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None
) -> go.Figure:
    """
    Создает унифицированный график прогнозирования с использованием Plotly.
    
    Parameters:
    -----------
    original_series : pd.Series
        Исходный временной ряд
    train_predictions : pd.Series, optional
        Прогнозы на обучающей выборке
    test_predictions : pd.Series, optional
        Прогнозы на тестовой выборке
    future_predictions : pd.Series, optional
        Прогнозы на будущие периоды
    train_data : pd.Series, optional
        Обучающие данные (для выделения области)
    test_data : pd.Series, optional
        Тестовые данные (для выделения области)
    title : str
        Заголовок графика
    height : int
        Высота графика в пикселях
    show_confidence_interval : bool
        Показывать ли доверительный интервал
    confidence_lower : pd.Series, optional
        Нижняя граница доверительного интервала
    confidence_upper : pd.Series, optional
        Верхняя граница доверительного интервала
    
    Returns:
    --------
    go.Figure
        График Plotly
    """
    fig = go.Figure()
    
    # Определяем частоту временного ряда
    freq, _ = detect_time_frequency(original_series.index)
    
    # Исходные данные
    fig.add_trace(go.Scatter(
        x=original_series.index,
        y=original_series.values,
        mode='lines',
        name='Исходные данные',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Исходные данные</b><br>' +
                     'Время: %{x}<br>' +
                     'Значение: %{y:.4f}<br>' +
                     '<extra></extra>'
    ))
    
    # Прогнозы на обучающей выборке
    if train_predictions is not None:
        fig.add_trace(go.Scatter(
            x=train_predictions.index,
            y=train_predictions.values,
            mode='lines',
            name='Прогноз (обучение)',
            line=dict(color='#ff7f0e', width=3, dash='dot'),  # Увеличиваем толщину с 2 до 3
            hovertemplate='<b>Прогноз (обучение)</b><br>' +
                         'Время: %{x}<br>' +
                         'Значение: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
    
    # Прогнозы на тестовой выборке
    if test_predictions is not None:
        fig.add_trace(go.Scatter(
            x=test_predictions.index,
            y=test_predictions.values,
            mode='lines',
            name='Прогноз (тест)',
            line=dict(color='#2ca02c', width=3, dash='dash'),  # Увеличиваем толщину с 2 до 3
            hovertemplate='<b>Прогноз (тест)</b><br>' +
                         'Время: %{x}<br>' +
                         'Значение: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
    
    # Прогнозы на будущие периоды
    if future_predictions is not None:
        fig.add_trace(go.Scatter(
            x=future_predictions.index,
            y=future_predictions.values,
            mode='lines',
            name='Прогноз (будущее)',
            line=dict(color='#d62728', width=3, dash='dash'),
            hovertemplate='<b>Прогноз (будущее)</b><br>' +
                         'Время: %{x}<br>' +
                         'Значение: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
    
    # Доверительный интервал
    if show_confidence_interval and confidence_lower is not None and confidence_upper is not None:
        # Добавляем заливку доверительного интервала
        fig.add_trace(go.Scatter(
            x=confidence_lower.index.tolist() + confidence_upper.index.tolist()[::-1],
            y=confidence_lower.values.tolist() + confidence_upper.values.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(214, 39, 40, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% доверительный интервал',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # Вертикальные линии для разделения обучающей и тестовой выборок
    if train_data is not None and test_data is not None and len(test_data) > 0:
        split_point = test_data.index[0]
        # Конвертируем timestamp для совместимости с Plotly
        split_point_converted = _safe_convert_timestamp_for_plotly(split_point)
        
        try:
            # Пробуем использовать add_shape напрямую как более надежный метод
            # Получаем пределы Y-оси для рисования линии
            all_y_values = []
            if original_series is not None:
                all_y_values.extend(original_series.values)
            if train_predictions is not None:
                all_y_values.extend(train_predictions.values)
            if test_predictions is not None:
                all_y_values.extend(test_predictions.values)
            if future_predictions is not None:
                all_y_values.extend(future_predictions.values)
            
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                
                fig.add_shape(
                    type="line",
                    x0=split_point_converted,
                    y0=y_min,
                    x1=split_point_converted,
                    y1=y_max,
                    line=dict(color="gray", dash="dot", width=1),
                )
                
                # Добавляем аннотацию
                fig.add_annotation(
                    x=split_point_converted,
                    y=y_max,
                    text="Начало тестирования",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="gray", size=10)
                )
        except Exception as e:
            # Если и это не работает, просто игнорируем линию
            print(f"Не удалось добавить разделительную линию: {e}")
    
    # Настройка осей и макета
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Время",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat=None,  # Plotly автоматически выберет формат
        ),
        yaxis=dict(
            title="Значение",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Увеличиваем отступ с -0.15 до -0.2
            xanchor="center",
            x=0.5
        ),
        height=height,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_unified_forecast_plot_matplotlib(
    original_series: pd.Series,
    train_predictions: Optional[pd.Series] = None,
    test_predictions: Optional[pd.Series] = None,
    future_predictions: Optional[pd.Series] = None,
    train_data: Optional[pd.Series] = None,
    test_data: Optional[pd.Series] = None,
    title: str = "Прогнозирование временного ряда",
    figsize: Tuple[int, int] = (12, 6),
    show_confidence_interval: bool = False,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None
) -> plt.Figure:
    """
    Создает унифицированный график прогнозирования с использованием Matplotlib.
    
    Parameters:
    -----------
    Аналогичны create_unified_forecast_plot_plotly
    
    Returns:
    --------
    plt.Figure
        График Matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Определяем частоту временного ряда
    _, format_str = detect_time_frequency(original_series.index)
    
    # Исходные данные
    ax.plot(original_series.index, original_series.values, 
           label='Исходные данные', color='#1f77b4', linewidth=2)
    
    # Прогнозы на обучающей выборке
    if train_predictions is not None:
        ax.plot(train_predictions.index, train_predictions.values, 
               label='Прогноз (обучение)', color='#ff7f0e', linewidth=3, linestyle=':')  # Увеличиваем толщину с 2 до 3
    
    # Прогнозы на тестовой выборке
    if test_predictions is not None:
        ax.plot(test_predictions.index, test_predictions.values, 
               label='Прогноз (тест)', color='#2ca02c', linewidth=3, linestyle='--')  # Увеличиваем толщину с 2 до 3
    
    # Прогнозы на будущие периоды
    if future_predictions is not None:
        ax.plot(future_predictions.index, future_predictions.values, 
               label='Прогноз (будущее)', color='#d62728', linewidth=3, linestyle='--')
    
    # Доверительный интервал
    if show_confidence_interval and confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(confidence_lower.index, confidence_lower.values, confidence_upper.values,
                       alpha=0.2, color='#d62728', label='95% доверительный интервал')
    
    # Вертикальная линия для разделения обучающей и тестовой выборок
    if train_data is not None and test_data is not None and len(test_data) > 0:
        split_point = test_data.index[0]
        ax.axvline(x=split_point, color='gray', linestyle=':', alpha=0.7)
        ax.text(split_point, ax.get_ylim()[1] * 0.95, 'Начало тестирования', 
               rotation=90, verticalalignment='top', alpha=0.7)
    
    # Настройка осей
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Настройка временной оси
    if isinstance(original_series.index, pd.DatetimeIndex):
        # Автоматический выбор формата даты в зависимости от периода
        time_span = original_series.index[-1] - original_series.index[0]
        
        if time_span <= timedelta(hours=1):
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_span <= timedelta(days=1):
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_span <= timedelta(days=30):
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif time_span <= timedelta(days=365):
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Легенда снизу
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)  # Увеличиваем отступ с -0.15 до -0.2
    
    plt.tight_layout()
    return fig


def create_simple_time_series_plot(
    series: pd.Series,
    title: str = "Временной ряд",
    height: int = 400
) -> go.Figure:
    """
    Создает простой график временного ряда с унифицированным форматированием.
    """
    fig = go.Figure()
    
    freq, _ = detect_time_frequency(series.index)
    
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name='Временной ряд',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Временной ряд</b><br>' +
                     'Время: %{x}<br>' +
                     'Значение: %{y:.4f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Время",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Значение",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    return fig
