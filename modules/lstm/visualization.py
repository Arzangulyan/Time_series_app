"""
Функции визуализации для LSTM моделей прогнозирования временных рядов.

Включает функции для:
- Визуализации исходных временных рядов
- Отображения результатов обучения и прогнозирования
- Анализа и диагностики точности моделей
- Интерактивной визуализации с использованием Plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
import warnings


def plot_time_series(time_series: pd.Series, title: str = "Временной ряд", 
                   y_label: str = "Значение") -> plt.Figure:
    """
    Строит график временного ряда.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для визуализации
    title : str, default="Временной ряд"
        Заголовок графика
    y_label : str, default="Значение"
        Название оси Y
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_series.index, time_series.values, color='#1f77b4')
    
    # Добавление сетки и подписей
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Время')
    ax.set_ylabel(y_label)
    
    # Если индекс - это даты, поворачиваем метки для лучшей читаемости
    if isinstance(time_series.index, pd.DatetimeIndex):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_train_test_results(original_series: pd.Series, train_pred: pd.Series, 
                          test_pred: pd.Series, title: str = "Результаты LSTM модели") -> go.Figure:
    """
    Строит график с результатами обучения и тестирования модели.
    
    Параметры:
    -----------
    original_series : pd.Series
        Исходный временной ряд
    train_pred : pd.Series
        Прогнозы на обучающей выборке
    test_pred : pd.Series
        Прогнозы на тестовой выборке
    title : str, default="Результаты LSTM модели"
        Заголовок графика
        
    Возвращает:
    -----------
    go.Figure
        Объект графика Plotly
    """
    fig = go.Figure()
    
    # Добавляем исходный ряд
    fig.add_trace(go.Scatter(
        x=original_series.index, 
        y=original_series.values,
        mode='lines',
        name='Исходные данные',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    # Добавляем прогнозы на обучающей выборке
    fig.add_trace(go.Scatter(
        x=train_pred.index, 
        y=train_pred.values,
        mode='lines',
        name='Прогноз (обучение)',
        line=dict(color='#ff7f0e', width=1.5, dash='dot')
    ))
    
    # Добавляем прогнозы на тестовой выборке
    fig.add_trace(go.Scatter(
        x=test_pred.index, 
        y=test_pred.values,
        mode='lines',
        name='Прогноз (тест)',
        line=dict(color='#2ca02c', width=1.5, dash='dot')
    ))
    
    # Настраиваем внешний вид
    fig.update_layout(
        title=title,
        xaxis_title='Время',
        yaxis_title='Значение',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_forecast(original_series: pd.Series, forecast_series: pd.Series, 
                title: str = "Прогноз LSTM модели") -> go.Figure:
    """
    Строит график с прогнозом на будущее.
    
    Параметры:
    -----------
    original_series : pd.Series
        Исходный временной ряд
    forecast_series : pd.Series
        Прогноз на будущее
    title : str, default="Прогноз LSTM модели"
        Заголовок графика
        
    Возвращает:
    -----------
    go.Figure
        Объект графика Plotly
    """
    fig = go.Figure()
    
    # Добавляем исходный ряд
    fig.add_trace(go.Scatter(
        x=original_series.index, 
        y=original_series.values,
        mode='lines',
        name='Исторические данные',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    # Добавляем прогноз
    fig.add_trace(go.Scatter(
        x=forecast_series.index, 
        y=forecast_series.values,
        mode='lines',
        name='Прогноз',
        line=dict(color='#ff7f0e', width=1.5, dash='dot')
    ))
    
    # Добавляем вертикальную линию, разделяющую исторические данные и прогноз
    if len(original_series) > 0 and len(forecast_series) > 0:
        fig.add_vline(
            x=original_series.index[-1], 
            line=dict(color='gray', width=1, dash='dash')
        )
    
    # Настраиваем внешний вид
    fig.update_layout(
        title=title,
        xaxis_title='Время',
        yaxis_title='Значение',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_training_history(history: Dict[str, List[float]]) -> plt.Figure:
    """
    Строит график процесса обучения модели.
    
    Параметры:
    -----------
    history : dict
        Словарь с историей обучения (потери на каждой эпохе)
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика Matplotlib
    """
    if not history or 'loss' not in history:
        raise ValueError("В истории обучения отсутствуют данные о потерях")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Строим график обучающих потерь
    ax.plot(history['loss'], label='Обучение', color='#1f77b4')
    
    # Строим график валидационных потерь, если они есть
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Валидация', color='#ff7f0e')
    
    # Настраиваем внешний вид
    ax.set_title('График потерь во время обучения')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Потери (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Строит гистограмму распределения ошибок прогноза.
    
    Параметры:
    -----------
    y_true : array-like
        Фактические значения
    y_pred : array-like
        Предсказанные значения
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика Matplotlib
    """
    # Преобразуем в одномерные массивы
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Вычисляем ошибки
    errors = y_true - y_pred
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Строим гистограмму ошибок
    sns.histplot(errors, kde=True, ax=ax, color='#1f77b4')
    
    # Добавляем вертикальную линию на нуле
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Вычисляем и отображаем статистики
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Настраиваем внешний вид
    ax.set_title('Распределение ошибок прогноза')
    ax.set_xlabel('Ошибка')
    ax.set_ylabel('Частота')
    ax.text(0.05, 0.95, f'Средняя ошибка: {mean_error:.4f}\nСтандартное отклонение: {std_error:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def display_model_information(model_params: Dict[str, Any]) -> None:
    """
    Отображает информацию о модели в Streamlit.
    
    Параметры:
    -----------
    model_params : dict
        Словарь с параметрами модели
    """
    st.subheader("Параметры модели LSTM")
    
    # Создаем таблицу с ключевыми параметрами
    params_table = {
        "Параметр": [],
        "Значение": []
    }
    
    # Добавляем основные параметры
    if 'sequence_length' in model_params:
        params_table["Параметр"].append("Длина последовательности")
        params_table["Значение"].append(model_params["sequence_length"])
    
    if 'units' in model_params:
        params_table["Параметр"].append("Архитектура (нейроны в слоях)")
        params_table["Значение"].append(str(model_params["units"]))
    
    if 'dropout_rate' in model_params:
        params_table["Параметр"].append("Коэффициент dropout")
        params_table["Значение"].append(f"{model_params['dropout_rate']:.2f}")
    
    if 'bidirectional' in model_params:
        params_table["Параметр"].append("Двунаправленный LSTM")
        params_table["Значение"].append("Да" if model_params["bidirectional"] else "Нет")
    
    # Отображаем таблицу
    st.table(pd.DataFrame(params_table))


def display_metrics(metrics: Dict[str, float]) -> None:
    """
    Отображает метрики качества модели в Streamlit.
    
    Параметры:
    -----------
    metrics : dict
        Словарь с метриками модели
    """
    st.subheader("Метрики качества модели")
    
    # Создаем три колонки для отображения метрик
    col1, col2, col3 = st.columns(3)
    
    # Отображаем основные метрики в колонках
    with col1:
        st.metric(
            label="RMSE",
            value=f"{metrics.get('rmse', 'N/A'):.4f}" if not pd.isna(metrics.get('rmse', np.nan)) else "N/A"
        )
    
    with col2:
        st.metric(
            label="MAE",
            value=f"{metrics.get('mae', 'N/A'):.4f}" if not pd.isna(metrics.get('mae', np.nan)) else "N/A"
        )
    
    with col3:
        if 'mape' in metrics and not pd.isna(metrics['mape']):
            st.metric(
                label="MAPE",
                value=f"{metrics['mape']:.2f}%"
            )
        else:
            st.metric(
                label="MAPE",
                value="N/A"
            )
    
    # Отображаем дополнительные метрики
    st.write("Дополнительные метрики:")
    
    # Создаем таблицу для дополнительных метрик
    additional_metrics = {
        "Метрика": ["R²", "Adjusted R²", "MASE"],
        "Значение": [
            f"{metrics.get('r2', 'N/A'):.4f}" if not pd.isna(metrics.get('r2', np.nan)) else "N/A",
            f"{metrics.get('adj_r2', 'N/A'):.4f}" if not pd.isna(metrics.get('adj_r2', np.nan)) else "N/A",
            f"{metrics.get('mase', 'N/A'):.4f}" if not pd.isna(metrics.get('mase', np.nan)) else "N/A"
        ]
    }
    
    # Отображаем таблицу
    st.table(pd.DataFrame(additional_metrics)) 