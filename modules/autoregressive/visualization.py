"""
Функции для визуализации и диагностики авторегрессионных моделей.

Включает функции для:
- Построения графиков временных рядов и прогнозов
- Визуализации ACF и PACF
- Анализа и диагностики остатков модели
- Сравнения нескольких моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import streamlit as st

from .models import BaseTimeSeriesModel
from .core import check_stationarity, calculate_acf_pacf, suggest_arima_params


def plot_time_series(time_series: pd.Series, title: str = "Временной ряд", 
                    y_label: str = "Значение") -> plt.Figure:
    """
    Строит простой график временного ряда.
    
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


def plot_train_test_split(train_data: pd.Series, test_data: pd.Series, 
                         title: str = "Разделение на обучающую и тестовую выборки") -> plt.Figure:
    """
    Визуализирует разделение временного ряда на обучающую и тестовую выборки.
    
    Параметры:
    -----------
    train_data : pd.Series
        Обучающая выборка
    test_data : pd.Series
        Тестовая выборка
    title : str, default="Разделение на обучающую и тестовую выборки"
        Заголовок графика
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Строим обучающие данные
    ax.plot(train_data.index, train_data.values, color='blue', label='Обучающая выборка')
    
    # Строим тестовые данные
    ax.plot(test_data.index, test_data.values, color='red', label='Тестовая выборка')
    
    # Добавляем вертикальную линию, разделяющую выборки
    ax.axvline(x=test_data.index[0], color='black', linestyle='--', alpha=0.7)
    
    # Добавление сетки и подписей
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.legend()
    
    # Если индекс - это даты, поворачиваем метки для лучшей читаемости
    if isinstance(train_data.index, pd.DatetimeIndex):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_acf_pacf_plotly(time_series: pd.Series, lags: int = 40, 
                        title: str = "Анализ ACF и PACF") -> go.Figure:
    """
    Создает интерактивный график ACF и PACF с использованием Plotly.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для анализа
    lags : int, default=40
        Количество лагов для отображения
    title : str, default="Анализ ACF и PACF"
        Заголовок графика
        
    Возвращает:
    -----------
    go.Figure
        Объект графика Plotly
    """
    # Вычисляем ACF и PACF
    acf_values, pacf_values = calculate_acf_pacf(time_series, lags)
    
    if len(acf_values) == 0 or len(pacf_values) == 0:
        # Если не удалось рассчитать, возвращаем пустой график
        fig = make_subplots(rows=2, cols=1, subplot_titles=["ACF не удалось рассчитать", "PACF не удалось рассчитать"])
        return fig
    
    # Создаем subplot
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Автокорреляционная функция (ACF)", 
                                                         "Частичная автокорреляционная функция (PACF)"])
    
    # Добавляем ACF
    fig.add_trace(
        go.Bar(x=list(range(len(acf_values))), y=acf_values, name="ACF"),
        row=1, col=1
    )
    
    # Добавляем PACF
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name="PACF"),
        row=2, col=1
    )
    
    # Добавляем доверительный интервал
    conf_level = 1.96 / np.sqrt(len(time_series))
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(acf_values))),
            y=[conf_level] * len(acf_values),
            mode='lines',
            line=dict(dash="dash", color="red", width=1),
            name="95% CI",
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(acf_values))),
            y=[-conf_level] * len(acf_values),
            mode='lines',
            line=dict(dash="dash", color="red", width=1),
            name="95% CI",
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pacf_values))),
            y=[conf_level] * len(pacf_values),
            mode='lines',
            line=dict(dash="dash", color="red", width=1),
            name="95% CI",
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pacf_values))),
            y=[-conf_level] * len(pacf_values),
            mode='lines',
            line=dict(dash="dash", color="red", width=1),
            name="95% CI"
        ),
        row=2, col=1
    )
    
    # Обновляем макет
    fig.update_layout(
        title=title,
        xaxis_title="Лаг",
        xaxis2_title="Лаг",
        yaxis_title="ACF",
        yaxis2_title="PACF",
        height=600,
        showlegend=True
    )
    
    return fig


def plot_forecast(model: BaseTimeSeriesModel, steps: int, original_data: Optional[pd.Series] = None,
                train_data: Optional[pd.Series] = None, test_data: Optional[pd.Series] = None,
                include_history: bool = True, conf_int: bool = True, 
                alpha: float = 0.05, title: str = "Прогноз") -> plt.Figure:
    """
    Построение графика прогноза с доверительным интервалом.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    steps : int
        Количество шагов для прогноза
    original_data : pd.Series, optional
        Исходный временной ряд (если отличается от train_data)
    train_data : pd.Series, optional
        Обучающая выборка
    test_data : pd.Series, optional
        Тестовая выборка
    include_history : bool, default=True
        Включать ли исторические данные в график
    conf_int : bool, default=True
        Отображать ли доверительный интервал
    alpha : float, default=0.05
        Уровень значимости для доверительного интервала (1-alpha)*100%
    title : str, default="Прогноз"
        Заголовок графика
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика
    """
    if not model.is_fitted:
        raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
    
    # Получаем прогноз
    forecast = model.predict(steps=steps)
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Строим исторические данные
    if include_history:
        if original_data is not None:
            ax.plot(original_data.index, original_data.values, color='gray', alpha=0.7, label='Исходные данные')
        
        if train_data is not None:
            ax.plot(train_data.index, train_data.values, color='blue', label='Обучающая выборка')
        
        if test_data is not None:
            ax.plot(test_data.index, test_data.values, color='green', label='Тестовая выборка')
            
            # Отображаем вертикальную линию, разделяющую историю и прогноз
            if test_data.index[0] != forecast.index[0]:
                ax.axvline(x=test_data.index[0], color='black', linestyle='--', alpha=0.5)
    
    # Строим прогноз
    ax.plot(forecast.index, forecast.values, color='red', linestyle='-', marker='o', markersize=4, label='Прогноз')
    
    # Добавляем доверительный интервал, если доступен и нужен
    if conf_int and hasattr(model.fitted_model, 'get_forecast'):
        try:
            forecast_obj = model.fitted_model.get_forecast(steps=steps)
            conf_interval = forecast_obj.conf_int(alpha=alpha)
            
            lower_bound = conf_interval.iloc[:, 0]
            upper_bound = conf_interval.iloc[:, 1]
            
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='red', alpha=0.2, 
                           label=f'{(1-alpha)*100:.0f}% доверительный интервал')
        except Exception as e:
            warnings.warn(f"Не удалось построить доверительный интервал: {str(e)}")
    
    # Добавление сетки и подписей
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.legend()
    
    # Если индекс - это даты, поворачиваем метки для лучшей читаемости
    if isinstance(forecast.index, pd.DatetimeIndex):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def plot_forecast_plotly(model: BaseTimeSeriesModel, steps: int, original_data: Optional[pd.Series] = None,
                train_data: Optional[pd.Series] = None, test_data: Optional[pd.Series] = None,
                include_history: bool = True, conf_int: bool = True, 
                alpha: float = 0.05, title: str = "Прогноз") -> go.Figure:
    """
    Построение интерактивного графика прогноза с доверительным интервалом (Plotly версия).
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    steps : int
        Количество шагов для прогноза
    original_data : pd.Series, optional
        Исходный временной ряд (если отличается от train_data)
    train_data : pd.Series, optional
        Обучающая выборка
    test_data : pd.Series, optional
        Тестовая выборка
    include_history : bool, default=True
        Включать ли исторические данные в график
    conf_int : bool, default=True
        Отображать ли доверительный интервал
    alpha : float, default=0.05
        Уровень значимости для доверительного интервала (1-alpha)*100%
    title : str, default="Прогноз"
        Заголовок графика
        
    Возвращает:
    -----------
    go.Figure
        Объект графика Plotly
    """
    if not model.is_fitted:
        raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
    
    # Получаем прогноз
    forecast = model.predict(steps=steps)
    
    # Создаем график Plotly
    fig = go.Figure()
    
    # Строим исторические данные
    if include_history:
        if original_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=original_data.index,
                    y=original_data.values,
                    mode='lines',
                    line=dict(color='gray', width=1.5),
                    opacity=0.7,
                    name='Исходные данные'
                )
            )
        
        if train_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=train_data.index,
                    y=train_data.values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Обучающая выборка'
                )
            )
        
        if test_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=test_data.index,
                    y=test_data.values,
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Тестовая выборка'
                )
            )
            
            # Отображаем вертикальную линию, разделяющую историю и прогноз
            if test_data.index[0] != forecast.index[0]:
                fig.add_vline(
                    x=test_data.index[0], 
                    line=dict(color='black', width=1, dash="dash"),
                    opacity=0.5
                )
    
    # Строим прогноз
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            name='Прогноз'
        )
    )
    
    # Добавляем доверительный интервал, если доступен и нужен
    if conf_int and hasattr(model.fitted_model, 'get_forecast'):
        try:
            forecast_obj = model.fitted_model.get_forecast(steps=steps)
            conf_interval = forecast_obj.conf_int(alpha=alpha)
            
            lower_bound = conf_interval.iloc[:, 0]
            upper_bound = conf_interval.iloc[:, 1]
            
            # Добавляем верхнюю границу доверительного интервала
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            # Добавляем нижнюю границу доверительного интервала
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    name=f'{(1-alpha)*100:.0f}% доверительный интервал'
                )
            )
        except Exception as e:
            warnings.warn(f"Не удалось построить доверительный интервал: {str(e)}")
    
    # Обновляем макет графика
    fig.update_layout(
        title=title,
        xaxis_title='Время',
        yaxis_title='Значение',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=500
    )
    
    # Настраиваем сетку
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.1)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.1)'
    )
    
    # Если индекс - это даты, настраиваем формат дат
    if isinstance(forecast.index, pd.DatetimeIndex):
        fig.update_xaxes(
            tickangle=45,
            tickformat="%Y-%m-%d"
        )
    
    return fig


def plot_forecast_matplotlib(model, steps=None, original_data=None, train_data=None, test_data=None, title=None):
    """
    Создает график прогноза с помощью matplotlib (для экспорта в PDF отчеты).
    """
    if steps is None:
        if test_data is not None:
            steps = len(test_data)
        else:
            steps = 10
    
    # Получаем прогноз на тестовый период
    forecast = model.predict(steps=steps)
    
    # Создаем фигуру с достаточным размером для читаемости в PDF
    plt.figure(figsize=(10, 6))
    
    # Строим график исходных данных, если они предоставлены
    if original_data is not None:
        plt.plot(original_data.index, original_data, 'gray', alpha=0.5, label='Исходные данные')
    
    # Строим график обучающих данных, если они предоставлены
    if train_data is not None:
        plt.plot(train_data.index, train_data, 'b-', label='Обучающая выборка')
    
    # Строим график тестовых данных, если они предоставлены
    if test_data is not None:
        plt.plot(test_data.index, test_data, 'g-', label='Тестовая выборка')
    
    # Строим график прогноза
    plt.plot(forecast.index, forecast, 'r-', linewidth=2, label='Прогноз')
    
    # Заголовок, оси и легенда
    if title:
        plt.title(title, fontsize=14)
    plt.xlabel('Время', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    
    # Улучшенная легенда
    legend = plt.legend(loc='upper left', fontsize=10, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Улучшения для читаемости
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Уменьшаем число делений на оси X для лучшей читаемости
    ax = plt.gca()
    if len(forecast) > 20:
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    
    # Поворачиваем метки, если они перекрываются
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return plt.gcf()  # Возвращает текущую фигуру


def plot_residuals_diagnostic(model: BaseTimeSeriesModel) -> go.Figure:
    """
    Создает диагностические графики для остатков модели.
    
    Параметры:
    -----------
    model : объект модели с атрибутом fitted_model
        Обученная авторегрессионная модель
        
    Возвращает:
    -----------
    plotly.graph_objs.Figure
        Фигура с графиками диагностики остатков
    """
    if not hasattr(model, 'fitted_model') or model.fitted_model is None:
        raise ValueError("Модель не обучена. Отсутствует атрибут 'fitted_model'.")
    
    residuals = model.fitted_model.resid.dropna()
    
    # Создаем график с подграфиками (2x2)
    fig = make_subplots(
        rows=2, 
        cols=2, 
        subplot_titles=(
            'Остатки',
            'Гистограмма остатков',
            'QQ-график остатков',
            'Остатки vs. Предсказанные значения'
        )
    )
    
    # 1. График остатков во времени
    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals,
            mode='lines',
            name='Остатки',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Добавляем горизонтальную линию y=0
    fig.add_trace(
        go.Scatter(
            x=[residuals.index[0], residuals.index[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y=0'
        ),
        row=1, col=1
    )
    
    # 2. Гистограмма остатков
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Гистограмма',
            marker=dict(color='blue')
        ),
        row=1, col=2
    )
    
    # Добавляем кривую нормального распределения
    x_range = np.linspace(min(residuals), max(residuals), 100)
    mean = np.mean(residuals)
    std = np.std(residuals)
    pdf = stats.norm.pdf(x_range, mean, std)
    
    # Масштабируем PDF к высоте гистограммы
    hist_count, _ = np.histogram(residuals, bins=30)
    scale_factor = max(hist_count) / max(pdf) if max(pdf) > 0 else 1
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=pdf * scale_factor,
            mode='lines',
            name='Нормальное распределение',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # 3. QQ-график
    try:
        qq = stats.probplot(residuals, dist='norm')
        x_values = np.array(qq[0][0])
        y_values = np.array(qq[0][1])
        
        # Добавляем линию y=x (теоретическое нормальное распределение)
        min_val = min(min(x_values), min(y_values))
        max_val = max(max(x_values), max(y_values))
        line_x = np.array([min_val, max_val])
        line_y = np.array([min_val, max_val])
        
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Теоретическая линия',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Добавляем точки QQ-графика
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name='QQ-график',
                marker=dict(color='blue')
            ),
            row=2, col=1
        )
    except Exception as e:
        warnings.warn(f"Ошибка при построении QQ-графика: {str(e)}")
    
    # 4. Остатки vs. Предсказанные значения
    try:
        fitted_values = model.fitted_model.fittedvalues.dropna()
        common_index = residuals.index.intersection(fitted_values.index)
        
        if len(common_index) > 0:
            res_common = residuals.loc[common_index]
            fit_common = fitted_values.loc[common_index]
            
            fig.add_trace(
                go.Scatter(
                    x=fit_common,
                    y=res_common,
                    mode='markers',
                    name='Остатки vs. Предсказанные',
                    marker=dict(color='blue')
                ),
                row=2, col=2
            )
            
            # Добавляем горизонтальную линию y=0
            fig.add_trace(
                go.Scatter(
                    x=[min(fit_common), max(fit_common)],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='y=0'
                ),
                row=2, col=2
            )
        else:
            warnings.warn("Не удалось построить график остатков vs. предсказанные значения из-за несовпадения индексов.")
    except Exception as e:
        warnings.warn(f"Ошибка при построении графика остатков vs. предсказанные значения: {str(e)}")
    
    # Обновляем макет графика
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Диагностика остатков модели"
    )
    
    return fig


def plot_residuals_diagnostic_matplotlib(model: BaseTimeSeriesModel) -> plt.Figure:
    """
    Создает диагностические графики для остатков модели с использованием matplotlib.
    
    Параметры:
    -----------
    model : объект модели с атрибутом fitted_model
        Обученная авторегрессионная модель
        
    Возвращает:
    -----------
    plt.Figure
        Фигура с графиками диагностики остатков для отчетов
    """
    if not hasattr(model, 'fitted_model') or model.fitted_model is None:
        raise ValueError("Модель не обучена. Отсутствует атрибут 'fitted_model'.")
    
    residuals = model.fitted_model.resid.dropna()
    
    # Создаем график с подграфиками (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1. График остатков во времени
    axes[0, 0].plot(residuals.index, residuals, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Остатки')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Гистограмма остатков
    axes[0, 1].hist(residuals, bins=20, color='blue', alpha=0.7)
    axes[0, 1].set_title('Гистограмма остатков')
    
    # Добавляем кривую нормального распределения
    x_range = np.linspace(min(residuals), max(residuals), 100)
    mean = np.mean(residuals)
    std = np.std(residuals)
    pdf = stats.norm.pdf(x_range, mean, std)
    
    # Масштабируем PDF к высоте гистограммы
    hist_count, _ = np.histogram(residuals, bins=20)
    scale_factor = max(hist_count) / max(pdf) if max(pdf) > 0 else 1
    axes[0, 1].plot(x_range, pdf * scale_factor, 'r-', linewidth=2)
    
    # 3. QQ-график
    try:
        qq = stats.probplot(residuals, dist='norm')
        axes[1, 0].plot(qq[0][0], qq[0][1], 'o', color='blue')
        axes[1, 0].plot(qq[0][0], qq[0][0], 'r-')
        axes[1, 0].set_title('QQ-график')
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"Ошибка построения QQ-графика: {str(e)}", 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Остатки vs. Предсказанные значения
    try:
        fitted_values = model.fitted_model.fittedvalues.dropna()
        common_index = residuals.index.intersection(fitted_values.index)
        
        if len(common_index) > 0:
            res_common = residuals.loc[common_index]
            fit_common = fitted_values.loc[common_index]
            
            axes[1, 1].scatter(fit_common, res_common, color='blue', alpha=0.7)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_title('Остатки vs. Предсказанные')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f"Ошибка построения графика: {str(e)}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    return fig


def display_model_information(model: BaseTimeSeriesModel) -> str:
    """
    Формирует строку с информацией о модели.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Модель для описания
        
    Возвращает:
    -----------
    str
        Строка с информацией о модели
    """
    if model is None:
        return "Модель не определена"
    
    params = model.get_params()
    model_type = params.get('type', model.__class__.__name__)
    
    if model_type == 'ARMA':
        p = params.get('p', 0)
        q = params.get('q', 0)
        info = f"ARMA({p}, {q})"
    elif model_type == 'ARIMA':
        p = params.get('p', 0)
        d = params.get('d', 0)
        q = params.get('q', 0)
        info = f"ARIMA({p}, {d}, {q})"
    elif model_type == 'SARIMA':
        p = params.get('p', 0)
        d = params.get('d', 0)
        q = params.get('q', 0)
        P = params.get('P', 0)
        D = params.get('D', 0)
        Q = params.get('Q', 0)
        m = params.get('m', 12)
        info = f"SARIMA({p}, {d}, {q})({P}, {D}, {Q}){m}"
    else:
        info = str(params)
    
    # Информация о обучении
    if model.is_fitted:
        info += " [Обучена]"
    else:
        info += " [Не обучена]"
    
    # Информационные критерии, если доступны
    if hasattr(model, 'fitted_model'):
        if hasattr(model.fitted_model, 'aic'):
            info += f", AIC: {model.fitted_model.aic:.2f}"
        if hasattr(model.fitted_model, 'bic'):
            info += f", BIC: {model.fitted_model.bic:.2f}"
    
    return info


def display_differencing_effect(original_series: pd.Series, differenced_series: pd.Series, 
                               order: str = "Первого", title: str = "Эффект дифференцирования") -> plt.Figure:
    """
    Визуализирует эффект дифференцирования временного ряда.
    
    Параметры:
    -----------
    original_series : pd.Series
        Исходный временной ряд
    differenced_series : pd.Series
        Дифференцированный ряд
    order : str, default="Первого"
        Порядок дифференцирования для отображения (текстовое описание)
    title : str, default="Эффект дифференцирования"
        Заголовок графика
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Строим исходный ряд
    ax1.plot(original_series.index, original_series.values, color='blue')
    ax1.set_title("Исходный временной ряд")
    ax1.set_ylabel("Значение")
    ax1.grid(True, alpha=0.3)
    
    # Строим дифференцированный ряд
    ax2.plot(differenced_series.index, differenced_series.values, color='red')
    ax2.set_title(f"Ряд после дифференцирования {order} порядка")
    ax2.set_xlabel("Время")
    ax2.set_ylabel("Значение")
    ax2.grid(True, alpha=0.3)
    
    # Если индекс - это даты, поворачиваем метки для лучшей читаемости
    if isinstance(original_series.index, pd.DatetimeIndex):
        plt.xticks(rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def analyze_residuals(model: BaseTimeSeriesModel, significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Анализирует остатки модели на соответствие предположениям.
    
    Проверяет:
    1. Отсутствие автокорреляции (тест Льюнга-Бокса)
    2. Нормальность распределения (тест Харке-Бера и Шапиро-Уилка)
    3. Гомоскедастичность (визуальный анализ)
    
    Параметры:
    -----------
    model : объект модели с атрибутом result
        Обученная авторегрессионная модель
    significance_level : float
        Уровень значимости для тестов
        
    Возвращает:
    -----------
    dict
        Словарь с результатами анализа и тестов
    """
    if not hasattr(model, 'fitted_model') or model.fitted_model is None:
        raise ValueError("Модель не обучена. Отсутствует атрибут 'fitted_model'.")
    
    residuals = model.fitted_model.resid
    
    # Удаляем пропущенные значения
    residuals = residuals.dropna()
    
    # 1. Тест на автокорреляцию (Льюнга-Бокса)
    lb_lags = min(10, int(len(residuals) / 5))  # Количество лагов для теста
    if lb_lags < 1:
        lb_lags = 1
        
    try:
        lb_test = acorr_ljungbox(residuals, lags=lb_lags)
        lb_stat = lb_test.iloc[-1, 0]  # Берем статистику для максимального лага
        lb_pvalue = lb_test.iloc[-1, 1]  # Берем p-значение для максимального лага
        lb_result = {
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'no_autocorrelation': lb_pvalue > significance_level
        }
    except Exception as e:
        warnings.warn(f"Ошибка при выполнении теста Льюнга-Бокса: {str(e)}")
        lb_result = {
            'statistic': None,
            'p_value': None,
            'no_autocorrelation': None,
            'error': str(e)
        }
    
    # 2. Тест на нормальность (Харке-Бера)
    try:
        from statsmodels.stats.stattools import jarque_bera
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
        jb_result = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'skewness': skew,
            'kurtosis': kurtosis,
            'is_normal': jb_pvalue > significance_level
        }
    except Exception as e:
        warnings.warn(f"Ошибка при выполнении теста Харке-Бера: {str(e)}")
        jb_result = {
            'statistic': None,
            'p_value': None,
            'skewness': None,
            'kurtosis': None,
            'is_normal': None,
            'error': str(e)
        }
    
    # 3. Тест Шапиро-Уилка на нормальность
    try:
        # Ограничиваем количество точек для теста (максимум 5000 из-за ограничений метода)
        if len(residuals) > 5000:
            test_residuals = residuals.sample(5000)
        else:
            test_residuals = residuals
            
        sw_stat, sw_pvalue = stats.shapiro(test_residuals)
        sw_result = {
            'statistic': sw_stat,
            'p_value': sw_pvalue,
            'is_normal': sw_pvalue > significance_level
        }
    except Exception as e:
        warnings.warn(f"Ошибка при выполнении теста Шапиро-Уилка: {str(e)}")
        sw_result = {
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'error': str(e)
        }
    
    # Базовая статистика остатков
    basic_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'count': len(residuals)
    }
    
    return {
        'basic_stats': basic_stats,
        'ljung_box_test': lb_result,
        'jarque_bera_test': jb_result,
        'shapiro_test': sw_result,
        'residuals': residuals
    }


def display_residuals_analysis(model: BaseTimeSeriesModel) -> None:
    """
    Отображает результаты анализа остатков в Streamlit.
    
    Параметры:
    -----------
    model : объект модели с атрибутом fitted_model
        Обученная авторегрессионная модель
    """
    try:
        # Анализ остатков
        residuals_analysis = analyze_residuals(model)
        
        # Отображаем графики
        st.subheader("Диагностика остатков модели")
        fig = plot_residuals_diagnostic(model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Отображаем результаты тестов
        st.subheader("Статистические тесты")
        
        # Тест Льюнга-Бокса
        st.write("#### 1. Тест Льюнга-Бокса (автокорреляция)")
        lb_test = residuals_analysis['ljung_box_test']
        if lb_test['p_value'] is not None:
            col1, col2 = st.columns(2)
            col1.metric("Статистика", f"{lb_test['statistic']:.4f}")
            col2.metric("p-значение", f"{lb_test['p_value']:.4f}")
            
            if lb_test['no_autocorrelation']:
                st.success("✅ Нет автокорреляции в остатках (p > 0.05). Модель корректно учитывает структуру временного ряда.")
            else:
                st.warning("⚠️ Обнаружена автокорреляция в остатках (p ≤ 0.05). Модель может не полностью учитывать структуру временного ряда.")
        else:
            st.error(f"Ошибка при выполнении теста Льюнга-Бокса: {lb_test.get('error', 'Неизвестная ошибка')}")
        
        # Тест Харке-Бера
        st.write("#### 2. Тест Харке-Бера (нормальность)")
        jb_test = residuals_analysis['jarque_bera_test']
        if jb_test['p_value'] is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Статистика", f"{jb_test['statistic']:.4f}")
            col2.metric("p-значение", f"{jb_test['p_value']:.4f}")
            col3.metric("Асимметрия", f"{jb_test['skewness']:.4f}")
            col4.metric("Эксцесс", f"{jb_test['kurtosis']:.4f}")
            
            if jb_test['is_normal']:
                st.success("✅ Остатки имеют нормальное распределение (p > 0.05). Предположение о нормальности остатков выполняется.")
            else:
                st.warning("⚠️ Остатки не имеют нормального распределения (p ≤ 0.05). Доверительные интервалы могут быть неточными.")
        else:
            st.error(f"Ошибка при выполнении теста Харке-Бера: {jb_test.get('error', 'Неизвестная ошибка')}")
        
        # Тест Шапиро-Уилка
        st.write("#### 3. Тест Шапиро-Уилка (нормальность)")
        sw_test = residuals_analysis['shapiro_test']
        if sw_test['p_value'] is not None:
            col1, col2 = st.columns(2)
            col1.metric("Статистика", f"{sw_test['statistic']:.4f}")
            col2.metric("p-значение", f"{sw_test['p_value']:.4f}")
            
            if sw_test['is_normal']:
                st.success("✅ Остатки имеют нормальное распределение (p > 0.05). Предположение о нормальности остатков выполняется.")
            else:
                st.warning("⚠️ Остатки не имеют нормального распределения (p ≤ 0.05). Доверительные интервалы могут быть неточными.")
        else:
            st.error(f"Ошибка при выполнении теста Шапиро-Уилка: {sw_test.get('error', 'Неизвестная ошибка')}")
        
        # Общая оценка качества модели
        st.subheader("Общая оценка качества модели")
        
        # Оценка автокорреляции
        autocorr_ok = lb_test.get('no_autocorrelation', False)
        
        # Оценка нормальности (хотя бы один из тестов показывает нормальность)
        normality_ok = jb_test.get('is_normal', False) or sw_test.get('is_normal', False)
        
        # Оценка ожидания (среднее близко к нулю)
        mean_ok = abs(residuals_analysis['basic_stats']['mean']) < 0.1 * residuals_analysis['basic_stats']['std']
        
        # Общая оценка
        if autocorr_ok and normality_ok and mean_ok:
            st.success("✅ Модель соответствует всем базовым предположениям. Прогнозы и доверительные интервалы можно считать надежными.")
        elif autocorr_ok and mean_ok:
            st.warning("⚠️ Модель удовлетворительно моделирует временной ряд, но остатки могут не соответствовать нормальному распределению. "
                      "Доверительные интервалы могут быть неточными.")
        elif normality_ok and mean_ok:
            st.warning("⚠️ Остатки имеют нормальное распределение, но присутствует автокорреляция. "
                      "Модель может не полностью учитывать структуру временного ряда.")
        else:
            st.error("❌ Модель не соответствует нескольким базовым предположениям. Рекомендуется пересмотреть спецификацию модели.")
        
    except Exception as e:
        st.error(f"Ошибка при анализе остатков: {str(e)}")


def auto_detect_seasonality(time_series: pd.Series) -> int:
    """
    Автоматически определяет период сезонности временного ряда.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для анализа
        
    Возвращает:
    -----------
    int
        Предполагаемый период сезонности
    """
    # Исключаем пропущенные значения
    time_series = time_series.dropna()
    
    # Проверяем, достаточно ли данных для анализа
    if len(time_series) < 10:
        return 12  # Возвращаем стандартное значение, если данных мало
    
    # Проверяем частоту ряда, если она задана
    if hasattr(time_series.index, 'freq') and time_series.index.freq is not None:
        freq = time_series.index.freq
        freq_str = str(freq)
        
        # Известные частоты и их сезонные периоды
        if 'M' in freq_str:  # Месячная частота
            return 12
        elif 'W' in freq_str:  # Недельная частота
            return 52
        elif 'D' in freq_str:  # Дневная частота
            return 7
        elif 'H' in freq_str:  # Часовая частота
            return 24
        elif 'Q' in freq_str:  # Квартальная частота
            return 4
        elif 'A' in freq_str or 'Y' in freq_str:  # Годовая частота
            return 1
    
    # Если частота не задана, проверяем расстояния между наблюдениями
    if isinstance(time_series.index, pd.DatetimeIndex):
        try:
            # Пытаемся определить частоту из данных
            inferred_freq = pd.infer_freq(time_series.index)
            if inferred_freq:
                if 'M' in inferred_freq:
                    return 12
                elif 'W' in inferred_freq:
                    return 52
                elif 'D' in inferred_freq:
                    return 7
                elif 'H' in inferred_freq:
                    return 24
                elif 'Q' in inferred_freq:
                    return 4
                elif 'A' in inferred_freq or 'Y' in inferred_freq:
                    return 1
        except:
            pass
    
    # Если не удалось определить частоту по индексу, анализируем автокорреляции
    try:
        n = len(time_series)
        max_lag = min(n // 2, 120)  # Максимальный лаг для поиска сезонности
        
        # Рассчитываем автокорреляции
        acf_values, _ = calculate_acf_pacf(time_series, max_lag)
        
        if len(acf_values) < 3:
            return 12  # Не достаточно данных для анализа
        
        # Ищем пики в автокорреляции (не считая лага 0)
        peaks = []
        for i in range(2, len(acf_values) - 1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1] and acf_values[i] > 0.2:
                peaks.append((i, acf_values[i]))
        
        # Сортируем пики по значению автокорреляции
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            # Берем лаг с наибольшей автокорреляцией (исключая первый пик, который может быть артефактом)
            for lag, _ in peaks:
                if lag > 1:  # Исключаем очень короткие лаги
                    # Проверяем, что это действительно сезонный пик
                    if lag in [2, 3, 4, 6, 7, 12, 24, 52]:
                        return lag
            
            # Если не нашли в стандартных периодах, возвращаем первый найденный пик
            return peaks[0][0]
        
        # Если не нашли пиков, проверяем точечно наиболее частые сезонности
        common_seasonalities = [12, 4, 7, 24, 52]
        for s in common_seasonalities:
            if s < len(acf_values) and acf_values[s] > 0.2:
                return s
    except:
        pass
    
    # По умолчанию предполагаем месячные данные
    return 12


def compare_models(models: List[BaseTimeSeriesModel], test_data: pd.Series, 
                  metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Сравнивает несколько моделей на основе их производительности на тестовом наборе.
    
    Параметры:
    -----------
    models : List[BaseTimeSeriesModel]
        Список обученных моделей для сравнения
    test_data : pd.Series
        Тестовый набор данных
    metrics : List[str], optional
        Список метрик для сравнения. Если None, используются все доступные метрики
        
    Возвращает:
    -----------
    pd.DataFrame
        Таблица с метриками для каждой модели
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'smape', 'theil_u2']
    
    # Получаем тренировочные данные из первой модели для расчета метрик
    if not models or not hasattr(models[0], 'train_data'):
        raise ValueError("Модели должны быть обучены и иметь атрибут train_data")
    
    train_data = models[0].train_data
    
    # Формируем DataFrame с результатами
    results = []
    
    for i, model in enumerate(models):
        if not model.is_fitted:
            continue
            
        # Выполняем прогноз
        forecast = model.predict(steps=len(test_data))
        
        # Рассчитываем метрики
        model_metrics = {}
        model_metrics['model_name'] = display_model_information(model)
        
        # Базовые метрики
        if 'mse' in metrics:
            model_metrics['mse'] = mean_squared_error(test_data, forecast)
        
        if 'rmse' in metrics:
            model_metrics['rmse'] = np.sqrt(model_metrics.get('mse', mean_squared_error(test_data, forecast)))
        
        if 'mae' in metrics:
            model_metrics['mae'] = mean_absolute_error(test_data, forecast)
        
        if 'mape' in metrics:
            # Избегаем деления на нуль
            test_nonzero = test_data.copy()
            test_nonzero = test_nonzero.replace(0, 1e-10)
            try:
                model_metrics['mape'] = mean_absolute_percentage_error(test_nonzero, forecast)
            except:
                model_metrics['mape'] = np.nan
        
        if 'smape' in metrics:
            # Симметричная MAPE
            denominator = (np.abs(test_data) + np.abs(forecast)) / 2
            model_metrics['smape'] = np.mean(np.abs(test_data - forecast) / denominator)
        
        if 'theil_u2' in metrics:
            # Коэффициент Тейла-2
            naive_forecast = pd.Series([train_data.iloc[-1]] * len(test_data), index=test_data.index)
            mse_naive = mean_squared_error(test_data, naive_forecast)
            model_metrics['theil_u2'] = model_metrics.get('mse', mean_squared_error(test_data, forecast)) / mse_naive if mse_naive > 0 else float('inf')
        
        results.append(model_metrics)
    
    # Преобразуем в DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df