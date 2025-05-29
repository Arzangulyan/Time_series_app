"""
Страница для работы с авторегрессионными моделями (ARMA, ARIMA, SARIMA).
Предоставляет интерфейс для анализа временных рядов, выбора параметров моделей, обучения и прогнозирования.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import plotly.graph_objects as go
import time
import logging

# Устанавливаем конфигурацию страницы в самом начале (до любых других вызовов st)
st.set_page_config(page_title="Авторегрессионные модели", page_icon="📈", layout="wide")

# Добавляем корневую директорию проекта в путь Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем модули авторегрессионных моделей
from modules.autoregressive.core import (
    check_stationarity, apply_differencing, detect_frequency,
    suggest_arima_params
)
from modules.autoregressive.models import (
    ARMAModel, ARIMAModel, SARIMAModel
)
from modules.autoregressive.visualization import (
    plot_time_series, plot_train_test_split, plot_acf_pacf_plotly,
    plot_forecast, plot_forecast_plotly, plot_forecast_matplotlib, 
    display_model_information, display_differencing_effect,
    auto_detect_seasonality
)
# Импортируем функции выбора и оценки моделей
from modules.autoregressive.model_selection import (
    check_stationarity as check_stationarity_advanced,
    estimate_differencing_order, detect_seasonality,
    auto_arima, evaluate_model_performance, split_train_test,  # Добавляем импорт split_train_test
    plot_model_results, generate_model_report
)

# Импортируем вспомогательные модули из проекта
from modules.page_template import (
    load_time_series,
    run_calculations_on_button_click,
)
import modules.reporting as reporting
from modules.utils import nothing_selected

# Импорт унифицированных графиков
from modules.visualization.unified_plots import (
    create_unified_forecast_plot_plotly,
    create_unified_forecast_plot_matplotlib,
    create_simple_time_series_plot
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Log to stdout so it appears in terminal
)
logger = logging.getLogger('arima_app')

# Инициализация состояния сессии - добавим инициализацию ar_model и ar_results
if 'ar_model' not in st.session_state:
    st.session_state.ar_model = None
if 'ar_results' not in st.session_state:
    st.session_state.ar_results = None
if 'run_future_forecast' not in st.session_state:
    st.session_state.run_future_forecast = False
if 'future_steps' not in st.session_state:
    st.session_state.future_steps = 12
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'current_active_model' not in st.session_state:
    st.session_state.current_active_model = None
if 'last_trained_on' not in st.session_state:
    st.session_state.last_trained_on = None
if 'auto_tuning_experiments' not in st.session_state:
    st.session_state.auto_tuning_experiments = []

def initialize_session_state():
    """Инициализирует переменные состояния сессии."""
    # Модель
    if 'selected_model_type' not in st.session_state:
        st.session_state.selected_model_type = "ARIMA"
    
    # Данные
    if 'time_series' not in st.session_state:
        st.session_state.time_series = None
    
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    
    # Параметры моделей
    if 'arma_params' not in st.session_state:
        st.session_state.arma_params = {'p': 1, 'q': 1}
    
    if 'arima_params' not in st.session_state:
        st.session_state.arima_params = {'p': 1, 'd': 1, 'q': 1}
    
    if 'sarima_params' not in st.session_state:
        st.session_state.sarima_params = {'p': 1, 'd': 1, 'q': 1, 'P': 0, 'D': 0, 'Q': 0, 's': 12}
    
    # Обученные модели
    if 'current_arma_model' not in st.session_state:
        st.session_state.current_arma_model = None
    
    if 'current_arima_model' not in st.session_state:
        st.session_state.current_arima_model = None
    
    if 'current_sarima_model' not in st.session_state:
        st.session_state.current_sarima_model = None


def get_current_model():
    """Возвращает текущую выбранную модель из session_state."""
    model_type = st.session_state.selected_model_type
    
    if model_type == "ARMA":
        return st.session_state.current_arma_model
    elif model_type == "ARIMA":
        return st.session_state.current_arima_model
    elif model_type == "SARIMA":
        return st.session_state.current_sarima_model
    else:
        return None


def set_current_model(model):
    """Устанавливает текущую модель в session_state."""
    model_type = st.session_state.selected_model_type
    
    if model_type == "ARMA":
        st.session_state.current_arma_model = model
    elif model_type == "ARIMA":
        st.session_state.current_arima_model = model
    elif model_type == "SARIMA":
        st.session_state.current_sarima_model = model


def fit_selected_model(data, model_type=None):
    """
    Обучает выбранную модель на данных.
    
    Параметры:
    -----------
    data : pandas.Series
        Временной ряд для обучения
    model_type : str или None
        Тип модели для обучения. Если None, используется выбранный тип из session_state.
        
    Возвращает:
    -----------
    object
        Обученная модель
    """
    if model_type is None:
        model_type = st.session_state.selected_model_type
    
    try:
        if model_type == "ARMA":
            params = st.session_state.arma_params
            model = ARMAModel(p=params['p'], q=params['q'])
        elif model_type == "ARIMA":
            params = st.session_state.arima_params
            model = ARIMAModel(p=params['p'], d=params['d'], q=params['q'])
        elif model_type == "SARIMA":
            params = st.session_state.sarima_params
            model = SARIMAModel(
                p=params['p'], d=params['d'], q=params['q'],
                P=params['P'], D=params['D'], Q=params['Q'], m=params['s']
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        # Обучаем модель с различными методами подгонки в случае ошибки
        try:
            # Сначала пробуем стандартный метод
            model.fit(data)
        except Exception as e:
            if "LU decomposition error" in str(e) or "SVD did not converge" in str(e):
                # Пробуем альтернативный метод 1: с другим солвером
                st.warning("Попытка использовать альтернативный метод подгонки модели (LBFGS)...")
                
                # Создаем новую модель с тем же типом
                if model_type == "ARMA":
                    new_model = ARMAModel(p=params['p'], q=params['q'])
                elif model_type == "ARIMA":
                    new_model = ARIMAModel(p=params['p'], d=params['d'], q=params['q'])
                elif model_type == "SARIMA":
                    new_model = SARIMAModel(
                        p=params['p'], d=params['d'], q=params['q'],
                        P=params['P'], D=params['D'], Q=params['Q'], m=params['s']
                    )
                
                # Добавляем атрибут для передачи альтернативного солвера
                if hasattr(new_model, 'fit_options'):
                    new_model.fit_options = {'method': 'lbfgs', 'maxiter': 50}
                
                try:
                    new_model.fit(data)
                    model = new_model
                except Exception as e2:
                    # Если и второй метод не работает, пробуем упростить модель
                    st.warning("Попытка упростить модель (снизить порядок параметров)...")
                    
                    # Упрощаем параметры
                    if model_type == "ARMA":
                        simple_model = ARMAModel(p=max(1, params['p']-1), q=max(1, params['q']-1))
                    elif model_type == "ARIMA":
                        simple_model = ARIMAModel(p=max(1, params['p']-1), d=params['d'], q=max(1, params['q']-1))
                    elif model_type == "SARIMA":
                        simple_model = SARIMAModel(
                            p=max(1, params['p']-1), d=params['d'], q=max(1, params['q']-1),
                            P=max(0, params['P']-1), D=params['D'], Q=max(0, params['Q']-1), m=params['s']
                        )
                    
                    try:
                        simple_model.fit(data)
                        model = simple_model
                        st.info(f"Использована упрощенная модель с пониженным порядом параметров.")
                    except Exception as e3:
                        # Если все методы не сработали, вызываем исключение
                        raise ValueError(f"Не удалось обучить модель. Попробуйте выбрать другие параметры модели. Ошибка: {str(e3)}")
            else:
                # Если ошибка не связана с декомпозицией, пробрасываем её дальше
                raise e
        
        # Сохраняем модель в session_state
        set_current_model(model)
        
        return model
    
    except Exception as e:
        st.error(f"Ошибка при обучении модели {model_type}: {str(e)}")
        return None


def get_forecast_with_data(model, steps, original_data, title):
    """
    Генерирует прогноз и возвращает как график, так и DataFrame с данными.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    steps : int
        Количество шагов для прогноза
    original_data : pd.Series
        Исходные данные
    title : str
        Заголовок графика
        
    Возвращает:
    -----------
    Tuple[go.Figure, pd.DataFrame]
        Кортеж (график Plotly, DataFrame с прогнозом)
    """
    # Получаем прогноз
    forecast = model.predict(steps=steps)
    
    # Получаем график
    fig = plot_forecast_plotly(model, steps=steps, original_data=original_data, title=title)
    
    # Создаем DataFrame с прогнозными значениями
    forecast_df = pd.DataFrame({'forecast': forecast.values}, index=forecast.index)
    
    # Если доступен доверительный интервал, добавляем его
    if hasattr(model.fitted_model, 'get_forecast'):
        try:
            forecast_obj = model.fitted_model.get_forecast(steps=steps)
            conf_interval = forecast_obj.conf_int(alpha=0.05)  # 95% доверительный интервал
            
            lower_bound = conf_interval.iloc[:, 0]
            upper_bound = conf_interval.iloc[:, 1]
            
            forecast_df['lower_bound'] = lower_bound.values
            forecast_df['upper_bound'] = upper_bound.values
        except Exception as e:
            warnings.warn(f"Не удалось получить доверительный интервал: {str(e)}")
    
    # Возвращаем график и данные
    return fig, forecast_df


# Функция для форматирования MAPE
def format_mape(mape_value):
    if pd.isna(mape_value):
        return "Н/Д"
    if mape_value > 10:  # Если MAPE больше 1000%
        return ">1000%"
    # Округляем до 2 десятичных знаков и добавляем знак %
    return f"{mape_value * 100:.2f}%"


def display_model_metrics(results_dict, model_type_key):
    st.write("### 📊 Метрики качества модели")
    
    # Определяем метрики для отображения
    metrics_to_display = {
        'R²': results_dict.get('r2', 'Н/Д'),
        'Adjusted R²': results_dict.get('adj_r2', 'Н/Д'),
        'MSE': results_dict.get('mse', 'Н/Д'),
        'RMSE': results_dict.get('rmse', 'Н/Д'),
        'MAE': results_dict.get('mae', 'Н/Д'),
        'MAPE': format_mape(results_dict.get('mape', np.nan)),
        'MASE': results_dict.get('mase', 'Н/Д')
    }
    
    # Создаем метрики в три колонки
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{metrics_to_display['R²']:.4f}" if isinstance(metrics_to_display['R²'], (int, float)) else metrics_to_display['R²'])
        st.metric("Adjusted R²", f"{metrics_to_display['Adjusted R²']:.4f}" if isinstance(metrics_to_display['Adjusted R²'], (int, float)) else metrics_to_display['Adjusted R²'])
        st.metric("MSE", f"{metrics_to_display['MSE']:.4f}" if isinstance(metrics_to_display['MSE'], (int, float)) else metrics_to_display['MSE'])
    with col2:
        st.metric("RMSE", f"{metrics_to_display['RMSE']:.4f}" if isinstance(metrics_to_display['RMSE'], (int, float)) else metrics_to_display['RMSE'])
        st.metric("MAE", f"{metrics_to_display['MAE']:.4f}" if isinstance(metrics_to_display['MAE'], (int, float)) else metrics_to_display['MAE'])
    with col3:
        st.metric("MAPE", metrics_to_display['MAPE'])
        st.metric("MASE", f"{metrics_to_display['MASE']:.4f}" if isinstance(metrics_to_display['MASE'], (int, float)) else metrics_to_display['MASE'])


def train_model_and_predict(data, test_data, model_type=None, parameters=None, model_title=None):
    """
    Унифицированная функция для обучения модели и создания прогноза.
    
    Параметры:
    -----------
    data : pandas.Series
        Временной ряд для обучения модели
    test_data : pandas.Series
        Тестовые данные для оценки качества прогноза
    model_type : str или None
        Тип модели для обучения. Если None, используется выбранный тип из session_state
    parameters : dict или None
        Параметры модели. Если None, параметры берутся из session_state
    model_title : str или None
        Заголовок для графика прогноза
        
    Возвращает:
    -----------
    tuple или None
        (модель, метрики, график) если обучение успешно, иначе None
    """
    # Определяем тип модели
    if model_type is None:
        model_type = st.session_state.selected_model_type
    
    # Обучаем модель
    with st.spinner(f"Обучение модели {model_type}..."):
        model = fit_selected_model(data, model_type)
        
        if not model or not model.is_fitted:
            st.error(f"Не удалось обучить модель {model_type}. Проверьте параметры.")
            return None
        
        # Показываем информацию о модели
        st.success(f"Модель {model_type} успешно обучена!")
        display_model_information(model)
        
        # Прогнозирование на тестовый период
        with st.spinner("Создание прогноза на тестовый период..."):
            if not model_title:
                model_title = f"Прогноз модели {model_type} на тестовый период"
                
            # Создаем график прогноза
            forecast_fig = plot_forecast_plotly(
                model,
                steps=len(test_data),
                train_data=data,
                test_data=test_data,
                title=model_title
            )
            
            # Вычисляем метрики
            metrics = evaluate_model_performance(model, data, test_data)
            
            # Возвращаем результаты
            return model, metrics, forecast_fig
    
    return None


def make_future_forecast(model, data, steps, title=None):
    """
    Создает прогноз на будущие периоды.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    data : pandas.Series
        Исходные данные (для графика)
    steps : int
        Количество шагов для прогноза
    title : str или None
        Заголовок для графика прогноза
    
    Возвращает:
    -----------
    tuple или None
        (график, прогноз) если прогнозирование успешно, иначе None
    """
    if not model or not model.is_fitted:
        st.error("Модель не обучена. Сначала обучите модель.")
        return None
    
    with st.spinner("Создание прогноза на будущие периоды..."):
        if not title:
            model_type = st.session_state.selected_model_type
            title = f"Прогноз модели {model_type} на {steps} периодов вперед"
        
        # Создаем прогноз
        future_fig, future_df = get_forecast_with_data(
            model,
            steps=steps,
            original_data=data,
            title=title
        )
        
        return future_fig, future_df


def display_auto_tuning_experiments():
    """Отображает результаты автоматического подбора параметров."""
    experiments_count = len(st.session_state.auto_tuning_experiments) if st.session_state.auto_tuning_experiments else 0
    logger.info(f"Display function called - auto_tuning_experiments count: {experiments_count}")
    
    if not st.session_state.auto_tuning_experiments or len(st.session_state.auto_tuning_experiments) == 0:
        # Debug information when experiments are not available
        experiments_exists = 'auto_tuning_experiments' in st.session_state
        experiments_length = len(st.session_state.auto_tuning_experiments) if experiments_exists else 0
        
        logger.info(f"No experiments to display: exists={experiments_exists}, length={experiments_length}")
        
        if experiments_length == 0 and experiments_exists:
            st.warning("Автоматический подбор завершился, но не удалось сохранить результаты экспериментов. Попробуйте запустить подбор заново.")
        else:
            st.info("Нет доступных экспериментов для отображения. Запустите автоматический подбор параметров для получения результатов.")
        
        if st.button("🔍 Показать состояние session_state (отладка)"):
            debug_info = {
                "auto_tuning_experiments_exists": experiments_exists,
                "auto_tuning_experiments_length": experiments_length,
                "ar_results_exists": 'ar_results' in st.session_state,
                "ar_model_exists": 'ar_model' in st.session_state,
                "session_state_keys": list(st.session_state.keys())
            }
            st.json(debug_info)
            
            # Показать первые несколько экспериментов, если они есть
            if experiments_length > 0:
                st.markdown("**Первый эксперимент (пример):**")
                first_exp = st.session_state.auto_tuning_experiments[0]
                st.json({
                    "model_info": first_exp.get('model_info', 'N/A'),
                    "params": first_exp.get('params', {}),
                    "has_train_metrics": 'train_metrics' in first_exp,
                    "has_test_metrics": 'test_metrics' in first_exp,
                    "rank": first_exp.get('rank', 'N/A')
                })
        return
    
    logger.info("Displaying auto-tuning experiments section")
    st.subheader("🔍 Результаты автоматического подбора параметров")
    
    # Отладочная информация
    st.info(f"🔍 Найдено {len(st.session_state.auto_tuning_experiments)} моделей для анализа")

    # Get criterion info from session_state
    try:
        criterion_info = st.session_state.get('last_info_criterion', 'AIC').upper()
    except:
        criterion_info = "AIC"  # Default fallback
    
    st.info(f"Выбор модели основан на критерии {criterion_info}. Меньшее значение = лучшая модель.")
    st.info(f"Всего протестировано моделей: {len(st.session_state.auto_tuning_experiments)}")
    
    # Create a DataFrame for comparison with enhanced metrics
    models_data = []
    for exp in st.session_state.auto_tuning_experiments:
        # Extract metrics for readability
        train_metrics = exp['train_metrics']
        test_metrics = exp['test_metrics']
        
        # Use total_time if available, otherwise fall back to train_time
        display_time = exp.get('total_time', exp.get('train_time', 0))
        
        model_data = {
            'Модель': exp['model_info'],
            'Ранг': exp['rank'],
            f'{criterion_info}': exp.get('info_criterion', 'Н/Д'),
            
            # Training metrics
            'R² (обуч)': train_metrics.get('r2', 'Н/Д'),
            'Adjusted R²': train_metrics.get('adj_r2', 'Н/Д'),
            'MSE': train_metrics.get('mse', 'Н/Д'),
            'RMSE': train_metrics.get('rmse', 'Н/Д'),
            'MAE': train_metrics.get('mae', 'Н/Д'),
            'MAPE': train_metrics.get('mape', 'Н/Д'),
            'SMAPE': train_metrics.get('smape', 'Н/Д'),
            'MASE': train_metrics.get('mase', 'Н/Д'),
            'Theil U2': train_metrics.get('theil_u2', 'Н/Д'),
            
            # Test metrics
            'R² (тест)': test_metrics.get('r2', 'Н/Д'),
            'Adjusted R² (тест)': test_metrics.get('adj_r2', 'Н/Д'),
            'MSE (тест)': test_metrics.get('mse', 'Н/Д'),
            'RMSE (тест)': test_metrics.get('rmse', 'Н/Д'),
            'MAE (тест)': test_metrics.get('mae', 'Н/Д'),
            'MAPE (тест)': test_metrics.get('mape', 'Н/Д'),
            'SMAPE (тест)': test_metrics.get('smape', 'Н/Д'),
            'MASE (тест)': test_metrics.get('mase', 'Н/Д'),
            'Theil U2 (тест)': test_metrics.get('theil_u2', 'Н/Д'),
            
            'Время (сек)': display_time
        }
        models_data.append(model_data)
        logger.info(f"Added model to comparison table: {model_data['Модель']} with {criterion_info}: {model_data[f'{criterion_info}']}")
    
    # Sort by criterion value (not by rank which might be wrong)
    models_df = pd.DataFrame(models_data)
    models_df[f'{criterion_info}'] = pd.to_numeric(models_df[f'{criterion_info}'], errors='coerce')
    models_df = models_df.sort_values(f'{criterion_info}')
    
    # Re-assign ranks based on sorted criterion values
    models_df['Ранг'] = range(1, len(models_df) + 1)
    
    # Обновляем ранги в исходных экспериментах на основе отсортированной таблицы
    for i, original_idx in enumerate(models_df.index):
        st.session_state.auto_tuning_experiments[original_idx]['rank'] = i + 1
    
    logger.info(f"Created comparison DataFrame with {len(models_df)} rows, sorted by {criterion_info}")
    
    # Add filter options for the table
    st.subheader("Таблица сравнения моделей")
    
    # Create metric display options
    col1, col2 = st.columns(2)
    with col1:
        show_train_metrics = st.checkbox("Показать метрики обучающей выборки", value=True)
    with col2:
        show_test_metrics = st.checkbox("Показать метрики тестовой выборки", value=True)
    
    # Filter columns based on user selection
    display_columns = ['Модель', 'Ранг', f'{criterion_info}']
    
    if show_train_metrics:
        train_metric_cols = [col for col in models_df.columns if '(обуч)' in col]
        display_columns.extend(train_metric_cols)
    
    if show_test_metrics:
        test_metric_cols = [col for col in models_df.columns if '(тест)' in col]
        display_columns.extend(test_metric_cols)
    
    display_columns.append('Время (сек)')
    
    # Display the filtered comparison table
    st.dataframe(models_df[display_columns], use_container_width=True)
    
    # Allow user to download the complete comparison table
    csv = models_df.to_csv(index=False)
    st.download_button(
        label="Скачать полную таблицу сравнения (CSV)",
        data=csv,
        file_name=f"arima_models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    logger.info("Displayed comparison table")
    
    # Detailed view of experiments with tabs
    if len(models_df) > 0:
        st.subheader("Подробная информация о всех протестированных моделях")
        
        # Get ALL models based on the sorted DataFrame order (исправляем нумерацию)
        sorted_experiments = []
        for i, original_idx in enumerate(models_df.index):
            exp = st.session_state.auto_tuning_experiments[original_idx].copy()
            exp['rank'] = i + 1  # Устанавливаем правильный ранг
            sorted_experiments.append(exp)
        
        # Create tabs for ALL experiments with правильной нумерацией
        experiment_tabs = st.tabs([f"#{i+1}: {exp['model_info']}" 
                                 for i, exp in enumerate(sorted_experiments)])
        
        # Fill each tab with details
        for i, tab in enumerate(experiment_tabs):
            if i < len(sorted_experiments):
                exp = sorted_experiments[i]
                with tab:
                    # Show rank and criterion value prominently
                    criterion_value = exp.get('info_criterion', 'Н/Д')
                    formatted_criterion = f"{criterion_value:.4f}" if isinstance(criterion_value, (int, float)) else str(criterion_value)
                    
                    st.markdown(f"### Модель #{i+1}: {exp['model_info']}")
                    st.markdown(f"**{criterion_info}**: {formatted_criterion}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Обучающая выборка:**")
                        metrics_train = exp['train_metrics']
                        st.metric("RMSE", f"{metrics_train.get('rmse', 'Н/Д'):.4f}")
                        st.metric("MAE", f"{metrics_train.get('mae', 'Н/Д'):.4f}")
                        st.metric("MAPE", f"{metrics_train.get('mape', 'Н/Д'):.4f}" if 'mape' in metrics_train else "Н/Д")
                        st.metric("SMAPE", f"{metrics_train.get('smape', 'Н/Д'):.4f}" if 'smape' in metrics_train else "Н/Д")
                        st.metric("MASE", f"{metrics_train.get('mase', 'Н/Д'):.4f}" if 'mase' in metrics_train else "Н/Д")
                        st.metric("R²", f"{metrics_train.get('r2', np.nan):.4f}")
                        st.metric("Adjusted R²", f"{metrics_train.get('adj_r2', np.nan):.4f}")
                        
                    with col2:
                        st.markdown("**Тестовая выборка:**")
                        metrics_test = exp['test_metrics']
                        st.metric("RMSE", f"{metrics_test.get('rmse', 'Н/Д'):.4f}")
                        st.metric("MAE", f"{metrics_test.get('mae', 'Н/Д'):.4f}")
                        st.metric("MAPE", f"{metrics_test.get('mape', 'Н/Д'):.4f}" if 'mape' in metrics_test else "Н/Д")
                        st.metric("SMAPE", f"{metrics_test.get('smape', 'Н/Д'):.4f}" if 'smape' in metrics_test else "Н/Д")
                        st.metric("MASE", f"{metrics_test.get('mase', 'Н/Д'):.4f}" if 'mase' in metrics_test else "Н/Д")
                        st.metric("R²", f"{metrics_test.get('r2', np.nan):.4f}")
                        st.metric("Adjusted R²", f"{metrics_test.get('adj_r2', np.nan):.4f}")
                    
                    # Generate forecast figure for this experiment model
                    try:
                        if st.session_state.ar_results and "original_series" in st.session_state.ar_results and "train" in st.session_state.ar_results and "test" in st.session_state.ar_results:
                            # Create forecast figure for this experiment model
                            forecast_fig = plot_forecast_plotly(
                                model=exp['model'],
                                steps=len(st.session_state.ar_results['test']),
                                original_data=st.session_state.ar_results['original_series'],
                                train_data=st.session_state.ar_results['train'],
                                test_data=st.session_state.ar_results['test'],
                                title=f"Прогноз модели #{i+1}: {exp['model_info']}"
                            )
                            st.plotly_chart(forecast_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось отобразить прогноз для модели: {str(e)}")
                    
                    # Add download report button for this model
                    try:
                        # Create forecast figure for this experiment model
                        forecast_fig = plot_forecast_matplotlib(
                            model=exp['model'],
                            steps=len(st.session_state.ar_results['test']),
                            original_data=st.session_state.ar_results['original_series'],
                            train_data=st.session_state.ar_results['train'],
                            test_data=st.session_state.ar_results['test'],
                            title=f"Прогноз модели #{i+1}: {exp['model_info']}"
                        )
                        forecast_img_base64 = reporting.save_plot_to_base64(forecast_fig, backend='matplotlib')
                        
                        # Create empty loss figure
                        loss_fig, ax = plt.subplots(figsize=(8, 4))
                        ax.text(0.5, 0.5, "Для авторегрессионных моделей график потерь не применим", 
                            ha='center', va='center', fontsize=12)
                        ax.set_axis_off()
                        loss_img_base64 = reporting.save_plot_to_base64(loss_fig, backend='matplotlib')
                        
                        # Generate report
                        md_report = reporting.generate_markdown_report(
                            title=f"Отчет по модели #{i+1}: {exp['model_info']}",
                            description=f"Авторегрессионная модель {exp['model_info']} из автоматического подбора параметров (ранг {i+1}).",
                            metrics_train=exp['train_metrics'],
                            metrics_test=exp['test_metrics'],
                            train_time=exp.get('train_time', 0),
                            forecast_img_base64=forecast_img_base64,
                            loss_img_base64=loss_img_base64,
                            params=exp['params'],
                            early_stopping=False,
                            early_stopping_epoch=None
                        )
                        
                        # Generate PDF
                        pdf_bytes = None
                        try:
                            pdf_bytes = reporting.markdown_to_pdf(md_report)
                        except Exception as e:
                            st.warning(f"Не удалось сгенерировать PDF: {e}")
                        
                        # Add download buttons
                        st.markdown("### Скачать отчет по этой модели")
                        reporting.download_report_buttons(
                            md_report, 
                            pdf_bytes, 
                            md_filename=f"arima_model_{i+1}_report.md", 
                            pdf_filename=f"arima_model_{i+1}_report.pdf"
                        )
                    except Exception as e:
                        st.error(f"Не удалось создать отчет по эксперименту: {str(e)}")

        # Add consolidated report option
        st.subheader("Сводный отчет по автоматическому подбору")
        if st.button("Сгенерировать сводный отчет"):
            try:
                # Create a consolidated report with all experiments
                consolidated_md = "# Сводный отчет по автоматическому подбору параметров\n\n"
                consolidated_md += f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                consolidated_md += "## Сравнение моделей\n\n"
                
                # Enhanced comparison table with more metrics
                table_md = "| Ранг | Модель | " + criterion_info + " | R² (тест) | Adj R² (тест) | RMSE (тест) | MAE (тест) | MAPE (тест) | SMAPE (тест) | MASE (тест) | Theil U2 (тест) |\n"
                table_md += "|------|--------|" + "-" * len(criterion_info) + "|----------|-------------|------------|------------|-------------|--------------|------------|----------------|\n"
                
                # Sort experiments by rank for the table
                sorted_experiments = sorted(st.session_state.auto_tuning_experiments, key=lambda x: x['rank'])
                
                for exp in sorted_experiments:
                    metrics = exp['test_metrics']
                    criterion_val = exp.get('info_criterion', 'Н/Д')
                    
                    # Format values correctly by evaluating conditional expressions outside the f-string
                    formatted_criterion = f"{criterion_val:.4f}" if isinstance(criterion_val, (int, float)) else str(criterion_val)
                    formatted_r2 = f"{metrics.get('r2', 'Н/Д'):.4f}" if isinstance(metrics.get('r2', 'Н/Д'), (int, float)) else str(metrics.get('r2', 'Н/Д'))
                    formatted_adj_r2 = f"{metrics.get('adj_r2', 'Н/Д'):.4f}" if isinstance(metrics.get('adj_r2', 'Н/Д'), (int, float)) else str(metrics.get('adj_r2', 'Н/Д'))
                    formatted_rmse = f"{metrics.get('rmse', 'Н/Д'):.4f}" if isinstance(metrics.get('rmse', 'Н/Д'), (int, float)) else str(metrics.get('rmse', 'Н/Д'))
                    formatted_mae = f"{metrics.get('mae', 'Н/Д'):.4f}" if isinstance(metrics.get('mae', 'Н/Д'), (int, float)) else str(metrics.get('mae', 'Н/Д'))
                    formatted_mape = f"{metrics.get('mape', 'Н/Д'):.4f}" if isinstance(metrics.get('mape', 'Н/Д'), (int, float)) else str(metrics.get('mape', 'Н/Д'))
                    formatted_smape = f"{metrics.get('smape', 'Н/Д'):.4f}" if isinstance(metrics.get('smape', 'Н/Д'), (int, float)) else str(metrics.get('smape', 'Н/Д'))
                    formatted_mase = f"{metrics.get('mase', 'Н/Д'):.4f}" if isinstance(metrics.get('mase', 'Н/Д'), (int, float)) else str(metrics.get('mase', 'Н/Д'))
                    formatted_theil = f"{metrics.get('theil_u2', 'Н/Д'):.4f}" if isinstance(metrics.get('theil_u2', 'Н/Д'), (int, float)) else str(metrics.get('theil_u2', 'Н/Д'))
                    
                    # Add row to table with proper formatting
                    table_md += f"| {exp['rank']} | {exp['model_info']} | {formatted_criterion} | "
                    table_md += f"{formatted_r2} | {formatted_adj_r2} | {formatted_rmse} | "
                    table_md += f"{formatted_mae} | {formatted_mape} | {formatted_smape} | "
                    table_md += f"{formatted_mase} | {formatted_theil} |\n"
                
                consolidated_md += table_md + "\n\n"
                
                # Add details for each model (all models, not just top 5)
                for i, exp in enumerate(sorted_experiments):
                    consolidated_md += f"## Модель #{i+1}: {exp['model_info']}\n\n"
                    
                    # Format parameters
                    params_text = "\n".join([f"- **{k}**: {v}" for k, v in exp['params'].items()])
                    consolidated_md += f"### Параметры\n{params_text}\n\n"
                    
                    # Add timing information
                    train_time = exp.get('train_time', 0)
                    metrics_time = exp.get('metrics_time', 0)
                    total_time = exp.get('total_time', train_time)
                    
                    consolidated_md += f"### Время выполнения\n"
                    consolidated_md += f"- **Обучение модели**: {train_time:.4f} сек.\n"
                    if metrics_time > 0:
                        consolidated_md += f"- **Расчет метрик**: {metrics_time:.4f} сек.\n"
                    consolidated_md += f"- **Общее время**: {total_time:.4f} сек.\n\n"
                    
                    # Format metrics with all available metrics
                    consolidated_md += "### Метрики\n\n"
                    consolidated_md += "**Обучающая выборка:**\n"
                    train_metrics = exp['train_metrics']
                    for metric_name, nice_name in [
                        ('r2', 'R²'), ('adj_r2', 'Adjusted R²'), ('mse', 'MSE'),
                        ('rmse', 'RMSE'), ('mae', 'MAE'), ('mape', 'MAPE'),
                        ('smape', 'SMAPE'), ('mase', 'MASE'), ('theil_u2', 'Theil U2')
                    ]:
                        value = train_metrics.get(metric_name, 'Н/Д')
                        formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                        consolidated_md += f"- **{nice_name}**: {formatted_value}\n"
                    
                    consolidated_md += "\n"
                    
                    # Test metrics with proper formatting
                    consolidated_md += "**Тестовая выборка:**\n"
                    test_metrics = exp['test_metrics']
                    
                    # Apply proper formatting for each metric
                    for metric_name, nice_name in [
                        ('r2', 'R²'), ('adj_r2', 'Adjusted R²'), ('mse', 'MSE'),
                        ('rmse', 'RMSE'), ('mae', 'MAE'), ('mape', 'MAPE'),
                        ('smape', 'SMAPE'), ('mase', 'MASE'), ('theil_u2', 'Theil U2')
                    ]:
                        value = test_metrics.get(metric_name, 'Н/Д')
                        formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                        consolidated_md += f"- **{nice_name}**: {formatted_value}\n"
                    
                    consolidated_md += "\n"
                    
                    # Add separator
                    consolidated_md += "---\n\n"
                
                # Generate and offer consolidated report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Скачать сводный отчет (.md)",
                    data=consolidated_md,
                    file_name=f"arima_autotuning_report_{timestamp}.md",
                    mime="text/markdown"
                )
                
                # Try to generate PDF
                try:
                    pdf_bytes = reporting.markdown_to_pdf(consolidated_md)
                    st.download_button(
                        label="Скачать сводный отчет (.pdf)",
                        data=pdf_bytes,
                        file_name=f"arima_autotuning_report_{timestamp}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.warning(f"Не удалось сгенерировать PDF для сводного отчета: {e}")
                
            except Exception as e:
                st.error(f"Ошибка при создании сводного отчета: {str(e)}")

def main():
    # Заголовок
    st.title("Авторегрессионные модели (ARIMA/SARIMA)")
    
    # Загрузка данных
    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return
    
    # Отображаем исходный ряд с унифицированным графиком
    st.subheader("Исходный временной ряд")
    ts_series = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series
    fig = create_simple_time_series_plot(ts_series, title="Исходный временной ряд")
    st.plotly_chart(fig, use_container_width=True)
    
    # Добавляем анализ АКФ и ЧАКФ
    st.subheader("Анализ автокорреляции")
    
    with st.expander("О функциях автокорреляции", expanded=False):
        st.markdown("""
        **АКФ (Автокорреляционная функция)** показывает корреляцию между значениями ряда, сдвинутыми на различное количество периодов (лагов). Помогает определить:
        - Сезонность в данных
        - Подходящий порядок MA (q) для ARIMA модели
        
        **ЧАКФ (Частная автокорреляционная функция)** показывает корреляцию между значениями после исключения влияния промежуточных лагов. Помогает определить:
        - Подходящий порядок AR (p) для ARIMA модели
        - Наличие прямых зависимостей между отдаленными периодами
        
        **Интерпретация:**
        - Значения внутри доверительных интервалов (голубая область) считаются незначимыми
        - Выход за пределы интервалов указывает на значимую корреляцию
        - Постепенное затухание в АКФ указывает на нестационарность
        - Резкий обрыв в ЧАКФ указывает на порядок AR процесса
        """)
    
    # Параметры для АКФ и ЧАКФ
    col1, col2 = st.columns(2)
    with col1:
        max_lags_acf = st.slider(
            "Максимальное количество лагов для АКФ", 
            min_value=10, 
            max_value=min(100, len(ts_series)//4), 
            value=min(40, len(ts_series)//10),
            help="Большее количество лагов дает более полную картину корреляций"
        )
    with col2:
        max_lags_pacf = st.slider(
            "Максимальное количество лагов для ЧАКФ", 
            min_value=10, 
            max_value=min(100, len(ts_series)//4), 
            value=min(40, len(ts_series)//10),
            help="Большее количество лагов помогает лучше определить порядок AR"
        )
    
    try:
        # Строим АКФ и ЧАКФ - используем стандартные параметры функции
        acf_pacf_fig = plot_acf_pacf_plotly(
            ts_series, 
            title="Функции автокорреляции временного ряда"
        )
        st.plotly_chart(acf_pacf_fig, use_container_width=True)
        
        # Автоматические рекомендации на основе АКФ/ЧАКФ
        with st.expander("Автоматические рекомендации параметров", expanded=False):
            try:
                suggested_params = suggest_arima_params(ts_series)
                st.markdown("**Рекомендуемые параметры ARIMA на основе анализа АКФ/ЧАКФ:**")
                st.json(suggested_params)
                st.info("Эти параметры основаны на анализе автокорреляций. Используйте их как отправную точку для ручной настройки.")
            except Exception as e:
                st.warning(f"Не удалось получить автоматические рекомендации: {str(e)}")
    
    except Exception as e:
        st.error(f"Ошибка при построении функций автокорреляции: {str(e)}")

    # Боковая панель с параметрами
    st.sidebar.subheader("Настройки модели")
    
    # Вкладки для различных режимов
    tabs = st.sidebar.tabs(["Автоматический подбор", "Ручная настройка"])
    
    with tabs[0]:
        st.header("Автоматический подбор параметров")
        
        train_size = st.slider(
            "Размер обучающей выборки", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.8, 
            step=0.05,
            help="Доля данных для обучения (остальное - для тестирования)"
        )
        
        seasonal = st.checkbox("Учитывать сезонность", value=True)
        
        # Настройки для автоматического подбора
        with st.expander("Расширенные настройки", expanded=False):
            info_criterion = st.selectbox(
                "Информационный критерий", 
                options=["aic", "bic", "aicc", "oob"],
                index=0,
                help="Критерий для выбора оптимальной модели"
            )
            
            max_p = st.slider("Максимальный порядок AR (p)", 0, 10, 5)  
            max_d = st.slider("Максимальный порядок дифференцирования (d)", 0, 3, 2)  
            max_q = st.slider("Максимальный порядок MA (q)", 0, 10, 5)  
            
            if seasonal:
                max_P = st.slider("Максимальный сезонный порядок AR (P)", 0, 3, 2, key="auto_max_P")  
                max_D = st.slider("Максимальный сезонный порядок дифференцирования (D)", 0, 2, 1, key="auto_max_D")  
                max_Q = st.slider("Максимальный сезонный порядок MA (Q)", 0, 3, 2, key="auto_max_Q")  
                m = st.slider("Сезонный период (m)", 2, 365, 24, key="auto_seasonal_period")
        
        forecast_steps = st.slider(
            "Шаги прогноза вперед", 
            min_value=0, 
            max_value=100, 
            value=10, 
            step=5,
            help="Количество периодов для прогноза в будущее"
        )
        
        auto_button = st.button("Запустить автоматический подбор")
        
        if auto_button:
            with st.spinner("Выполняется автоматический подбор параметров..."):
                try:
                    # Очистка предыдущих экспериментов
                    logger.info("Clearing previous experiments")
                    st.session_state.auto_tuning_experiments = []
                    
                    # Отслеживаем время выполнения
                    start_time = time.perf_counter()
                    
                    # Подготовка данных
                    ts_series = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series
                    train, test = split_train_test(ts_series, train_size)
                    
                    # Параметры для auto_arima
                    auto_params = {
                        'seasonal': seasonal,
                        'information_criterion': info_criterion,
                        'max_p': max_p,
                        'max_d': max_d,
                        'max_q': max_q,
                        'return_all_models': True,  
                        'verbose': True,  
                    }
                    
                    if seasonal:
                        auto_params.update({
                            'max_P': max_P,
                            'max_D': max_D,
                            'max_Q': max_Q,
                            'm': m
                        })
                    
                    # Сохраняем последний использованный критерий в session_state для использования позже
                    st.session_state.last_info_criterion = info_criterion
                    
                    # Запускаем автоматический подбор
                    logger.info(f"Starting auto_arima with params: {auto_params}")
                    auto_results = auto_arima(train, **auto_params)
                    logger.info(f"auto_arima completed, keys in result: {list(auto_results.keys())}")
                    
                    # Теперь auto_results - это словарь с 'best_model' и, возможно, другими моделями
                    model = auto_results['best_model']
                    
                    # Сохраняем все протестированные модели в состоянии сессии, если они доступны
                    if 'all_models' in auto_results:
                        logger.info(f"Found {len(auto_results['all_models'])} models in auto_results['all_models']")
                        
                        # Добавляем информацию о критерии
                        criterion_used = auto_results.get('criterion_used', info_criterion).upper()
                        st.info(f"Модели ранжированы по критерию {criterion_used}. Меньше значение = лучшая модель.")
                        
                        # Очищаем список экспериментов перед добавлением новых
                        experiments_list = []
                        logger.info("Creating new experiments list")
                        
                        for i, model_info in enumerate(auto_results['all_models']):
                            logger.info(f"Processing model {i+1}/{len(auto_results['all_models'])}")
                            
                            try:
                                experiment_model = model_info['model']
                                experiment_params = experiment_model.get_params()
                                
                                # Получаем время обучения из model_info, обеспечивая минимальное ненулевое значение
                                fit_time = model_info.get('fit_time', None)
                                
                                # Если время отсутствует или равно 0, установим минимальное значение
                                if fit_time is None or fit_time < 0.001:
                                    logger.warning(f"Model {i+1} ({experiment_model.__class__.__name__}) has no fit_time recorded, using a default minimal value")
                                    fit_time = 0.001  # 1 миллисекунда как минимальное время
                                
                                # Вычисляем метрики для этого эксперимента
                                logger.info(f"Calculating metrics for model {i+1}")
                                
                                # Засекаем дополнительное время для расчета метрик
                                metrics_start_time = time.perf_counter()
                                
                                train_pred = experiment_model.predict_in_sample()
                                test_pred = experiment_model.predict(steps=len(test))
                                train_metrics = evaluate_model_performance(experiment_model, train, train)
                                test_metrics = evaluate_model_performance(experiment_model, train, test)
                                
                                # Дополнительное время на расчет метрик и прогнозов
                                metrics_time = time.perf_counter() - metrics_start_time
                                
                                # Создаем запись эксперимента
                                experiment = {
                                    'model': experiment_model,
                                    'model_info': display_model_information(experiment_model),
                                    'params': experiment_params,
                                    'train_metrics': train_metrics,
                                    'test_metrics': test_metrics,
                                    'info_criterion': model_info.get('criterion_value', None),
                                    'train_time': fit_time,  
                                    'metrics_time': metrics_time,  
                                    'total_time': fit_time + metrics_time,  
                                    'rank': i + 1  
                                }
                                
                                experiments_list.append(experiment)
                                logger.info(f"Successfully added experiment {i+1}: {experiment['model_info']} to list. Total experiments now: {len(experiments_list)}")
                                
                            except Exception as exp_e:
                                logger.error(f"Error processing model {i+1}: {str(exp_e)}", exc_info=True)
                                st.warning(f"Не удалось обработать модель {i+1}: {str(exp_e)}")
                        
                        # Присваиваем сразу всю коллекцию
                        st.session_state.auto_tuning_experiments = experiments_list
                        logger.info(f"Finished processing all models. Total experiments saved to session_state: {len(st.session_state.auto_tuning_experiments)}")
                        
                        # Принудительно обновляем состояние
                        if len(st.session_state.auto_tuning_experiments) > 0:
                            logger.info("Auto-tuning experiments successfully saved to session_state")
                        else:
                            logger.error("Failed to save any experiments to session_state")
                            
                    else:
                        logger.warning("No 'all_models' key found in auto_results")
                    
                    # Рассчитываем время выполнения
                    train_time = time.perf_counter() - start_time
                    
                    # Вычисляем метрики
                    train_pred = model.predict_in_sample()
                    test_pred = model.predict(steps=len(test))
                    
                    train_metrics = evaluate_model_performance(model, train, train)
                    test_metrics = evaluate_model_performance(model, train, test)
                    
                    # Сохраняем результаты в состоянии сессии
                    st.session_state.ar_model = model
                    st.session_state.ar_results = {
                        'train': train,
                        'test': test,
                        'train_predictions': train_pred,
                        'test_predictions': test_pred,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'original_series': ts_series,
                        'train_time': train_time,
                        'model_info': display_model_information(model),
                        'params': model.get_params()
                    }
                    
                    st.success(f"Автоматический подбор завершен! Выбрана модель: {display_model_information(model)}")
                    logger.info(f"Auto-tuning completed with {len(st.session_state.auto_tuning_experiments)} experiments")
                    
                except Exception as e:
                    logger.error(f"Error in auto-tuning: {str(e)}", exc_info=True)
                    st.error(f"Ошибка при автоматическом подборе: {str(e)}")
    
    # Отображение результатов автоматического подбора - вызываем отдельную функцию
    display_auto_tuning_experiments()

    with tabs[1]:
        st.header("Ручная настройка параметров")
        
        train_size = st.slider(
            "Размер обучающей выборки", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.8, 
            step=0.05,
            key="manual_train_size",
            help="Доля данных для обучения (остальное - для тестирования)"
        )
        
        model_type = st.radio("Тип модели", ["ARIMA", "SARIMA"])
        
        # Параметры ARIMA
        p = st.slider("Порядок AR (p)", 0, 5, 1, key="manual_p")
        d = st.slider("Порядок дифференцирования (d)", 0, 2, 1, key="manual_d")
        q = st.slider("Порядок MA (q)", 0, 5, 1, key="manual_q")
        
        # Дополнительные параметры для SARIMA
        if model_type == "SARIMA":
            P = st.slider("Сезонный порядок AR (P)", 0, 2, 1, key="manual_P")
            D = st.slider("Сезонный порядок дифференцирования (D)", 0, 1, 1, key="manual_D")
            Q = st.slider("Сезонный порядок MA (Q)", 0, 2, 1, key="manual_Q")
            m = st.slider("Сезонный период (m)", 2, 365, 12, key="manual_seasonal_period")
        
        forecast_steps = st.slider(
            "Шаги прогноза вперед", 
            min_value=0, 
            max_value=100, 
            value=10, 
            step=5,
            key="manual_forecast_steps",
            help="Количество периодов для прогноза в будущее"
        )
        
        manual_button = st.button("Запустить обучение модели")
        
        if manual_button:
            with st.spinner("Выполняется обучение модели..."):
                try:
                    # Отслеживаем время выполнения
                    start_time = time.perf_counter()
                    
                    # Подготовка данных
                    ts_series = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series
                    train, test = split_train_test(ts_series, train_size)
                    
                    # Создаем модель
                    if model_type == "ARIMA":
                        model = ARIMAModel(p=p, d=d, q=q)
                    else:  # SARIMA
                        model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m)
                    
                    # Обучаем модель
                    model.fit(train)
                    
                    # Рассчитываем время выполнения
                    train_time = time.perf_counter() - start_time
                    
                    # Вычисляем метрики
                    train_pred = model.predict_in_sample()
                    test_pred = model.predict(steps=len(test))
                    
                    train_metrics = evaluate_model_performance(model, train, train)
                    test_metrics = evaluate_model_performance(model, train, test)
                    
                    # Сохраняем результаты в состоянии сессии
                    st.session_state.ar_model = model
                    st.session_state.ar_results = {
                        'train': train,
                        'test': test,
                        'train_predictions': train_pred,
                        'test_predictions': test_pred,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'original_series': ts_series,
                        'train_time': train_time,
                        'model_info': display_model_information(model),
                        'params': model.get_params()
                    }
                    
                    st.success(f"Модель успешно обучена: {display_model_information(model)}")
                    
                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {str(e)}")
    
    # Секция выбора модели для отображения результатов
    if st.session_state.auto_tuning_experiments and len(st.session_state.auto_tuning_experiments) > 0:
        st.subheader("🎯 Выбор модели для детального анализа")
        
        # Создаем список моделей для выбора, используя отсортированный порядок
        model_options = []
        # Получаем отсортированные эксперименты
        sorted_df = pd.DataFrame([{'original_idx': i, 'criterion': exp.get('info_criterion', float('inf'))} 
                                 for i, exp in enumerate(st.session_state.auto_tuning_experiments)])
        sorted_df = sorted_df.sort_values('criterion')
        
        for display_rank, row in enumerate(sorted_df.itertuples(), 1):
            original_idx = row.original_idx
            exp = st.session_state.auto_tuning_experiments[original_idx]
            criterion_value = exp.get('info_criterion', 'Н/Д')
            criterion_str = f"{criterion_value:.4f}" if isinstance(criterion_value, (int, float)) else str(criterion_value)
            
            # Получаем критерий из session_state
            criterion_name = st.session_state.get('last_info_criterion', 'AIC').upper();
            
            model_options.append(f"#{display_rank}: {exp['model_info']} ({criterion_name}: {criterion_str})")
        
        # Селектбокс для выбора модели
        selected_model_index = st.selectbox(
            "Выберите модель для детального анализа:",
            range(len(model_options)),
            format_func=lambda x: model_options[x],
            index=0,  # По умолчанию выбрана лучшая модель
            help="Выберите модель из списка экспериментов для отображения подробных результатов внизу страницы"
        )
        
        # Кнопка для применения выбора
        if st.button("📊 Показать результаты выбранной модели"):
            try:
                # Получаем выбранный эксперимент по отсортированному индексу
                sorted_original_idx = int(sorted_df.iloc[selected_model_index]['original_idx'])  # Преобразуем в int
                selected_exp = st.session_state.auto_tuning_experiments[sorted_original_idx]
                selected_model = selected_exp['model']
                
                # Пересоздаем данные для выбранной модели
                if st.session_state.ar_results:
                    # Используем те же данные разбиения, что и в автоматическом подборе
                    train = st.session_state.ar_results['train']
                    test = st.session_state.ar_results['test']
                    ts_series = st.session_state.ar_results['original_series']
                    
                    # Получаем прогнозы выбранной модели
                    train_pred = selected_model.predict_in_sample()
                    test_pred = selected_model.predict(steps=len(test))
                    
                    # Обновляем результаты в session_state
                    st.session_state.ar_model = selected_model
                    st.session_state.ar_results.update({
                        'train_predictions': train_pred,
                        'test_predictions': test_pred,
                        'train_metrics': selected_exp['train_metrics'],
                        'test_metrics': selected_exp['test_metrics'],
                        'model_info': selected_exp['model_info'],
                        'params': selected_exp['params'],
                        'train_time': selected_exp.get('train_time', 0)
                    })
                    
                    st.success(f"✅ Выбрана модель: {selected_exp['model_info']}")
                    st.info("Результаты обновлены. Прокрутите страницу вниз, чтобы увидеть детальный анализ выбранной модели.")
                    
                    # Принудительный rerun для обновления отображения
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Ошибка при выборе модели: {str(e)}")
        
        # Информация о текущей выбранной модели
        if st.session_state.ar_model is not None:
            current_model_info = display_model_information(st.session_state.ar_model)
            st.info(f"🔍 Текущая модель для анализа: {current_model_info}")

    # Отображение результатов, если они есть
    if st.session_state.ar_results is not None:
        results = st.session_state.ar_results
        
        # Отображение метрик
        st.subheader("Метрики качества прогноза")
        
        # Вывод времени обучения
        st.caption(f"Время обучения модели: {results['train_time']:.2f} сек.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Обучающая выборка:**")
            metrics_train = results['train_metrics']
            st.metric("RMSE", f"{metrics_train['rmse']:.4f}")
            st.metric("MAE", f"{metrics_train['mae']:.4f}")
            st.metric("MAPE", f"{metrics_train.get('mape', np.nan):.4f}")
            st.metric("SMAPE", f"{metrics_train.get('smape', np.nan):.4f}")
            st.metric("MASE", f"{metrics_train.get('mase', np.nan):.4f}")
            st.metric("R²", f"{metrics_train.get('r2', np.nan):.4f}")
            st.metric("Adjusted R²", f"{metrics_train.get('adj_r2', np.nan):.4f}")
            
            with st.expander("Что означают эти метрики?"):
                st.markdown("""
                - **RMSE** (Root Mean Squared Error) - среднеквадратичная ошибка. Показывает среднюю величину ошибки прогноза в тех же единицах измерения, что и данные.
                - **MAE** (Mean Absolute Error) - средняя абсолютная ошибка. Более устойчива к выбросам, чем RMSE.
                - **MAPE** (Mean Absolute Percentage Error) - средняя абсолютная процентная ошибка. Показывает ошибку в процентах.
                - **SMAPE** (Symmetric Mean Absolute Percentage Error) - симметричная средняя абсолютная процентная ошибка. Более надежна, чем MAPE, при значениях близких к нулю.
                - **MASE** (Mean Absolute Scaled Error) - масштабированная средняя абсолютная ошибка. Значения < 1 означают, что модель лучше наивного прогноза.
                - **R²** - коэффициент детерминации. Показывает долю дисперсии, объясненную моделью (1.0 - идеальный прогноз).
                - **Adjusted R²** - скорректированный R². Учитывает сложность модели, помогает выявить переобучение.
                """)
        
        with col2:
            st.markdown("**Тестовая выборка:**")
            metrics_test = results['test_metrics']
            st.metric("RMSE", f"{metrics_test['rmse']:.4f}")
            st.metric("MAE", f"{metrics_test['mae']:.4f}")
            st.metric("MAPE", f"{metrics_test.get('mape', np.nan):.4f}")
            st.metric("SMAPE", f"{metrics_test.get('smape', np.nan):.4f}")
            st.metric("MASE", f"{metrics_test.get('mase', np.nan):.4f}")
            st.metric("R²", f"{metrics_test.get('r2', np.nan):.4f}")
            st.metric("Adjusted R²", f"{metrics_test.get('adj_r2', np.nan):.4f}")
            
            with st.expander("Как интерпретировать результаты?"):
                st.markdown("""
                **Хорошими показателями** считаются:
                
                1. **RMSE и MAE** - чем меньше, тем лучше. Сравнивайте с масштабом ваших данных.
                
                2. **MAPE**:
                   - < 10%: отличный прогноз
                   - 10-20%: хороший прогноз
                   - 20-50%: приемлемый прогноз
                   - > 50%: плохой прогноз
                
                3. **MASE**:
                   - < 1: модель лучше наивного прогноза
                   - ≈ 1: сопоставимо с наивным прогнозом
                   - > 1: модель хуже наивного прогноза
                
                4. **R² и Adjusted R²**:
                   - > 0.9: отличный результат
                   - 0.7-0.9: хороший результат
                   - 0.5-0.7: удовлетворительный результат
                   - < 0.5: модель требует улучшения
                
                5. **Theil's U2**:
                   - < 0.8: модель значительно лучше наивного прогноза
                   - 0.8-1: модель сопоставима с наивным прогнозом
                   - > 1: модель хуже наивного прогноза
                
                Если метрики на тестовой выборке значительно хуже, чем на обучающей, это может указывать на переобучение.
                """)
        
        # Информация о модели
        st.subheader("Информация о модели")
        st.info(results['model_info'])
        
        if hasattr(st.session_state.ar_model, 'fitted_model') and hasattr(st.session_state.ar_model.fitted_model, 'summary'):
            with st.expander("Подробная статистика модели"):
                try:
                    st.text(st.session_state.ar_model.fitted_model.summary())
                except:
                    st.warning("Не удалось получить подробную статистику модели.")
        
        # Отображение прогнозов
        st.subheader("Результаты прогнозирования")
        
        # Проверяем наличие необходимых ключей для построения графика
        if all(key in results for key in ['original_series', 'train', 'test', 'test_predictions']):
            # Создаем прогнозы на обучающей выборке для полного отображения
            train_predictions = None
            if st.session_state.ar_model and hasattr(st.session_state.ar_model, 'predict_in_sample'):
                try:
                    train_predictions = st.session_state.ar_model.predict_in_sample()
                except:
                    train_predictions = None
            
            # Используем унифицированный график для plotly
            fig = create_unified_forecast_plot_plotly(
                original_series=results['original_series'],
                train_predictions=train_predictions,
                test_predictions=results['test_predictions'],
                train_data=results['train'],
                test_data=results['test'],
                title="Результаты прогнозирования",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Создание и скачивание CSV с результатами моделирования
            st.markdown("### 📊 Экспорт данных")
            
            # Создаем сводную таблицу с результатами
            export_data = pd.DataFrame(index=results['original_series'].index)
            export_data['Исходные_данные'] = results['original_series'].values
            
            # Отмечаем принадлежность к обучающей/тестовой выборке
            export_data['Тип_данных'] = 'Обучение'
            if 'test' in results and len(results['test']) > 0:
                test_indices = results['test'].index
                export_data.loc[test_indices, 'Тип_данных'] = 'Тест'
            
            # Добавляем прогнозы на обучающей выборке
            if train_predictions is not None:
                export_data['Прогноз_обучение'] = np.nan
                export_data.loc[train_predictions.index, 'Прогноз_обучение'] = train_predictions.values
            
            # Добавляем прогнозы на тестовой выборке
            if 'test_predictions' in results and results['test_predictions'] is not None:
                export_data['Прогноз_тест'] = np.nan
                export_data.loc[results['test_predictions'].index, 'Прогноз_тест'] = results['test_predictions'].values
            
            # Добавляем остатки (ошибки прогноза)
            if train_predictions is not None:
                export_data['Остатки_обучение'] = np.nan
                train_residuals = results['train'].loc[train_predictions.index] - train_predictions
                export_data.loc[train_predictions.index, 'Остатки_обучение'] = train_residuals.values
            
            if 'test_predictions' in results and results['test_predictions'] is not None:
                export_data['Остатки_тест'] = np.nan
                test_residuals = results['test'].loc[results['test_predictions'].index] - results['test_predictions']
                export_data.loc[results['test_predictions'].index, 'Остатки_тест'] = test_residuals.values
            
            # Добавляем абсолютные ошибки
            if train_predictions is not None:
                export_data['Абс_ошибка_обучение'] = np.nan
                abs_errors_train = np.abs(results['train'].loc[train_predictions.index] - train_predictions)
                export_data.loc[train_predictions.index, 'Абс_ошибка_обучение'] = abs_errors_train.values
            
            if 'test_predictions' in results and results['test_predictions'] is not None:
                export_data['Абс_ошибка_тест'] = np.nan
                abs_errors_test = np.abs(results['test'].loc[results['test_predictions'].index] - results['test_predictions'])
                export_data.loc[results['test_predictions'].index, 'Абс_ошибка_тест'] = abs_errors_test.values
            
            # Предварительный просмотр данных
            st.markdown("**Предварительный просмотр данных для экспорта:**")
            st.dataframe(export_data.head(10), use_container_width=True)
            
            # Кнопка скачивания
            csv_export = export_data.to_csv(index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('model_info', 'ARIMA').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            
            st.download_button(
                label="📥 Скачать данные моделирования (CSV)",
                data=csv_export,
                file_name=f"arima_modeling_results_{model_name}_{timestamp}.csv",
                mime="text/csv",
                help="Скачать CSV файл со всеми данными: исходные значения, прогнозы, остатки, ошибки"
            )
            
            # Информация о содержимом файла
            with st.expander("📋 Описание столбцов в CSV файле"):
                st.markdown("""
                **Столбцы в экспортируемом файле:**
                
                - **Индекс** - временные метки (дата/время)
                - **Исходные_данные** - оригинальные значения временного ряда
                - **Тип_данных** - принадлежность к обучающей выборке ('Обучение') или тестовой ('Тест')
                - **Прогноз_обучение** - прогнозы модели на обучающей выборке
                - **Прогноз_тест** - прогнозы модели на тестовой выборке
                - **Остатки_обучение** - разность между реальными и прогнозными значениями (обучение)
                - **Остатки_тест** - разность между реальными и прогнозными значениями (тест)
                - **Абс_ошибка_обучение** - абсолютная ошибка прогноза (обучение)
                - **Абс_ошибка_тест** - абсолютная ошибка прогноза (тест)
                
                **Применение:** Эти данные можно использовать для дополнительного анализа, построения 
                собственных графиков, расчета дополнительных метрик или импорта в другие аналитические системы.
                """)

        # СЕКЦИЯ: Прогноз в будущее по уже обученной модели
        if st.session_state.ar_model is not None:
            st.subheader("Прогноз на будущее по обученной модели")
            future_steps = st.number_input(
                "Шаги прогноза вперед", min_value=1, max_value=100, value=10, step=1, key="future_steps")
            
            if st.button("Сделать прогноз в будущее"):
                try:
                    future_preds = st.session_state.ar_model.predict(steps=int(future_steps))
                    
                    # Используем унифицированный график прогноза
                    future_fig = create_unified_forecast_plot_plotly(
                        original_series=results['original_series'],
                        future_predictions=future_preds,
                        train_data=results['train'],
                        title="Прогноз на будущее"
                    )
                    st.plotly_chart(future_fig, use_container_width=True)
                    
                    # Таблица прогноза
                    st.dataframe(pd.DataFrame({'Прогнозируемое значение': future_preds}))
                    
                    # Кнопка для скачивания
                    csv = future_preds.to_csv(index=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Скачать прогноз (CSV)",
                        data=csv,
                        file_name=f"arima_forecast_{timestamp}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Ошибка при прогнозе в будущее: {str(e)}")
    # Если модель еще не обучена, покажем инструкции
    else:
        st.info("Выберите режим настройки и нажмите 'Запустить обучение' для начала анализа.")


if __name__ == "__main__":
    main()