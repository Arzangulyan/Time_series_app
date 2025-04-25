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
    plot_forecast, plot_forecast_plotly, display_model_information, display_differencing_effect,
    analyze_residuals, plot_residuals_diagnostic, compare_models,
    display_stationarity_results, display_acf_pacf, auto_detect_seasonality, display_residuals_analysis
)
# Импортируем функции выбора и оценки моделей
from modules.autoregressive.model_selection import (
    check_stationarity as check_stationarity_advanced,
    estimate_differencing_order, detect_seasonality,
    auto_arima, evaluate_model_performance,
    plot_model_results, generate_model_report
)

# Импортируем вспомогательные модули из проекта
from modules.page_template import (
    setup_page,
    load_time_series,
    run_calculations_on_button_click,
)

# Метаданные моделей для отображения
MODEL_METADATA = {
    "ARMA": {
        "name": "ARMA",
        "full_name": "Auto-Regressive Moving Average",
        "description": """
        **ARMA (Auto-Regressive Moving Average)** - модель для стационарных временных рядов,
        объединяющая авторегрессионный компонент (AR) и компонент скользящего среднего (MA).
        
        Математическая запись: ARMA(p, q)
        
        $$y_t = c + \\sum_{i=1}^{p} \\phi_i y_{t-i} + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j} + \\varepsilon_t$$
        
        где:
        - $y_t$ - значение временного ряда в момент времени $t$
        - $c$ - константа
        - $\\phi_i$ - параметры авторегрессии
        - $\\theta_j$ - параметры скользящего среднего
        - $\\varepsilon_t$ - белый шум
        
        **Когда использовать:** для стационарных временных рядов без выраженного тренда и сезонности.
        """
    },
    "ARIMA": {
        "name": "ARIMA",
        "full_name": "Auto-Regressive Integrated Moving Average",
        "description": """
        **ARIMA (Auto-Regressive Integrated Moving Average)** - расширение модели ARMA для нестационарных временных рядов
        с добавлением компонента интегрирования (дифференцирования).
        
        Математическая запись: ARIMA(p, d, q)
        
        $$\\nabla^d y_t = c + \\sum_{i=1}^{p} \\phi_i \\nabla^d y_{t-i} + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j} + \\varepsilon_t$$
        
        где:
        - $\\nabla^d$ - оператор дифференцирования порядка $d$
        - $y_t$ - значение временного ряда в момент времени $t$
        - $c$ - константа
        - $\\phi_i$ - параметры авторегрессии
        - $\\theta_j$ - параметры скользящего среднего
        - $\\varepsilon_t$ - белый шум
        
        **Когда использовать:** для временных рядов с трендом, но без выраженной сезонности.
        """
    },
    "SARIMA": {
        "name": "SARIMA",
        "full_name": "Seasonal Auto-Regressive Integrated Moving Average",
        "description": """
        **SARIMA (Seasonal Auto-Regressive Integrated Moving Average)** - расширение модели ARIMA
        для учета сезонности во временных рядах.
        
        Математическая запись: SARIMA(p, d, q)(P, D, Q, s)
        
        $$\\Phi_P(B^s)\\phi_p(B)(1-B)^d(1-B^s)^D y_t = c + \\Theta_Q(B^s)\\theta_q(B)\\varepsilon_t$$
        
        где:
        - $B$ - оператор сдвига назад: $By_t = y_{t-1}$
        - $\\phi_p(B)$ - несезонный AR полином порядка $p$
        - $\\Phi_P(B^s)$ - сезонный AR полином порядка $P$
        - $(1-B)^d$ - несезонное дифференцирование порядка $d$
        - $(1-B^s)^D$ - сезонное дифференцирование порядка $D$
        - $\\theta_q(B)$ - несезонный MA полином порядка $q$
        - $\\Theta_Q(B^s)$ - сезонный MA полином порядка $Q$
        - $s$ - сезонный период
        - $\\varepsilon_t$ - белый шум
        
        **Когда использовать:** для временных рядов с трендом и сезонностью.
        """
    }
}

st.set_page_config(page_title="Авторегрессионные модели", page_icon="📈", layout="wide")

# Инициализация состояния сессии
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
                        st.info(f"Использована упрощенная модель с пониженным порядком параметров.")
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
        'MSE': results_dict.get('mse', 'Н/Д'),
        'RMSE': results_dict.get('rmse', 'Н/Д'),
        'MAE': results_dict.get('mae', 'Н/Д'),
        'MAPE': format_mape(results_dict.get('mape', np.nan))
    }
    
    # Создаем метрики в две колонки
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{metrics_to_display['R²']:.4f}" if isinstance(metrics_to_display['R²'], (int, float)) else metrics_to_display['R²'])
        st.metric("MSE", f"{metrics_to_display['MSE']:.4f}" if isinstance(metrics_to_display['MSE'], (int, float)) else metrics_to_display['MSE'])
    with col2:
        st.metric("RMSE", f"{metrics_to_display['RMSE']:.4f}" if isinstance(metrics_to_display['RMSE'], (int, float)) else metrics_to_display['RMSE'])
        st.metric("MAE", f"{metrics_to_display['MAE']:.4f}" if isinstance(metrics_to_display['MAE'], (int, float)) else metrics_to_display['MAE'])
    with col3:
        st.metric("MAPE", metrics_to_display['MAPE'])


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


def main():
    # Инициализация страницы (без вызова set_page_config, так как он уже вызван ранее)
    # setup_page("Прогнозирование временных рядов с авторегрессионными моделями", "Настройки AR-моделей")
    st.title("Прогнозирование временных рядов с авторегрессионными моделями")
    st.sidebar.header("Настройки AR-моделей")
    
    # Инициализация session_state
    initialize_session_state()
    
    # Боковая панель: выбор модели
    st.sidebar.subheader("Выбор модели")
    model_type = st.sidebar.radio(
        "Тип модели:",
        ["ARMA", "ARIMA", "SARIMA"],
        index=["ARMA", "ARIMA", "SARIMA"].index(st.session_state.selected_model_type)
    )
    
    # Обновляем тип модели в session_state
    st.session_state.selected_model_type = model_type
    
    # Отображаем информацию о выбранной модели
    st.markdown(f"# {MODEL_METADATA[model_type]['full_name']} ({MODEL_METADATA[model_type]['name']})")
    
    with st.expander("Описание модели"):
        st.markdown(MODEL_METADATA[model_type]['description'])
    
    # Загрузка временного ряда
    time_series = load_time_series()
    
    if time_series is None:
        st.warning("Пожалуйста, загрузите временной ряд")
        return
    
    # Получаем первый столбец, если данные представлены в формате DataFrame
    if isinstance(time_series, pd.DataFrame) and time_series.shape[1] > 0:
        data = time_series.iloc[:, 0]
    else:
        data = time_series
    
    # Сохраняем данные в session_state
    st.session_state.time_series = data
    
    # Отображаем исходный временной ряд
    st.subheader("Исходный временной ряд")
    fig = plot_time_series(data, title="Исходный временной ряд")
    st.pyplot(fig)
    
    # Информация о данных
    freq = detect_frequency(data)
    st.info(f"""
    **Информация о временном ряде:**
    - Количество наблюдений: {len(data)}
    - Период: с {data.index[0].strftime('%d.%m.%Y')} по {data.index[-1].strftime('%d.%m.%Y')}
    - Определенная частота: {freq or "Не определена"}
    """)
    
    # Разделение на обучающую и тестовую выборки
    st.sidebar.subheader("Разделение данных")
    train_size = st.sidebar.slider("Доля обучающей выборки", 0.5, 0.95, 0.8, 0.05)
    
    # Разделяем данные
    n = len(data)
    train_idx = int(n * train_size)
    train_data = data.iloc[:train_idx]
    test_data = data.iloc[train_idx:]
    
    # Сохраняем разделение в session_state
    st.session_state.train_data = train_data
    st.session_state.test_data = test_data
    
    # Отображаем разделение
    st.subheader("Разделение на обучающую и тестовую выборки")
    split_fig = plot_train_test_split(train_data, test_data)
    st.pyplot(split_fig)
    
    # Секция анализа стационарности
    st.subheader("Анализ стационарности")
    stationarity_results = check_stationarity(train_data)
    display_stationarity_results(stationarity_results)
    
    # Дифференцирование для нестационарных рядов
    st.subheader("Дифференцирование")
    
    # Параметры дифференцирования в зависимости от типа модели
    if model_type == "ARMA":
        st.info("Модель ARMA предназначена для стационарных временных рядов и не использует дифференцирование.")
        differenced_data = train_data
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            d = st.number_input("Порядок дифференцирования (d)", 
                               min_value=0, max_value=2, 
                               value=1 if not stationarity_results['is_stationary'] else 0,
                               step=1)
        
        # Для SARIMA добавляем параметры сезонного дифференцирования
        if model_type == "SARIMA":
            with col2:
                D = st.number_input("Порядок сезонного дифференцирования (D)", 
                                   min_value=0, max_value=1, 
                                   value=0, step=1)
            
            s = st.number_input("Сезонный период (s)", 
                               min_value=1, max_value=52, 
                               value=12 if freq in ['M', 'ME'] else 
                                    4 if freq in ['Q', 'QE'] else 
                                    7 if freq in ['W', 'WE'] else 
                                    24 if freq in ['H'] else 12,
                               step=1)
            
            # Обновляем параметры SARIMA в session_state
            st.session_state.sarima_params['d'] = d
            st.session_state.sarima_params['D'] = D
            st.session_state.sarima_params['s'] = s
            
            # Применяем дифференцирование
            if d > 0 or D > 0:
                differenced_data = apply_differencing(train_data, d, D, s)
                
                # Отображаем эффект дифференцирования
                diff_order = f"{'первого' if d == 1 else 'второго' if d == 2 else ''} порядка"
                if D > 0:
                    diff_order += f" и сезонного порядка {D}"
                
                diff_fig = display_differencing_effect(train_data, differenced_data, diff_order)
                st.pyplot(diff_fig)
            else:
                differenced_data = train_data
                st.info("Дифференцирование не применялось.")
        
        else:  # ARIMA
            # Обновляем параметр d в session_state
            st.session_state.arima_params['d'] = d
            
            # Применяем дифференцирование
            if d > 0:
                differenced_data = apply_differencing(train_data, d)
                
                # Отображаем эффект дифференцирования
                diff_order = f"{'первого' if d == 1 else 'второго' if d == 2 else ''} порядка"
                diff_fig = display_differencing_effect(train_data, differenced_data, diff_order)
                st.pyplot(diff_fig)
            else:
                differenced_data = train_data
                st.info("Дифференцирование не применялось.")
    
    # Проверяем стационарность после дифференцирования
    if model_type != "ARMA" and (d > 0 or (model_type == "SARIMA" and D > 0)):
        st.subheader("Стационарность после дифференцирования")
        diff_stationarity = check_stationarity(differenced_data.dropna())
        display_stationarity_results(diff_stationarity)
    
    # Анализ ACF и PACF
    st.subheader("Анализ ACF и PACF")
    acf_pacf_fig = plot_acf_pacf_plotly(differenced_data)
    st.plotly_chart(acf_pacf_fig, use_container_width=True)
    
    # Получаем рекомендации по параметрам
    seasonal = model_type == "SARIMA"
    seasonal_period = s if seasonal and 's' in locals() else None
    suggested_params = suggest_arima_params(differenced_data, seasonal, seasonal_period)
    
    # Настройка параметров модели - вкладки для ручного и автоматического подбора
    st.subheader("Параметры модели")
    
    param_tabs = st.tabs(["Ручной подбор параметров", "Автоматический подбор параметров"])
    
    with param_tabs[0]:  # Ручной подбор параметров
        # Отображаем параметры в зависимости от типа модели
        if model_type == "ARMA":
            col1, col2 = st.columns(2)
            
            with col1:
                p = st.number_input("Порядок AR (p)", min_value=0, max_value=5, 
                                   value=suggested_params['p'], step=1)
            
            with col2:
                q = st.number_input("Порядок MA (q)", min_value=0, max_value=5, 
                                   value=suggested_params['q'], step=1)
            
            # Обновляем параметры в session_state
            st.session_state.arma_params['p'] = p
            st.session_state.arma_params['q'] = q
            
        elif model_type == "ARIMA":
            col1, col2 = st.columns(2)
            
            with col1:
                p = st.number_input("Порядок AR (p)", min_value=0, max_value=5, 
                                   value=suggested_params['p'], step=1)
            
            with col2:
                q = st.number_input("Порядок MA (q)", min_value=0, max_value=5, 
                                   value=suggested_params['q'], step=1)
            
            # Отображаем параметр d, но не даем его изменить здесь
            st.info(f"Порядок дифференцирования (d): {d}")
            
            # Обновляем параметры в session_state
            st.session_state.arima_params['p'] = p
            st.session_state.arima_params['q'] = q
            
        elif model_type == "SARIMA":
            st.markdown("### Несезонные компоненты")
            col1, col2 = st.columns(2)
            
            with col1:
                p = st.number_input("Порядок AR (p)", min_value=0, max_value=5, 
                                   value=suggested_params['p'], step=1)
            
            with col2:
                q = st.number_input("Порядок MA (q)", min_value=0, max_value=5, 
                                   value=suggested_params['q'], step=1)
            
            # Отображаем параметр d, но не даем его изменить здесь
            st.info(f"Порядок дифференцирования (d): {d}")
            
            st.markdown("### Сезонные компоненты")
            col1, col2 = st.columns(2)
            
            with col1:
                P = st.number_input("Сезонный AR (P)", min_value=0, max_value=2, 
                                   value=suggested_params['P'], step=1)
            
            with col2:
                Q = st.number_input("Сезонный MA (Q)", min_value=0, max_value=2, 
                                   value=suggested_params['Q'], step=1)
            
            # Отображаем параметры D и s, но не даем их изменить здесь
            st.info(f"Сезонное дифференцирование (D): {D}")
            st.info(f"Сезонный период (s): {s}")
            
            # Обновляем параметры в session_state
            st.session_state.sarima_params['p'] = p
            st.session_state.sarima_params['q'] = q
            st.session_state.sarima_params['P'] = P
            st.session_state.sarima_params['Q'] = Q
            
        # Добавляем кнопку применения параметров и обучения модели
        if st.button("Применить параметры и обучить модель", key="apply_manual_params"):
            result = train_model_and_predict(
                train_data, 
                test_data, 
                model_title=f"Прогноз модели {model_type} с заданными параметрами на тестовый период"
            )
            
            if result:
                model, metrics, forecast_fig = result
                
                # Отображаем график прогноза
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Отображаем метрики
                display_model_metrics(metrics, model_type)
                
                # Анализ остатков
                st.subheader("Анализ остатков")
                display_residuals_analysis(model)

    with param_tabs[1]:  # Автоматический подбор параметров
        st.markdown("### Автоматический подбор параметров")
        
        st.write("""
        Автоматический подбор использует алгоритм, который перебирает различные комбинации параметров 
        и выбирает модель с наилучшим информационным критерием.
        """)
        
        # Настройки для auto_arima
        auto_col1, auto_col2 = st.columns(2)
        
        with auto_col1:
            information_criterion = st.selectbox(
                "Информационный критерий", 
                ["aic", "bic", "aicc", "oob"],
                index=0,
                help="Критерий для выбора лучшей модели (меньше = лучше)"
            )
        
        with auto_col2:
            n_jobs = st.slider(
                "Количество параллельных процессов", 
                min_value=1, 
                max_value=8, 
                value=1,
                help="Использование нескольких ядер ускоряет подбор, но требует больше памяти"
            )
        
        # Параметры поиска
        st.subheader("Диапазон параметров для поиска")
        
        param_cols = st.columns(3)
        
        with param_cols[0]:
            max_p = st.slider("Максимальный порядок p", 0, 5, 2)
            if model_type != "ARMA":
                max_d = st.slider("Максимальный порядок d", 0, 2, d)
            max_q = st.slider("Максимальный порядок q", 0, 5, 2)
        
        if model_type == "SARIMA":
            with param_cols[1]:
                max_P = st.slider("Максимальный порядок P", 0, 2, 1)
                max_D = st.slider("Максимальный порядок D", 0, 1, D)
                max_Q = st.slider("Максимальный порядок Q", 0, 2, 1)
            
            with param_cols[2]:
                auto_detect_s = st.checkbox("Автоматически определить период сезонности", value=True)
                
                if not auto_detect_s:
                    seasonal_m = st.number_input(
                        "Сезонный период m", 
                        min_value=1,
                        max_value=52,
                        value=s,
                        help="Период сезонности (например, 12 для ежемесячных данных)"
                    )
                else:
                    with st.spinner("Определение периода сезонности..."):
                        detected_s = auto_detect_seasonality(train_data)
                        seasonal_m = st.number_input(
                            "Определенный период сезонности", 
                            min_value=1,
                            max_value=52,
                            value=detected_s,
                            help="Автоматически определенный период сезонности (можно скорректировать)"
                        )
                        st.info(f"Определен период сезонности: {detected_s}")
        
        # Кнопка для запуска автоматического подбора
        if st.button("Запустить автоматический подбор"):
            with st.spinner("Выполняется подбор оптимальных параметров..."):
                try:
                    # Параметры для auto_arima
                    auto_params = {
                        "information_criterion": information_criterion,
                        "n_jobs": n_jobs,
                        "return_all_models": True,
                        "max_p": max_p,
                        "max_q": max_q
                    }
                    
                    if model_type != "ARMA":
                        auto_params["max_d"] = max_d
                    
                    if model_type == "SARIMA":
                        auto_params["seasonal"] = True
                        auto_params["max_P"] = max_P
                        auto_params["max_D"] = max_D
                        auto_params["max_Q"] = max_Q
                        auto_params["m"] = seasonal_m
                    else:
                        auto_params["seasonal"] = False
                    
                    # Запускаем auto_arima
                    auto_result = auto_arima(train_data, **auto_params)
                    
                    # Отображаем результаты
                    st.success("Подбор параметров успешно завершен!")
                    
                    if model_type == "SARIMA":
                        st.markdown(f"""
                        ### Лучшая модель: SARIMA{auto_result['order']}{auto_result['seasonal_order']}
                        - AIC: {auto_result['aic']:.2f}
                        - BIC: {auto_result['bic']:.2f}
                        """)
                        
                        # Обновляем параметры модели в session_state
                        p, d, q = auto_result['order']
                        P, D, Q, m = auto_result['seasonal_order']
                        
                        st.session_state.sarima_params = {
                            'p': p, 'd': d, 'q': q,
                            'P': P, 'D': D, 'Q': Q, 's': m
                        }
                    elif model_type == "ARIMA":
                        st.markdown(f"""
                        ### Лучшая модель: ARIMA{auto_result['order']}
                        - AIC: {auto_result['aic']:.2f}
                        - BIC: {auto_result['bic']:.2f}
                        """)
                        
                        # Обновляем параметры модели в session_state
                        p, d, q = auto_result['order']
                        st.session_state.arima_params = {'p': p, 'd': d, 'q': q}
                    else:  # ARMA
                        st.markdown(f"""
                        ### Лучшая модель: ARMA({auto_result['order'][0]}, {auto_result['order'][2]})
                        - AIC: {auto_result['aic']:.2f}
                        - BIC: {auto_result['bic']:.2f}
                        """)
                        
                        # Обновляем параметры модели в session_state
                        p, _, q = auto_result['order']
                        st.session_state.arma_params = {'p': p, 'q': q}
                    
                    # Сохраняем лучшую модель
                    best_model = auto_result['best_model']
                    
                    # Сохраняем лучшую модель как текущую
                    set_current_model(best_model)
                    
                    # Сохраняем модель как текущую активную для всех функций
                    st.session_state.current_active_model = best_model
                    st.session_state.last_trained_on = 'train'
                    
                    # Сохраняем параметры лучшей модели для отображения
                    if model_type == "ARMA":
                        p, _, q = auto_result['order']
                        st.session_state.model_params = {'p': p, 'q': q}
                    elif model_type == "ARIMA":
                        p, d, q = auto_result['order']
                        st.session_state.model_params = {'p': p, 'd': d, 'q': q}
                    elif model_type == "SARIMA":
                        p, d, q = auto_result['order']
                        P, D, Q, m = auto_result['seasonal_order']
                        st.session_state.model_params = {
                            'p': p, 'd': d, 'q': q,
                            'P': P, 'D': D, 'Q': Q, 's': m
                        }
                    
                    # Генерируем прогноз
                    forecast_fig = plot_forecast_plotly(
                        best_model, 
                        steps=len(test_data),
                        train_data=train_data,
                        test_data=test_data,
                        title="Прогноз лучшей модели на тестовом периоде"
                    )
                    
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Рассчитываем метрики
                    metrics = evaluate_model_performance(best_model, train_data, test_data)
                    
                    display_model_metrics(metrics, model_type)
                    
                except Exception as e:
                    st.error(f"Ошибка при автоматическом подборе параметров: {str(e)}")
    
    # Обучение модели и прогнозирование
    st.subheader("Обучение модели и прогнозирование")
    
    # Создаем вкладки для обучения и прогнозирования
    train_forecast_tabs = st.tabs(["Оценка на тестовой выборке", "Прогноз в будущее"])
    
    # Сбрасываем флаг run_future_forecast при первой загрузке страницы
    # и когда модель меняет тип
    tab_key = f"tab_{model_type}"
    if 'last_tab' not in st.session_state or st.session_state.last_tab != tab_key:
        st.session_state.run_future_forecast = False
        st.session_state.last_tab = tab_key
    
    with train_forecast_tabs[0]:
        # Проверяем, есть ли уже обученная модель
        if st.session_state.get('current_active_model'):
            st.success(f"Модель {model_type} уже обучена с параметрами: {display_model_information(st.session_state.current_active_model)}")
        
        st.write("Нажмите кнопку ниже, чтобы обучить модель и произвести тестирование на отложенной выборке.")
        
        if st.button("Обучить и протестировать модель", key="train_and_test"):
            # Если уже есть обученная модель, используем ее
            if st.session_state.get('current_active_model'):
                # Получаем тип модели
                current_model = st.session_state.current_active_model
                if hasattr(current_model, 'model_name'):
                    model_type_from_obj = current_model.model_name
                else:
                    model_type_from_obj = current_model.get_params().get('type')
                    
                if model_type_from_obj == model_type:
                    model = current_model
                    st.info(f"Используется ранее обученная модель {model_type}")
                    
                    # Генерируем прогноз и метрики
                    forecast_fig = plot_forecast_plotly(
                        model, 
                        steps=len(test_data),
                        train_data=train_data,
                        test_data=test_data,
                        title=f"Оценка модели {model_type} на тестовой выборке"
                    )
                    
                    # Отображаем график прогноза
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Отображаем метрики
                    metrics = evaluate_model_performance(model, train_data, test_data)
                    display_model_metrics(metrics, model_type)
                    
                    # Анализ остатков
                    st.subheader("Анализ остатков")
                    display_residuals_analysis(model)
                    
                    # Устанавливаем состояние обучения
                    st.session_state.last_trained_on = 'train'
                else:
                    result = train_model_and_predict(
                        train_data, 
                        test_data, 
                        model_title=f"Оценка модели {model_type} на тестовой выборке"
                    )
                    
                    if result:
                        model, metrics, forecast_fig = result
                        
                        # Сохраняем модель как текущую активную
                        st.session_state.current_active_model = model
                        st.session_state.last_trained_on = 'train'
                        
                        # Отображаем график прогноза
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Отображаем метрики
                        display_model_metrics(metrics, model_type)
                        
                        # Анализ остатков
                        st.subheader("Анализ остатков")
                        display_residuals_analysis(model)
            else:
                result = train_model_and_predict(
                    train_data, 
                    test_data, 
                    model_title=f"Оценка модели {model_type} на тестовой выборке"
                )
                
                if result:
                    model, metrics, forecast_fig = result
                    
                    # Сохраняем модель как текущую активную
                    st.session_state.current_active_model = model
                    st.session_state.last_trained_on = 'train'
                    
                    # Отображаем график прогноза
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Отображаем метрики
                    display_model_metrics(metrics, model_type)
                    
                    # Анализ остатков
                    st.subheader("Анализ остатков")
                    display_residuals_analysis(model)
    
    with train_forecast_tabs[1]:
        st.write("Эта вкладка позволяет сделать прогноз за пределы имеющихся данных.")
        st.info("Для прогноза на будущее модель будет переобучена на всем наборе данных, включая тестовую выборку.")
        
        # Получаем текущую модель
        current_model = st.session_state.get('current_active_model')
        
        if current_model and current_model.is_fitted:
            # Проверяем, соответствует ли текущая модель выбранному типу
            # Сначала пробуем получить тип через атрибут model_name, если он есть
            if hasattr(current_model, 'model_name'):
                model_type_from_obj = current_model.model_name
            else:
                # Если атрибута нет, используем get_params()
                model_type_from_obj = current_model.get_params().get('type')
            
            # Отображаем информацию о текущей модели
            st.success(f"Текущая модель: {model_type_from_obj} с параметрами: {display_model_information(current_model)}")
            
            if model_type_from_obj != model_type:
                st.warning(f"Текущая обученная модель ({model_type_from_obj}) отличается от выбранного типа ({model_type}). Для продолжения сначала обучите модель {model_type}.")
            else:
                # Определяем разумное максимальное количество шагов для прогноза
                max_steps = min(int(len(data) * 0.5), 100)
                
                # Используем session_state для сохранения значения слайдера
                if 'future_steps' not in st.session_state:
                    st.session_state.future_steps = min(12, max_steps)
                
                future_steps = st.slider("Количество периодов для прогноза", 1, max_steps, st.session_state.future_steps, key='future_steps_slider')
                st.session_state.future_steps = future_steps
                
                if st.button("Сделать прогноз в будущее", key='future_forecast_button'):
                    st.session_state.run_future_forecast = True
                    # Добавляем перезагрузку страницы для активации изменения состояния
                    st.rerun()
                
                if st.session_state.get('run_future_forecast', False):
                    # Проверяем, нужно ли переобучать модель на полных данных
                    need_retrain = True
                    if st.session_state.last_trained_on == 'full':
                        need_retrain = False
                    
                    if need_retrain:
                        # Обучаем модель на всех данных
                        with st.spinner("Переобучаем модель на полном наборе данных..."):
                            full_model = fit_selected_model(data)
                            
                            if full_model:
                                # Сохраняем модель и ее параметры
                                st.session_state.current_active_model = full_model
                                st.session_state.last_trained_on = 'full'
                                
                                # Сохраняем параметры модели в session_state для отображения
                                if model_type == "ARMA":
                                    st.session_state.model_params = {'p': full_model.p, 'q': full_model.q}
                                elif model_type == "ARIMA":
                                    st.session_state.model_params = {'p': full_model.p, 'd': full_model.d, 'q': full_model.q}
                                elif model_type == "SARIMA":
                                    st.session_state.model_params = {
                                        'p': full_model.p, 'd': full_model.d, 'q': full_model.q,
                                        'P': full_model.P, 'D': full_model.D, 'Q': full_model.Q, 's': full_model.m
                                    }
                                
                                st.success(f"Модель успешно переобучена на полных данных с параметрами: {display_model_information(full_model)}")
                            else:
                                st.error("Не удалось переобучить модель на полных данных.")
                                st.stop()
                    else:
                        full_model = current_model
                        st.info("Используется ранее обученная модель на полных данных")
                        
                        # Отображаем сохраненные параметры модели
                        if hasattr(st.session_state, 'model_params'):
                            params_str = ", ".join([f"{k}={v}" for k, v in st.session_state.model_params.items()])
                            st.write(f"Параметры модели: {params_str}")
                    
                    if full_model and full_model.is_fitted:
                        try:
                            future_result = make_future_forecast(
                                full_model, 
                                data, 
                                future_steps,
                                title=f"Прогноз модели {model_type} на {future_steps} периодов вперед"
                            )
                            
                            if future_result:
                                future_fig, future_df = future_result
                                
                                # Проверка, что future_fig является объектом Figure
                                if hasattr(future_fig, 'update_layout'):
                                    st.plotly_chart(future_fig, use_container_width=True)
                                else:
                                    st.warning("Не удалось создать график прогноза. Используйте только DataFrame.")
                                
                                # Отображаем таблицу с прогнозом
                                st.dataframe(future_df)
                                
                                # Предлагаем скачать прогноз
                                csv = future_df.to_csv()
                                st.download_button(
                                    label="Скачать прогноз как CSV",
                                    data=csv,
                                    file_name=f'{model_type}_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                    mime='text/csv',
                                )
                                
                                # Сохраняем результат прогноза в session_state
                                st.session_state.forecast_results = future_df
                                
                                # Добавляем кнопку для сброса прогноза
                                if st.button("Сбросить прогноз", key="reset_forecast"):
                                    st.session_state.run_future_forecast = False
                                    st.rerun()
                                
                        except Exception as e:
                            st.error(f"Ошибка при создании прогноза: {str(e)}")
                            st.exception(e)
                    else:
                        st.error("Не удалось обучить модель на полном наборе данных.")
        else:
            st.warning("Сначала обучите модель на вкладке 'Оценка на тестовой выборке'.")


if __name__ == "__main__":
    main() 