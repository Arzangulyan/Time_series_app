"""
Модуль для подбора параметров и оценки качества авторегрессионных моделей.

Включает функции для проверки стационарности, определения параметров 
дифференцирования и сезонности, а также автоматический подбор ARIMA/SARIMA моделей.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima as pm_auto_arima
import itertools
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
from .models import BaseTimeSeriesModel, ARMAModel, ARIMAModel, SARIMAModel
from .core import detect_frequency
import logging
import time  # Add explicit import of time module
logger = logging.getLogger('arima_app.model_selection')


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Проверяет стационарность временного ряда с использованием тестов ADF и KPSS.
    
    Тест ADF (нулевая гипотеза: ряд нестационарный)
    Тест KPSS (нулевая гипотеза: ряд стационарный)
    
    Параметры:
    -----------
    series : pd.Series
        Временной ряд для проверки
    alpha : float, default=0.05
        Уровень значимости для тестов
        
    Возвращает:
    -----------
    Dict[str, Any]
        Словарь с результатами тестов и выводами о стационарности
    """
    result = {}
    
    # Тест ADF
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        adf_result = adfuller(series.dropna())
    
    adf_p_value = adf_result[1]
    adf_statistic = adf_result[0]
    adf_critical_values = adf_result[4]
    
    # Тест KPSS
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        kpss_result = kpss(series.dropna())
    
    kpss_p_value = kpss_result[1]
    kpss_statistic = kpss_result[0]
    kpss_critical_values = kpss_result[3]
    
    # Формирование вывода
    result['adf_statistic'] = adf_statistic
    result['adf_p_value'] = adf_p_value
    result['adf_critical_values'] = adf_critical_values
    result['adf_is_stationary'] = adf_p_value < alpha
    
    result['kpss_statistic'] = kpss_statistic
    result['kpss_p_value'] = kpss_p_value
    result['kpss_critical_values'] = kpss_critical_values
    result['kpss_is_stationary'] = kpss_p_value >= alpha
    
    # Общий вывод на основе обоих тестов
    if result['adf_is_stationary'] and result['kpss_is_stationary']:
        result['conclusion'] = "Ряд стационарный (по обоим тестам)"
    elif result['adf_is_stationary'] and not result['kpss_is_stationary']:
        result['conclusion'] = "Возможно ряд стационарный (ADF показывает стационарность, KPSS показывает нестационарность)"
    elif not result['adf_is_stationary'] and result['kpss_is_stationary']:
        result['conclusion'] = "Вероятно ряд нестационарный (ADF показывает нестационарность, KPSS показывает стационарность)"
    else:
        result['conclusion'] = "Ряд нестационарный (по обоим тестам)"
    
    return result


def estimate_differencing_order(series: pd.Series, max_d: int = 2) -> int:
    """
    Оценивает оптимальный порядок дифференцирования для достижения стационарности.
    
    Параметры:
    -----------
    series : pd.Series
        Временной ряд для анализа
    max_d : int, default=2
        Максимальный порядок дифференцирования для проверки
        
    Возвращает:
    -----------
    int
        Оптимальный порядок дифференцирования
    """
    d = 0
    is_stationary = False
    test_series = series.copy()
    
    while d <= max_d and not is_stationary:
        # Проверяем текущий ряд на стационарность
        stat_result = check_stationarity(test_series)
        is_stationary = stat_result['adf_is_stationary']
        
        if is_stationary:
            break
        
        # Если ряд нестационарный, делаем дифференцирование
        d += 1
        if d <= max_d:
            test_series = test_series.diff().dropna()
    
    return d


def detect_seasonality(series: pd.Series, periodicity: Optional[int] = None, 
                      test_values: List[int] = None) -> Dict[str, Any]:
    """
    Выявляет сезонность во временном ряде.
    
    Параметры:
    -----------
    series : pd.Series
        Временной ряд для анализа
    periodicity : int, optional
        Предполагаемый период сезонности (если известен)
    test_values : List[int], optional
        Список периодов для проверки, если periodicity не указан.
        По умолчанию проверяются периоды [4, 7, 12, 24, 52, 30, 180, 365] для различных 
        типичных временных интервалов (квартал, неделя, месяц, сутки, год, месяц, полугодие, год)
        
    Возвращает:
    -----------
    Dict[str, Any]
        Словарь с результатами анализа сезонности
        
    Метод:
    ------
    Функция использует seasonal_decompose для разложения временного ряда на компоненты: тренд, 
    сезонность и остатки. Затем вычисляется "сила сезонности" как отношение дисперсии сезонного 
    компонента к сумме дисперсий сезонного компонента и остатков:
    
    seasonal_strength = дисперсия_сезонного_компонента / (дисперсия_сезонного_компонента + дисперсия_остатков)
    
    Значение близкое к 1 указывает на сильную сезонность, близкое к 0 - на слабую.
    
    Если period не указан, функция перебирает различные периоды из test_values 
    и выбирает тот, который даёт наибольшую силу сезонности. Сезонность считается 
    значимой, если seasonal_strength > 0.3.
    """
    if test_values is None:
        test_values = [4, 7, 12, 24, 52, 30, 180, 365]
    
    result = {}
    
    # Если период известен, используем его
    if periodicity is not None:
        try:
            # Сезонная декомпозиция 
            decomposition = seasonal_decompose(series, model='additive', period=periodicity)
            
            result['periodicity'] = periodicity
            result['seasonal'] = decomposition.seasonal
            result['trend'] = decomposition.trend
            result['residual'] = decomposition.resid
            result['observed'] = decomposition.observed
            
            # Вычисляем силу сезонности (отношение дисперсии сезонной компоненты к сумме дисперсий)
            variance_seasonal = decomposition.seasonal.var()
            variance_residual = decomposition.resid.dropna().var()
            seasonal_strength = variance_seasonal / (variance_seasonal + variance_residual)
            
            result['seasonal_strength'] = seasonal_strength
            result['has_seasonality'] = seasonal_strength > 0.3  # Пороговое значение
            
            return result
        except:
            # Если не удалось выполнить декомпозицию, пробуем найти лучший период
            periodicity = None
    
    # Если период не указан или не сработал, проверяем разные периоды
    if periodicity is None:
        best_strength = 0
        best_period = None
        decomp_results = {}
        
        for period in test_values:
            if len(series) <= 2 * period:
                continue  # Пропускаем, если не хватает данных
                
            try:
                decomposition = seasonal_decompose(series, model='additive', period=period)
                
                variance_seasonal = decomposition.seasonal.var()
                variance_residual = decomposition.resid.dropna().var()
                seasonal_strength = variance_seasonal / (variance_seasonal + variance_residual)
                
                decomp_results[period] = {
                    'strength': seasonal_strength,
                    'decomposition': decomposition
                }
                
                if seasonal_strength > best_strength:
                    best_strength = seasonal_strength
                    best_period = period
            except:
                continue
        
        if best_period is not None:
            best_decomp = decomp_results[best_period]['decomposition']
            
            result['periodicity'] = best_period
            result['seasonal'] = best_decomp.seasonal
            result['trend'] = best_decomp.trend
            result['residual'] = best_decomp.resid
            result['observed'] = best_decomp.observed
            result['seasonal_strength'] = best_strength
            result['has_seasonality'] = best_strength > 0.3
            result['all_periods_tested'] = list(decomp_results.keys())
            result['all_strengths'] = {p: decomp_results[p]['strength'] for p in decomp_results}
        else:
            result['periodicity'] = None
            result['has_seasonality'] = False
            result['all_periods_tested'] = test_values
    
    return result


def auto_arima(series: pd.Series, seasonal: bool = True, 
              max_p: int = 5, max_d: int = 2, max_q: int = 5,
              max_P: int = 2, max_D: int = 1, max_Q: int = 2,
              m: Optional[int] = None, test_values: Optional[List[int]] = None,
              information_criterion: str = 'aic', n_jobs: int = 1,
              return_all_models: bool = False, verbose: bool = False, **kwargs) -> Union[SARIMAModel, ARIMAModel, Dict]:
    """
    Автоматически подбирает лучшую ARIMA или SARIMA модель на основе заданных параметров.
    
    Параметры:
    -----------
    series : pd.Series
        Временной ряд для анализа
    seasonal : bool, default=True
        Учитывать ли сезонность
    max_p, max_d, max_q : int
        Максимальные значения для параметров p, d, q
    max_P, max_D, max_Q : int
        Максимальные значения для сезонных параметров P, D, Q
    m : int, optional
        Период сезонности (если известен)
    test_values : List[int], optional
        Список периодов сезонности для проверки, если m не указан
    information_criterion : str, default='aic'
        Критерий для выбора модели ('aic', 'bic', 'aicc', 'oob')
    n_jobs : int, default=1
        Количество параллельных процессов для выполнения
    return_all_models : bool, default=False
        Если True, возвращает словарь с результатами всех тестируемых моделей
    verbose : bool, default=False
        Выводить ли дополнительную информацию о процессе подбора моделей
    **kwargs : dict
        Дополнительные параметры для pmdarima.auto_arima
        
    Возвращает:
    -----------
    Union[SARIMAModel, ARIMAModel, Dict]
        Лучшая модель или словарь с результатами всех моделей, если return_all_models=True
    """
    logger.info(f"Starting auto_arima with seasonal={seasonal}, return_all_models={return_all_models}")
    
    # Track all models if requested
    all_models_info = []
    
    # Если сезонность включена, но период не указан, определяем его
    if seasonal and m is None:
        seasonality_result = detect_seasonality(series, test_values=test_values)
        m = seasonality_result.get('periodicity', 12)  # По умолчанию 12, если не удалось определить
        logger.info(f"Detected seasonality period: {m}")
        
        # Если сезонность не обнаружена, отключаем её
        if not seasonality_result.get('has_seasonality', False):
            logger.info("No seasonality detected, turning off seasonal mode")
            seasonal = False
    
    # Remove callback parameter that was causing errors
    if 'callback' in kwargs:
        del kwargs['callback']
    
    # Используем pmdarima.auto_arima для поиска лучшей модели
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        logger.info(f"Running pmdarima.auto_arima with trace={kwargs.get('trace', False)}")
        try:
            best_model = pm_auto_arima(
                series,
                max_p=max_p,
                d=None, max_d=max_d,
                max_q=max_q,
                max_P=max_P if seasonal else 0,
                max_Q=max_Q if seasonal else 0,
                D=None, max_D=max_D if seasonal else 0,
                seasonal=seasonal,
                m=m if seasonal else 1,
                information_criterion=information_criterion,
                n_jobs=n_jobs,
                **kwargs
            )
            logger.info("pmdarima.auto_arima completed")
        except Exception as e:
            logger.error(f"Error in pmdarima.auto_arima: {str(e)}")
            raise e
    
    # Get best model parameters
    order = best_model.order
    seasonal_order = best_model.seasonal_order if seasonal else (0, 0, 0, 0)
    
    # Check for (0,0,0) model which is invalid
    p, d, q = order
    if p == 0 and q == 0:
        # Если выбрана модель (0,0), принудительно установим параметры на основе анализа ACF/PACF
        from .core import suggest_arima_params
        suggested_params = suggest_arima_params(series, seasonal, m)
        p = max(suggested_params['p'], 1)  # Используем p минимум 1
        q = max(suggested_params['q'], 1)  # Используем q минимум 1
        order = (p, d, q)
        
        # Обновляем seasonal_order если модель сезонная
        if seasonal and seasonal_order[0] == 0 and seasonal_order[2] == 0:
            P = max(suggested_params.get('P', 0), 1)  # Используем P минимум 1
            Q = max(suggested_params.get('Q', 0), 1)  # Используем Q минимум 1
            D = seasonal_order[1]
            seasonal_order = (P, D, Q, m)
    
    # Создаем соответствующую модель из нашего пакета
    if seasonal and seasonal_order != (0, 0, 0, 0):
        p, d, q = order
        P, D, Q, m = seasonal_order
        model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m)
    else:
        p, d, q = order
        model = ARIMAModel(p=p, d=d, q=q)
    
    # Обучаем модель
    model.fit(series)
    
    # Возвращаем все модели, если запрошено
    if return_all_models:
        # For simplicity, let's create a subset of the parameter space that was searched
        logger.info("Creating candidate models for comparison")
        
        # Generate top parameter combinations based on typical ARIMA models
        parameter_combinations = []
        
        # Add the best model first
        parameter_combinations.append((order, seasonal_order))
        
        # Add some variations around the best model
        p, d, q = order
        for p_var in range(max(0, p-1), min(p+2, max_p+1)):
            for q_var in range(max(0, q-1), min(q+2, max_q+1)):
                if (p_var, d, q_var) != order:  # Skip the best model we already added
                    if seasonal:
                        P, D, Q, m_val = seasonal_order
                        for P_var in range(max(0, P-1), min(P+2, max_P+1)):
                            for Q_var in range(max(0, Q-1), min(Q+2, max_Q+1)):
                                if (P_var, D, Q_var, m_val) != seasonal_order:  # Skip duplicates
                                    parameter_combinations.append(
                                        ((p_var, d, q_var), (P_var, D, Q_var, m_val))
                                    )
                    else:
                        parameter_combinations.append(
                            ((p_var, d, q_var), (0, 0, 0, 0))
                        )
        
        # Limit to a reasonable number of combinations
        parameter_combinations = parameter_combinations[:10]
        
        # Fit each model and calculate metrics
        for i, (order_params, seasonal_params) in enumerate(parameter_combinations):
            if i == 0:  # Best model already fitted
                continue
                
            try:
                p, d, q = order_params
                
                if seasonal and seasonal_params != (0, 0, 0, 0):
                    P, D, Q, m_val = seasonal_params
                    candidate_model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m_val)
                else:
                    candidate_model = ARIMAModel(p=p, d=d, q=q)
                
                # Fit the model
                start_time = time.perf_counter()
                candidate_model.fit(series)
                fit_time = time.perf_counter() - start_time
                
                # Get criterion value
                criterion_value = None
                if hasattr(candidate_model.fitted_model, information_criterion):
                    criterion_value = getattr(candidate_model.fitted_model, information_criterion)
                
                # Add to results
                all_models_info.append({
                    'model': candidate_model,
                    'order': order_params,
                    'seasonal_order': seasonal_params,
                    'criterion_value': criterion_value,
                    'fit_time': fit_time
                })
                
                logger.info(f"Added model {order_params}{seasonal_params} to candidate models")
            except Exception as e:
                logger.warning(f"Failed to fit model with params {order_params}, {seasonal_params}: {str(e)}")
        
        # Add best model to the beginning
        all_models_info.insert(0, {
            'model': model,
            'order': order,
            'seasonal_order': seasonal_order,
            'criterion_value': getattr(model.fitted_model, information_criterion, None),
            'fit_time': 0.0  # We don't track time for the best model from pmdarima
        })
        
        logger.info(f"Created {len(all_models_info)} candidate models for comparison")
        
        # Sort all_models_info by criterion_value (lower is better)
        all_models_info = sorted(all_models_info, key=lambda x: x.get('criterion_value', float('inf')))
        
        # Make sure the best model is actually the one with lowest criterion value
        if all_models_info and 'criterion_value' in all_models_info[0]:
            best_model_criterion = all_models_info[0]['criterion_value']
            logger.info(f"Best model has {information_criterion}={best_model_criterion}")
            
            # Fix: Call the method to get actual value
            criterion_method = getattr(best_model, information_criterion, None)
            if criterion_method and callable(criterion_method):
                best_model_value = criterion_method()
                logger.info(f"pmdarima model has {information_criterion}={best_model_value}")
                
                # Compare the values - need to handle cases where method isn't callable
                if best_model_criterion < best_model_value:
                    logger.info(f"Replacing pmdarima's model with our true best model")
                    model = all_models_info[0]['model']
            
        result_dict = {
            'best_model': model,  # This is now guaranteed to be the best model by criterion
            'pmdarima_model': best_model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': best_model.aic() if hasattr(best_model, 'aic') else None,
            'bic': best_model.bic() if hasattr(best_model, 'bic') else None,
            'aicc': best_model.aicc() if hasattr(best_model, 'aicc') else None,
            'is_seasonal': seasonal and seasonal_order != (0, 0, 0, 0),
            'all_models': all_models_info,
            'criterion_used': information_criterion
        }
        return result_dict
    
    return model


def evaluate_model_performance(model, train_data, test_data):
    """
    Оценивает производительность модели на тестовых данных.
    
    Параметры:
    -----------
    model : Обученная модель с методом predict
    train_data : Обучающие данные
    test_data : Тестовые данные
        
    Возвращает:
    -----------
    dict : Словарь с метриками производительности
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    # Получаем прогноз на тестовый период
    y_pred = model.predict(steps=len(test_data))
    y_true = test_data.values if hasattr(test_data, 'values') else test_data
    
    # Рассчитываем основные метрики
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R-squared и adjusted R-squared
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = 1  # количество предикторов (для авторегрессии можно взять порядок модели)
    if hasattr(model, 'get_params'):
        params = model.get_params()
        if 'p' in params:
            p = max(1, params['p'])
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # MAPE (Mean Absolute Percentage Error)
    # Избегаем деления на 0
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if np.sum(mask) > 0:
        smape = 2.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = np.nan
    
    # Theil's U2 (компаративная точность по сравнению с наивным прогнозом)
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]  # первое значение остается тем же
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.mean((y_true - y_naive) ** 2))
    if denominator > 0:
        theil_u2 = numerator / denominator
    else:
        theil_u2 = np.nan
    
    # MASE (Mean Absolute Scaled Error)
    # Вычисляем MAE наивного прогноза (one-step сезонный)
    # Для несезонных данных используем смещение на 1 шаг
    seasonal_period = 1
    
    # Для SARIMA можно использовать сезонный период s
    if hasattr(model, 'get_params'):
        params = model.get_params()
        if 'm' in params and params['m'] > 1:
            seasonal_period = params['m']
    
    if len(train_data) > seasonal_period:
        # Вычисляем MAE наивного сезонного прогноза на обучающих данных
        naive_errors = np.abs(train_data[seasonal_period:].values - train_data[:-seasonal_period].values)
        scale = np.mean(naive_errors)
        if scale > 0:
            mase = mae / scale
        else:
            mase = np.nan
    else:
        mase = np.nan
    
    # Собираем метрики в словарь
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape / 100,  # Переводим в доли для совместимости с другими метриками
        'smape': smape / 100,  # Переводим в доли для совместимости с другими метриками
        'theil_u2': theil_u2,
        'r2': r2,
        'adj_r2': adj_r2,
        'mase': mase
    }
    
    return metrics


def plot_model_results(model: BaseTimeSeriesModel, train: pd.Series, test: Optional[pd.Series] = None,
                      forecast_steps: Optional[int] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Строит график с фактическими данными, подогнанной моделью и прогнозом.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    train : pd.Series
        Обучающий набор данных
    test : pd.Series, optional
        Тестовый набор данных
    forecast_steps : int, optional
        Количество шагов для прогноза (если отличается от длины test)
    figsize : Tuple[int, int], default=(12, 8)
        Размер графика
        
    Возвращает:
    -----------
    plt.Figure
        Объект графика
    """
    if not model.is_fitted:
        raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
    
    # Определяем количество шагов для прогноза
    if forecast_steps is None and test is not None:
        forecast_steps = len(test)
    elif forecast_steps is None:
        forecast_steps = 10  # Значение по умолчанию, если тестовый набор не предоставлен
    
    # Делаем прогноз
    forecast = model.predict(steps=forecast_steps)
    
    # Создаем график
    fig, ax = plt.subplots(figsize=figsize)
    
    # Строим обучающие данные
    ax.plot(train.index, train, label='Обучающие данные', color='blue')
    
    # Если есть тестовые данные, строим их
    if test is not None:
        ax.plot(test.index, test, label='Тестовые данные', color='green')
    
    # Строим подогнанные значения модели
    if hasattr(model.fitted_model, 'fittedvalues'):
        fitted_values = model.fitted_model.fittedvalues
        ax.plot(fitted_values.index, fitted_values, label='Подогнанные значения', color='red', alpha=0.7)
    
    # Строим прогноз
    ax.plot(forecast.index, forecast, label='Прогноз', color='purple', linestyle='--')
    
    # Если есть тестовые данные и прогноз, добавляем область прогноза
    if hasattr(model.fitted_model, 'get_forecast') and test is not None:
        try:
            forecast_obj = model.fitted_model.get_forecast(steps=len(test))
            conf_int = forecast_obj.conf_int(alpha=0.05)
            
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]
            
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% доверительный интервал')
        except:
            pass
    
    # Настройка графика
    ax.set_title('Результаты моделирования временного ряда')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Значение')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Поворачиваем метки на оси X для лучшей читаемости
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def generate_model_report(model: BaseTimeSeriesModel, train: pd.Series, test: Optional[pd.Series] = None,
                         forecast_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Генерирует отчет, обобщающий результаты моделирования.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    train : pd.Series
        Обучающий набор данных
    test : pd.Series, optional
        Тестовый набор данных
    forecast_steps : int, optional
        Количество шагов для прогноза (если отличается от длины test)
        
    Возвращает:
    -----------
    Dict[str, Any]
        Словарь с результатами моделирования
    """
    if not model.is_fitted:
        raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
    
    report = {}
    
    # Параметры модели
    report['model_type'] = model.__class__.__name__
    report['model_params'] = model.get_params()
    
    # Если есть fitted_model, добавляем его сводку
    if hasattr(model, 'fitted_model') and hasattr(model.fitted_model, 'summary'):
        try:
            report['model_summary'] = str(model.fitted_model.summary())
        except:
            report['model_summary'] = "Сводка модели недоступна"
    
    # Прогноз
    if forecast_steps is None and test is not None:
        forecast_steps = len(test)
    elif forecast_steps is None:
        forecast_steps = 10
    
    forecast = model.predict(steps=forecast_steps)
    report['forecast'] = forecast
    
    # Если есть тестовые данные, вычисляем метрики
    if test is not None:
        metrics = evaluate_model_performance(model, train, test)
        report['metrics'] = metrics
    
    # Информационные критерии
    if hasattr(model.fitted_model, 'aic'):
        report['aic'] = model.fitted_model.aic
    
    if hasattr(model.fitted_model, 'bic'):
        report['bic'] = model.fitted_model.bic
    
    if hasattr(model.fitted_model, 'aicc'):
        report['aicc'] = model.fitted_model.aicc
    
    # Диагностика остатков
    if hasattr(model.fitted_model, 'resid'):
        residuals = model.fitted_model.resid
        report['residuals'] = residuals
        
        # Статистика остатков
        report['residuals_mean'] = residuals.mean()
        report['residuals_std'] = residuals.std()
        
        # Проверяем остатки на стационарность
        try:
            residuals_stationarity = check_stationarity(residuals)
            report['residuals_stationarity'] = residuals_stationarity['conclusion']
        except:
            report['residuals_stationarity'] = "Не удалось выполнить проверку стационарности"
    
    return report


def split_train_test(time_series, train_size=0.8):
    """
    Разделяет временной ряд на обучающую и тестовую выборки.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для разделения
    train_size : float, default=0.8
        Доля данных для обучающей выборки (от 0 до 1)
        
    Возвращает:
    -----------
    tuple
        Кортеж (train, test) с обучающей и тестовой выборками
    """
    n = len(time_series)
    train_idx = int(n * train_size)
    
    train = time_series.iloc[:train_idx]
    test = time_series.iloc[train_idx:]
    
    return train, test


def create_candidate_models(series, best_params, max_params, seasonal=True, information_criterion='aic'):
    """Helper function to create candidate models around the best parameters"""
    all_models_info = []
    
    p, d, q = best_params['order']
    max_p, max_d, max_q = max_params['max_p'], max_params['max_d'], max_params['max_q']
    
    # Create parameters for the best model first
    model_configs = []
    
    # Add the best model parameters
    if seasonal and 'seasonal_order' in best_params:
        P, D, Q, m = best_params['seasonal_order']
        model_configs.append((p, d, q, P, D, Q, m))
    else:
        model_configs.append((p, d, q))
    
    # Add some variations around best parameters
    for p_var in range(max(0, p-1), min(p+2, max_p+1)):
        for q_var in range(max(0, q-1), min(q+2, max_q+1)):
            if seasonal and 'seasonal_order' in best_params:
                P, D, Q, m = best_params['seasonal_order']
                max_P, max_D, max_Q = max_params.get('max_P', 2), max_params.get('max_D', 1), max_params.get('max_Q', 2)
                
                for P_var in range(max(0, P-1), min(P+2, max_P+1)):
                    for Q_var in range(max(0, Q-1), min(Q+2, max_Q+1)):
                        # Skip duplicate of the best model
                        if (p_var, d, q_var, P_var, D, Q_var, m) != (p, d, q, P, D, Q, m):
                            model_configs.append((p_var, d, q_var, P_var, D, Q_var, m))
            else:
                # Skip duplicate of the best model
                if (p_var, d, q_var) != (p, d, q):
                    model_configs.append((p_var, d, q_var))
    
    # Limit to a reasonable number (10)
    model_configs = model_configs[:10]
    
    # Create and fit each model
    for i, config in enumerate(model_configs):
        try:
            start_time = time.perf_counter()
            
            if len(config) == 3:  # ARIMA
                p, d, q = config
                model = ARIMAModel(p=p, d=d, q=q)
                model.fit(series)
                
                model_info = {
                    'model': model,
                    'order': (p, d, q),
                    'seasonal_order': (0, 0, 0, 0),
                    'criterion_value': getattr(model.fitted_model, information_criterion)(),
                    'fit_time': time.perf_counter() - start_time
                }
            else:  # SARIMA
                p, d, q, P, D, Q, m = config
                model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m)
                model.fit(series)
                
                model_info = {
                    'model': model,
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, m),
                    'criterion_value': getattr(model.fitted_model, information_criterion)(),
                    'fit_time': time.perf_counter() - start_time
                }
            
            all_models_info.append(model_info)
            logger.info(f"Successfully fitted model with config {config}")
            
        except Exception as e:
            logger.warning(f"Failed to fit model with params {config}: {str(e)}")
    
    # Sort models by criterion value
    all_models_info = sorted(all_models_info, key=lambda x: x.get('criterion_value', float('inf')))
    
    return all_models_info