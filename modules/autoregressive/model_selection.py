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
              m: Optional[int] = None, test_values: List[int] = None,
              information_criterion: str = 'aic', n_jobs: int = 1,
              return_all_models: bool = False, **kwargs) -> Union[SARIMAModel, ARIMAModel, Dict]:
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
    **kwargs : dict
        Дополнительные параметры для pmdarima.auto_arima
        
    Возвращает:
    -----------
    Union[SARIMAModel, ARIMAModel, Dict]
        Лучшая модель или словарь с результатами всех моделей, если return_all_models=True
    """
    # Если сезонность включена, но период не указан, определяем его
    if seasonal and m is None:
        seasonality_result = detect_seasonality(series, test_values=test_values)
        m = seasonality_result.get('periodicity', 12)  # По умолчанию 12, если не удалось определить
        
        # Если сезонность не обнаружена, отключаем её
        if not seasonality_result.get('has_seasonality', False):
            seasonal = False
    
    # Используем pmdarima.auto_arima для поиска лучшей модели
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Базовые настройки, которые могут быть переопределены через kwargs
        default_params = {
            'start_p': 1,  # Начинаем с p=1 вместо p=0
            'start_q': 1,  # Начинаем с q=1 вместо q=0
            'start_P': 0,
            'start_Q': 0,
            'error_action': 'ignore',
            'suppress_warnings': True,
            'stepwise': False,  # Используем полный поиск вместо поэтапного
            'trace': True if return_all_models else False,
            'allow_ar_only': True,  # Разрешаем чистые AR модели
            'allow_ma_only': True,  # Разрешаем чистые MA модели
            'enforce_stationarity': True if max_d == 0 else False,  # Только для ARMA
            'enforce_invertibility': True,
        }
        
        # Обновляем параметры теми, что пришли в kwargs
        for key, value in default_params.items():
            kwargs.setdefault(key, value)
        
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
    
    # Получаем параметры лучшей модели
    order = best_model.order
    seasonal_order = best_model.seasonal_order if seasonal else (0, 0, 0, 0)
    
    # Проверка, не выбрана ли модель (0,0)
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
    
    if return_all_models:
        return {
            'best_model': model,
            'pmdarima_model': best_model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': best_model.aic() if hasattr(best_model, 'aic') else None,
            'bic': best_model.bic() if hasattr(best_model, 'bic') else None,
            'aicc': best_model.aicc() if hasattr(best_model, 'aicc') else None,
            'is_seasonal': seasonal and seasonal_order != (0, 0, 0, 0)
        }
    
    return model


def evaluate_model_performance(model: BaseTimeSeriesModel, train: pd.Series, test: pd.Series) -> Dict[str, float]:
    """
    Оценивает производительность обученной модели на тестовом наборе данных.
    
    Параметры:
    -----------
    model : BaseTimeSeriesModel
        Обученная модель
    train : pd.Series
        Обучающий набор данных
    test : pd.Series
        Тестовый набор данных
        
    Возвращает:
    -----------
    Dict[str, float]
        Словарь с метриками качества модели
    """
    if not model.is_fitted:
        raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
    
    # Делаем прогноз на длину тестового набора
    forecast = model.predict(steps=len(test))
    
    # Вычисляем метрики
    metrics = {}
    
    # Средняя квадратичная ошибка (MSE)
    metrics['mse'] = mean_squared_error(test, forecast)
    
    # Корень из средней квадратичной ошибки (RMSE)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Средняя абсолютная ошибка (MAE)
    metrics['mae'] = mean_absolute_error(test, forecast)
    
    # Средняя абсолютная процентная ошибка (MAPE)
    # Избегаем деления на нуль, заменяя нулевые значения на очень маленькие
    test_nonzero = test.copy()
    test_nonzero = test_nonzero.replace(0, 1e-10)
    try:
        metrics['mape'] = mean_absolute_percentage_error(test_nonzero, forecast)
    except:
        metrics['mape'] = np.nan
    
    # Симметричная MAPE (sMAPE) - альтернативная мера, более устойчивая к выбросам
    denominator = (np.abs(test) + np.abs(forecast)) / 2
    smape = np.mean(np.abs(test - forecast) / denominator)
    metrics['smape'] = smape
    
    # Коэффициент Тейла-2 (Theil's U2) - отношение MSE модели к MSE наивной модели
    naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    mse_naive = mean_squared_error(test, naive_forecast)
    metrics['theil_u2'] = metrics['mse'] / mse_naive if mse_naive > 0 else np.nan
    
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