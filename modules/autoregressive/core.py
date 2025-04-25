"""
Основные функции для анализа и выбора параметров авторегрессионных моделей.

Включает функции для:
- Проверки стационарности временных рядов
- Дифференцирования и определения оптимального порядка дифференцирования
- Обнаружения сезонности
- Автоматического подбора параметров моделей ARIMA/SARIMA
- Оценки качества моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
from pmdarima import auto_arima as pm_auto_arima

# Импортируем модели из модуля models
from .models import BaseTimeSeriesModel, ARMAModel, ARIMAModel, SARIMAModel


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
    
    # Проверяем, что у нас есть достаточно данных
    if len(series) < 8:
        return {
            'stationarity': "Недостаточно данных для проверки стационарности",
            'adf_test': None,
            'kpss_test': None,
            'is_stationary': False,
            'conclusion': "Недостаточно данных для проверки стационарности"
        }
    
    # Удаляем пропущенные значения
    series = series.dropna()
    
    # Тест ADF
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        adf_result = adfuller(series, regression='ct')
    
    adf_p_value = adf_result[1]
    adf_statistic = adf_result[0]
    adf_critical_values = adf_result[4]
    adf_is_stationary = adf_p_value < alpha
    
    # Тест KPSS
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            kpss_result = kpss(series, regression='ct', nlags='auto')
            kpss_p_value = kpss_result[1]
            kpss_statistic = kpss_result[0]
            kpss_critical_values = kpss_result[3]
            kpss_is_stationary = kpss_p_value >= alpha
        except:
            # Если KPSS тест не выполнился, предполагаем стационарность
            kpss_p_value = None
            kpss_statistic = None
            kpss_critical_values = {}
            kpss_is_stationary = True
    
    # Формирование вывода
    result['adf_statistic'] = adf_statistic
    result['adf_p_value'] = adf_p_value
    result['adf_critical_values'] = adf_critical_values
    result['adf_is_stationary'] = adf_is_stationary
    
    result['kpss_statistic'] = kpss_statistic
    result['kpss_p_value'] = kpss_p_value
    result['kpss_critical_values'] = kpss_critical_values
    result['kpss_is_stationary'] = kpss_is_stationary
    
    # Общий вывод на основе обоих тестов
    is_stationary = adf_is_stationary and kpss_is_stationary
    
    if adf_is_stationary and kpss_is_stationary:
        conclusion = "Ряд стационарный (по обоим тестам)"
    elif adf_is_stationary and not kpss_is_stationary:
        conclusion = "Возможно ряд стационарный (ADF показывает стационарность, KPSS показывает нестационарность)"
    elif not adf_is_stationary and kpss_is_stationary:
        conclusion = "Вероятно ряд нестационарный (ADF показывает нестационарность, KPSS показывает стационарность)"
    else:
        conclusion = "Ряд нестационарный (по обоим тестам)"
    
    result['is_stationary'] = is_stationary
    result['conclusion'] = conclusion
    
    return result


def apply_differencing(time_series: pd.Series, d: int = 0, D: int = 0, seasonal_period: Optional[int] = None) -> pd.Series:
    """
    Применяет обычное и сезонное дифференцирование к временному ряду.
    
    Математически:
    - Обычное дифференцирование: ∇y_t = y_t - y_{t-1}
    - Сезонное дифференцирование: ∇_s y_t = y_t - y_{t-s}, где s - сезонный период
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для дифференцирования
    d : int
        Порядок обычного дифференцирования
    D : int
        Порядок сезонного дифференцирования
    seasonal_period : int или None
        Сезонный период для сезонного дифференцирования
        
    Возвращает:
    -----------
    pd.Series
        Дифференцированный временной ряд
    """
    if d < 0 or D < 0:
        raise ValueError("Порядок дифференцирования не может быть отрицательным")
    
    result = time_series.copy()
    
    # Применяем обычное дифференцирование
    for _ in range(d):
        result = result.diff().dropna()
    
    # Применяем сезонное дифференцирование, если указан сезонный период
    if D > 0 and seasonal_period:
        for _ in range(D):
            result = result.diff(seasonal_period).dropna()
    
    return result


def detect_frequency(time_series: pd.Series) -> Optional[str]:
    """
    Определяет частоту временного ряда на основе его индекса.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд с DatetimeIndex
        
    Возвращает:
    -----------
    str или None
        Строковое представление определенной частоты или None, если не удалось определить
    """
    if not isinstance(time_series.index, pd.DatetimeIndex):
        try:
            time_series.index = pd.DatetimeIndex(time_series.index)
        except:
            return None
    
    # Если частота уже задана, возвращаем ее
    if time_series.index.freq is not None:
        return time_series.index.freq
    
    # Пытаемся определить частоту с помощью pandas
    inferred_freq = pd.infer_freq(time_series.index)
    if inferred_freq:
        return inferred_freq
    
    # Если pandas не смог определить частоту, рассчитываем её на основе медианной разницы
    if len(time_series) < 2:
        return None
    
    # Рассчитываем разницу между точками в днях
    deltas = []
    for i in range(1, min(30, len(time_series))):
        delta = (time_series.index[i] - time_series.index[i-1]).total_seconds() / (24 * 3600)  # в днях
        deltas.append(delta)
    
    # Определяем медианное расстояние
    median_days = np.median(deltas)
    
    # Определяем частоту на основе медианного расстояния
    if median_days < 1/24:  # Меньше часа
        return 'min'
    elif median_days < 1:  # Меньше дня
        return 'H'
    elif median_days < 7:  # Меньше недели
        return 'D'
    elif median_days < 31:  # Меньше месяца
        return 'W'
    elif median_days < 92:  # Меньше квартала
        return 'M'
    elif median_days < 366:  # Меньше года
        return 'Q'
    else:
        return 'Y'


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
        is_stationary = stat_result['is_stationary']
        
        if is_stationary:
            break
        
        # Если ряд нестационарный, делаем дифференцирование
        d += 1
        if d <= max_d:
            test_series = test_series.diff().dropna()
    
    return d


def calculate_acf_pacf(time_series: pd.Series, lags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Рассчитывает функции автокорреляции (ACF) и частичной автокорреляции (PACF) для временного ряда.
    
    Математически:
    - ACF: ρ(k) = Cov(y_t, y_{t-k}) / Var(y_t)
    - PACF: φ(k,k) - корреляция между y_t и y_{t-k} за вычетом влияния y_{t-1}, y_{t-2}, ..., y_{t-k+1}
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для анализа
    lags : int
        Количество лагов для расчета
        
    Возвращает:
    -----------
    Tuple[np.ndarray, np.ndarray]
        (acf_values, pacf_values)
    """
    # Ограничиваем количество лагов половиной длины ряда
    n = len(time_series)
    if lags > n // 2:
        lags = max(1, n // 2)
    
    # Удаляем пропущенные значения
    clean_series = time_series.dropna()
    
    try:
        # Расчет ACF
        acf_values = acf(clean_series, nlags=lags, fft=True)
        
        # Расчет PACF
        pacf_values = pacf(clean_series, nlags=lags, method='ywmle')
        
        return acf_values, pacf_values
    except Exception as e:
        warnings.warn(f"Ошибка при расчете ACF/PACF: {str(e)}")
        return np.array([]), np.array([])


def detect_seasonality(series: pd.Series, periodicity: Optional[int] = None, 
                     test_values: Optional[List[int]] = None) -> Dict[str, Any]:
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


def split_train_test(series: pd.Series, train_size: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.
    
    Параметры:
    -----------
    series : pd.Series
        Исходный временной ряд
    train_size : float, default=0.8
        Доля данных для обучающей выборки (от 0 до 1)
        
    Возвращает:
    -----------
    Tuple[pd.Series, pd.Series]
        Кортеж (train, test) с обучающей и тестовой выборками
    """
    n = len(series)
    train_idx = int(n * train_size)
    
    train = series.iloc[:train_idx]
    test = series.iloc[train_idx:]
    
    return train, test


def suggest_arima_params(time_series: pd.Series, seasonal: bool = False, 
                       seasonal_period: Optional[int] = None, max_order: int = 5) -> Dict[str, int]:
    """
    Предлагает параметры для моделей ARIMA и SARIMA на основе анализа ACF и PACF.
    
    Параметры:
    -----------
    time_series : pd.Series
        Временной ряд для анализа
    seasonal : bool
        Учитывать ли сезонность при подборе параметров
    seasonal_period : int или None
        Сезонный период для сезонного компонента
    max_order : int
        Максимальный рассматриваемый порядок
        
    Возвращает:
    -----------
    dict
        Словарь с рекомендуемыми параметрами
    """
    # Проверяем стационарность
    stationarity_result = check_stationarity(time_series)
    is_stationary = stationarity_result['is_stationary']
    
    # Определяем d на основе результатов теста
    d = 0 if is_stationary else 1
    
    # Рассчитываем ACF и PACF для оригинального или дифференцированного ряда
    if d > 0:
        diff_series = apply_differencing(time_series, d=d)
        acf_values, pacf_values = calculate_acf_pacf(diff_series, max_order * 2)
    else:
        acf_values, pacf_values = calculate_acf_pacf(time_series, max_order * 2)
    
    # Если не удалось рассчитать ACF/PACF, возвращаем значения по умолчанию
    if len(acf_values) == 0 or len(pacf_values) == 0:
        return {
            'p': 1, 'd': d, 'q': 1,
            'P': 0, 'D': 0, 'Q': 0, 'seasonal_period': seasonal_period
        }
    
    # Определяем p на основе PACF
    p = 0
    for i in range(1, min(max_order + 1, len(pacf_values))):
        if abs(pacf_values[i]) > 1.96 / np.sqrt(len(time_series)):
            p = max(p, i)
    
    # Определяем q на основе ACF
    q = 0
    for i in range(1, min(max_order + 1, len(acf_values))):
        if abs(acf_values[i]) > 1.96 / np.sqrt(len(time_series)):
            q = max(q, i)
    
    # Определяем сезонные параметры
    P, D, Q = 0, 0, 0
    
    if seasonal and seasonal_period:
        # Проверяем сезонную стационарность
        if not is_stationary:
            D = 1
        
        # Для расчета сезонных параметров используем ACF и PACF на сезонных лагах
        seasonal_lags = [seasonal_period * i for i in range(1, 4)]
        
        # Определяем P на основе сезонных значений PACF
        for lag in seasonal_lags:
            if lag < len(pacf_values) and abs(pacf_values[lag]) > 1.96 / np.sqrt(len(time_series)):
                P = 1
                break
        
        # Определяем Q на основе сезонных значений ACF
        for lag in seasonal_lags:
            if lag < len(acf_values) and abs(acf_values[lag]) > 1.96 / np.sqrt(len(time_series)):
                Q = 1
                break
    
    return {
        'p': p, 'd': d, 'q': q,
        'P': P, 'D': D, 'Q': Q, 'seasonal_period': seasonal_period
    }


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