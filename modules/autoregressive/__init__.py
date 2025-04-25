"""
Пакет для работы с авторегрессионными моделями временных рядов.

Включает классы моделей ARMA, ARIMA, SARIMA и функции для подбора параметров,
анализа временных рядов, оценки качества моделей и визуализации результатов.
"""

# Импортируем классы моделей
from .models import (
    BaseTimeSeriesModel,
    ARMAModel, 
    ARIMAModel, 
    SARIMAModel
)

# Импортируем основные функции для анализа временных рядов
from .core import (
    check_stationarity,
    apply_differencing,
    detect_frequency,
    estimate_differencing_order,
    calculate_acf_pacf,
    detect_seasonality,
    auto_arima,
    evaluate_model_performance,
    split_train_test,
    suggest_arima_params,
    generate_model_report
)

# Импортируем функции визуализации
from .visualization import (
    plot_time_series,
    plot_train_test_split,
    plot_acf_pacf_plotly,
    plot_forecast,
    plot_forecast_plotly,
    display_model_information,
    display_differencing_effect,
    analyze_residuals,
    plot_residuals_diagnostic,
    compare_models,
    display_stationarity_results,
    display_acf_pacf
)

# Импортируем вспомогательные функции
from .utils import (
    check_input_series,
    validate_time_series,
    convert_to_datetime_index,
    estimate_forecast_horizon,
    generate_forecast_index,
    format_model_params,
    clean_time_series
)

# Определяем публичный API пакета
__all__ = [
    # Классы моделей
    'BaseTimeSeriesModel',
    'ARMAModel',
    'ARIMAModel',
    'SARIMAModel',
    
    # Основные функции анализа
    'check_stationarity',
    'apply_differencing',
    'detect_frequency',
    'estimate_differencing_order',
    'calculate_acf_pacf',
    'detect_seasonality',
    'auto_arima',
    'evaluate_model_performance',
    'split_train_test',
    'suggest_arima_params',
    'generate_model_report',
    
    # Функции визуализации
    'plot_time_series',
    'plot_train_test_split',
    'plot_acf_pacf_plotly',
    'plot_forecast',
    'plot_forecast_plotly',
    'display_model_information',
    'display_differencing_effect',
    'analyze_residuals',
    'plot_residuals_diagnostic',
    'compare_models',
    'display_stationarity_results',
    'display_acf_pacf',
    
    # Вспомогательные функции
    'check_input_series',
    'validate_time_series',
    'convert_to_datetime_index',
    'estimate_forecast_horizon',
    'generate_forecast_index',
    'format_model_params',
    'clean_time_series'
] 