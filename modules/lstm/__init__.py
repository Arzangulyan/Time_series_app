"""
Пакет для прогнозирования временных рядов с помощью LSTM (Long Short-Term Memory).

Включает классы моделей нейросетей, функции для подготовки данных, оценки качества
и визуализации результатов прогнозирования.
"""

# Импортируем классы моделей
from .models import (
    BaseTimeSeriesModel,
    LSTMModel
)

# Импортируем основные функции для работы с LSTM
from .core import (
    create_sequences,
    train_test_split_ts,
    auto_tune_lstm_params,
    forecast_future,
    calculate_metrics,
    prepare_data_for_forecast
)

# Импортируем функции визуализации
from .visualization import (
    plot_time_series,
    plot_train_test_results,
    plot_forecast,
    plot_training_history,
    plot_error_distribution,
    display_model_information,
    display_metrics
)

# Импортируем вспомогательные функции
from .utils import (
    check_input_series,
    scale_time_series,
    create_future_index,
    generate_forecast_index,
    save_results_to_csv,
    load_model_from_file
)

# Определяем публичный API пакета
__all__ = [
    # Классы моделей
    'BaseTimeSeriesModel',
    'LSTMModel',
    
    # Основные функции анализа
    'create_sequences',
    'train_test_split_ts',
    'auto_tune_lstm_params',
    'forecast_future',
    'calculate_metrics',
    'prepare_data_for_forecast',
    
    # Функции визуализации
    'plot_time_series',
    'plot_train_test_results',
    'plot_forecast',
    'plot_training_history',
    'plot_error_distribution',
    'display_model_information',
    'display_metrics',
    
    # Вспомогательные функции
    'check_input_series',
    'scale_time_series',
    'create_future_index',
    'generate_forecast_index',
    'save_results_to_csv',
    'load_model_from_file'
] 