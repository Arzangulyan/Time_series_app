# Модуль LSTM для прогнозирования временных рядов

## Описание

Модуль `lstm` предоставляет инструменты для прогнозирования временных рядов с использованием нейронных сетей LSTM (Long Short-Term Memory). Модуль включает реализацию моделей LSTM, функции для подготовки данных, обучения моделей, оценки их качества и визуализации результатов.

## Математическое описание

LSTM (Long Short-Term Memory) - это тип рекуррентной нейронной сети, специально разработанный для эффективной обработки последовательностей данных. В отличие от обычных рекуррентных нейронных сетей, LSTM способен запоминать долговременные зависимости благодаря своей архитектуре с "ячейками памяти".

### Ключевые компоненты LSTM ячейки:

1. **Вентиль забывания (forget gate)** - определяет, какую информацию нужно удалить из состояния ячейки:
   $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

2. **Вентиль входа (input gate)** - определяет, какую новую информацию добавить в состояние ячейки:
   $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
   $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

3. **Обновление состояния ячейки**:
   $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$

4. **Вентиль выхода (output gate)** - определяет, какую информацию выдавать на выход:
   $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
   $h_t = o_t * \tanh(C_t)$

где:
- $\sigma$ - сигмоидальная функция активации
- $\tanh$ - гиперболический тангенс
- $W_f, W_i, W_C, W_o$ - весовые матрицы
- $b_f, b_i, b_C, b_o$ - векторы смещения
- $h_{t-1}$ - выход с предыдущего шага
- $x_t$ - вход на текущем шаге
- $C_{t-1}$ - состояние ячейки на предыдущем шаге
- $C_t$ - состояние ячейки на текущем шаге
- $h_t$ - выход на текущем шаге

## Структура модуля

Модуль состоит из следующих файлов:

1. **models.py** - Содержит классы моделей:
   - `BaseTimeSeriesModel` - абстрактный базовый класс для моделей временных рядов
   - `LSTMModel` - модель LSTM для прогнозирования временных рядов

2. **core.py** - Основные функции для работы с LSTM:
   - `create_sequences` - создание последовательностей для обучения LSTM
   - `train_test_split_ts` - разделение временного ряда на обучающую и тестовую выборки
   - `auto_tune_lstm_params` - автоматический подбор параметров LSTM
   - `forecast_future` - прогнозирование будущих значений
   - `calculate_metrics` - расчет метрик качества
   - `prepare_data_for_forecast` - подготовка данных для прогнозирования

3. **visualization.py** - Функции для визуализации и диагностики:
   - `plot_time_series` - построение графика временного ряда
   - `plot_train_test_results` - визуализация результатов обучения и тестирования
   - `plot_forecast` - визуализация прогноза
   - `plot_training_history` - график процесса обучения
   - `plot_error_distribution` - анализ распределения ошибок
   - `display_model_information` - отображение информации о модели в Streamlit
   - `display_metrics` - отображение метрик качества в Streamlit

4. **utils.py** - Вспомогательные функции:
   - `check_input_series` - проверка и преобразование входных данных
   - `scale_time_series` - масштабирование временного ряда
   - `create_future_index` - создание индекса для будущих прогнозов
   - `generate_forecast_index` - генерация индекса для прогнозных значений
   - `save_results_to_csv` - сохранение результатов в CSV
   - `load_model_from_file` - загрузка модели из файла

## Как использовать модуль

### Импорт необходимых компонентов

```python
from modules.lstm import (
    LSTMModel, 
    create_sequences, 
    train_test_split_ts, 
    auto_tune_lstm_params, 
    calculate_metrics,
    plot_forecast,
    plot_training_history
)
```

### Подготовка данных и подбор параметров

```python
import pandas as pd

# Загрузка данных
data = pd.read_csv('timeseries_data.csv', parse_dates=['date'], index_col='date')
time_series = data['value']

# Автоматический подбор параметров
params = auto_tune_lstm_params(time_series)
print(f"Рекомендуемые параметры LSTM: {params}")

# Разделение на обучающую и тестовую выборки
train_size = 0.8
train_data, test_data = train_test_split_ts(time_series, train_size)
```

### Создание и обучение модели

```python
# Создание модели с выбранными параметрами
model = LSTMModel(
    sequence_length=params['sequence_length'],
    units=params['units'],
    dropout_rate=params['dropout_rate'],
    bidirectional=params['bidirectional']
)

# Обучение модели
trained_model = model.fit(
    series=train_data,
    epochs=params['epochs'],
    batch_size=params['batch_size'],
    validation_split=params['validation_split'],
    early_stopping=params['early_stopping'],
    patience=params['patience'],
    verbose=1
)

# Визуализация процесса обучения
history_plot = plot_training_history(model.training_history)
```

### Оценка качества модели

```python
# Прогнозирование на тестовой выборке
test_predictions = model.predict(steps=len(test_data))

# Расчет метрик качества
metrics = calculate_metrics(test_data.values, test_predictions.values)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

### Прогнозирование будущих значений

```python
# Прогноз на 30 шагов вперед
forecast = model.predict(steps=30)

# Визуализация прогноза
forecast_plot = plot_forecast(time_series, forecast)

# Сохранение модели для последующего использования
model.save_model('models/lstm_model')
```

### Интеграция с Streamlit

```python
import streamlit as st
from modules.lstm.visualization import display_model_information, display_metrics

# Отображение информации о модели
display_model_information(model.get_params())

# Отображение метрик качества
display_metrics(metrics)

# Отображение графика прогноза
st.plotly_chart(forecast_plot)
```

## Требования к данным

Для оптимальной работы LSTM моделей рекомендуется:

1. Использовать временные ряды с не менее чем 100 наблюдениями
2. Предварительно обрабатывать данные (удалять выбросы, заполнять пропуски)
3. Масштабировать данные перед подачей в модель (функция `scale_time_series`)
4. Выбирать подходящую длину последовательности (sequence_length) в зависимости от характеристик ряда

## Ограничения

- LSTM требует значительных вычислительных ресурсов для обучения на больших объемах данных
- Для рядов с сильной периодичностью может потребоваться дополнительная предобработка
- Качество прогноза снижается при увеличении горизонта прогнозирования

## Дополнительная информация

LSTM модели особенно эффективны для:
- Временных рядов с долговременными зависимостями
- Данных с сложной структурой, которую сложно моделировать статистическими методами
- Рядов с нелинейными взаимосвязями

Для улучшения результатов рекомендуется:
- Экспериментировать с различными архитектурами (изменять количество слоев и нейронов)
- Использовать раннюю остановку для предотвращения переобучения
- Комбинировать прогнозы нескольких моделей для повышения точности