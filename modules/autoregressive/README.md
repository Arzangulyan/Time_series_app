# Модуль авторегрессионных методов для анализа временных рядов

## Описание

Модуль `autoregressive` предоставляет инструменты для анализа временных рядов, подбора параметров авторегрессионных моделей (ARMA, ARIMA, SARIMA), оценки их качества и визуализации результатов.

## Структура модуля

Модуль состоит из следующих файлов:

1. **models.py** - Содержит классы моделей:
   - `BaseTimeSeriesModel` - абстрактный базовый класс
   - `ARMAModel` - модель для стационарных рядов
   - `ARIMAModel` - модель для рядов с трендом
   - `SARIMAModel` - модель для рядов с сезонностью

2. **core.py** - Основные функции для анализа и выбора параметров моделей:
   - `check_stationarity` - проверка стационарности временного ряда
   - `apply_differencing` - выполнение дифференцирования
   - `detect_seasonality` - определение сезонности
   - `auto_arima` - автоматический подбор параметров ARIMA/SARIMA
   - `evaluate_model_performance` - оценка качества модели

3. **visualization.py** - Функции для визуализации и диагностики:
   - `plot_time_series` - построение графика временного ряда
   - `plot_forecast` - визуализация прогноза
   - `analyze_residuals` - анализ остатков модели
   - `compare_models` - сравнение моделей

4. **utils.py** - Вспомогательные функции:
   - `check_input_series` - валидация входных данных
   - `clean_time_series` - очистка временного ряда
   - `split_train_test` - разделение на обучающую и тестовую выборки

5. **compatibility.py** - Функции для обеспечения обратной совместимости

## Как использовать модуль

### Импорт необходимых компонентов

```python
from modules.autoregressive import (
    ARIMAModel, SARIMAModel, 
    check_stationarity, auto_arima, 
    evaluate_model_performance, plot_forecast
)
```

### Проверка стационарности и определение параметров

```python
# Проверка стационарности временного ряда
stationarity_result = check_stationarity(time_series)
print(f"Ряд стационарен: {stationarity_result['is_stationary']}")

# Автоматический подбор модели
best_model = auto_arima(time_series, seasonal=True)
```

### Обучение модели и прогнозирование

```python
# Разделение на обучающую и тестовую выборки
from modules.autoregressive import split_train_test
train, test = split_train_test(time_series, train_size=0.8)

# Создание и обучение модели
model = SARIMAModel(p=1, d=1, q=1, P=1, D=1, Q=1, m=12)
model.fit(train)

# Прогнозирование
forecast = model.predict(steps=len(test))
```

### Оценка качества модели

```python
# Расчет метрик качества
metrics = evaluate_model_performance(model, train, test)
print(f"RMSE: {metrics['rmse']}")
print(f"MAE: {metrics['mae']}")
print(f"MAPE: {metrics['mape']}")

# Визуализация результатов
from modules.autoregressive import plot_forecast
fig = plot_forecast(model, steps=len(test), train_data=train, test_data=test)
```

### Анализ остатков и диагностика модели

```python
from modules.autoregressive import analyze_residuals, plot_residuals_diagnostic

# Анализ остатков
residuals_analysis = analyze_residuals(model)
print(f"Остатки нормальны: {residuals_analysis['shapiro_test']['is_normal']}")
print(f"Автокорреляция отсутствует: {residuals_analysis['ljung_box_test']['no_autocorrelation']}")

# Визуализация диагностики
fig = plot_residuals_diagnostic(model)
```

### Сравнение нескольких моделей

```python
from modules.autoregressive import compare_models

# Создаем несколько моделей
models = [
    ARIMAModel(p=1, d=1, q=1).fit(train),
    ARIMAModel(p=2, d=1, q=2).fit(train),
    SARIMAModel(p=1, d=1, q=1, P=1, D=1, Q=1, m=12).fit(train)
]

# Сравниваем модели
comparison = compare_models(models, test)
print(comparison)
```

## Дополнительная информация

Этот модуль объединяет функциональность нескольких предыдущих модулей и оптимизирован для использования в проекте TimeSeriesApp. Он предоставляет унифицированный интерфейс для работы с различными типами авторегрессионных моделей и может быть расширен для поддержки дополнительных методов анализа временных рядов. 