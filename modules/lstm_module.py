"""
Модуль для прогнозирования временных рядов с помощью LSTM (Long Short-Term Memory).

Включает функционал для подготовки данных, обучения модели, оценки её точности
и построения прогнозов на будущие периоды.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Подавляем предупреждения TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


def auto_tune_lstm_params(time_series, complexity_level='auto'):
    """
    Автоматически подбирает параметры LSTM на основе характеристик временного ряда.
    
    Параметры:
    ----------
    time_series : pd.Series или pd.DataFrame
        Входной временной ряд
    complexity_level : str
        Уровень сложности модели ('simple', 'medium', 'complex', 'auto')
        
    Возвращает:
    -----------
    dict
        Словарь с подобранными параметрами
    """
    # Определяем длину ряда
    ts_length = len(time_series)
    
    # Определяем периодичность (если есть)
    freq = None
    if hasattr(time_series, 'index') and isinstance(time_series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(time_series.index)
    
    # Подбираем параметры в зависимости от длины ряда
    if complexity_level == 'auto':
        if ts_length < 500:
            complexity_level = 'simple'
        elif ts_length < 2000:
            complexity_level = 'medium'
        else:
            complexity_level = 'complex'
    
    # Настройки по уровням сложности
    params = {
        'simple': {
            'sequence_length': 5,
            'units': [64],
            'dropout_rate': 0.1,
            'batch_size': 16,
            'epochs': 50
        },
        'medium': {
            'sequence_length': 7,
            'units': [128, 64],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 100
        },
        'complex': {
            'sequence_length': 10,
            'units': [256, 128, 64],
            'dropout_rate': 0.3,
            'batch_size': 64,
            'epochs': 150
        }
    }
    
    selected_params = params[complexity_level]
    
    # Корректируем параметры в зависимости от частоты данных
    if freq:
        if 'D' in str(freq):  # Дневные данные
            selected_params['sequence_length'] = min(7, selected_params['sequence_length'])
        elif 'W' in str(freq):  # Недельные данные
            selected_params['sequence_length'] = min(8, selected_params['sequence_length'])
        elif 'M' in str(freq):  # Месячные данные
            selected_params['sequence_length'] = min(12, selected_params['sequence_length'])
    
    # Всегда включаем раннюю остановку
    selected_params['early_stopping'] = True
    selected_params['patience'] = 15
    selected_params['validation_split'] = 0.1
    
    return selected_params


def create_sequences(data, sequence_length):
    """
    Создает последовательности для обучения LSTM модели.
    
    Параметры:
    ----------
    data : array-like
        Входной временной ряд (масштабированный)
    sequence_length : int
        Длина входной последовательности (лаг)
        
    Возвращает:
    -----------
    tuple
        (X - входные последовательности, y - целевые значения)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)


def train_test_split_ts(data, train_size=0.8):
    """
    Разделяет временной ряд на обучающую и тестовую выборки.
    
    Параметры:
    ----------
    data : pd.Series или np.array
        Входной временной ряд
    train_size : float
        Доля данных для обучения (0 < train_size < 1)
        
    Возвращает:
    -----------
    tuple
        (train_data, test_data)
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.values.reshape(-1, 1)
    elif not isinstance(data, np.ndarray):
        data = np.array(data).reshape(-1, 1)
    
    # Проверка размерности
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n = len(data)
    train_size_idx = int(n * train_size)
    
    # Разделение данных
    train_data = data[:train_size_idx]
    test_data = data[train_size_idx:]
    
    return train_data, test_data


def build_lstm_model(input_shape, units=[100, 50], dropout_rate=0.2):
    """
    Создает модель LSTM с заданной архитектурой.
    
    Параметры:
    ----------
    input_shape : tuple
        Форма входных данных (sequence_length, features)
    units : list
        Список с количеством нейронов в каждом слое LSTM
    dropout_rate : float
        Коэффициент прореживания (dropout)
        
    Возвращает:
    -----------
    tensorflow.keras.models.Sequential
        Скомпилированная модель LSTM
    """
    model = Sequential()
    
    # Добавляем первый слой LSTM
    if len(units) > 1:
        model.add(LSTM(units[0], 
                      activation='relu', 
                      return_sequences=True, 
                      input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Добавляем промежуточные слои LSTM, если они есть
        for i in range(1, len(units) - 1):
            model.add(LSTM(units[i], activation='relu', return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Последний слой LSTM
        model.add(LSTM(units[-1], activation='relu'))
        model.add(Dropout(dropout_rate))
    else:
        # Если только один слой
        model.add(LSTM(units[0], activation='relu', input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    
    # Выходной слой
    model.add(Dense(1))
    
    # Компиляция модели
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def calculate_metrics(y_true, y_pred):
    """
    Рассчитывает метрики качества прогноза.
    
    Параметры:
    ----------
    y_true : array-like
        Фактические значения
    y_pred : array-like
        Предсказанные значения
        
    Возвращает:
    -----------
    dict
        Словарь с метриками (RMSE, MAE, MASE, R², Adjusted R²)
    """
    # Преобразуем в одномерные массивы
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Удаляем пропущенные значения
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:  # Нужно минимум 2 точки для расчета метрик
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mase': np.nan,
            'r2': np.nan,
            'adj_r2': np.nan
        }
    
    # Базовые метрики
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Расчет MASE (Mean Absolute Scaled Error)
    # Используем сдвиг на 1 как наивный прогноз для знаменателя
    naive_errors = np.abs(np.diff(y_true))
    if len(naive_errors) > 0 and np.mean(naive_errors) != 0:
        mase = np.mean(np.abs(y_pred - y_true)) / np.mean(naive_errors)
    else:
        mase = np.nan
    
    # Расчет R² (коэффициент детерминации)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot != 0:
        r2 = 1 - (ss_res / ss_tot)
        # Скорректированный R² учитывает количество предикторов
        n = len(y_true)
        p = 1  # количество предикторов (для временных рядов обычно 1)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        r2 = np.nan
        adj_r2 = np.nan
    
    # Ограничиваем R² и adj_r2 снизу значением -1 для более понятной интерпретации
    r2 = max(r2, -1) if not np.isnan(r2) else np.nan
    adj_r2 = max(adj_r2, -1) if not np.isnan(adj_r2) else np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mase': mase,
        'r2': r2,
        'adj_r2': adj_r2
    }


@st.cache_data(ttl=3600)
def train_lstm_model(time_series, sequence_length=5, epochs=50, batch_size=32, 
                     train_size=0.8, units=[100, 50], dropout_rate=0.2,
                     validation_split=0.1, early_stopping=True, patience=10):
    """
    Обучает модель LSTM на временном ряде и оценивает её качество.
    
    Параметры:
    ----------
    time_series : pd.Series или array-like
        Входной временной ряд
    sequence_length : int
        Длина входной последовательности (лаг)
    epochs : int
        Количество эпох обучения
    batch_size : int
        Размер батча
    train_size : float
        Доля данных для обучения
    units : list
        Список с количеством нейронов в каждом слое LSTM
    dropout_rate : float
        Коэффициент прореживания
    validation_split : float
        Доля обучающих данных, используемых для валидации
    early_stopping : bool
        Использовать ли раннюю остановку
    patience : int
        Количество эпох ожидания улучшения при ранней остановке
        
    Возвращает:
    -----------
    dict
        Результаты обучения и прогнозирования
    """
    # Проверяем входные данные
    if isinstance(time_series, pd.DataFrame):
        time_series = time_series.iloc[:, 0]
    
    # Преобразуем в numpy массив
    data = np.array(time_series).reshape(-1, 1)
    
    # Разделение на обучающую и тестовую выборки
    train_data, test_data = train_test_split_ts(data, train_size)
    
    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Создание последовательностей
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    
    # Проверка наличия данных после создания последовательностей
    if len(X_train) == 0 or len(X_test) == 0:
        return {
            'success': False,
            'error': f"Недостаточно данных для создания последовательностей. Уменьшите sequence_length (текущее значение: {sequence_length})"
        }
    
    # Построение модели
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, units, dropout_rate)
    
    # Настройка ранней остановки
    callbacks = []
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)
    
    # Обучение модели
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
    except Exception as e:
        return {
            'success': False,
            'error': f"Ошибка при обучении модели: {str(e)}"
        }
    
    # Прогнозы
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)
    
    # Обратное масштабирование
    train_predict = scaler.inverse_transform(train_predict)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Расчет метрик
    train_metrics = calculate_metrics(y_train_actual, train_predict)
    test_metrics = calculate_metrics(y_test_actual, test_predict)
    
    # Создание индексов для прогнозов
    train_index = time_series.index[sequence_length:len(train_data)] if hasattr(time_series, 'index') else np.arange(sequence_length, len(train_data))
    test_index = time_series.index[len(train_data) + sequence_length:] if hasattr(time_series, 'index') else np.arange(len(train_data) + sequence_length, len(data))
    
    # Формирование результатов для визуализации
    results = {
        'success': True,
        'model': model,
        'scaler': scaler,
        'history': history.history,
        'sequence_length': sequence_length,
        'input_shape': input_shape,
        'train_data': time_series.iloc[:len(train_data)] if hasattr(time_series, 'iloc') else train_data,
        'test_data': time_series.iloc[len(train_data):] if hasattr(time_series, 'iloc') else test_data,
        'train_predict': pd.Series(train_predict.flatten(), index=train_index),
        'test_predict': pd.Series(test_predict.flatten(), index=test_index),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    return results


def forecast_future(model, last_sequence, scaler, steps=10):
    """
    Прогнозирует будущие значения на основе обученной модели.
    
    Параметры:
    ----------
    model : tensorflow.keras.models.Sequential
        Обученная модель LSTM
    last_sequence : numpy.ndarray
        Последняя известная последовательность (масштабированная)
    scaler : sklearn.preprocessing.MinMaxScaler
        Масштабировщик для обратного преобразования
    steps : int
        Количество шагов для прогноза вперед
        
    Возвращает:
    -----------
    numpy.ndarray
        Массив с прогнозами
    """
    # Копируем последнюю последовательность
    current_sequence = last_sequence.copy()
    future_predictions = []
    
    # Генерируем прогнозы по одному шагу за раз
    for _ in range(steps):
        # Форматируем входные данные
        current_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        # Получаем прогноз
        next_pred = model.predict(current_input, verbose=0)
        future_predictions.append(next_pred[0, 0])
        # Обновляем последовательность для следующего шага прогноза
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = next_pred[0, 0]
    
    # Преобразуем прогнозы обратно
    future_preds_array = np.array(future_predictions).reshape(-1, 1)
    future_preds_rescaled = scaler.inverse_transform(future_preds_array)
    
    return future_preds_rescaled.flatten()


def prepare_data_for_display(results, forecast_future_steps=0):
    """
    Подготавливает данные для отображения в Streamlit.
    
    Параметры:
    ----------
    results : dict
        Результаты обучения и прогнозирования LSTM модели
    forecast_future_steps : int
        Количество шагов для прогноза вперед
        
    Возвращает:
    -----------
    pandas.DataFrame
        Датафрейм с данными для отображения
    """
    if not results['success']:
        return None
    
    # Получаем индексы для объединенного датафрейма
    if hasattr(results['train_data'], 'index'):
        all_indices = results['train_data'].index.tolist() + results['test_data'].index.tolist()
    else:
        # Создаем искусственные индексы, если нет временного индекса
        n_train = len(results['train_data'])
        n_test = len(results['test_data'])
        all_indices = pd.date_range(start='2000-01-01', periods=n_train + n_test)
    
    # Создаем датафрейм с исходными данными
    original_data = pd.Series(
        np.concatenate([
            results['train_data'].values if hasattr(results['train_data'], 'values') else results['train_data'].flatten(),
            results['test_data'].values if hasattr(results['test_data'], 'values') else results['test_data'].flatten()
        ]), 
        index=all_indices
    )
    
    # Создаем полный датафрейм
    df = pd.DataFrame({
        'Исходные данные': original_data
    })
    
    # Добавляем прогнозы для обучающего периода
    df['Прогноз (обучение)'] = np.nan
    df.loc[results['train_predict'].index, 'Прогноз (обучение)'] = results['train_predict']
    
    # Добавляем прогнозы для тестового периода
    df['Прогноз (тест)'] = np.nan
    df.loc[results['test_predict'].index, 'Прогноз (тест)'] = results['test_predict']
    
    # Прогноз на будущие периоды
    if forecast_future_steps > 0:
        # Получаем последнюю последовательность для прогноза
        sequence_length = results['sequence_length']
        last_data = original_data.iloc[-sequence_length:].values.reshape(-1, 1)
        
        # Масштабируем последовательность
        last_sequence = results['scaler'].transform(last_data)
        
        # Получаем прогноз
        future_preds = forecast_future(results['model'], last_sequence, results['scaler'], steps=forecast_future_steps)
        
        # Создаем индексы для будущих значений
        if isinstance(all_indices, pd.DatetimeIndex):
            # Если индекс временной, продолжаем его
            last_date = all_indices[-1]
            freq = pd.infer_freq(all_indices)
            if freq is None:
                # Если частота не определена, используем среднее расстояние
                avg_delta = (all_indices[-1] - all_indices[0]) / (len(all_indices) - 1)
                future_indices = pd.date_range(
                    start=last_date + avg_delta, 
                    periods=forecast_future_steps, 
                    freq=avg_delta
                )
            else:
                future_indices = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=freq), 
                    periods=forecast_future_steps, 
                    freq=freq
                )
        else:
            # Если индекс не временной, просто продолжаем нумерацию
            future_indices = np.arange(len(all_indices), len(all_indices) + forecast_future_steps)
        
        # Добавляем будущие прогнозы к датафрейму
        future_df = pd.DataFrame(
            {
                'Исходные данные': np.nan,
                'Прогноз (обучение)': np.nan,
                'Прогноз (тест)': np.nan
            }, 
            index=future_indices
        )
        future_df['Прогноз (будущее)'] = future_preds
        df = pd.concat([df, future_df])
        df['Прогноз (будущее)'] = df['Прогноз (будущее)']
    
    return df


def plot_training_history(history):
    """
    Строит график процесса обучения модели.
    
    Параметры:
    ----------
    history : dict
        Словарь с историей обучения
        
    Возвращает:
    -----------
    matplotlib.figure.Figure
        Фигура с графиком потерь
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['loss'], label='Обучение')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Валидация')
    ax.set_title('График потерь во время обучения')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Потери (MSE)')
    ax.legend()
    ax.grid(True)
    
    return fig
