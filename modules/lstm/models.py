"""
Классы моделей нейронных сетей для прогнозирования временных рядов.

Реализованы следующие модели:
- Базовый абстрактный класс BaseTimeSeriesModel
- LSTM (Long Short-Term Memory) модель
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from sklearn.preprocessing import MinMaxScaler
import warnings

# Подавляем предупреждения TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


class BaseTimeSeriesModel(ABC):
    """
    Базовый абстрактный класс для всех моделей временных рядов.
    Определяет общий интерфейс для работы с моделями прогнозирования.
    """
    
    def __init__(self):
        """
        Инициализация базовой модели
        """
        self.model = None
        self.is_fitted = False
        self.scaler = None
        self.sequence_length = None
        self.input_shape = None
        self.training_history = None
    
    @abstractmethod
    def fit(self, series: pd.Series, **kwargs) -> 'BaseTimeSeriesModel':
        """
        Обучение модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
        **kwargs : dict
            Дополнительные параметры для обучения
            
        Возвращает:
        -----------
        BaseTimeSeriesModel
            Обученная модель
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int = 1) -> pd.Series:
        """
        Прогнозирование будущих значений временного ряда.
        
        Параметры:
        -----------
        steps : int, default=1
            Количество шагов для прогноза
            
        Возвращает:
        -----------
        pd.Series
            Прогнозные значения
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Получение параметров модели.
        
        Возвращает:
        -----------
        Dict
            Словарь с параметрами модели
        """
        pass
    
    def plot_forecast(self, steps: int = 10, ax=None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Построение графика прогноза.
        
        Параметры:
        -----------
        steps : int, default=10
            Количество шагов для прогноза
        ax : matplotlib.axes, optional
            Объект осей для построения графика
        figsize : Tuple[int, int], default=(12, 6)
            Размер графика
            
        Возвращает:
        -----------
        plt.Figure
            Объект графика
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Получаем прогнозные данные
        forecast = self.predict(steps=steps)
        
        # Создаем график
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Строим прогноз
        ax.plot(forecast.index, forecast.values, 'r--', label='Прогноз')
        
        # Добавляем подписи и легенду
        ax.set_title('Прогноз временного ряда')
        ax.set_xlabel('Время')
        ax.set_ylabel('Значение')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_model(self, filepath: str) -> None:
        """
        Сохранение модели в файл.
        
        Параметры:
        -----------
        filepath : str
            Путь для сохранения модели
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Создаем директорию, если ее не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Сохраняем модель и ее параметры
        model_params = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'input_shape': self.input_shape,
            'is_fitted': self.is_fitted,
            'model_params': self.get_params()
        }
        
        # Сохраняем нейросетевую модель отдельно
        model_path = filepath + '.keras'
        if self.model:
            self.model.save(model_path)
        
        # Сохраняем остальные параметры
        joblib.dump(model_params, filepath + '.params')
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseTimeSeriesModel':
        """
        Загрузка модели из файла.
        
        Параметры:
        -----------
        filepath : str
            Путь к файлу модели
            
        Возвращает:
        -----------
        BaseTimeSeriesModel
            Загруженная модель
        """
        if not os.path.exists(filepath + '.params'):
            raise FileNotFoundError(f"Файл параметров модели {filepath}.params не найден")
        if not os.path.exists(filepath + '.keras'):
            raise FileNotFoundError(f"Файл модели {filepath}.keras не найден")
        
        # Загружаем параметры
        model_params = joblib.load(filepath + '.params')
        
        # Создаем экземпляр модели
        model_instance = cls()
        model_instance.scaler = model_params['scaler']
        model_instance.sequence_length = model_params['sequence_length']
        model_instance.input_shape = model_params['input_shape']
        model_instance.is_fitted = model_params['is_fitted']
        
        # Загружаем модель
        model_instance.model = load_model(filepath + '.keras')
        
        return model_instance


class LSTMModel(BaseTimeSeriesModel):
    """
    LSTM (Long Short-Term Memory) модель для прогнозирования временных рядов.
    
    LSTM - это тип рекуррентной нейронной сети, способный обучаться долгосрочным зависимостям
    во временных рядах. LSTM имеет специальную архитектуру с "ячейками памяти", которые
    могут сохранять информацию в течение длительных периодов времени.
    
    Математическая формулировка:
    - Входной вентиль: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
    - Вентиль забывания: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
    - Выходной вентиль: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
    - Кандидат на новую память: C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
    - Обновление ячейки памяти: C_t = f_t * C_{t-1} + i_t * C̃_t
    - Выход: h_t = o_t * tanh(C_t)
    
    Параметры:
    -----------
    sequence_length : int
        Длина входной последовательности (лаг)
    units : list
        Список с количеством нейронов в каждом слое LSTM
    dropout_rate : float
        Коэффициент прореживания (dropout)
    bidirectional : bool
        Использовать ли двунаправленный LSTM
    """
    
    def __init__(self, sequence_length: int = 10, units: List[int] = [100, 50], 
                 dropout_rate: float = 0.2, bidirectional: bool = False):
        """
        Инициализация LSTM модели.
        
        Параметры:
        -----------
        sequence_length : int, default=10
            Длина входной последовательности (лаг)
        units : list, default=[100, 50]
            Список с количеством нейронов в каждом слое LSTM
        dropout_rate : float, default=0.2
            Коэффициент прореживания (dropout)
        bidirectional : bool, default=False
            Использовать ли двунаправленный LSTM
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.last_sequence = None
        
        # Для хранения обучающих и тестовых данных
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_actual = None
        self.y_test_actual = None
        self.train_indices = None
        self.test_indices = None
        self.training_history = None
    
    def _build_model(self):
        """
        Создает и компилирует модель LSTM с заданной архитектурой.
        """
        model = Sequential()
        
        # Добавляем первый слой LSTM
        if len(self.units) > 1:
            if self.bidirectional:
                model.add(Bidirectional(
                    LSTM(self.units[0], activation='relu', return_sequences=True),
                    input_shape=self.input_shape
                ))
            else:
                model.add(LSTM(
                    self.units[0], activation='relu', return_sequences=True,
                    input_shape=self.input_shape
                ))
            model.add(Dropout(self.dropout_rate))
            
            # Добавляем промежуточные слои LSTM, если они есть
            for i in range(1, len(self.units) - 1):
                if self.bidirectional:
                    model.add(Bidirectional(LSTM(self.units[i], activation='relu', return_sequences=True)))
                else:
                    model.add(LSTM(self.units[i], activation='relu', return_sequences=True))
                model.add(Dropout(self.dropout_rate))
            
            # Последний слой LSTM
            if self.bidirectional:
                model.add(Bidirectional(LSTM(self.units[-1], activation='relu')))
            else:
                model.add(LSTM(self.units[-1], activation='relu'))
            model.add(Dropout(self.dropout_rate))
        else:
            # Если только один слой
            if self.bidirectional:
                model.add(Bidirectional(
                    LSTM(self.units[0], activation='relu'),
                    input_shape=self.input_shape
                ))
            else:
                model.add(LSTM(
                    self.units[0], activation='relu',
                    input_shape=self.input_shape
                ))
            model.add(Dropout(self.dropout_rate))
        
        # Выходной слой
        model.add(Dense(1))
        
        # Компиляция модели
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def fit(self, series: pd.Series, epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.1, early_stopping: bool = True,
            patience: int = 10, verbose: int = 0, train_size: float = 0.8) -> 'LSTMModel':
        """
        Обучение модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
        epochs : int, default=100
            Количество эпох обучения
        batch_size : int, default=32
            Размер батча
        validation_split : float, default=0.1
            Доля обучающих данных, используемых для валидации
        early_stopping : bool, default=True
            Использовать ли раннюю остановку
        patience : int, default=10
            Количество эпох ожидания улучшения при ранней остановке
        verbose : int, default=0
            Уровень вывода информации во время обучения
        train_size : float, default=0.8
            Доля данных для обучения
            
        Возвращает:
        -----------
        LSTMModel
            Обученная модель
        """
        # Проверяем входные данные
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        
        # Сохраняем индексы для создания прогнозов потом
        if hasattr(series, 'index'):
            n_train = int(len(series) * train_size)
            self.train_indices = series.index[:n_train]
            self.test_indices = series.index[n_train:]
        
        # Преобразуем в numpy массив
        data = np.array(series).reshape(-1, 1)
        
        # Разделение на обучающую и тестовую выборки
        train_data, test_data = self._train_test_split(data, train_size)
        
        # Масштабирование данных
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Создание последовательностей
        self.X_train, self.y_train = self._create_sequences(train_scaled, self.sequence_length)
        self.X_test, self.y_test = self._create_sequences(test_scaled, self.sequence_length)
        
        # Сохраняем исходные значения для расчета метрик
        self.y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        
        # Проверка наличия данных после создания последовательностей
        if len(self.X_train) == 0:
            raise ValueError(f"Недостаточно данных для создания последовательностей. "
                           f"Уменьшите sequence_length (текущее значение: {self.sequence_length})")
        
        # Устанавливаем форму входных данных
        self.input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        # Создаем и компилируем модель
        self.model = self._build_model()
        
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
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            self.training_history = history.history
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении модели: {str(e)}")
        
        # Сохраняем последнюю последовательность для будущих прогнозов
        self.last_sequence = train_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Отмечаем модель как обученную
        self.is_fitted = True
        
        return self
    
    def predict(self, steps: int = 1, input_sequence: Optional[np.ndarray] = None) -> pd.Series:
        """
        Прогнозирование будущих значений временного ряда.
        
        Параметры:
        -----------
        steps : int, default=1
            Количество шагов для прогноза
        input_sequence : np.ndarray, optional
            Входная последовательность для прогноза. Если не указана, используется последняя
            последовательность из обучающих данных.
            
        Возвращает:
        -----------
        pd.Series
            Прогнозные значения
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Используем заданную последовательность или последнюю из обучения
        current_sequence = input_sequence if input_sequence is not None else self.last_sequence.copy()
        
        if current_sequence is None:
            raise ValueError("Отсутствует входная последовательность для прогноза")
        
        future_predictions = []
        
        # Генерируем прогнозы по одному шагу за раз
        for _ in range(steps):
            # Получаем прогноз
            next_pred = self.model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            # Обновляем последовательность для следующего шага прогноза
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Преобразуем прогнозы обратно в исходный масштаб
        future_preds_array = np.array(future_predictions).reshape(-1, 1)
        future_preds_rescaled = self.scaler.inverse_transform(future_preds_array)
        
        # Создаем Series с прогнозами
        future_preds = pd.Series(future_preds_rescaled.flatten())
        
        return future_preds
    
    def get_params(self) -> Dict:
        """
        Получение параметров модели.
        
        Возвращает:
        -----------
        Dict
            Словарь с параметрами модели
        """
        return {
            'sequence_length': self.sequence_length,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional
        }
    
    def _train_test_split(self, data, train_size):
        """
        Вспомогательный метод для разделения данных на обучающую и тестовую выборки.
        """
        n = len(data)
        train_size_idx = int(n * train_size)
        
        train_data = data[:train_size_idx]
        test_data = data[train_size_idx:]
        
        return train_data, test_data
    
    def _create_sequences(self, data, sequence_length):
        """
        Вспомогательный метод для создания последовательностей.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def predict_train(self) -> pd.Series:
        """
        Делает прогноз на обучающей выборке.
        
        Возвращает:
        -----------
        pd.Series
            Прогноз на обучающей выборке
        """
        if not self.is_fitted or self.X_train is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Предсказание на обучающей выборке
        train_predict = self.model.predict(self.X_train, verbose=0)
        train_predict = self.scaler.inverse_transform(train_predict)
        
        # Если есть индексы для возврата Series
        if self.train_indices is not None:
            # Учитываем sequence_length при создании индексов
            train_indices = self.train_indices[self.sequence_length:]
            if len(train_indices) != len(train_predict):
                train_indices = train_indices[:len(train_predict)]
            return pd.Series(train_predict.flatten(), index=train_indices)
        else:
            return pd.Series(train_predict.flatten())
    
    def predict_test(self) -> pd.Series:
        """
        Делает прогноз на тестовой выборке.
        
        Возвращает:
        -----------
        pd.Series
            Прогноз на тестовой выборке
        """
        if not self.is_fitted or self.X_test is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Предсказание на тестовой выборке
        test_predict = self.model.predict(self.X_test, verbose=0)
        test_predict = self.scaler.inverse_transform(test_predict)
        
        # Если есть индексы для возврата Series
        if self.test_indices is not None:
            # Учитываем sequence_length при создании индексов
            test_indices = self.test_indices[self.sequence_length:]
            if len(test_indices) != len(test_predict):
                test_indices = test_indices[:len(test_predict)]
            return pd.Series(test_predict.flatten(), index=test_indices)
        else:
            return pd.Series(test_predict.flatten())
    
    def forecast(self, steps: int = 10) -> pd.Series:
        """
        Прогнозирует будущие значения, продолжая ряд.
        Это алиас для метода predict, оставленный для совместимости.
        
        Параметры:
        -----------
        steps : int, default=10
            Количество шагов для прогноза
            
        Возвращает:
        -----------
        pd.Series
            Прогнозные значения
        """
        return self.predict(steps=steps) 