"""
Классы авторегрессионных моделей для анализа временных рядов.

Реализованы следующие модели:
- Базовый абстрактный класс BaseTimeSeriesModel
- ARMA (AutoRegressive Moving Average)
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal AutoRegressive Integrated Moving Average)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os


class BaseTimeSeriesModel(ABC):
    """
    Базовый абстрактный класс для всех авторегрессионских моделей.
    Определяет общий интерфейс для работы с моделями временных рядов.
    """
    
    def __init__(self):
        """
        Инициализация базовой модели
        """
        self.fitted_model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseTimeSeriesModel':
        """
        Обучение модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
            
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
        
        # Получаем исторические и прогнозные данные
        historical_data = self.fitted_model.model.endog
        if isinstance(historical_data, np.ndarray):
            historical_data = pd.Series(historical_data)
        
        forecast = self.predict(steps=steps)
        
        # Создаем график
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Строим исторические данные
        ax.plot(historical_data.index if hasattr(historical_data, 'index') else range(len(historical_data)), 
               historical_data, label='Исторические данные')
        
        # Строим прогноз
        ax.plot(forecast.index if hasattr(forecast, 'index') else range(len(historical_data), len(historical_data) + len(forecast)), 
               forecast, 'r--', label='Прогноз')
        
        # Добавляем подписи и легенду
        ax.set_title('Временной ряд и прогноз')
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
        
        # Сохраняем модель
        joblib.dump(self, filepath)
    
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл модели {filepath} не найден")
        
        return joblib.load(filepath)
    
    def predict_in_sample(self) -> pd.Series:
        """
        Возвращает предсказанные значения для обучающей выборки.
        
        Возвращает:
        -----------
        pd.Series
            Предсказанные значения на исторических данных, которые использовались при обучении
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        if hasattr(self.fitted_model, 'fittedvalues'):
            # Для моделей из statsmodels (ARIMA, SARIMA и др.)
            fitted_values = self.fitted_model.fittedvalues
            
            # Обработка возможных пропусков (NaN) в начале
            if isinstance(fitted_values, pd.Series):
                fitted_values = fitted_values.dropna()
            
            return fitted_values
        else:
            # Базовая реализация для других моделей
            # Предсказываем значения на тех же данных, что и обучались
            if not hasattr(self, 'train_data'):
                raise ValueError("Отсутствуют данные обучения. Нельзя предсказать значения.")
                
            return self.predict(steps=0)  # Просим предсказание без прогноза вперед


class ARMAModel(BaseTimeSeriesModel):
    """
    ARMA (AutoRegressive Moving Average) модель.
    
    Параметры:
    -----------
    p : int
        Порядок авторегрессии (AR)
    q : int
        Порядок скользящего среднего (MA)
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Инициализация ARMA модели.
        
        Параметры:
        -----------
        p : int, default=1
            Порядок авторегрессии (AR)
        q : int, default=1
            Порядок скользящего среднего (MA)
        """
        super().__init__()
        self.p = p
        self.q = q
        self.model_name = "ARMA"
        self.fit_options = {}  # Дополнительные опции для метода fit
    
    def fit(self, series: pd.Series) -> 'ARMAModel':
        """
        Обучение ARMA модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
            
        Возвращает:
        -----------
        ARMAModel
            Обученная модель
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                # ARMA - это частный случай ARIMA с d=0
                model = SARIMAX(series, order=(self.p, 0, self.q))
                
                # Используем дополнительные опции подгонки, если они есть
                if hasattr(self, 'fit_options') and self.fit_options:
                    method = self.fit_options.get('method', None)
                    maxiter = self.fit_options.get('maxiter', None)
                    
                    if method is not None and maxiter is not None:
                        self.fitted_model = model.fit(method=method, maxiter=maxiter, disp=False)
                    elif method is not None:
                        self.fitted_model = model.fit(method=method, disp=False)
                    else:
                        self.fitted_model = model.fit(disp=False)
                else:
                    self.fitted_model = model.fit(disp=False)
                
                self.is_fitted = True
                self.train_data = series
            except Exception as e:
                warnings.warn(f"Ошибка при обучении ARMA модели: {str(e)}")
                raise e
        
        return self
    
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
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Делаем прогноз
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def get_params(self) -> Dict:
        """
        Получение параметров модели.
        
        Возвращает:
        -----------
        Dict
            Словарь с параметрами модели
        """
        return {'p': self.p, 'q': self.q, 'type': 'ARMA'}


class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) модель.
    
    Параметры:
    -----------
    p : int
        Порядок авторегрессии (AR)
    d : int
        Порядок интегрирования/дифференцирования (I)
    q : int
        Порядок скользящего среднего (MA)
    """
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        """
        Инициализация ARIMA модели.
        
        Параметры:
        -----------
        p : int, default=1
            Порядок авторегрессии (AR)
        d : int, default=1
            Порядок интегрирования/дифференцирования (I)
        q : int, default=1
            Порядок скользящего среднего (MA)
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model_name = "ARIMA"
        self.fit_options = {}  # Дополнительные опции для метода fit
    
    def fit(self, series: pd.Series) -> 'ARIMAModel':
        """
        Обучение ARIMA модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
            
        Возвращает:
        -----------
        ARIMAModel
            Обученная модель
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = ARIMA(series, order=(self.p, self.d, self.q))
                
                # Используем дополнительные опции подгонки, если они есть
                if hasattr(self, 'fit_options') and self.fit_options:
                    method = self.fit_options.get('method', None)
                    maxiter = self.fit_options.get('maxiter', None)
                    
                    if method is not None:
                        self.fitted_model = model.fit(method=method)
                    elif method is not None and maxiter is not None:
                        self.fitted_model = model.fit(method=method, maxiter=maxiter)
                    else:
                        self.fitted_model = model.fit()
                else:
                    self.fitted_model = model.fit()
                    
                self.is_fitted = True
                self.train_data = series
            except Exception as e:
                warnings.warn(f"Ошибка при обучении ARIMA модели: {str(e)}")
                raise e
        
        return self
    
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
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Делаем прогноз
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def get_params(self) -> Dict:
        """
        Получение параметров модели.
        
        Возвращает:
        -----------
        Dict
            Словарь с параметрами модели
        """
        return {'p': self.p, 'd': self.d, 'q': self.q, 'type': 'ARIMA'}


class SARIMAModel(BaseTimeSeriesModel):
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) модель.
    
    Параметры:
    -----------
    p : int
        Порядок авторегрессии (AR)
    d : int
        Порядок интегрирования/дифференцирования (I)
    q : int
        Порядок скользящего среднего (MA)
    P : int
        Порядок сезонной авторегрессии (SAR)
    D : int
        Порядок сезонного интегрирования/дифференцирования (SI)
    Q : int
        Порядок сезонного скользящего среднего (SMA)
    m : int
        Период сезонности
    """
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1, 
                P: int = 1, D: int = 1, Q: int = 1, m: int = 12):
        """
        Инициализация SARIMA модели.
        
        Параметры:
        -----------
        p : int, default=1
            Порядок авторегрессии (AR)
        d : int, default=1
            Порядок интегрирования/дифференцирования (I)
        q : int, default=1
            Порядок скользящего среднего (MA)
        P : int, default=1
            Порядок сезонной авторегрессии (SAR)
        D : int, default=1
            Порядок сезонного интегрирования/дифференцирования (SI)
        Q : int, default=1
            Порядок сезонного скользящего среднего (SMA)
        m : int, default=12
            Период сезонности
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.model_name = "SARIMA"
        self.fit_options = {}  # Дополнительные опции для метода fit
    
    def fit(self, series: pd.Series) -> 'SARIMAModel':
        """
        Обучение SARIMA модели на временном ряде.
        
        Параметры:
        -----------
        series : pd.Series
            Временной ряд для обучения
            
        Возвращает:
        -----------
        SARIMAModel
            Обученная модель
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = SARIMAX(
                    series, 
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m)
                )
                
                # Используем дополнительные опции подгонки, если они есть
                if hasattr(self, 'fit_options') and self.fit_options:
                    method = self.fit_options.get('method', None)
                    maxiter = self.fit_options.get('maxiter', None)
                    
                    if method is not None and maxiter is not None:
                        self.fitted_model = model.fit(method=method, maxiter=maxiter, disp=False)
                    elif method is not None:
                        self.fitted_model = model.fit(method=method, disp=False)
                    else:
                        self.fitted_model = model.fit(disp=False)
                else:
                    self.fitted_model = model.fit(disp=False)
                
                self.is_fitted = True
                self.train_data = series
            except Exception as e:
                warnings.warn(f"Ошибка при обучении SARIMA модели: {str(e)}")
                raise e
                return None
        return self
    
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
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")
        
        # Делаем прогноз
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def get_params(self) -> Dict:
        """
        Получение параметров модели.
        
        Возвращает:
        -----------
        Dict
            Словарь с параметрами модели
        """
        return {
            'p': self.p, 'd': self.d, 'q': self.q,
            'P': self.P, 'D': self.D, 'Q': self.Q, 'm': self.m,
            'type': 'SARIMA'
        }