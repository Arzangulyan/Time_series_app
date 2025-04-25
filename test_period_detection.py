import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Импортируем функции для обнаружения периодичностей
from modules.fourier_module import find_periods_using_periodogram
from modules.wavelet_module import find_significant_periods_wavelet

# Переименовываем функцию для обнаружения периодов методом Фурье
def detect_periods_with_fourier(time_series, max_periods=10):
    """
    Обертка для функции find_periods_using_periodogram для метода Фурье
    """
    return find_periods_using_periodogram(time_series, max_periods=max_periods)

def generate_test_data(periods, amplitudes, noise_level=1.0, trend=0.01, sample_count=1000):
    """
    Генерирует временной ряд с заданными периодами и амплитудами.
    
    Параметры:
    ----------
    periods : list
        Список периодов (в днях)
    amplitudes : list
        Список амплитуд для каждого периода
    noise_level : float, optional
        Уровень шума
    trend : float, optional
        Наклон тренда
    sample_count : int, optional
        Количество точек во временном ряде
    
    Возвращает:
    -----------
    pandas.DataFrame
        Временной ряд с датами в индексе
    """
    # Проверяем, что количество периодов и амплитуд совпадает
    if len(periods) != len(amplitudes):
        raise ValueError("Количество периодов и амплитуд должно совпадать")
    
    # Создаем временную шкалу
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(sample_count)]
    
    # Инициализируем временной ряд
    time_series = np.zeros(sample_count)
    
    # Добавляем периодические компоненты
    for period, amplitude in zip(periods, amplitudes):
        # Преобразуем период из дней в индексы
        period_idx = period
        
        # Создаем синусоидальную компоненту
        component = amplitude * np.sin(2 * np.pi * np.arange(sample_count) / period_idx)
        
        # Добавляем компоненту к временному ряду
        time_series += component
    
    # Добавляем линейный тренд
    time_series += trend * np.arange(sample_count)
    
    # Добавляем шум
    noise = noise_level * np.random.randn(sample_count)
    time_series += noise
    
    # Создаем DataFrame
    df = pd.DataFrame({'value': time_series}, index=dates)
    
    return df

def evaluate_detection(detected_periods, true_periods, tolerance=0.2):
    """
    Оценивает точность обнаружения периодов.
    
    Параметры:
    ----------
    detected_periods : array-like
        Обнаруженные периоды
    true_periods : array-like
        Истинные периоды
    tolerance : float, optional
        Допустимая относительная ошибка
        
    Возвращает:
    -----------
    float
        Доля правильно обнаруженных периодов
    """
    if len(detected_periods) == 0:
        return 0.0
    
    # Для каждого истинного периода проверяем, был ли он обнаружен
    detected_count = 0
    detected_periods_list = []
    
    for true_period in true_periods:
        # Ищем ближайший обнаруженный период
        min_error = float('inf')
        best_match = None
        
        for detected_period in detected_periods:
            # Вычисляем относительную ошибку
            rel_error = abs(detected_period - true_period) / true_period
            
            if rel_error < min_error:
                min_error = rel_error
                best_match = detected_period
        
        # Если ошибка меньше допустимой, считаем период обнаруженным
        if min_error <= tolerance:
            detected_count += 1
            detected_periods_list.append(best_match)
            
            # Выводим информацию о совпадении
            print(f"✅ Период {true_period} обнаружен как {best_match:.2f} (ошибка {min_error*100:.1f}%)")
        else:
            print(f"❌ Период {true_period} не обнаружен")
    
    # Вычисляем точность как долю обнаруженных периодов
    accuracy = detected_count / len(true_periods)
    
    return accuracy, detected_periods_list

# Функция для проверки точности обнаружения периодов
def check_detection_accuracy(true_periods, detected_periods, tolerance=0.2):
    """
    Проверяет точность обнаружения периодов.
    
    Параметры:
    ----------
    true_periods : array-like
        Истинные периоды
    detected_periods : array-like
        Обнаруженные периоды
    tolerance : float, optional
        Допустимая относительная ошибка
        
    Возвращает:
    -----------
    float
        Доля правильно обнаруженных периодов
    list
        Список обнаруженных периодов, соответствующих истинным
    """
    return evaluate_detection(detected_periods, true_periods, tolerance)

# Функция для вывода сравнения обнаруженных и фактических периодов
def print_comparison(true_periods, detected_periods):
    """
    Выводит сравнение обнаруженных и фактических периодов.
    
    Параметры:
    ----------
    true_periods : array-like
        Истинные периоды
    detected_periods : array-like
        Обнаруженные периоды
    """
    if not detected_periods:
        return
    
    print("Сравнение обнаруженных и фактических периодов:")
    print("  Фактический   Обнаруженный     Ошибка, %   ")
    print("---------------------------------------------")
    
    for i, (true, detected) in enumerate(zip(true_periods, detected_periods)):
        error_percent = abs(detected - true) / true * 100
        print(f"     {true:.2f}          {detected:.2f}           {error_percent:.1f}      ")

def test_period_detection(test_name, periods, amplitudes, noise_level=1.0, trend_slope=0.01, num_points=1000):
    """
    Тестирует обнаружение периодов различными методами.
    
    Параметры:
    ----------
    test_name : str
        Название теста
    periods : list
        Список периодов для генерации временного ряда
    amplitudes : list
        Список амплитуд для каждого периода
    noise_level : float, optional
        Уровень шума
    trend_slope : float, optional
        Наклон тренда
    num_points : int, optional
        Количество точек во временном ряде
    """
    print(f"=== {test_name} ===")
    
    # Генерируем временной ряд с заданными периодами
    df = generate_test_data(
        periods=periods,
        amplitudes=amplitudes,
        noise_level=noise_level,
        trend=trend_slope,
        sample_count=num_points
    )
    
    print(f"Сгенерирован временной ряд со следующими параметрами:")
    print(f"Периоды: {periods}")
    print(f"Амплитуды: {amplitudes}")
    print(f"Шум: {noise_level}, Тренд: {trend_slope}")
    print(f"Количество точек: {num_points}")
    print("\n")
    
    # Получаем временной ряд и даты
    time_series = df['value'].values
    dates = df.index
    
    # Анализируем методом Фурье
    print("=== Результаты методом Фурье ===")
    fourier_periods = detect_periods_with_fourier(
        df['value'], 
        max_periods=10
    )
    
    # Выводим результаты
    if not fourier_periods.empty:
        print("Обнаруженные периоды:")
        print(fourier_periods[['Период', 'Мощность', 'Интерпретация']])
        
        # Проверяем точность обнаружения
        fourier_accuracy, fourier_detected = evaluate_detection(fourier_periods['Период'].values, periods)
        
        # Выводим сравнение
        print_comparison(periods, fourier_detected)
        
        print(f"Точность обнаружения: {fourier_accuracy:.2f}")
        print()
    else:
        print("Не обнаружено значимых периодов методом Фурье.")
        fourier_accuracy = 0.0
        fourier_detected = []
    
    # Анализируем методом Периодограммы
    print("=== Результаты методом Периодограммы ===")
    periodogram_periods = find_periods_using_periodogram(
        df['value'],
        max_periods=10
    )
    
    # Выводим результаты
    if not periodogram_periods.empty:
        print("Обнаруженные периоды:")
        print(periodogram_periods[['Период', 'Мощность', 'Интерпретация']])
        
        # Проверяем точность обнаружения
        periodogram_accuracy, periodogram_detected = evaluate_detection(periodogram_periods['Период'].values, periods)
        
        # Выводим сравнение
        print_comparison(periods, periodogram_detected)
        
        print(f"Точность обнаружения: {periodogram_accuracy:.2f}")
        print()
    else:
        print("Не обнаружено значимых периодов методом Периодограммы.")
        periodogram_accuracy = 0.0
        periodogram_detected = []
    
    # Анализируем методом Вейвлет-преобразования
    print("=== Результаты методом Вейвлет-преобразования ===")
    wavelet_periods = find_significant_periods_wavelet(
        df, 
        mother_wavelet="Морле",
        power_threshold=0.1,
        max_periods=10
    )
    
    # Выводим результаты
    if not wavelet_periods.empty:
        print("Обнаруженные периоды:")
        print(wavelet_periods)
        
        # Проверяем точность обнаружения
        wavelet_accuracy, wavelet_detected = evaluate_detection(wavelet_periods['Период'].values, periods)
        
        # Выводим сравнение
        print_comparison(periods, wavelet_detected)
        
        print(f"Точность обнаружения: {wavelet_accuracy:.2f}")
        print()
    else:
        print("Не обнаружено значимых периодов методом Вейвлет-преобразования.")
        wavelet_accuracy = 0.0
        wavelet_detected = []
    
    # Визуализируем результаты
    plt.figure(figsize=(15, 10))
    
    # График временного ряда
    plt.subplot(4, 1, 1)
    plt.plot(dates, time_series)
    plt.title(f"{test_name}: Временной ряд")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    
    # График спектра Фурье
    plt.subplot(4, 1, 2)
    if not fourier_periods.empty:
        periods_fourier = fourier_periods['Период'].values
        amplitudes_fourier = fourier_periods['Мощность'].values
        plt.stem(periods_fourier, amplitudes_fourier, linefmt='b-', markerfmt='bo', basefmt='r-')
        
        # Добавляем вертикальные линии для истинных периодов
        for period in periods:
            plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"Спектр Фурье (точность: {fourier_accuracy:.2f})")
    plt.xlabel("Период")
    plt.ylabel("Мощность")
    plt.xscale('log')
    
    # График периодограммы
    plt.subplot(4, 1, 3)
    if not periodogram_periods.empty:
        periods_periodogram = periodogram_periods['Период'].values
        power_periodogram = periodogram_periods['Мощность'].values
        plt.stem(periods_periodogram, power_periodogram, linefmt='g-', markerfmt='go', basefmt='r-')
        
        # Добавляем вертикальные линии для истинных периодов
        for period in periods:
            plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"Периодограмма (точность: {periodogram_accuracy:.2f})")
    plt.xlabel("Период")
    plt.ylabel("Мощность")
    plt.xscale('log')
    
    # График вейвлет-спектра
    plt.subplot(4, 1, 4)
    if not wavelet_periods.empty:
        periods_wavelet = wavelet_periods['Период'].values
        power_wavelet = wavelet_periods['Нормализованная мощность'].values
        plt.stem(periods_wavelet, power_wavelet, linefmt='m-', markerfmt='mo', basefmt='r-')
        
        # Добавляем вертикальные линии для истинных периодов
        for period in periods:
            plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"Вейвлет-спектр (точность: {wavelet_accuracy:.2f})")
    plt.xlabel("Период")
    plt.ylabel("Нормализованная мощность")
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Сохраняем результаты
    filename = f"detection_results_{test_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Результаты сохранены в файл: {filename}")
    print(f"Точность методов:")
    print(f"Фурье: {fourier_accuracy:.2f}, Периодограмма: {periodogram_accuracy:.2f}, Вейвлет: {wavelet_accuracy:.2f}")
    
    return fourier_accuracy, periodogram_accuracy, wavelet_accuracy

if __name__ == "__main__":
    # Тестирование с разными наборами параметров
    
    print("=== Тест 1: Базовые периоды ===")
    test_period_detection(
        test_name="Тест 1: Базовые периоды",
        periods=[7, 30, 365],  # Недельный, месячный и годовой циклы
        amplitudes=[3, 5, 10],  # Амплитуды для каждого цикла
        noise_level=1.0,        # Уровень шума
        trend_slope=0.02,       # Наклон тренда
        num_points=1000
    )
    
    print("\n\n=== Тест 2: Короткие периоды ===")
    test_period_detection(
        test_name="Тест 2: Короткие периоды",
        periods=[3, 12, 50],    # Короткие периоды
        amplitudes=[2, 4, 6],   # Амплитуды для каждого цикла
        noise_level=0.8,        # Уровень шума
        trend_slope=0.01,       # Наклон тренда
        num_points=1000
    )
    
    print("\n\n=== Тест 3: Высокий уровень шума ===")
    test_period_detection(
        test_name="Тест 3: Высокий уровень шума",
        periods=[10, 90, 180],  # Разные периоды
        amplitudes=[5, 8, 12],  # Высокие амплитуды
        noise_level=3.0,        # Высокий уровень шума
        trend_slope=0.05,       # Заметный тренд
        num_points=1000
    )
    
    # Тест 4: Длинные периоды с большей амплитудой
    print("\n=== Тест 4: Длинные периоды с большей амплитудой ===")
    test_period_detection(
        "Тест 4: Длинные периоды с большей амплитудой",
        periods=[30, 90, 365],
        amplitudes=[5, 10, 20],  # Увеличенные амплитуды для длинных периодов
        noise_level=2.0,
        trend_slope=0.03,
        num_points=1500  # Больше точек для лучшего обнаружения длинных периодов
    ) 