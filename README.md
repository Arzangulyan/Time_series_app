# Комплекс анализа временных рядов

Комплексное приложение для анализа, обработки и прогнозирования временных рядов метеорологических данных с использованием Streamlit. Приложение предоставляет широкий спектр методов от классических статистических подходов до современных нейронных сетей.

## 🚀 Возможности

### Основная функциональность
- **Загрузка и генерация данных**: Импорт CSV файлов или создание синтетических временных рядов
- **Предобработка данных**: Фильтрация, ресемплинг, обработка пропусков, заполнение временной сетки
- **Визуализация**: Интерактивные графики с использованием Plotly
- **Экспорт результатов**: Сохранение обработанных данных и отчетов

### Методы анализа временных рядов
1. **AR модели** - Авторегрессионные модели (ARMA, ARIMA)
2. **Вейвлет-преобразование** - Многомасштабный анализ сигналов
3. **Фурье-преобразование** - Частотный анализ временных рядов
4. **Морфологический анализ** - Структурный анализ данных
5. **LSTM нейронные сети** - Глубокое обучение для прогнозирования
6. **Обнаружение аномалий** - Выявление выбросов и нештатных ситуаций

## 📁 Структура проекта

```
TimeSeriesApp_experimental/
├── main.py                          # Главная страница с предобработкой
├── config.py                        # Конфигурационные параметры
├── requirements.txt                 # Зависимости проекта
├── App_descriptions_streamlit.py    # Описания методов для UI
├── process_fix.py                   # Утилиты для исправления данных
├── TODO.md                          # Список задач для разработки
│
├── pages/                           # Страницы Streamlit
│   ├── 1_AR_models.py              # AR/ARMA/ARIMA модели
│   ├── 2_Wavelets.py               # Вейвлет-анализ
│   ├── 3_Fourier.py                # Фурье-преобразование
│   ├── 4_Morphology.py             # Морфологический анализ
│   ├── 5_LSTM.py                   # LSTM нейронные сети
│   └── 6_Anomaly.py                # Обнаружение аномалий
│
├── modules/                         # Модули функциональности
│   ├── data_processing.py          # Обработка и предобработка данных
│   ├── synthetic_data.py           # Генерация синтетических данных
│   ├── time_series_analysis.py     # Основные функции анализа временных рядов
│   ├── visualization.py            # Функции визуализации
│   ├── utils.py                    # Вспомогательные утилиты
│   ├── reporting.py                # Генерация отчетов
│   ├── page_template.py            # Шаблон для страниц
│   ├── wavelet_module.py           # Функции вейвлет-анализа
│   ├── fourier_module.py           # Функции Фурье-преобразования
│   ├── lstm_module.py              # LSTM модели и утилиты
│   ├── anomaly_module.py           # Методы обнаружения аномалий
│   ├── morfology_module.py         # Морфологический анализ
│   ├── autoregressive/             # Продвинутый модуль AR моделей
│   │   ├── __init__.py
│   │   ├── core.py                 # Основные функции AR
│   │   ├── models.py               # AR/ARMA/ARIMA модели
│   │   ├── model_selection.py      # Выбор оптимальных параметров
│   │   ├── helper.py               # Вспомогательные функции
│   │   ├── utils.py                # Утилиты для AR моделей
│   │   ├── visualization.py        # Визуализация AR моделей
│   │   └── README.md               # Документация AR модуля
│   └── lstm/                       # Продвинутый LSTM модуль
│       ├── __init__.py
│       ├── core.py                 # Основные функции LSTM
│       ├── models.py               # LSTM модели
│       ├── utils.py                # Утилиты для LSTM
│       ├── visualization.py        # Визуализация LSTM
│       └── README.md               # Документация LSTM
│
├── method_descriptions/             # Теоретические описания
│   ├── ARMA.py                     # Описание ARMA моделей
│   ├── ARIMA.py                    # Описание ARIMA моделей
│   ├── SARIMA.py                   # Описание SARIMA моделей
│   ├── Wavelet.py                  # Описание вейвлет-метода
│   └── Fourier.py                  # Описание Фурье-преобразования
│
├── logs/                           # Логи приложения
│   └── app_*.log                   # Файлы логов по дням
│
├── .streamlit/                     # Конфигурация Streamlit
├── .vscode/                        # Настройки VS Code
├── .devcontainer/                  # Конфигурация для Dev Container
│
└── ignore-folder/                  # Вспомогательные файлы
    ├── about.md                    # Информация о проекте
    ├── architecture_diagram.md     # Диаграмма архитектуры
    ├── mermaid.md                  # Mermaid диаграммы
    ├── test_period_detection.py    # Тестовые скрипты
    └── wavelet_improvements.txt    # Заметки по улучшениям
```

## 🛠 Установка

### Предварительные требования

- Python 3.8 или выше
- pip (менеджер пакетов Python)

### Шаги по установке

1. Клонируйте репозиторий:
   ```bash
   git clone <url-репозитория>
   cd TimeSeriesApp_experimental
   ```

2. Создайте виртуальное окружение (рекомендуется):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # На Linux/Mac
   # или
   .venv\Scripts\activate  # На Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Запуск приложения

После установки всех зависимостей запустите приложение командой:

```bash
streamlit run main.py
```

Приложение будет доступно в браузере по адресу `http://localhost:8501`.

## 📊 Руководство пользователя

### 1. Главная страница (Предобработка)

**Загрузка данных:**
- Импорт CSV файлов с автоматическим определением разделителей
- Генерация синтетических данных с настраиваемыми параметрами (тренд, циклы, шум)

**Обработка данных:**
- Выбор временной колонки и колонки значений
- Фильтрация по временному интервалу
- Ресемплинг с различными частотами (дни, недели, месяцы, и т.д.)
- Обработка пропущенных значений (интерполяция, заполнение средним/медианой)
- Заполнение пропусков временной сетки

### 2. AR модели
- Построение ARMA и ARIMA моделей
- Автоматический подбор параметров
- Анализ автокорреляционных функций
- Прогнозирование с доверительными интервалами

### 3. Вейвлет-анализ
- Непрерывное вейвлет-преобразование
- Выбор материнского вейвлета (Морле, Гаусс, Мексиканская шляпа)
- Анализ частотно-временных характеристик
- Обнаружение периодичностей и аномалий

### 4. Фурье-преобразование
- Быстрое преобразование Фурье (FFT)
- Спектральный анализ
- Фильтрация частотных компонент
- Восстановление сигнала

### 5. Морфологический анализ
- Структурный анализ временных рядов
- Выделение морфологических особенностей
- Анализ формы сигнала

### 6. LSTM нейронные сети
- Автоматическая настройка архитектуры
- Три уровня сложности (простая, средняя, сложная)
- Ручная настройка параметров
- Прогнозирование на будущие периоды
- Оценка качества моделей

### 7. Обнаружение аномалий
- Генерация синтетических аномалий (точечные, протяженные, сбои датчиков)
- Методы обнаружения: Z-score, IQR, фильтр Хампеля, детекция плато
- Автоматический подбор параметров
- Численные эксперименты для оптимизации
- Оценка качества детекции (precision, recall, F1-score)

## 🔧 Технические детали

### Основные зависимости

```
streamlit              # Веб-интерфейс
pandas                 # Обработка данных
numpy                  # Численные вычисления
plotly                 # Интерактивная визуализация
scipy                  # Научные вычисления
statsmodels           # Статистические модели
tensorflow            # Глубокое обучение (LSTM)
pywt                  # Вейвлет-преобразования
scikit-learn          # Машинное обучение
```

### Кэширование и производительность

Приложение использует `@st.cache_data` для оптимизации производительности:
- Кэширование загрузки данных
- Кэширование результатов вычислений
- Сохранение состояния между страницами

### Состояние приложения

Данные сохраняются в `st.session_state` между страницами:
- `time_series`: Обработанные временные ряды
- `main_column`: Основная колонка для анализа
- Параметры обработки и фильтрации

## 🧪 Возможности для исследователей

### Экспериментальные функции
- Численные эксперименты с параметрами алгоритмов
- Сравнение эффективности различных методов
- Генерация отчетов в формате YAML
- Настраиваемые метрики качества

### Расширяемость
- Модульная архитектура для добавления новых методов
- Стандартизированные интерфейсы для алгоритмов
- Готовые шаблоны для новых страниц анализа

## 🎯 Применение

Приложение предназначено для:
- **Метеорологов**: Анализ погодных данных и климатических трендов
- **Исследователей**: Изучение временных закономерностей в данных
- **Аналитиков данных**: Предобработка и анализ временных рядов
- **Студентов**: Изучение методов анализа временных рядов

## 📝 Примечания для разработчиков

### Добавление нового метода анализа

1. Создайте файл страницы в `pages/`
2. Добавьте модуль функций в `modules/`
3. При необходимости добавьте описание в `method_descriptions/`
4. Следуйте соглашениям о кэшировании и состоянии

### Принципы разработки
- Модульная структура кода
- Подробные комментарии и документация
- Обработка ошибок для предотвращения сбоев
- Использование кэширования для производительности
- Совместимость с различными форматами данных

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте логи приложения
2. Используйте кнопку "Очистить состояние и кэш" на главной странице
3. Убедитесь в корректности формата входных данных
4. Проверьте наличие всех зависимостей

## 📄 Лицензия

Проект разработан для научных и образовательных целей.