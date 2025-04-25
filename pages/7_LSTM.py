import streamlit as st
# Очищаем кэш при запуске
st.cache_data.clear()
st.cache_resource.clear()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import App_descriptions_streamlit as txt
from modules.lstm import (
    LSTMModel,
    train_test_split_ts,
    auto_tune_lstm_params,
    plot_train_test_results,
    plot_training_history,
    plot_forecast,
    calculate_metrics,
    prepare_data_for_forecast,
    create_future_index,
    save_results_to_csv
)
from modules.utils import nothing_selected
from modules.page_template import setup_page, load_time_series, display_data, run_calculations_on_button_click
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Инициализация состояния сессии
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'lstm_results' not in st.session_state:
    st.session_state.lstm_results = None
if 'lstm_display_df' not in st.session_state:
    st.session_state.lstm_display_df = None
if 'lstm_forecast' not in st.session_state:
    st.session_state.lstm_forecast = None

def main():
    setup_page(
        "Прогнозирование с помощью LSTM",
        "Настройки модели LSTM"
    )
    
    # Загрузка данных
    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return
    
    # Основная секция
    st.title("Прогнозирование временных рядов с помощью LSTM")
    
    with st.expander("Что такое LSTM?", expanded=False):
        st.markdown("""
        **LSTM (Long Short-Term Memory)** - это особый вид рекуррентных нейронных сетей, способный обучаться долгосрочным зависимостям во временных рядах. LSTM имеет внутренние механизмы, называемые вентилями, которые позволяют ей помнить или забывать информацию, что делает её эффективной для задач прогнозирования временных рядов.
        
        **Преимущества LSTM:**
        - Способность улавливать долгосрочные зависимости в данных
        - Устойчивость к проблеме затухающего градиента
        - Возможность работать с последовательностями различной длины
        - Высокая точность прогноза при правильной настройке
        
        **Применение:**
        - Прогнозирование финансовых временных рядов
        - Предсказание нагрузки на сервера или потребления энергии
        - Анализ данных сенсоров и IoT-устройств
        - Прогнозирование спроса и продаж
        """)
    
    # Отображение временного ряда
    st.subheader("Исходный временной ряд")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_series.index,
        y=time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series,
        mode="lines",
        name="Временной ряд"
    ))
    
    fig.update_layout(
        xaxis_title="Время",
        yaxis_title="Значение",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Боковая панель с параметрами
    st.sidebar.subheader("Настройки LSTM")
    
    with st.sidebar.expander("О параметрах модели", expanded=True):
        st.markdown("""
        **Основные параметры LSTM:**
        
        1. **Размер обучающей выборки**: определяет, сколько данных использовать для обучения модели
        2. **Длина последовательности**: сколько предыдущих точек использовать для предсказания следующей
        3. **Сложность модели**: влияет на способность модели улавливать сложные паттерны
        4. **Количество эпох**: сколько раз модель просмотрит все данные при обучении
        5. **Прогноз вперед**: на сколько шагов вперед делать прогноз
        """)
    
    # Основные параметры с значениями по умолчанию
    train_size = st.sidebar.slider(
        "Размер обучающей выборки", 
        min_value=0.5, 
        max_value=0.95, 
        value=0.8, 
        step=0.05,
        help="Доля данных для обучения (остальное - для тестирования)"
    )
    
    sequence_length = st.sidebar.slider(
        "Длина входной последовательности",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Количество предыдущих точек для предсказания следующего значения"
    )
    
    model_complexity = st.sidebar.select_slider(
        "Сложность модели",
        options=["Низкая", "Средняя", "Высокая"],
        value="Средняя",
        help="Влияет на количество слоев и нейронов в модели"
    )
    
    # Преобразование сложности модели в параметры
    complexity_map = {
        "Низкая": "simple", 
        "Средняя": "medium", 
        "Высокая": "complex"
    }
    
    epochs = st.sidebar.slider(
        "Количество эпох", 
        min_value=10, 
        max_value=200, 
        value=50, 
        step=10,
        help="Количество проходов по обучающей выборке"
    )
    
    forecast_steps = st.sidebar.slider(
        "Шаги прогноза вперед", 
        min_value=0, 
        max_value=100, 
        value=10, 
        step=5,
        help="Количество периодов для прогноза в будущее"
    )
    
    # Кнопка для запуска обучения
    run_button = st.sidebar.button("Запустить обучение")
    
    # Запуск обучения
    if run_button:
        with st.spinner("Обучение LSTM модели..."):
            try:
                # Получаем автоматически настроенные параметры LSTM
                params = auto_tune_lstm_params(
                    time_series, 
                    complexity_level=complexity_map[model_complexity]
                )
                
                # Заменяем некоторые параметры пользовательскими
                params['sequence_length'] = sequence_length
                params['epochs'] = epochs
                
                # Создаем и обучаем модель
                lstm_model = LSTMModel(
                    sequence_length=params['sequence_length'],
                    units=params['units'],
                    dropout_rate=params['dropout_rate'],
                    bidirectional=params['bidirectional']
                )
                
                # Подготавливаем серию (преобразуем DataFrame в Series, если нужно)
                ts_series = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series
                
                # Обучаем модель
                lstm_model.fit(
                    series=ts_series,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=params['validation_split'],
                    early_stopping=params['early_stopping'],
                    patience=params['patience'],
                    verbose=1,
                    train_size=train_size
                )
                
                # Получаем результаты обучения и прогнозирования
                train_predictions = lstm_model.predict_train()
                test_predictions = lstm_model.predict_test()
                
                # Извлекаем реальные значения для метрик
                train_actual = ts_series[train_predictions.index]
                test_actual = ts_series[test_predictions.index]
                
                # Создаем метрики
                train_metrics = calculate_metrics(
                    train_actual.values, 
                    train_predictions.values
                )
                test_metrics = calculate_metrics(
                    test_actual.values, 
                    test_predictions.values
                )
                
                # Если нужен прогноз на будущее
                future_preds = None
                if forecast_steps > 0:
                    try:
                        # Создаем индекс для будущих прогнозов, обрабатывая возможные исключения
                        future_index = create_future_index(ts_series.index, forecast_steps)
                        future_preds = lstm_model.predict(steps=forecast_steps)
                        future_preds = pd.Series(future_preds, index=future_index)
                    except Exception as e:
                        st.warning(f"Не удалось создать прогноз на будущее: {str(e)}")
                        future_preds = None
                
                # Сохраняем в session_state
                st.session_state.lstm_model = lstm_model
                st.session_state.lstm_results = {
                    'train_predictions': train_predictions,
                    'test_predictions': test_predictions,
                    'future_predictions': future_preds,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'original_series': ts_series  # Гарантируем наличие оригинального ряда
                }
                
                # Проверяем доступность истории обучения
                if hasattr(lstm_model, 'training_history') and lstm_model.training_history is not None:
                    st.session_state.lstm_results['history'] = lstm_model.training_history
                
                st.success("Модель успешно обучена!")
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {str(e)}")
                return
    
    # Отображение результатов, если они есть
    if st.session_state.lstm_results is not None:
        results = st.session_state.lstm_results
        
        # Отображение метрик, если они есть
        if 'train_metrics' in results and 'test_metrics' in results:
            st.subheader("Метрики качества прогноза")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Обучающая выборка:**")
                metrics_train = results['train_metrics']
                st.metric("RMSE", f"{metrics_train['rmse']:.4f}")
                st.metric("MAE", f"{metrics_train['mae']:.4f}")
                st.metric("MASE", f"{metrics_train['mase']:.4f}")
                st.metric("R²", f"{metrics_train['r2']:.4f}")
                st.metric("Adjusted R²", f"{metrics_train['adj_r2']:.4f}")
                
                with st.expander("Что означают эти метрики?"):
                    st.markdown("""
                    - **RMSE** (Root Mean Squared Error) - среднеквадратичная ошибка. Показывает среднюю величину ошибки прогноза в тех же единицах измерения, что и данные.
                    - **MAE** (Mean Absolute Error) - средняя абсолютная ошибка. Более устойчива к выбросам, чем RMSE.
                    - **MASE** (Mean Absolute Scaled Error) - масштабированная средняя абсолютная ошибка. Значения < 1 означают, что модель лучше наивного прогноза.
                    - **R²** - коэффициент детерминации. Показывает долю дисперсии, объясненную моделью (1.0 - идеальный прогноз).
                    - **Adjusted R²** - скорректированный R². Учитывает сложность модели, помогает выявить переобучение.
                    """)
            
            with col2:
                st.markdown("**Тестовая выборка:**")
                metrics_test = results['test_metrics']
                st.metric("RMSE", f"{metrics_test['rmse']:.4f}")
                st.metric("MAE", f"{metrics_test['mae']:.4f}")
                st.metric("MASE", f"{metrics_test['mase']:.4f}")
                st.metric("R²", f"{metrics_test['r2']:.4f}")
                st.metric("Adjusted R²", f"{metrics_test['adj_r2']:.4f}")
                
                with st.expander("Как интерпретировать результаты?"):
                    st.markdown("""
                    **Хорошими показателями** считаются:
                    
                    1. **RMSE и MAE** - чем меньше, тем лучше. Сравнивайте с масштабом ваших данных.
                    
                    2. **MASE**:
                       - < 1: модель лучше наивного прогноза
                       - ≈ 1: сопоставимо с наивным прогнозом
                       - > 1: модель хуже наивного прогноза
                    
                    3. **R² и Adjusted R²**:
                       - > 0.9: отличный результат
                       - 0.7-0.9: хороший результат
                       - 0.5-0.7: удовлетворительный результат
                       - < 0.5: модель требует улучшения
                    
                    Если метрики на тестовой выборке значительно хуже, чем на обучающей, это может указывать на переобучение.
                    """)
        else:
            st.warning("Метрики качества прогноза недоступны.")
        
        # Отображение графика обучения
        st.subheader("График процесса обучения")
        if 'history' in results:
            history_fig = plot_training_history(results['history'])
            st.pyplot(history_fig)
        else:
            st.warning("Невозможно отобразить график процесса обучения из-за отсутствия данных.")
        
        # Отображение прогнозов
        st.subheader("Результаты прогнозирования")
        
        # Проверяем наличие необходимых ключей для построения графика
        if all(key in results for key in ['original_series', 'train_predictions', 'test_predictions']):
            # Создаем график с результатами
            fig = plot_train_test_results(
                original_series=results['original_series'],
                train_pred=results['train_predictions'],
                test_pred=results['test_predictions'],
                title="Результаты прогнозирования LSTM"
            )
            
            # Добавляем прогноз на будущее, если он есть
            if results.get('future_predictions') is not None:
                fig.add_trace(go.Scatter(
                    x=results['future_predictions'].index,
                    y=results['future_predictions'].values,
                    mode='lines+markers',
                    name='Прогноз (будущее)',
                    line=dict(color='purple'),
                    marker=dict(size=6)
                ))
            
            # Настройка макета графика
            fig.update_layout(
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Не удалось отобразить график прогнозирования из-за отсутствия необходимых данных.")
        
        # Показываем таблицу с прогнозами на будущее, если есть
        if results.get('future_predictions') is not None and forecast_steps > 0:
            st.subheader("Прогноз на будущие периоды")
            future_df = pd.DataFrame({
                'Прогнозируемое значение': results['future_predictions']
            })
            st.dataframe(future_df)
            
            # Добавляем возможность скачать прогноз
            csv = future_df.to_csv(index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Скачать прогноз (CSV)",
                data=csv,
                file_name=f"lstm_forecast_{timestamp}.csv",
                mime="text/csv"
            )
    # Если модель еще не обучена, покажем инструкции
    else:
        st.info("Выберите режим настройки и нажмите 'Запустить обучение' для начала анализа.")


if __name__ == "__main__":
    main()
