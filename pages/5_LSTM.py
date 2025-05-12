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
import tensorflow as tf
import time
import modules.reporting as reporting
from modules.lstm.visualization import plot_train_test_results_matplotlib

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

class StreamlitStopTrainingCallback:
    pass  # Больше не нужен, оставлен для совместимости, не используется

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
        max_value=100,
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
    
    # Кнопка для остановки обучения
    if "lstm_stop_training" not in st.session_state:
        st.session_state.lstm_stop_training = False
    if st.sidebar.button("Остановить обучение"):
        st.session_state.lstm_stop_training = True
    
    # Кнопка для запуска обучения
    run_button = st.sidebar.button("Запустить обучение")
    
    # Запуск обучения
    if run_button:
        st.session_state.lstm_stop_training = False  # Сброс флага остановки
        st.subheader("Прогресс обучения")
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)
        status_text = status_text_placeholder.empty()
        with st.spinner("Обучение LSTM модели..."):
            try:
                # Получаем автоматически настроенные параметры LSTM
                params = auto_tune_lstm_params(
                    time_series, 
                    complexity_level=complexity_map[model_complexity]
                )
                params['sequence_length'] = sequence_length
                params['epochs'] = epochs
                # Создаем и обучаем модель
                lstm_model = LSTMModel(
                    sequence_length=params['sequence_length'],
                    units=params['units'],
                    dropout_rate=params['dropout_rate'],
                    bidirectional=params['bidirectional']
                )
                ts_series = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series
                # Подготовка данных и инициализация модели (fit с 0 эпох)
                lstm_model.fit(
                    series=ts_series,
                    epochs=0,
                    batch_size=params['batch_size'],
                    validation_split=params['validation_split'],
                    early_stopping=False,
                    patience=params['patience'],
                    verbose=0,
                    train_size=train_size,
                    callbacks=None
                )
                # Явный цикл по эпохам
                num_epochs = params['epochs']
                patience = params.get('patience', 10)
                best_val_loss = float('inf')
                best_weights = None
                wait = 0
                history = {'loss': [], 'val_loss': []}
                start_time = time.perf_counter()
                early_stopping_epoch = None
                for epoch in range(num_epochs):
                    hist = lstm_model.model.fit(
                        lstm_model.X_train, lstm_model.y_train,
                        epochs=1,
                        batch_size=params['batch_size'],
                        validation_split=params['validation_split'],
                        verbose=0
                    )
                    loss = hist.history['loss'][0]
                    val_loss = hist.history['val_loss'][0] if 'val_loss' in hist.history else None
                    history['loss'].append(loss)
                    history['val_loss'].append(val_loss)
                    # EarlyStopping вручную
                    if val_loss is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_weights = lstm_model.model.get_weights()
                            wait = 0
                        else:
                            wait += 1
                            if wait >= patience and params.get('early_stopping', True):
                                status_text.warning(f"Ранняя остановка на эпохе {epoch+1} (val_loss не улучшается {patience} эпох)")
                                early_stopping_epoch = epoch + 1
                                break
                    # Обновление прогресса
                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    msg = f"Эпоха {epoch + 1}/{num_epochs} | loss: {loss:.4f}"
                    if val_loss is not None:
                        msg += f" | val_loss: {val_loss:.4f}"
                    status_text.text(msg)
                    if st.session_state.lstm_stop_training:
                        st.warning("Обучение было остановлено пользователем.")
                        break
                train_time = time.perf_counter() - start_time
                # Восстановление лучших весов, если была ранняя остановка
                if best_weights is not None:
                    lstm_model.model.set_weights(best_weights)
                lstm_model.training_history = history
                st.success("Модель успешно обучена!")
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
                        future_index = create_future_index(ts_series.index, int(forecast_steps))
                        future_preds = lstm_model.predict(steps=int(forecast_steps))
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
                    'original_series': ts_series,  # Гарантируем наличие оригинального ряда
                    'train_time': train_time,
                    'early_stopping': early_stopping_epoch is not None,
                    'early_stopping_epoch': early_stopping_epoch
                }
                # Проверяем доступность истории обучения
                if hasattr(lstm_model, 'training_history') and lstm_model.training_history is not None:
                    st.session_state.lstm_results['history'] = lstm_model.training_history
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {str(e)}")
                return
    
    # Отображение результатов, если они есть
    if st.session_state.lstm_results is not None:
        results = st.session_state.lstm_results
        
        # Отображение метрик, если они есть
        if 'train_metrics' in results and 'test_metrics' in results:
            st.subheader("Метрики качества прогноза")
            # Вывод времени обучения
            if 'train_time' in results:
                st.caption(f"Время обучения модели: {results['train_time']:.2f} сек.")
            if results.get('early_stopping', False):
                st.info(f"Обучение завершено досрочно на эпохе {results['early_stopping_epoch']} (ранняя остановка)")
            
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
            # Создаем график с результатами (plotly для Streamlit)
            fig = plot_train_test_results(
                original_series=results['original_series'],
                train_pred=results['train_predictions'],
                test_pred=results['test_predictions'],
                title="Результаты прогнозирования LSTM"
            )
            fig.update_layout(
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            # --- Секция: СКАЧАТЬ ОТЧЕТ ---
            # График прогноза (matplotlib -> base64)
            forecast_fig = plot_train_test_results_matplotlib(
                results['original_series'],
                results['train_predictions'],
                results['test_predictions'],
                title="Результаты прогнозирования LSTM"
            )
            forecast_img_base64 = reporting.save_plot_to_base64(forecast_fig, backend='matplotlib')
            # График потерь (matplotlib -> base64)
            if 'history' in results:
                loss_fig = plot_training_history(results['history'])
                loss_img_base64 = reporting.save_plot_to_base64(loss_fig, backend='matplotlib')
            else:
                loss_img_base64 = ''
            # Описание и параметры
            description = "Прогнозирование временного ряда с помощью LSTM."
            params = {
                'Размер обучающей выборки': train_size,
                'Длина входной последовательности': sequence_length,
                'Сложность модели': model_complexity,
                'Количество эпох': epochs,
                'Шаги прогноза вперед': forecast_steps
            }
            md_report = reporting.generate_markdown_report(
                title="Отчет по эксперименту LSTM",
                description=description,
                metrics_train=results['train_metrics'],
                metrics_test=results['test_metrics'],
                train_time=results.get('train_time', 0),
                forecast_img_base64=forecast_img_base64,
                loss_img_base64=loss_img_base64,
                params=params,
                early_stopping=results.get('early_stopping', False),
                early_stopping_epoch=results.get('early_stopping_epoch')
            )
            # Генерируем PDF (если возможно)
            try:
                pdf_bytes = reporting.markdown_to_pdf(md_report)
            except Exception as e:
                pdf_bytes = None
                st.warning(f"Не удалось сгенерировать PDF: {e}")
            reporting.download_report_buttons(md_report, pdf_bytes, md_filename="lstm_report.md", pdf_filename="lstm_report.pdf")
            # --- Конец секции отчета ---
        else:
            st.warning("Не удалось отобразить график прогнозирования из-за отсутствия необходимых данных.")
        
        # СЕКЦИЯ: Прогноз в будущее по уже обученной модели
        if st.session_state.lstm_model is not None:
            st.subheader("Прогноз на будущее по обученной модели")
            forecast_steps = st.number_input(
                "Шаги прогноза вперед", min_value=1, max_value=100, value=10, step=1, key="future_steps")
            if st.button("Сделать прогноз в будущее"):
                try:
                    future_preds = st.session_state.lstm_model.forecast(steps=int(forecast_steps))
                    # Создаём индекс для будущих дат
                    future_index = create_future_index(results['original_series'].index, int(forecast_steps))
                    future_preds = pd.Series(future_preds.values, index=future_index)
                    # График прогноза
                    st.plotly_chart(plot_forecast(results['original_series'], future_preds, title="Прогноз на будущее (LSTM)"), use_container_width=True)
                    # Таблица прогноза
                    st.dataframe(pd.DataFrame({'Прогнозируемое значение': future_preds}))
                    # Кнопка для скачивания
                    csv = future_preds.to_csv(index=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Скачать прогноз (CSV)",
                        data=csv,
                        file_name=f"lstm_forecast_{timestamp}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Ошибка при прогнозе в будущее: {str(e)}")
    # Если модель еще не обучена, покажем инструкции
    else:
        st.info("Выберите режим настройки и нажмите 'Запустить обучение' для начала анализа.")

    # СЕКЦИЯ: Эксперимент с тройным разбиением
    if st.session_state.lstm_model is not None and st.session_state.lstm_results is not None:
        st.subheader("Эксперимент: тройное разбиение (train/val/future)")
        if st.button("Запустить эксперимент тройного разбиения"):
            try:
                ts_series = st.session_state.lstm_results['original_series']
                n = len(ts_series)
                part = n // 3
                train = ts_series.iloc[:part]
                val = ts_series.iloc[part:2*part]
                future = ts_series.iloc[2*part:]
                # Обучаем новую модель только на train
                exp_model = LSTMModel(
                    sequence_length=st.session_state.lstm_model.sequence_length,
                    units=st.session_state.lstm_model.units,
                    dropout_rate=st.session_state.lstm_model.dropout_rate,
                    bidirectional=st.session_state.lstm_model.bidirectional
                )
                exp_model.fit(
                    series=train,
                    epochs=st.session_state.lstm_model.training_history and len(st.session_state.lstm_model.training_history['loss']) or 50,
                    batch_size=16,
                    validation_split=0.1,
                    early_stopping=True,
                    patience=10,
                    verbose=0,
                    train_size=1.0
                )
                # Прогноз на валидации (val)
                val_data = pd.concat([train[-exp_model.sequence_length:], val])
                # Приведение к Series, если DataFrame
                if isinstance(val_data, pd.DataFrame):
                    val_data = val_data.iloc[:, 0]
                if len(val_data) <= exp_model.sequence_length:
                    st.error("Слишком короткая валидационная часть для формирования хотя бы одного окна. Увеличьте длину ряда или уменьшите sequence_length.")
                    return
                prep = prepare_data_for_forecast(val_data, exp_model.sequence_length)
                X_val, y_val = prep['X'], prep['y']
                if len(X_val) == 0:
                    st.error("Не удалось сформировать ни одного окна для валидации. Попробуйте уменьшить sequence_length или увеличить размер данных.")
                    return
                val_pred = exp_model.model.predict(X_val, verbose=0)
                val_pred = exp_model.scaler.inverse_transform(val_pred).flatten()
                y_val = exp_model.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                # Прогноз в будущее (на длину future)
                future_pred = exp_model.forecast(steps=len(future))
                # Метрики
                val_metrics = calculate_metrics(y_val, val_pred)
                future_metrics = calculate_metrics(future.values, future_pred)
                st.markdown("**Метрики на валидации (2-я часть):**")
                st.json(val_metrics)
                st.markdown("**Метрики на будущем (3-я часть):**")
                st.json(future_metrics)
                # График
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_series.index, y=ts_series.values, mode='lines', name='Исходные данные'))
                fig.add_trace(go.Scatter(x=val.index[-len(val_pred):], y=val_pred, mode='lines', name='Прогноз (валидация)', line=dict(dash='dot', color='orange')))
                fig.add_trace(go.Scatter(x=future.index, y=future_pred, mode='lines', name='Прогноз (будущее)', line=dict(dash='dot', color='green')))
                fig.update_layout(title='Эксперимент: тройное разбиение', xaxis_title='Время', yaxis_title='Значение', height=500)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка в эксперименте тройного разбиения: {str(e)}")

if __name__ == "__main__":
    main()
