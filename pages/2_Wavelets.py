import streamlit as st
import numpy as np
import pandas as pd
import App_descriptions_streamlit as txt
import time # Импортируем time
from modules.wavelet_module import (
    wavelet_transform, 
    plot_wavelet_transform, 
    get_scale_ticks, 
    format_period, 
    find_significant_periods_wavelet,
    plot_wavelet_periodicity_analysis,
    TIME_UNITS,
    DEFAULT_TIME_UNIT,
    MEASUREMENT_UNIT_KEY,
    get_time_delta
)
from modules.utils import nothing_selected, initialize_session_state
from modules.page_template import (
    setup_page,
    load_time_series,
    display_data,
    run_calculations_on_button_click,
)
from method_descriptions.Wavelet import DESCRIPTION, PARAMS_CHOICE

# --- Колбэк для сброса флага рассчитанных результатов ---
def reset_wavelet_calculation_flag():
    st.session_state.wavelet_results_calculated = False
    # Также очистим предыдущие результаты, чтобы не показывать старые данные, если новый расчет не удастся
    st.session_state.wavelet_coef = None
    st.session_state.wavelet_periods_meas_original_scale = None
    st.session_state.wavelet_significant_periods_df = pd.DataFrame() # Пустой DataFrame
    st.session_state.wavelet_ts_processed_for_plot = None
    st.session_state.wavelet_freqs_calc = None
    st.session_state.wavelet_calculation_time = None # Сбрасываем время расчета

# --- Кэшированная функция для вычисления значимых периодов --- 
@st.cache_data
def _calculate_and_cache_significant_periods(
     # Идентификаторы для кэширования
     time_series_hash, # Используем хэш или другую простую репрезентацию
     mother_wavelet,
     num_scales,
     agg_rule, # Добавляем правило агрегации в ключ кэша
     threshold_percent, 
     max_periods,
     # Сам time_series передаем отдельно, чтобы не хэшировать его целиком
     time_series_data 
):
    """
    Обертка для кэширования вычисления значимых периодов.
    Применяет агрегацию перед вызовом wavelet_transform.
    """
    # --- Применение агрегации --- 
    processed_ts = time_series_data
    original_time_delta = get_time_delta(time_series_data.index)
    downsampling_factor = 1
    
    if agg_rule != 'none' and isinstance(processed_ts.index, pd.DatetimeIndex):
         try:
             # Сохраняем исходный для передачи в find_significant_periods
             original_ts_for_find = processed_ts.copy()
             
             # Ресемплируем. Используем mean(), можно сделать опцией.
             # to_period() перед resample может помочь с нерегулярными рядами
             # processed_ts = processed_ts.to_period(freq=agg_rule).resample(agg_rule).mean()
             # Пробуем без to_period сначала:
             resampled_ts = processed_ts.resample(agg_rule).mean()
             
             # Проверяем, что результат не пустой
             if not resampled_ts.empty:
                 processed_ts = resampled_ts
                 print(f"Данные агрегированы до '{agg_rule}'. Новый размер: {len(processed_ts)}")
                 # Оцениваем фактор даунсемплинга
                 if len(processed_ts) > 0:
                      downsampling_factor = len(time_series_data) / len(processed_ts)
             else:
                 print(f"Предупреждение: Агрегация до '{agg_rule}' дала пустой результат. Используются исходные данные.")
                 # Возвращаемся к исходному ряду, если агрегация не удалась
                 processed_ts = time_series_data 

         except Exception as e:
             print(f"Ошибка при агрегации данных до '{agg_rule}': {e}. Используются исходные данные.")
             processed_ts = time_series_data # Возвращаемся к исходному ряду при ошибке
             downsampling_factor = 1
    # --------------------------- 

    # 1. Получаем CWT результаты для (возможно) агрегированного ряда
    transform_result = wavelet_transform(processed_ts, mother_wavelet, num_scales=num_scales, return_periods=True)
    if len(transform_result) != 3:
        print("Ошибка в _calculate...: Неверный результат wavelet_transform")
        return pd.DataFrame(), None, None, None # Возвращаем больше None
    coef, _, periods_meas_processed = transform_result
    if coef.size == 0 or periods_meas_processed.size == 0:
        print("Ошибка в _calculate...: Пустой результат wavelet_transform")
        return pd.DataFrame(), coef, periods_meas_processed, None # Возвращаем то, что есть

    # --- Корректируем периоды на фактор даунсемплинга --- 
    # periods_meas_processed - это периоды в *новых*, агрегированных измерениях
    # Чтобы получить периоды в *исходных* измерениях, умножаем на фактор
    periods_meas_original_scale = periods_meas_processed * downsampling_factor
    # ----------------------------------------------------

    # 2. Вызываем поиск периодов
    # Передаем ОРИГИНАЛЬНЫЙ time_series (original_ts_for_find), 
    # но coef и periods_meas от ОБРАБОТАННОГО ряда
    significant_periods_df = find_significant_periods_wavelet(
        original_ts_for_find if 'original_ts_for_find' in locals() else time_series_data, # Передаем исходный ряд
        mother_wavelet=mother_wavelet,
        num_scales=num_scales,
        power_threshold=0.1, 
        threshold_percent=threshold_percent,
        max_periods=max_periods,
        coef=coef, # От обработанного ряда
        periods_meas=periods_meas_processed # От обработанного ряда (find_significant_periods не использует их напрямую для поиска пиков)
    )
    
    # Возвращаем результаты: DataFrame периодов (в исходных измерениях!), coef, периоды (в исходных измерениях), и сам обработанный ряд
    # Корректируем DataFrame, чтобы периоды были в исходном масштабе
    if 'Период (изм.)' in significant_periods_df.columns:
         significant_periods_df['Период (изм.)'] = significant_periods_df['Период (изм.)'].astype(float) * downsampling_factor # Убедимся, что тип float для умножения
         
    return significant_periods_df, coef, periods_meas_original_scale, processed_ts
# ------------------------------------------------------------

# --- Основная функция страницы (замена wavelet_run и main) --- 
def render_wavelet_page():
    setup_page("Wavelets", "Настройки вейвлетов")
    
    # Инициализация состояния для результатов вейвлет-анализа
    if 'wavelet_results_calculated' not in st.session_state:
        st.session_state.wavelet_results_calculated = False
    
    # Инициализируем остальные ключи, если их нет, значением None или пустым DataFrame
    for key, default_val in [
        ('wavelet_coef', None),
        ('wavelet_periods_meas_original_scale', None),
        ('wavelet_significant_periods_df', pd.DataFrame()),
        ('wavelet_ts_processed_for_plot', None),
        ('wavelet_freqs_calc', None),
        ('current_agg_rule', 'none'), # Для хранения выбранного правила агрегации
        ('wavelet_calculation_time', None) # Инициализируем время расчета
    ]:
        if key not in st.session_state:
            st.session_state[key] = default_val
    
    with st.expander("Что такое вейвлет-преобразование?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)
    with st.sidebar.expander("Типы вейвлетов"):
        # Используем f-string для вставки текста из markdown
        st.markdown(f"""### Характеристики различных типов вейвлетов:
        
        - **Морле** (Morlet): Комплексный вейвлет, хорошо локализованный как во временной, так и в частотной области. 
          Оптимален для анализа гармонических сигналов и выявления периодичностей.
          
        - **Гаусс** (Gaussian): Производная от функции Гаусса. Хорошо подходит для обнаружения 
          локальных изменений в сигнале, таких как скачки и разрывы.
          
        - **Мексиканская шляпа** (Mexican hat): Вторая производная функции Гаусса. 
          Эффективен для обнаружения особенностей сигнала, таких как пики и впадины.
          
        - **Симлет** (Symlet): Почти симметричный вейвлет с компактным носителем. 
          Хорошо сохраняет форму сигнала, подходит для сжатия и шумоподавления.
          
        - **Добеши** (Daubechies): Асимметричный вейвлет с компактным носителем. 
          Обеспечивает хорошую локализацию во временной области, эффективен для анализа 
          нестационарных сигналов.
          
        - **Койфлет** (Coiflet): Почти симметричный вейвлет, близкий к Добеши, но с лучшими 
          аппроксимационными свойствами. Хорошо подходит для сжатия сигналов.
        """, unsafe_allow_html=True)

    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд.")
        return

    # --- Виджеты выбора --- 
    st.sidebar.header("Параметры анализа")
    # Определяем опции и индекс для selected_unit_key более безопасно
    unit_options = list(TIME_UNITS.keys())
    default_unit_idx = 0
    if isinstance(time_series.index, pd.DatetimeIndex):
        if DEFAULT_TIME_UNIT in unit_options:
            default_unit_idx = unit_options.index(DEFAULT_TIME_UNIT)
        elif MEASUREMENT_UNIT_KEY in unit_options: # Запасной вариант, если DEFAULT_TIME_UNIT нет
            default_unit_idx = unit_options.index(MEASUREMENT_UNIT_KEY)
    elif MEASUREMENT_UNIT_KEY in unit_options:
        default_unit_idx = unit_options.index(MEASUREMENT_UNIT_KEY)
    # Если ни один из ключей не найден, default_unit_idx останется 0, что безопасно, если unit_options не пуст
    # (предполагается, что TIME_UNITS всегда содержит хотя бы MEASUREMENT_UNIT_KEY)

    wavelet_options = ["Морле", "Гаусс", "Мексиканская шляпа"]
    wavelet_select = st.sidebar.selectbox(
        label="Материнский вейвлет",
        options=wavelet_options,
        index=0, # По умолчанию первый вейвлет
        key='wavelet_type',
        on_change=reset_wavelet_calculation_flag
    )

    selected_unit_key = st.sidebar.selectbox(
         label="Единицы периода",
         options=unit_options,
         index=default_unit_idx,
         key='period_unit',
         help="Выберите, в каких единицах отображать найденные периоды."
    )
    
    default_max_scales = min(150, max(10, len(time_series) // 2)) 
    max_scales = st.sidebar.slider(
        "Макс. кол-во масштабов", 
        min_value=10, max_value=min(500, max(10, len(time_series) // 2)), 
        value=default_max_scales, 
        key='max_scales',
        help="Количество масштабов для анализа. Меньшее значение ускоряет расчет, но снижает детальность по частоте.",
        on_change=reset_wavelet_calculation_flag
    )
    
    st.sidebar.header("Параметры отображения и поиска пиков")
    threshold_percent_val = st.sidebar.slider(
        "Порог значимости (%)", 
        min_value=1.0, max_value=50.0, value=22.0, step=1.0,
        key='threshold_percent', # Ключ остается для session_state, если он где-то используется, но передаем _val
        help="Порог мощности (в % от максимума) для выделения значимых пиков.",
        on_change=reset_wavelet_calculation_flag
    )
    max_periods_display_val = st.sidebar.slider(
        "Макс. кол-во периодов", 
        min_value=1, max_value=20, value=13,
        key='max_periods_display', # Ключ остается
        help="Максимальное количество пиков для отображения в таблице и на графике.",
        on_change=reset_wavelet_calculation_flag
    )

    # --- Опция Агрегации --- 
    downsample_options = {"Нет": 'none'} 
    time_delta_agg_check = get_time_delta(time_series.index) # Используем другую переменную для проверки
    
    if len(time_series) > 500 and isinstance(time_delta_agg_check, pd.Timedelta):
         if time_delta_agg_check < pd.Timedelta(minutes=1):
             downsample_options["До минут"] = 'min'
         if time_delta_agg_check < pd.Timedelta(hours=1):
             downsample_options["До часов"] = 'h'
         if time_delta_agg_check < pd.Timedelta(days=1):
             downsample_options["До дней"] = 'D'
             
    selected_agg_label = "Нет" # Значение по умолчанию
    if len(downsample_options) > 1:
         selected_agg_label = st.sidebar.selectbox(
              "Ускорение: Агрегация данных",
              options=list(downsample_options.keys()),
              index=0, 
              key='agg_rule_label_selector', 
              help="Агрегирует данные до указанной частоты ПЕРЕД вейвлет-анализом. Ускоряет расчет для длинных рядов, но теряется информация о коротких периодах.",
              on_change=reset_wavelet_calculation_flag # При смене правила агрегации, нужно пересчитывать
         )
    
    # Обновляем current_agg_rule в session_state на основе выбора пользователя
    if selected_agg_label and selected_agg_label in downsample_options:
        st.session_state.current_agg_rule = downsample_options[selected_agg_label]
    else: # На случай, если selected_agg_label None или невалидный (хотя selectbox должен это предотвращать)
        st.session_state.current_agg_rule = 'none'

    # --- Кнопка для запуска расчетов ---
    run_button_clicked = st.sidebar.button("🚀 Рассчитать вейвлет-анализ", type="primary", key="run_wavelet_calculation_button")

    if run_button_clicked:
        if not wavelet_select:
            st.warning("Пожалуйста, выберите материнский вейвлет.")
            st.stop()
        
        st.session_state.wavelet_calculation_time = None # Сбрасываем перед новым расчетом
        with st.spinner("Выполнение вейвлет-преобразования и поиска пиков..."):
            start_time = time.time() # Время начала расчета
            ts_hash = pd.util.hash_pandas_object(time_series)
            actual_agg_rule = st.session_state.get('current_agg_rule', 'none')

            (st.session_state.wavelet_significant_periods_df, 
             st.session_state.wavelet_coef, 
             st.session_state.wavelet_periods_meas_original_scale, 
             st.session_state.wavelet_ts_processed_for_plot) = _calculate_and_cache_significant_periods(
                 time_series_hash=ts_hash,
                 mother_wavelet=str(wavelet_select), # Явное преобразование для линтера
                 num_scales=max_scales, 
                 agg_rule=actual_agg_rule,
                 threshold_percent=threshold_percent_val, # Используем прямое значение со слайдера
                 max_periods=max_periods_display_val,   # Используем прямое значение со слайдера
                 time_series_data=time_series
            )
            
            st.session_state.wavelet_freqs_calc = None 
            if st.session_state.wavelet_coef is not None and st.session_state.wavelet_ts_processed_for_plot is not None:
                series_for_freq_calc = st.session_state.wavelet_ts_processed_for_plot
                if isinstance(series_for_freq_calc, pd.DataFrame):
                    series_for_freq_calc = series_for_freq_calc.iloc[:, 0]

                freq_transform_result = wavelet_transform(
                    series_for_freq_calc, 
                    str(wavelet_select), # Явное преобразование
                    num_scales=max_scales,
                    return_periods=False 
                )
                if len(freq_transform_result) == 2:
                    _, st.session_state.wavelet_freqs_calc = freq_transform_result
                else:
                    st.warning("Не удалось получить частоты для вейвлет-спектра после расчета (неверный формат результата).")
            elif st.session_state.wavelet_coef is not None:
                 st.warning("Не удалось получить обработанный временной ряд для расчета частот, хотя коэффициенты есть.")

            if st.session_state.wavelet_coef is not None: # Проверяем хотя бы CWT
                st.session_state.wavelet_results_calculated = True
                if not st.session_state.wavelet_significant_periods_df.empty:
                    st.success("Расчеты завершены! Найдены значимые периоды.")
                else:
                    st.info("Вейвлет-преобразование выполнено, но значимых периодов с текущими настройками не найдено.")
            else:
                st.session_state.wavelet_results_calculated = False # Явный сброс
                st.error("Ошибка при выполнении расчетов. Коэффициенты вейвлет-преобразования не получены.")
                reset_wavelet_calculation_flag() # Очистка состояния
            
            end_time = time.time() # Время окончания расчета
            st.session_state.wavelet_calculation_time = end_time - start_time # Сохраняем время

    # --- Отображение результатов --- 
    if st.session_state.get('wavelet_results_calculated', False):
        # Проверяем, есть ли основные данные для отображения
        if st.session_state.wavelet_coef is None or \
           st.session_state.wavelet_periods_meas_original_scale is None or \
           st.session_state.wavelet_significant_periods_df is None: # significant_periods_df может быть пустым, это нормально
            st.error("Результаты расчета отсутствуют или некорректны для отображения. Попробуйте нажать 'Рассчитать'.")
            st.stop()
        
        # Отображаем время расчета, если оно есть
        if st.session_state.get('wavelet_calculation_time') is not None:
            st.caption(f"Время выполнения расчетов: {st.session_state.wavelet_calculation_time:.3f} сек.")
            
        time_delta = get_time_delta(time_series.index) 

        st.subheader("Вейвлет-спектр (Тепловая карта)")
        
        # Проверка перед использованием periods_meas_original_scale
        if st.session_state.wavelet_periods_meas_original_scale is None or len(st.session_state.wavelet_periods_meas_original_scale) == 0:
            st.warning("Данные о периодах для вейвлет-спектра отсутствуют. Тепловая карта не может быть построена.")
        # else: # Временно отключаем отображение тепловой карты для диагностики
        #     min_period_meas, max_period_meas = st.session_state.wavelet_periods_meas_original_scale.min(), st.session_state.wavelet_periods_meas_original_scale.max()
        #     tickvals_log, ticktext = get_scale_ticks(min_period_meas, max_period_meas, time_delta, str(selected_unit_key))
            
        #     fig_heatmap = plot_wavelet_transform(
        #         time_series, 
        #         st.session_state.wavelet_coef, 
        #         st.session_state.wavelet_freqs_calc, 
        #         st.session_state.wavelet_periods_meas_original_scale, 
        #         tickvals_log, 
        #         ticktext, 
        #         str(selected_unit_key) # Явное преобразование
        #     )
        #     st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Отображение тепловой карты временно отключено для диагностики.")

        st.subheader("Наиболее значимые периоды")
        if st.session_state.wavelet_significant_periods_df is not None and not st.session_state.wavelet_significant_periods_df.empty:
            display_df = st.session_state.wavelet_significant_periods_df.copy()
            # Попробуем заменить apply на list comprehension + Series constructor для обхода ошибки линтера с apply
            formatted_periods = [format_period(p, time_delta, str(selected_unit_key)) for p in display_df['Период (изм.)']]
            display_df['Период (формат.)'] = pd.Series(formatted_periods, index=display_df.index)
            
            measurement_periods = [format_period(p, None, MEASUREMENT_UNIT_KEY) for p in display_df['Период (изм.)']]
            display_df['Период (изм.)'] = pd.Series(measurement_periods, index=display_df.index)
            
            st.dataframe(display_df[['Период (формат.)', 'Период (изм.)', 'Мощность']], 
                        use_container_width=True,
                        column_config={
                            "Период (формат.)": st.column_config.TextColumn(
                                "Период", 
                                help=f"Период в выбранных единицах ({selected_unit_key})"
                            ),
                            "Период (изм.)": st.column_config.TextColumn(
                                "Период (в изм.)", 
                                help="Период в количестве измерений"
                            ),
                            "Мощность": st.column_config.NumberColumn(
                                "Норм. мощность", format="%.3f", 
                                help="Относительная мощность вейвлет-спектра (пики найдены с порогом >10%)"
                            )
                        },
                        hide_index=True
            )
        else:
            st.info("Значимых периодов не найдено с текущими параметрами.")
        
        st.subheader("Спектр мощности (Пики)")
        fig_periodicity = plot_wavelet_periodicity_analysis(
            time_series, 
            mother_wavelet=str(wavelet_select), # Явное преобразование
            max_scales=max_scales, 
            selected_unit_key=str(selected_unit_key), # Явное преобразование
            coef=st.session_state.wavelet_coef, 
            periods_meas=st.session_state.wavelet_periods_meas_original_scale, 
            significant_periods_df=st.session_state.wavelet_significant_periods_df 
        )
        st.plotly_chart(fig_periodicity, use_container_width=True)
    else:
        st.info("⬅️ Настройте параметры в боковой панели и нажмите **'🚀 Рассчитать вейвлет-анализ'** для запуска вычислений.")

if __name__ == "__main__":
    render_wavelet_page()