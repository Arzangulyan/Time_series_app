import streamlit as st
import numpy as np
import pandas as pd
import App_descriptions_streamlit as txt
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
         significant_periods_df['Период (изм.)'] = significant_periods_df['Период (изм.)'] * downsampling_factor
         
    return significant_periods_df, coef, periods_meas_original_scale, processed_ts
# ------------------------------------------------------------

# --- Основная функция страницы (замена wavelet_run и main) --- 
def render_wavelet_page():
    setup_page("Wavelets", "Настройки вейвлетов")
    
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
    wavelet_select = st.sidebar.selectbox(
        label="Материнский вейвлет",
        options=(["Морле", "Гаусс", "Мексиканская шляпа"]),
        key='wavelet_type'
    )

    selected_unit_key = st.sidebar.selectbox(
         label="Единицы периода",
         options=list(TIME_UNITS.keys()),
         index=list(TIME_UNITS.keys()).index(DEFAULT_TIME_UNIT) 
                 if DEFAULT_TIME_UNIT in TIME_UNITS and isinstance(time_series.index, pd.DatetimeIndex) 
                 else list(TIME_UNITS.keys()).index(MEASUREMENT_UNIT_KEY),
         key='period_unit',
         help="Выберите, в каких единицах отображать найденные периоды."
    )
    
    # Параметры расчета (влияют на CWT)
    default_max_scales = min(150, max(10, len(time_series) // 2)) # Уменьшаем дефолт с 218 до 150
    max_scales = st.sidebar.slider(
        "Макс. кол-во масштабов", 
        min_value=10, max_value=min(500, max(10, len(time_series) // 2)), 
        value=default_max_scales, # Используем новый дефолт
        key='max_scales',
        help="Количество масштабов для анализа. Меньшее значение ускоряет расчет, но снижает детальность по частоте."
    )
    
    # --- Опция Агрегации --- 
    # Предлагаем только если ряд достаточно длинный и есть временной шаг
    downsample_options = {"Нет": 'none'} # Используем строки для ключей агрегации
    time_delta_agg = get_time_delta(time_series.index)
    
    # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка time_delta_agg --- 
    if len(time_series) > 500 and isinstance(time_delta_agg, pd.Timedelta):
         # Добавляем разумные опции агрегации
         if time_delta_agg < pd.Timedelta(minutes=1):
             downsample_options["До минут"] = 'min'
         if time_delta_agg < pd.Timedelta(hours=1):
             downsample_options["До часов"] = 'h'
         if time_delta_agg < pd.Timedelta(days=1):
             downsample_options["До дней"] = 'D'
             
    # Показываем виджет, если есть опции кроме "Нет"
    agg_rule = 'none' # По умолчанию нет агрегации
    if len(downsample_options) > 1:
         selected_agg_label = st.sidebar.selectbox(
              "Ускорение: Агрегация данных",
              options=list(downsample_options.keys()),
              index=0, 
              key='agg_rule_label', # Меняем ключ, чтобы не конфликтовал с переменной
              help="Агрегирует данные до указанной частоты ПЕРЕД вейвлет-анализом. Ускоряет расчет для длинных рядов, но теряется информация о коротких периодах."
         )
         # --- ИСПРАВЛЕНИЕ ЛИНТЕРА: Проверка selected_agg_label --- 
         if selected_agg_label:
             agg_rule = downsample_options[selected_agg_label]
    # -----------------------

    # Параметры отображения/поиска пиков (НЕ влияют на CWT, но влияют на find_significant_periods)
    st.sidebar.header("Параметры отображения и поиска пиков")
    threshold_percent = st.sidebar.slider(
        "Порог значимости (%)", 
        min_value=1.0, max_value=50.0, value=22.0, step=1.0,
        key='threshold_percent',
        help="Порог мощности (в % от максимума) для выделения значимых пиков."
    )
    max_periods_display = st.sidebar.slider(
        "Макс. кол-во периодов", 
        min_value=1, max_value=20, value=13,
        key='max_periods_display',
        help="Максимальное количество пиков для отображения в таблице и на графике."
    )

    # --- Проверка выбора вейвлета --- 
    if not wavelet_select:
        st.warning("Пожалуйста, выберите материнский вейвлет в боковой панели.")
        st.stop()
        
    # --- Проверка выбора единиц --- 
    if not selected_unit_key:
        st.warning("Пожалуйста, выберите единицы измерения периода.")
        st.stop()

    # --- Выполнение вычислений --- 
    
    # --- Добавляем Spinner --- 
    with st.spinner("Выполнение вейвлет-преобразования..."):
        # 1. Основное вейвлет-преобразование (результат кэшируется внутри функции wavelet_transform)
        transform_result = wavelet_transform(time_series, wavelet_select, num_scales=max_scales, return_periods=True)

    if len(transform_result) == 3:
        coef, freqs, periods_meas = transform_result
        if coef.size == 0 or periods_meas.size == 0:
             st.error(f"Ошибка при вейвлет-преобразовании для {wavelet_select}. Нет данных.")
             return
    else:
        st.error(f"Ошибка при вейвлет-преобразовании для {wavelet_select}.")
        return

    # 2. Поиск значимых периодов (результат кэшируется функцией _calculate_and_cache_significant_periods)
    # --- Добавляем Spinner для поиска пиков (сработает, если кэш промахнулся) --- 
    with st.spinner("Поиск значимых периодов..."):
        # --- ИСПРАВЛЕНИЕ: Распаковка 4 значений и передача agg_rule --- 
        significant_periods_raw_df, coef_calc, periods_meas_original_scale, ts_processed_for_plot = _calculate_and_cache_significant_periods(
             # Ключи для кэша:
             pd.util.hash_pandas_object(time_series), # Используем хэш для идентификации данных
             wavelet_select, 
             max_scales, 
             agg_rule, # Правило агрегации
             threshold_percent, 
             max_periods_display,
             # Данные:
             time_series
        )
        # ----------------------------------------------------------------

    # Определение временного шага (быстро, не кэшируется)
    # Используем исходный индекс
    time_delta = get_time_delta(time_series.index)

    # --- Отображение результатов --- 
    
    # --- Проверка, что расчеты вернули данные --- 
    if coef_calc is None or periods_meas_original_scale is None:
         st.error("Не удалось получить результаты вейвлет-преобразования после кэширования.")
         st.stop()
         
    st.subheader("Вейвлет-спектр (Тепловая карта)")
    # ВАЖНО: Для heatmap используем ОРИГИНАЛЬНЫЙ time_series для оси X,
    # но coef и periods_meas_original_scale от (возможно) агрегированных данных.
    # Ось Y должна соответствовать coef.
    
    # Получаем freqs соответствующие coef_calc (если они нужны plot_wavelet_transform)
    # Вызываем wavelet_transform для получения частот.
    # Используем обработанный ряд ts_processed_for_plot, чтобы получить те же частоты,
    # которые соответствуют coef_calc.
    # Устанавливаем return_periods=False, так как периоды у нас уже есть (в исходном масштабе)
    freq_transform_result = wavelet_transform(ts_processed_for_plot, wavelet_select, num_scales=max_scales, return_periods=False)
    freqs_calc = None
    if len(freq_transform_result) == 2:
         _, freqs_calc = freq_transform_result # _ для coef
    # Обработка случая, если частоты получить не удалось (маловероятно, если coef есть)
    if freqs_calc is None:
         st.warning("Не удалось получить частоты для вейвлет-спектра.")
         # Можно попробовать создать фиктивные частоты или прервать отрисовку
         # Для простоты пока оставим None, plot_wavelet_transform должен это обработать (или исправить его)
    
    min_period_meas, max_period_meas = periods_meas_original_scale.min(), periods_meas_original_scale.max()
    # Передаем периоды в исходном масштабе для расчета тиков
    tickvals_log, ticktext = get_scale_ticks(min_period_meas, max_period_meas, time_delta, selected_unit_key)
    # Передаем оригинальный time_series для оси X, coef и periods_meas_original_scale для данных
    fig_heatmap = plot_wavelet_transform(time_series, coef_calc, freqs_calc, periods_meas_original_scale, tickvals_log, ticktext, selected_unit_key)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.subheader("Наиболее значимые периоды")
    # Для таблицы используем результат поиска пиков (significant_periods_raw_df)
    if significant_periods_raw_df is not None and not significant_periods_raw_df.empty:
         display_df = significant_periods_raw_df.copy()
         display_df['Период (формат.)'] = display_df['Период (изм.)'].apply(
             lambda p: format_period(p, time_delta, selected_unit_key)
         )
         # Форматируем колонку 'Период (изм.)' для единообразия
         display_df['Период (изм.)'] = display_df['Период (изм.)'].apply(
             lambda p: format_period(p, None, MEASUREMENT_UNIT_KEY)
         )
         
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
    # Для графика спектра мощности используем coef, periods_meas (шаг 1) 
    # и significant_periods_raw_df (шаг 2)
    fig_periodicity = plot_wavelet_periodicity_analysis(
        time_series, 
        mother_wavelet=wavelet_select,
        max_scales=max_scales, # Передаем на случай, если plot_wavelet_periodicity_analysis захочет вызвать CWT
        selected_unit_key=selected_unit_key, 
        coef=coef_calc, # Передаем результат CWT
        periods_meas=periods_meas_original_scale, # Передаем результат CWT
        significant_periods_df=significant_periods_raw_df # Передаем результат поиска пиков
    )
    st.plotly_chart(fig_periodicity, use_container_width=True)

if __name__ == "__main__":
    render_wavelet_page()