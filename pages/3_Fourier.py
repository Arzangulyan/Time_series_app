import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time # Импортируем time
from modules.page_template import setup_page, load_time_series
from modules.fourier_module import (
    find_significant_periods_fourier,
    plot_fourier_periodicity_analysis,
    next_power_of_2,
    TIME_UNITS,
    DEFAULT_TIME_UNIT,
    MEASUREMENT_UNIT_KEY,
    format_period,
    get_time_delta
)

def main():
    setup_page(
        "Спектральный анализ Фурье",
        "Настройки преобразования Фурье"
    )
    
    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return

    st.sidebar.subheader("Параметры анализа")
    
    # Определение количества точек FFT
    time_series_length = len(time_series)
    default_nfft = next_power_of_2(time_series_length)
    min_nfft = time_series_length
    max_nfft = max(min_nfft, default_nfft * 2)

    # Адаптивный шаг для nfft
    if default_nfft <= 128:
        nfft_step = 16
    elif default_nfft <= 512:
        nfft_step = 32
    elif default_nfft <= 2048:
        nfft_step = 64
    else:
        nfft_step = 128
    
    # Корректировка шага, если он слишком большой для диапазона
    if max_nfft == min_nfft: # Если диапазон нулевой (например, time_series_length - степень двойки)
        nfft_step = 1
    elif (max_nfft - min_nfft) < nfft_step:
        # Если диапазон меньше шага, делаем шаг меньше, чтобы было хотя бы одно изменение
        # Например, step = 1 или такой, чтобы покрыть диапазон в несколько шагов
        nfft_step = max(1, (max_nfft - min_nfft) // 2) if (max_nfft - min_nfft) > 1 else 1

    help_text_nfft = f"""Количество точек для БПФ (NFFT). Это значение влияет на детализацию частотного спектра.
- **Минимальное значение ({min_nfft}):** Равно длине вашего временного ряда ({time_series_length}). Использование этого значения анализирует ваш ряд без дополнения нулями.
- **Значения больше длины ряда:** Ряд дополняется нулями до указанной длины NFFT. Это НЕ добавляет новой информации в исходные данные, но может сделать спектр более гладким и помочь лучше выделить пики частот (эффект интерполяции).
- **Значение по умолчанию ({default_nfft}):** Следующая степень двойки от длины ряда. Степени двойки часто используются для оптимизации скорости вычислений БПФ, хотя современные библиотеки хорошо работают и с другими значениями.
- **Выбор значения:** Большие значения NFFT увеличивают разрешение по частоте (позволяют различать близкие частоты) и сглаживают спектр, но также увеличивают время расчета. Слишком малые значения (меньше длины ряда) приведут к потере информации.
- **Текущий диапазон: от {min_nfft} до {max_nfft}. Шаг: {nfft_step}.**"""
    
    # --- Выбор единиц измерения периода --- 
    unit_options = list(TIME_UNITS.keys())
    default_unit_idx = 0
    time_delta = get_time_delta(time_series.index) # Получаем time_delta для определения единиц по умолчанию

    if time_delta: # Если есть временной индекс
        if DEFAULT_TIME_UNIT in unit_options:
            default_unit_idx = unit_options.index(DEFAULT_TIME_UNIT)
    elif MEASUREMENT_UNIT_KEY in unit_options: # Если нет временного индекса, используем измерения
        default_unit_idx = unit_options.index(MEASUREMENT_UNIT_KEY)

    selected_unit_key_from_selectbox = st.sidebar.selectbox(
         label="Единицы периода",
         options=unit_options,
         index=default_unit_idx,
         key='fourier_period_unit',
         help="Выберите, в каких единицах отображать найденные периоды."
    )
    # Гарантируем, что selected_unit_key не None
    selected_unit_key = selected_unit_key_from_selectbox if selected_unit_key_from_selectbox is not None else MEASUREMENT_UNIT_KEY

    # --- Новые элементы управления для логарифмического режима ---
    st.sidebar.markdown("#### Режим анализа Фурье")
    analysis_mode = st.sidebar.radio(
        "Выберите режим анализа:",
        ("Стандартный", "Итеративный (экспериментальный)"),
        key='fourier_analysis_mode',
        help="Стандартный: классический поиск пиков. Итеративный: последовательное нахождение и вычитание сильнейших периодов."
    )

    st.sidebar.markdown("#### Общие параметры")
    # NFFT и Max Periods остаются общими
    nfft_value = st.sidebar.number_input(
        "Количество точек FFT", 
        min_value=min_nfft, 
        max_value=max_nfft,
        value=default_nfft, 
        step=nfft_step,
        help=help_text_nfft,
        key='fourier_nfft'
    )
    max_periods_to_show = st.sidebar.slider(
        "Максимальное количество периодов для вывода",
        min_value=1,
        max_value=25, # Немного увеличил максимум
        value=10,
        step=1,
        key='fourier_max_periods'
    )

    # Параметры для стандартного режима
    if analysis_mode == "Стандартный":
        st.sidebar.markdown("#### Настройки стандартного анализа")
        use_log_scale_y_checkbox = st.sidebar.checkbox(
            "Логарифмическая шкала Y для графика", 
            value=False,
            key='fourier_log_scale_y',
            help="Отображать ось Y (амплитуда/мощность) спектра в логарифмическом масштабе (log10). Помогает увидеть слабые пики."
        )
        use_log_for_peaks_checkbox = st.sidebar.checkbox(
            "Искать пики в логарифмическом спектре", 
            value=False,
            key='fourier_log_for_peaks',
            help="Искать значимые периоды на основе логарифма амплитуд. Может помочь выявить пики, которые малы в абсолютном значении, но выделяются на логарифмической шкале."
        )
        log_peak_factor_slider = 1.5 
        if use_log_for_peaks_checkbox:
            log_peak_factor_slider = st.sidebar.slider(
                "Фактор порога для лог. пиков (N в медиана + N*std)",
                min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                key='fourier_log_peak_factor',
                help="При поиске пиков в лог. спектре, порог высоты пика = медиана(лог.амплитуд) + N * std(лог.амплитуд). Этот слайдер задает N."
            )
        power_threshold_slider = st.sidebar.slider(
            "Порог мощности (для линейного поиска)",
            min_value=0.01, max_value=0.95, value=0.1, step=0.01,
            key='fourier_power_threshold',
            help="(Для линейного режима стандартного поиска) Относительный порог. Игнорируется, если включен поиск пиков в лог. спектре."
        )
    # Параметры для итеративного режима
    elif analysis_mode == "Итеративный (экспериментальный)":
        st.sidebar.markdown("#### Настройки итеративного анализа")
        num_iterations_slider = st.sidebar.slider(
            "Количество итераций",
            min_value=1,
            max_value=15, # Максимум итераций
            value=5,
            step=1,
            key='fourier_num_iterations',
            help="Количество последовательных удалений сильнейшего периода."
        )
        loop_gain_slider = st.sidebar.slider(
            "Коэффициент вычитания (loop gain)",
            min_value=0.1,
            max_value=1.0,
            value=0.6, # Значение по умолчанию, предложенное вами
            step=0.05,
            key='fourier_loop_gain',
            help="Доля амплитуды пика для вычитания на каждой итерации. Меньшие значения могут быть стабильнее, но требуют больше итераций."
        )
 
    calculation_time_str = ""
    fig = None # Инициализируем fig
    significant_periods_df = pd.DataFrame() # Инициализируем DataFrame

    try:
        start_time = time.time()

        if analysis_mode == "Стандартный":
            fig, significant_periods_df = plot_fourier_periodicity_analysis(
                time_series,
                num_points=nfft_value,
                max_periods=max_periods_to_show,
                power_threshold=power_threshold_slider,
                selected_unit_key=selected_unit_key,
                use_log_scale_y=use_log_scale_y_checkbox,
                use_log_for_peaks=use_log_for_peaks_checkbox,
                log_peak_threshold_factor=log_peak_factor_slider
            )
        elif analysis_mode == "Итеративный (экспериментальный)":
            # Импортируем новую функцию
            from modules.fourier_module import find_significant_periods_fourier_iterative
            
            st.info("Итеративный анализ: график спектра мощности для этого режима не отображается стандартным образом, т.к. спектр меняется на каждой итерации. Отображается таблица найденных периодов.")
            
            # Явное преобразование nfft_value в int, если оно не None
            nfft_int_value = int(nfft_value) if nfft_value is not None else None
            
            significant_periods_df = find_significant_periods_fourier_iterative(
                time_series=time_series.iloc[:,0] if isinstance(time_series, pd.DataFrame) else time_series, # Передаем pd.Series
                num_iterations=num_iterations_slider,
                nfft=nfft_int_value, # Используем преобразованное значение
                max_total_periods=max_periods_to_show,
                loop_gain=loop_gain_slider # Передаем значение loop_gain
            )
            # Для итеративного режима пока не будем создавать сложный график fig
            # Можно будет позже добавить простой график найденных периодов, если потребуется
            # fig = go.Figure() # Пустой график или график из significant_periods_df

        end_time = time.time()
        calculation_time_str = f"Время выполнения расчетов: {end_time - start_time:.3f} сек."
        
        st.subheader("Спектр мощности")
        if calculation_time_str:
            st.caption(calculation_time_str)
        
        if fig is not None and analysis_mode == "Стандартный": # Показываем график только для стандартного режима
            st.plotly_chart(fig, use_container_width=True)
        elif analysis_mode == "Итеративный (экспериментальный)" and not significant_periods_df.empty:
            # Можно отобразить простой график найденных периодов для итерационного режима
            if 'Период (изм.)' in significant_periods_df.columns and 'Норм. мощность' in significant_periods_df.columns:
                iter_fig = go.Figure(data=[
                    go.Bar(
                        x=significant_periods_df['Период (изм.)'].astype(str) + " изм. (Итер. " + significant_periods_df['Итерация'].astype(str) + ")", 
                        y=significant_periods_df['Норм. мощность'],
                        text=significant_periods_df['Норм. мощность'].round(3),
                        textposition='auto'
                    )
                ])
                iter_fig.update_layout(
                    title_text="Найденные периоды (итеративный анализ)",
                    xaxis_title="Период (измерения) и Итерация",
                    yaxis_title="Нормализованная мощность"
                )
                st.plotly_chart(iter_fig, use_container_width=True)
            else:
                st.info("Нет достаточных данных для построения графика итеративного анализа.")
        
        st.subheader("Наиболее значимые периоды")
        
        if significant_periods_df is not None and not significant_periods_df.empty:
            # Форматируем периоды для отображения
            display_df = significant_periods_df.copy()
            
            # Получаем time_delta еще раз, если он нужен здесь (хотя plot уже его использовал)
            # time_delta_for_table = get_time_delta(time_series.index) 
            # Он уже есть из selected_unit_key определения

            formatted_periods = [
                format_period(p_meas, time_delta, selected_unit_key) # selected_unit_key здесь точно str
                for p_meas in display_df['Период (изм.)']
            ]
            display_df['Период (формат.)'] = formatted_periods
            
            # Добавляем колонку с периодом только в измерениях для справки
            measurement_periods = [
                format_period(p_meas, None, MEASUREMENT_UNIT_KEY)
                for p_meas in display_df['Период (изм.)']
            ]
            display_df['Период (изм. только)'] = measurement_periods

            # Переименовываем и выбираем колонки для отображения
            display_df = display_df.rename(columns={'Мощность': 'Ампл.комп.', 'Норм. мощность': 'Норм. ампл.комп.'})
            
            # Колонки для отображения в зависимости от режима
            cols_to_show = ['Период (формат.)', 'Период (изм. только)']
            if 'Норм. ампл.комп.' in display_df.columns: # Для итеративного и стандартного (если есть)
                cols_to_show.append('Норм. ампл.комп.')
            elif 'Ампл.комп.' in display_df.columns: # Если только 'Мощность' (амплитуда компонента)
                 cols_to_show.append('Ампл.комп.')

            if analysis_mode == "Итеративный (экспериментальный)" and 'Итерация' in display_df.columns:
                cols_to_show.append('Итерация')
            
            display_df_final = display_df[cols_to_show]
             
            # Сортируем по убыванию мощности/амплитуды
            sort_col = 'Норм. ампл.комп.' if 'Норм. ампл.комп.' in display_df_final.columns else 'Ампл.комп.'
            if sort_col in display_df_final.columns:
                display_df_final = display_df_final.sort_values(sort_col, ascending=False)
             
            # Отображаем таблицу
            st.dataframe(
                display_df_final, 
                use_container_width=True,
                column_config={
                    "Период (формат.)": st.column_config.TextColumn(
                        "Период", 
                        help=f"Период в выбранных единицах ({selected_unit_key})"
                    ),
                    "Период (изм. только)": st.column_config.TextColumn(
                        "Период (в изм.)", 
                        help="Период в количестве измерений"
                    ),
                    "Норм. ампл.комп.": st.column_config.NumberColumn(
                        "Нормализованная мощность", format="%.3f", 
                        help="Относительная мощность Фурье-спектра"
                    )
                },
                hide_index=True
            )
        else:
            st.info("Не удалось обнаружить значимые периоды. Попробуйте изменить параметры анализа.")
            
    except Exception as e:
        st.error(f"Произошла ошибка при выполнении анализа Фурье: {str(e)}")
        st.info("Проверьте входные данные и параметры анализа.")

if __name__ == "__main__":
    main()