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
    plot_wavelet_periodicity_analysis
)
from modules.utils import nothing_selected, initialize_session_state
from modules.page_template import (
    setup_page,
    load_time_series,
    display_data,
    run_calculations_on_button_click,
)
from method_descriptions.Wavelet import DESCRIPTION, PARAMS_CHOICE

def wavelet_run(time_series, wavelet_select):
    # Настройки параметров анализа
    st.subheader("Параметры вейвлет-анализа")
    
    col1, col2 = st.columns(2)
    with col1:
        # Используем только 'measurements' как единицу измерения
        scale_unit = 'measurements'
        
        max_scales = st.slider(
            "Максимальное количество масштабов", 
            min_value=10, 
            max_value=min(500, len(time_series) // 2), 
            value=min(218, len(time_series) // 2),
            help="Большее количество масштабов дает более детальное разрешение по частоте"
        )
    
    with col2:
        threshold_percent = st.slider(
            "Порог значимости (%)", 
            min_value=1.0, 
            max_value=50.0, 
            value=22.0, 
            step=1.0,
            help="Периоды с мощностью ниже этого порога будут игнорироваться"
        )
        
        max_periods = st.slider(
            "Максимальное количество периодов", 
            min_value=1, 
            max_value=20, 
            value=13,
            help="Максимальное количество периодичностей для отображения"
        )

    # Вейвлет-преобразование для визуализации
    coef, freqs, periods = wavelet_transform(time_series, wavelet_select, num_scales=max_scales, return_periods=True)
    
    # Вычисление тиков для оси Y
    min_period, max_period = periods.min(), periods.max()
    ticks = get_scale_ticks(min_period, max_period)
    tickvals = np.log2(ticks)
    ticktext = [format_period(t, scale_unit) for t in ticks]
    
    # Визуализация вейвлет-спектра
    st.subheader("Вейвлет-спектр")
    fig = plot_wavelet_transform(time_series, coef, freqs, periods, tickvals, ticktext, scale_unit)
    st.plotly_chart(fig, use_container_width=True)
    
    # Поиск значимых периодичностей
    significant_periods = find_significant_periods_wavelet(
        time_series, 
        mother_wavelet=wavelet_select,
        num_scales=max_scales,
        threshold_percent=threshold_percent,
        max_periods=max_periods,
        power_threshold=0.1  # Снижаем порог мощности для лучшего обнаружения периодов
    )
    
    # Отображение таблицы со значимыми периодами
    st.subheader("Наиболее значимые периоды")
    st.dataframe(significant_periods[['Период', 'Период (округленно)', 'Нормализованная мощность']], 
                use_container_width=True)
    
    # Визуализация спектра мощности
    st.subheader("Спектр мощности")
    periodicity_fig, _ = plot_wavelet_periodicity_analysis(
        time_series, 
        mother_wavelet=wavelet_select,
        max_scales=max_scales
    )
    st.plotly_chart(periodicity_fig, use_container_width=True)


def main():
    setup_page("Wavelets", "Настройки вейвлетов")
    
    # Добавляем описание метода
    with st.expander("Что такое вейвлет-преобразование?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    
    # Добавляем описание выбора параметров
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)
        
    # Добавляем информацию о типах вейвлетов
    with st.sidebar.expander("Типы вейвлетов"):
        st.markdown("""
        ### Характеристики различных типов вейвлетов:
        
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
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return

    wavelet_select = st.sidebar.selectbox(
        label="Выберите материнский вейвлет",
        options=(["", "Морле", "Гаусс", "Мексиканская шляпа", "Симлет", "Добеши", "Койфлет"]),
    )

    if wavelet_select:
        run_calculations_on_button_click(wavelet_run, time_series, wavelet_select)
    else:
        st.warning("Пожалуйста, выберите материнский вейвлет.")

if __name__ == "__main__":
    main()