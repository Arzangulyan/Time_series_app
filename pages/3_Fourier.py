import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.page_template import setup_page, load_time_series
from modules.fourier_module import (
    find_significant_periods_fourier,
    plot_fourier_periodicity_analysis,
    next_power_of_2
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
    
    nfft = st.sidebar.number_input(
        "Количество точек FFT", 
        min_value=time_series_length, 
        value=default_nfft, 
        step=2**7
    )
    
    power_threshold = st.sidebar.slider(
        "Порог мощности",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Периоды с амплитудой ниже этого порога будут игнорироваться"
    )
    
    max_periods = st.sidebar.slider(
        "Максимальное количество периодов",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
 
    try:
        # Создаем график и получаем значимые периоды
        fig, significant_periods = plot_fourier_periodicity_analysis(
            time_series,
            num_points=nfft,
            max_periods=max_periods,
            power_threshold=power_threshold
        )
        
        # Отображаем график
        st.subheader("Спектр мощности")
        st.plotly_chart(fig, use_container_width=True)
        
        # Отображаем таблицу с периодами
        st.subheader("Наиболее значимые периоды")
        
        if not significant_periods.empty:
            # Выбираем и переименовываем колонки для отображения
            display_df = significant_periods[['Период', 'Период (округленно)', 'Амплитуда']].copy()
            display_df.columns = ['Период', 'Период (округленно)', 'Нормализованная мощность']
            
            # Сортируем по убыванию мощности
            display_df = display_df.sort_values('Нормализованная мощность', ascending=False)
            
            # Отображаем таблицу
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Не удалось обнаружить значимые периоды. Попробуйте изменить параметры анализа.")
            
    except Exception as e:
        st.error(f"Произошла ошибка при выполнении анализа Фурье: {str(e)}")
        st.info("Проверьте входные данные и параметры анализа.")

if __name__ == "__main__":
    main()