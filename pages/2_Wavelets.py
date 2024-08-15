import streamlit as st
import numpy as np
import pandas as pd
import App_descriptions_streamlit as txt
from modules.wavelet_module import wavelet_transform, plot_wavelet_transform, get_scale_ticks, format_period
from modules.utils import nothing_selected, initialize_session_state
from modules.page_template import (
    setup_page,
    load_time_series,
    display_data,
    run_calculations_on_button_click,
)
from method_descriptions.Wavelet import DESCRIPTION, PARAMS_CHOICE

def wavelet_run(time_series, wavelet_select):
    coef, freqs = wavelet_transform(time_series, wavelet_select)
    
    # Используем st.radio() для установки значения, но не присваиваем его напрямую
    scale_unit = st.radio("Единица измерения периода:", ('days', 'measurements'), key='scale_unit')
    
    periods = 1 / freqs
    min_period, max_period = periods.min(), periods.max()
    ticks = get_scale_ticks(min_period, max_period)
    tickvals = np.log2(ticks)
    
    # Добавляем проверку на None
    ticktext = [format_period(t, scale_unit or 'days') for t in ticks]
    
    fig = plot_wavelet_transform(time_series, coef, freqs, periods, tickvals, ticktext, scale_unit)
    st.plotly_chart(fig)


def main():
    setup_page("Wavelets", "Настройки вейвлетов")
    
    # Добавляем описание метода
    with st.expander("Что такое вейвлет-преобразование?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    
    # Добавляем описание выбора параметров
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)

    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return

    wavelet_select = st.sidebar.selectbox(
        label="Выберите материнский вейвлет",
        options=(["", "Морле", "Гаусс", "Мексиканская шляпа"]),
    )

    if wavelet_select:
        run_calculations_on_button_click(wavelet_run, time_series, wavelet_select)
    else:
        st.warning("Пожалуйста, выберите материнский вейвлет.")

if __name__ == "__main__":
    main()