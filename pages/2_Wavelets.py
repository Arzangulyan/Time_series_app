import streamlit as st
import numpy as np
import pandas as pd
import App_descriptions_streamlit as txt
from modules.wavelet_module import wavelet_transform, plot_wavelet_transform
from modules.utils import nothing_selected
from modules.page_template import (
    setup_page,
    load_time_series,
    display_data,
    run_calculations_on_button_click,
)


def wavelet_run(time_series, wavelet_select):
    fig = plot_wavelet_transform(time_series, wavelet_select)
    st.pyplot(fig)


def main():
    setup_page("Wavelets", "Настройки вейвлетов")
    txt.Wavelet_descr()

    time_series = load_time_series()

    wavelet_select = st.sidebar.selectbox(
        label="Выберите материнский вейвлет",
        options=(["", "Морле", "Гаусс", "Мексиканская шляпа"]),
    )

    nothing_selected(wavelet_select)

    run_calculations_on_button_click(wavelet_run, time_series, wavelet_select)
    # coef, freq = wavelet_transform(time_series, wavelet_select)


if __name__ == "__main__":
    main()
