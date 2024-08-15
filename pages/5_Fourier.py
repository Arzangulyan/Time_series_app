import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from modules.page_template import setup_page, load_time_series
from modules.fourier_module import fft_plus_power_dataframe, apply_window, next_power_of_2
from method_descriptions.Fourier import DESCRIPTION, PARAMS_CHOICE

def main():
    setup_page(
        "Выделение сезонностей во временных рядах с помощью Быстрого Фурье Преобразования",
        "Настройки Фурье преобразования"
    )
    
    with st.expander("Что такое Быстрое преобразование Фурье?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)

    time_series = load_time_series()
    if time_series is None or time_series.empty:
        st.error("Не удалось загрузить временной ряд. Пожалуйста, убедитесь, что данные загружены корректно.")
        return

    time = np.arange(0, time_series.shape[0])

    st.sidebar.subheader("Параметры FFT")
    window_type = st.sidebar.selectbox("Тип окна", ["—", "hamming", "hann", "blackman"])

    signal = time_series.iloc[:, 0] if isinstance(time_series, pd.DataFrame) else time_series

    if window_type != "—":
        signal = apply_window(signal, window_type)

    time_series_length = len(signal)
    default_nfft = next_power_of_2(time_series_length)

    nfft = st.sidebar.number_input(
        "Количество точек FFT", 
        min_value=time_series_length, 
        value=default_nfft, 
        step=2**7
    )
 
    try:
        fft_df, power_df = fft_plus_power_dataframe(time, signal, nfft)
        data = pd.concat([fft_df, power_df])

        # Перед созданием графика
        max_amplitude = data['Амплитуда'].max()
        threshold = max_amplitude * 0.01  # Например, 1% от максимальной амплитуды
        filtered_data = data[data['Амплитуда'] > threshold]

        # Используйте filtered_data для создания графика

        st.subheader("Наиболее значимые периоды")
        top_periods = filtered_data.sort_values('Амплитуда', ascending=False).head(5)
        st.write(top_periods[['Период', 'Амплитуда', 'Тип']])

        st.subheader("Спектр Фурье")
        alt_chart = alt.Chart(filtered_data).mark_line().encode(
        x=alt.X("Период", scale=alt.Scale(type="log")),
        y="Амплитуда",
        color="Тип"
    ).properties(
        width=700,
        height=400
    )
        st.altair_chart(alt_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Произошла ошибка при выполнении FFT: {str(e)}")


if __name__ == "__main__":
    main()