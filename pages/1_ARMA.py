import streamlit as st
import pandas as pd
import App_descriptions_streamlit as txt
from modules.arma_module import (
    calculate_acf_pacf,
    plot_acf_pacf,
    arma_processing,
    forecast_arma,
)
from modules.page_template import (
    setup_page,
    load_time_series,
    run_calculations_on_button_click,
)
from method_descriptions.ARMA import DESCRIPTION, PARAMS_CHOICE



def arma_run(time_series, p, q):
    model = arma_processing(time_series, p, q)
    forecast_steps = st.sidebar.number_input(
        "Количество шагов прогнозирования в будущее", min_value=1, value=5
    )
    forecast_df = forecast_arma(model, forecast_steps)

    st.write("Прогноз временного ряда:")
    st.write(forecast_df)

    ARMA_df = pd.DataFrame(
        {
            "Прогноз": pd.concat([time_series, forecast_df["mean"]]),
            "Исходные данные": time_series,
        }
    )
    st.line_chart(ARMA_df)


def main():
    setup_page(
        "Прогнозирование временных рядов с использованием ARMA", "Настройки ARMA"
    )
    with st.expander("Что такое метод ARMA?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)

    time_series = load_time_series()
    acf_df, pacf_df = calculate_acf_pacf(time_series)
    acf_chart, pacf_chart = plot_acf_pacf(acf_df, pacf_df)
    st.altair_chart(acf_chart)
    st.altair_chart(pacf_chart)
    st.line_chart(time_series)

    st.sidebar.title("Параметры модели ARMA")
    p = st.sidebar.number_input("Параметр AR (p)", min_value=1)
    q = st.sidebar.number_input("Параметр MA (q)", min_value=1)

    try:
        run_calculations_on_button_click(arma_run, time_series, p, q)
    except ValueError as e:
        st.write("Не удалось обучить модель ARMA. Проверьте параметры и данные.")
        st.write("Ошибка:", e)


if __name__ == "__main__":
    main()
