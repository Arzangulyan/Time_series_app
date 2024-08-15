import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from modules.page_template import (
    setup_page,
    load_time_series,
    run_calculations_on_button_click,
)
from modules.arima_module import (
    calculate_acf_pacf,
    plot_acf_pacf,
    arima_processing,
    forecast_arima,
    apply_differencing
)
from method_descriptions.ARIMA import DESCRIPTION, PARAMS_CHOICE
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def arima_processing(data, p, d, q):
    model = ARIMA(data, order=(p, d, q)).fit()
    return model

def forecast_arima(model, forecast_steps):
    forecast = model.forecast(steps=forecast_steps)
    return pd.DataFrame(forecast, columns=['mean'])

def arima_run(time_series, p, d, q):
    model = arima_processing(time_series, p, d, q)
    forecast_steps = st.sidebar.number_input(
        "Количество шагов прогнозирования в будущее", min_value=1, value=5
    )
    forecast_df = forecast_arima(model, forecast_steps)

    st.write("Прогноз временного ряда:")
    st.write(forecast_df)

    ARIMA_df = pd.DataFrame(
        {
            "Прогноз": pd.concat([time_series, forecast_df["mean"]]),
            "Исходные данные": time_series,
        }
    )
    st.line_chart(ARIMA_df)

def main():
    setup_page(
        "Прогнозирование временных рядов с использованием ARIMA", "Настройки ARIMA"
    )
    with st.expander("Что такое метод ARIMA?"):
        st.markdown(DESCRIPTION, unsafe_allow_html=True)
    
    with st.sidebar.expander("Как выбрать параметры?"):
        st.markdown(PARAMS_CHOICE, unsafe_allow_html=True)

    time_series = load_time_series()
    data = time_series.iloc[:, 0]

    st.subheader("Исходный временной ряд")
    st.line_chart(data)

    st.subheader("Анализ стационарности и выбор параметра d")
    
    d = st.number_input("Выберите порядок дифференцирования (d)", min_value=0, max_value=2, value=0, step=1)
    
    if d > 0:
        diff_data = apply_differencing(data, d)
        st.line_chart(diff_data)
        st.write(f"Временной ряд после дифференцирования (d={d})")
    else:
        diff_data = data
    
    adf_result = adfuller(diff_data)
    st.write("Результаты теста Дики-Фуллера:")
    st.write(f"ADF Statistic: {adf_result[0]}")
    st.write(f"p-value: {adf_result[1]}")
    for key, value in adf_result[4].items():
        st.write(f"Critical Value ({key}): {value}")
    
    if adf_result[1] <= 0.05:
        st.success("Временной ряд стационарен (p-value <= 0.05)")
    else:
        st.warning("Временной ряд не стационарен (p-value > 0.05). Рекомендуется увеличить порядок дифференцирования.")


    # В основной функции:
    st.subheader("Анализ ACF и PACF")
    acf_pacf_fig = plot_acf_pacf(diff_data)
    st.plotly_chart(acf_pacf_fig, use_container_width=True)

    st.sidebar.title("Параметры модели ARIMA")
    p = st.sidebar.number_input("Параметр AR (p)", min_value=0, value=1)
    q = st.sidebar.number_input("Параметр MA (q)", min_value=0, value=1)

    try:
        if st.button("Запустить ARIMA"):
            model = arima_processing(data, p, d, q)
            st.write("Сводка модели ARIMA:")
            st.text(str(model.summary()))
            
            forecast_steps = st.number_input("Количество шагов для прогноза", min_value=1, value=10)
            forecast = forecast_arima(model, forecast_steps)
            
            st.subheader("Прогноз")
            
            fig = go.Figure()
            
            # Исходные данные
            fig.add_trace(go.Scatter(x=data.index, y=data,
                                    mode='lines',
                                    name='Исходные данные'))
            
            # Прогноз
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast['mean'],
                                    mode='lines',
                                    name='Прогноз',
                                    line=dict(color='red')))
            
            # Доверительный интервал
            fig.add_trace(go.Scatter(
                x=forecast.index.tolist() + forecast.index[::-1].tolist(),
                y=(forecast['mean'] + 1.96 * forecast['std_error']).tolist() + 
                (forecast['mean'] - 1.96 * forecast['std_error'])[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255,192,203,0.3)',
                line=dict(color='rgba(255,192,203,0.3)'),
                name='95% Доверительный интервал'
            ))
            
            fig.update_layout(
                title='Прогноз ARIMA',
                xaxis_title='Дата',
                yaxis_title='Значение',
                legend_title='Легенда',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Произошла ошибка при выполнении ARIMA: {str(e)}")

if __name__ == "__main__":
    main()