import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_morfology as f
import sys
import heapq
sys.path.insert(0, 'TimeSeriesApp/functions_morfology.py')

##Monotonus numeric

st.set_page_config(page_title="Morphology")
st.title("Моделирование временных рядов с помощью морфологического анализа")
st.sidebar.header("Настройки морфологии")

def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 1])

def new_method_start():
    # st.session_state
    # st.session_state.final_dataframe.empty

    if not st.session_state.final_dataframe.empty:
        time_series = st.session_state.final_dataframe
    else:
        st.warning('Отсутствует ряд для анализа. Перейдите во вкладку «Time Series App»')
        st.stop()
    df_chart_display_iloc(time_series)
    return time_series

def Convex_numeric(time_series, period, delta):
    y = time_series.iloc[:,1]
    x = pd.Series(np.array(time_series.index)[:])
    # x = time_series.index.iloc[:]
    # y = np.array(df.S1CO2)[0:300]
    x = x-x[0]

    size = len(x)
    form_parameters = np.linspace(0, size, size*2//period + 1, dtype=int)
    inner_cycle = 2 * delta + 1
    outer_cycle = len(form_parameters) - 2
    for i in range(1, len(form_parameters)-1):
        h = []
        outer_done = i - 1

        for infl_p in range(form_parameters[i-1] + period//2 - delta, form_parameters[i-1] + period//2 + delta + 1):
            infl_p = min(infl_p, size)
            inner_done = infl_p - (form_parameters[i-1] + period//2 - delta)
            print(round((outer_done * inner_cycle + inner_done) / (outer_cycle * inner_cycle) * 100, 2), '%')

            form_parameters[i] = infl_p

            A = f.convex_constraints_matrix(form_parameters)
            app = f.numeric_sol(y, A)
            heapq.heappush(h, (np.linalg.norm(y - app), infl_p))

        form_parameters[i] = h[0][1]
    print(100, '%')

    A = f.convex_constraints_matrix(form_parameters)
    APPROXIMATION = f.numeric_sol(y, A)

    infl_p = form_parameters[1:-1]
    periods_1 = [infl_p[i+2] - infl_p[i] for i in range(len(infl_p)-2)]
    periods = np.array([(periods_1[i] + periods_1[i+1])/2 for i in range(0, len(periods_1)-1, 2)])
    avg_period = periods.mean()

    # graphs
    fig, ax1 = plt.subplots(1)
    ax1.plot(x, y, 'ro', markersize=2, label='experimental data')
    ax1.plot(x, APPROXIMATION, 'b', linewidth=1, label='result of morphological filtering')
    ax1.plot(x[infl_p], APPROXIMATION[infl_p], 'go', markersize=5, label='inflection points')
    ax1.set_xlabel("Time (1 tick = 30 min)")
    ax1.set_ylabel("CO2 concentration")
    legend1 = ax1.legend(loc='best', shadow=True, fontsize='large')
    return fig


def Monotonus_numeric(time_series, period, delta):


    y = time_series.iloc[:,1]
    x = pd.Series(np.array(time_series.index)[:])
    # x = time_series.index.iloc[:]
    # y = np.array(df.S1CO2)[0:300]
    x = x-x[0]

    size = len(x)
    form_parameters = np.linspace(0, size, size*2//period + 1, dtype=int)
    inner_cycle = 2 * delta + 1
    outer_cycle = len(form_parameters) - 2
    for i in range(1, len(form_parameters)-1):
        h = []
        outer_done = i - 1

        for infl_p in range(form_parameters[i-1] + period//2 - delta, form_parameters[i-1] + period//2 + delta + 1):
            infl_p = min(infl_p, size)
            inner_done = infl_p - (form_parameters[i-1] + period//2 - delta)
            ## Тут прогрес бар нужен
            # print(round((outer_done * inner_cycle + inner_done) / (outer_cycle * inner_cycle) * 100, 2), '%')

            form_parameters[i] = infl_p

            A = f.monotonous_constraints_matrix(form_parameters)
            app = f.numeric_sol(y, A)
            heapq.heappush(h, (np.linalg.norm(y - app), infl_p))

        form_parameters[i] = h[0][1]

    A = f.monotonous_constraints_matrix(form_parameters)
    APPROXIMATION = f.numeric_sol(y, A)

    infl_p = form_parameters[1:-1]
    periods_1 = [infl_p[i+2] - infl_p[i] for i in range(len(infl_p)-2)]
    periods = np.array([(periods_1[i] + periods_1[i+1])/2 for i in range(0, len(periods_1)-1, 2)])
    avg_period = periods.mean()

    # graphs
    fig, ax1 = plt.subplots(1)
    ax1.plot(x, y, 'ro', markersize=2, label='experimental data')
    ax1.plot(x, APPROXIMATION, 'b', linewidth=1, label='result of morphological filtering')
    ax1.plot(x[infl_p], APPROXIMATION[infl_p], 'go', markersize=5, label='inflection points')
    ax1.set_xlabel("Time (1 tick = 30 min)")
    ax1.set_ylabel("CO2 concentration")
    legend1 = ax1.legend(loc='best', shadow=True, fontsize='large')
    return fig

time_series = new_method_start()


# parameters settings (НАДО ЗАСУНУТЬ ИХ В INPUT)
#max_value = 1/2 от длины ряда, так как больше вряд-ли может быть
t_s_period = st.sidebar.slider('Выберите ожидаемую периодичность ряда', min_value=1, max_value=len(time_series)//2)
t_s_delta = st.sidebar.number_input('Выберите ожидаемую дельту ряда', value=1)

method_types = ['', 'Монотонность', 'Выпуклость']
method_radio = st.sidebar.selectbox('Выберите типа функций', method_types)

if method_radio == '':
    st.sidebar.warning('Необходимо выбрать тип функции')
    st.stop()

if method_radio == 'Монотонность':
    'Monotonus numeric'
    st.pyplot(Monotonus_numeric(time_series, period=t_s_period, delta=t_s_delta))

if method_radio == 'Выпуклость':
    st.pyplot(Convex_numeric(time_series, period=t_s_period, delta=t_s_delta))



# data import
# print("data import complete")




# '''
# # ax2.plot(
# #     x, y-APPROXIMATION, 'g', linewidth=1, label="residual series"
# #     )
# # ax2.set_xlabel("Time (1 tick = 30 min)")
# # ax2.set_ylabel("CO2 concentration")
# # legend2 = ax2.legend(loc='best', shadow=True, fontsize='large')
# '''
