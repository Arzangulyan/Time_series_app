import streamlit as st
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(page_title="Wavelets")


st.title("Комплекс для работы с временными рядами")

# df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vietnam_CO2_Temp.csv")
# st.line_chart(df['Temperature'])


st.sidebar.subheader("Настройки")
upload_file = st.sidebar.file_uploader(label="Загрузите файл CSV", type="CSV")

type_of_data = st.sidebar.checkbox ("Тестовый временной ряд", value=True)
if upload_file != None:
    df = pd.read_csv(upload_file)
elif type_of_data:
    x = np.linspace(0, 2000,2000)
    # df = pd.DataFrame(np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900), columns = ['Тестовый ВР'])
    df = pd.DataFrame(10*np.sin(2*np.pi*x/100) + 10*np.cos(2*np.pi*x/900), columns = ['Тестовый ВР'])


        # df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Vietnam_CO2_Temp.csv")

dframe = st.dataframe(df)
numeric_columns = list(df.select_dtypes(include=['int', 'float']).columns)

value_select = st.sidebar.radio(label='Выберите колонку для анализа', options=(["Выбрать"]+numeric_columns))

if value_select != "Выбрать":
    df_selected = pd.DataFrame(df.loc[:, value_select])
    dframe.write(df_selected)

    start_point, end_point = st.sidebar.slider(
    'Выберите диапазон данных',
    0, df_selected.shape[0]-1, (0, df_selected.shape[0]-1))
    dframe.write(df_selected.loc[start_point:end_point])
    T_s_len = end_point-start_point
    st.sidebar.write("Размер выбранного диапазона:", T_s_len)
    # value_select = value_select[start_point:end_point]

    st.sidebar.write("Усреднить скользящим средним")
    m_a_step = st.sidebar.number_input('Введите шаг скользящего среднего', min_value=1, max_value=df_selected.shape[0])
    st.write('Шаг скользящего среднего: ', m_a_step, value=1)
    df_selected['Averaged'] = df_selected.rolling(window=m_a_step, min_periods=1).mean()

    # dframe.write()
    # dframe.write(df_selected)
    linechart = st.line_chart(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])
    st.dataframe(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])
    # st.write(df_selected)
    # linechart2 = st.line_chart(df_selected.loc[start_point:end_point])

    # linechart = st.line_chart(df_selected.iloc[start_point:end_point, 1])

    stationar_test = st.sidebar.button("Тест на стационарность")
    if stationar_test:
        st.sidebar.write("Результаты теста на стационарность (p-value): ", adfuller(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])[1])

    operation_select = st.sidebar.radio(label='Что вы хотите сделать', options=(["Выбрать", "Анализ", "Восстановление данных", "Прогнозирование"]))
    if operation_select == "Анализ":


        pacf = plot_pacf(df, lags=25)
        acf = plot_acf(df, lags=25)
        # st.pyplot(plot_acf)




        method_select = st.sidebar.radio(label='Выберите метод анализа', options=(["Выбрать","Вейвлет преобразование", "Быстрое Фурье преобразование"]))

        if method_select == "Вейвлет преобразование":

            wavelet_select = st.sidebar.radio(label='Выберите материнский вейвлет', options=(["Выбрать", "Морле", "Гаусс", "Мексиканская шляпа", "Другой"]))
            mother_switcher = {
                "Морле": "morl",
                "Гаусс": "gaus1",
                "Мексиканская шляпа": "mexh",
                "Другой": ""
            }

            if wavelet_select != "Выбрать":

                # #TEST
                # x = np.linspace(0, 1000,1000)
                # y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900)
                # linechart = st.line_chart(y)
                # coef, freqs = pywt.cwt(y, np.arange(1,129), 'mexh')
                # fig, ax = plt.subplots()
                # ax.imshow(coef, cmap = 'copper', aspect = 'auto')
                # st.pyplot(fig)
                # # sns.heatmap(coef, ax = ax, cmap = 'copper')
                # # st.write(fig)
                #
                # I = np.empty((len(freqs)))
                # for j in range(len(freqs)-1):
                #     for i in range(len(y)):
                #         I[j] += ((coef[j, i])**2 + (coef[j+1,i])**2)/2
                # # st.write(I)
                # I_s = pd.DataFrame({'I':I, 'Freqs': freqs})
                # st.write(I_s)
                # fig2, ax2 = plt.subplots()
                # ax2.plot(freqs, I)
                # ax2.set_aspect('auto')
                # st.pyplot(fig2)
                #
                # # Intergral_spectrum = st.line_chart(I_s)
                # #TEST

                #REAL

                coef, freqs = pywt.cwt(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'], np.arange(1, T_s_len/4), mother_switcher.get(wavelet_select))
                fig1, ax1 = plt.subplots()
                ax1.imshow(coef, cmap = 'copper', aspect = 'auto')
                # sns.heatmap(coef, ax = ax, cmap = 'copper')
                # st.write(fig)
                ax1.set_title("Power Spectrum", fontsize=20)
                ax1.set_ylabel("Период", fontsize=18)
                ax1.set_xlabel("Время", fontsize=18)
                ax1.invert_yaxis()
                st.pyplot(fig1)



                # I = np.empty((len(freqs)))
                # for j in range(len(freqs)-1):
                #     for i in range(len(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])-1):
                #         I[j] += ((coef[j, i]) + (coef[j+1,i]))/2
                # # st.write(I)
                # Int_freq = pd.DataFrame({'I':I, 'Freqs': freqs})
                # st.write(Int_freq)
                # fig2, ax2 = plt.subplots()
                # ax2.plot(freqs, I)
                # ax2.set_aspect('auto')
                # plt.xscale("log")
                # st.pyplot(fig2)
                #REAL

        elif method_select == "Быстрое Фурье преобразование":

            def get_fft_values(y_values, T, N, f_s):
                f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
                fft_values_ = fft(y_values)
                fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
                return f_values, fft_values

            def plot_fft_plus_power(time, signal):
                dt = time[1] - time[0]
                N = len(signal)
                fs = 1/dt

                fig, ax = plt.subplots(figsize=(15, 3))
                variance = np.std(signal)**2
                f_values, fft_values = get_fft_values(signal, dt, N, fs)
                fft_power = variance * abs(fft_values) ** 2     # FFT power spectrum
                ax.plot(f_values, fft_values, 'r-', label='Фурье преобразование')
                ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
                ax.set_xlabel('Частота [Hz / year]', fontsize=18)
                ax.set_ylabel('Амплитуда', fontsize=18)
                ax.legend()
                st.pyplot(fig)

            signal = df_selected.loc[(start_point+m_a_step):end_point, 'Averaged']
            time = np.arange(0, T_s_len)
            plot_fft_plus_power(time, signal)
