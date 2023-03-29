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

from random import gauss

def nothing_selected(wiget):
    if wiget == '':
        st.sidebar.write('Ожидается ответ от пользователя...')
        st.stop()

def DF_wrapper(date_column, series_column):
    time_series_dict = {'Время': date_column,
                        'Ряд': series_column}
    time_series = pd.DataFrame(time_series_dict)
    return time_series

# СЕЙЧАС ЭТО КОСТЫЛЬ, НАДО СДЕЛАТЬ КРАСИВО И АККУРАТНО!!!
def syntetic_time_series(date_range, S_1_coef=0., S_1_freq=0., S_2_coef=0., S_2_freq=0., S_3_coef=0., S_3_freq=0., C_1_coef=0., C_1_freq=0., C_2_coef=0., C_2_freq=0., C_3_coef=0., C_3_freq=0., NoiseCoef=0., TrendSlope=0.):
    length = len(date_range)
    t = np.linspace(0, length-1 , length)
    # t = np.linspace(0, time_series_length, time_series_length+1)
    Sin_1 = S_1_coef * np.sin(2*np.pi * t * S_1_freq) #Генерация синуса по коэффициенту
    Sin_2 = S_2_coef * np.sin(2*np.pi * t * S_2_freq)
    Sin_3 = S_3_coef * np.sin(2*np.pi * t * S_3_freq)
    Cos_1 = C_1_coef * np.cos(2*np.pi * t * C_1_freq)
    Cos_2 = C_2_coef * np.cos(2*np.pi * t * C_2_freq)
    Cos_3 = C_3_coef * np.cos(2*np.pi * t * C_3_freq)
    TrendFunc = TrendSlope * t
    NoiseFunc = np.array([gauss(0.0, NoiseCoef) for i in t])
    time_series = TrendFunc + Sin_1 + Sin_2 + Sin_3 + Cos_1 + Cos_2 + Cos_3 + NoiseFunc
    time_series_df = DF_wrapper(date_range, time_series)
    return time_series_df

# trigonometric_args = [S_1_coef, S_1_freq, S_2_coef, S_2_freq, S_3_coef, S_3_freq, C_1_coef, C_1_freq, C_2_coef, C_2_freq, C_3_coef, C_3_freq]

st.title("Комплекс для работы с временными рядами")

# df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vietnam_CO2_Temp.csv")
# st.line_chart(df['Temperature'])

time_series = None

#Выбор типа данных
type_of_data = ["", "Искусственный ряд", "Загруженный ряд"]
data_radio = st.sidebar.selectbox ("Выберите данные для обработки", type_of_data)


nothing_selected (data_radio)

if data_radio == 'Загруженный ряд':
    upload_file = st.sidebar.file_uploader(label="Загрузите файл CSV", type="CSV")
    if upload_file != None:
        time_series = pd.read_csv(upload_file)
    else:
        st.stop()


if data_radio == "Искусственный ряд":
    start_date = st.sidebar.date_input ('Дата начала')
    end_date = st.sidebar.date_input ('Дата окончания')
    date_range = pd.date_range(start= start_date, end= end_date, freq='D')

    paramName_default = ['S_1_coef', 'S_1_freq', 'S_2_coef', 'S_2_freq', 'S_3_coef', 'S_3_freq', 'C_1_coef', 'C_1_freq', 'C_2_coef', 'C_2_freq', 'C_3_coef', 'C_3_freq', 'NoiseCoef', 'TrendSlope']    
    param_Name = st.sidebar.multiselect("Выберите параметры для генерируемого ряда", paramName_default)

    paramDict_default = {paramName_default[i]: 0 for i in range(len(paramName_default))}
    # paramDict_default

    #Создание строк input для каждой выбранной переменной (коэф/частота)
    for i in param_Name:
        i = st.sidebar.number_input(i, key="{}".format(i))
    
    #Создание словаря, в котором не измененные значения остануться нулями, а измененные поменяются
    param_Dict = {param_Name[i]: st.session_state[param_Name[i]] for i in range(len(param_Name))}
    paramDict_default.update(param_Dict)
    paramDict_default


    time_series = syntetic_time_series(date_range, **paramDict_default)

#Отображение ряда
dframe = st.dataframe(time_series)

#НАДО ПОНЯТЬ, КАК ИЗБЕГАТ ОШИБКУ ВЕЗДЕ, ПОКА Я ЕЩЕ НЕ ЗАРУЗИЛ ДАННЫЕ, А НЕ ПИСАТЬ ВЕЗДЕ if != None

numeric_columns = list(time_series.select_dtypes(include=['int', 'float']).columns)
other_columns = list(time_series.columns)

date_column_select = st.sidebar.selectbox(label='Выберите колонку отражающую время', options=([""]+other_columns))
value_select = st.sidebar.selectbox(label='Выберите колонку для анализа', options=([""]+numeric_columns))

nothing_selected(data_radio)
nothing_selected(value_select)


time_series_selected = time_series.loc[:,[date_column_select, value_select]]
dframe.write (time_series_selected)

start_point, end_point = st.sidebar.slider(
    'Выберите диапазон данных',
    0, time_series_selected.shape[0]-1, (0, time_series_selected.shape[0]-1), key='time series borders')
dframe.write(time_series_selected.loc[start_point:end_point])
T_s_len = end_point-start_point #Размер выбранного диапазона
st.sidebar.write("Размер выбранного диапазона:", T_s_len)

st.line_chart(time_series_selected.iloc[:, 1])