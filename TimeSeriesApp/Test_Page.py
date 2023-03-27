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

st.title("Комплекс для работы с временными рядами")

# df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vietnam_CO2_Temp.csv")
# st.line_chart(df['Temperature'])

time_series = None

#Выбор типа данных
type_of_data = ["Никакие", "Искусственный ряд", "Загруженный ряд"]
data_radio = st.sidebar.radio ("Выберите данные для обработки", type_of_data)
if data_radio == 'Никакие':
    time_series = None

if data_radio == 'Загруженный ряд':
    upload_file = st.sidebar.file_uploader(label="Загрузите файл CSV", type="CSV")
    if upload_file != None:
        time_series = pd.read_csv(upload_file)

if data_radio == "Искусственный ряд":
    x = np.linspace(0, 2000,2000)
    # df = pd.DataFrame(np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900), columns = ['Тестовый ВР'])
    time_series = pd.DataFrame(10*np.sin(2*np.pi*x/100) + 10*np.cos(2*np.pi*x/900), columns = ['Тестовый ВР'])


        # df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Vietnam_CO2_Temp.csv")

#Отображение ряда
dframe = st.dataframe(time_series)

#НАДО ПОНЯТЬ, КАК ИЗБЕГАТ ОШИБКУ ВЕЗДЕ, ПОКА Я ЕЩЕ НЕ ЗАРУЗИЛ ДАННЫЕ, А НЕ ПИСАТЬ ВЕЗДЕ if != None

numeric_columns = list(time_series.select_dtypes(include=['int', 'float']).columns)
other_columns = list(time_series.columns)

date_column_select = st.sidebar.selectbox(label='Выберите колонку отражающую время', options=(["Никакая"]+other_columns))
value_select = st.sidebar.selectbox(label='Выберите колонку для анализа', options=(["Никакая"]+numeric_columns))

time_series_selected = time_series.loc[:,[date_column_select, value_select]]
dframe.write (time_series_selected)

start_point, end_point = st.sidebar.slider(
    'Выберите диапазон данных',
    0, time_series_selected.shape[0]-1, (0, time_series_selected.shape[0]-1), key='time series borders')
dframe.write(time_series_selected.loc[start_point:end_point])
T_s_len = end_point-start_point #Размер выбранного диапазона
st.sidebar.write("Размер выбранного диапазона:", T_s_len)