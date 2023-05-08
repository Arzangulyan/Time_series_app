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
import App_descriptions_streamlit as txt

from random import gauss


def nothing_selected(wiget):
    if wiget == "":
        st.sidebar.write("Ожидается ответ от пользователя...")
        st.stop()


def DF_wrapper(date_column, series_column):
    time_series_dict = {"Время": date_column, "Ряд": series_column}
    time_series = pd.DataFrame(time_series_dict)
    return time_series


# СЕЙЧАС ЭТО КОСТЫЛЬ, НАДО СДЕЛАТЬ КРАСИВО И АККУРАТНО!!!
def syntetic_time_series(
    date_range,
    S_1_coef=0.0,
    S_1_freq=0.0,
    S_2_coef=0.0,
    S_2_freq=0.0,
    S_3_coef=0.0,
    S_3_freq=0.0,
    C_1_coef=0.0,
    C_1_freq=0.0,
    C_2_coef=0.0,
    C_2_freq=0.0,
    C_3_coef=0.0,
    C_3_freq=0.0,
    NoiseCoef=0.0,
    TrendSlope=0.0,
):
    length = len(date_range)
    t = np.linspace(0, length - 1, length)
    # t = np.linspace(0, time_series_length, time_series_length+1)
    Sin_1 = S_1_coef * np.sin(
        2 * np.pi * t * S_1_freq
    )  # Генерация синуса по коэффициенту
    Sin_2 = S_2_coef * np.sin(2 * np.pi * t * S_2_freq)
    Sin_3 = S_3_coef * np.sin(2 * np.pi * t * S_3_freq)
    Cos_1 = C_1_coef * np.cos(2 * np.pi * t * C_1_freq)
    Cos_2 = C_2_coef * np.cos(2 * np.pi * t * C_2_freq)
    Cos_3 = C_3_coef * np.cos(2 * np.pi * t * C_3_freq)
    TrendFunc = TrendSlope * t
    NoiseFunc = np.array([gauss(0.0, NoiseCoef) for i in t])
    time_series = TrendFunc + Sin_1 + Sin_2 + Sin_3 + Cos_1 + Cos_2 + Cos_3 + NoiseFunc
    time_series_df = DF_wrapper(date_range, time_series)
    return time_series_df


# @st.cache_data
def df_chart_display_loc(df, data_col_loc):
    st.write(df.loc[:])
    st.line_chart(df.loc[:, data_col_loc])


# @st.cache_data
def df_chart_display_iloc(df, data_col_iloc):
    st.write(df.iloc[:])
    st.line_chart(df.iloc[:, data_col_iloc])

def sample_csv_download_button():
    with st.expander("Что делать, если нет своего ряда?"):

        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")

        convert_df(pd.read_csv("arzangulyan/time_series_app/new-main/TimeSeriesApp/Vietnam_CO2_Temp.csv").iloc[:, 2:])
        st.write("Настоящий ряд для тестирования можно скачать тут")
        st.download_button(
            label="Download data as CSV",
            data=convert_df(pd.read_csv("arzangulyan/time_series_app/new-main/TimeSeriesApp/Vietnam_CO2_Temp.csv")),
            file_name="Vietnam.csv",
            # mime='text/csv',
        )

# @st.cache_data


if "final_dataframe" not in st.session_state:
    st.session_state.final_dataframe = pd.DataFrame(None)


# trigonometric_args = [S_1_coef, S_1_freq, S_2_coef, S_2_freq, S_3_coef, S_3_freq, C_1_coef, C_1_freq, C_2_coef, C_2_freq, C_3_coef, C_3_freq]

st.title("Комплекс для работы с временными ря дами")
st.sidebar.header("Выбор временного ряда для обработки")

txt.intro_text()

# st.session_state
sample_csv_download_button()

# df = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vietnam_CO2_Temp.csv")
# st.line_chart(df['Temperature'])

time_series = None

# Выбор типа данных
type_of_data = ["", "Искусственный ряд", "Загруженный ряд"]
data_radio = st.sidebar.selectbox("Выберите данные для обработки", type_of_data)

# if time_series == None:
#     nothing_selected (data_radio)

if data_radio == "":
    if st.session_state.final_dataframe.empty:
        st.warning("Отсутствует ряд для анализа. Загрузите или сгененируйте его в боковом меню.")
        # st.sidebar.write('Ожидается ответ от пользователя...')
        st.stop()
    else:
        time_series = st.session_state.final_dataframe


if data_radio == "Загруженный ряд":
    st.sidebar.info('Важно, чтобы временной индекс был в одной колонке в файле!', icon="ℹ️")

    upload_file = st.sidebar.file_uploader(label="Загрузите файл CSV", type="CSV")
    if upload_file == None:
        st.stop()
    else:
        time_series = pd.read_csv(upload_file)


if data_radio == "Искусственный ряд":
    start_date = st.sidebar.date_input("Дата начала")
    end_date = st.sidebar.date_input("Дата окончания")
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    paramName_default = [
        "S_1_coef",
        "S_1_freq",
        "S_2_coef",
        "S_2_freq",
        "S_3_coef",
        "S_3_freq",
        "C_1_coef",
        "C_1_freq",
        "C_2_coef",
        "C_2_freq",
        "C_3_coef",
        "C_3_freq",
        "NoiseCoef",
        "TrendSlope",
    ]
    param_Name = st.sidebar.multiselect(
        "Выберите параметры для генерируемого ряда", paramName_default
    )



    paramDict_default = {paramName_default[i]: 0 for i in range(len(paramName_default))}
    # paramDict_default

    # Создание строк input для каждой выбранной переменной (коэф/частота)
    for i in param_Name:
        i = st.sidebar.number_input(i, key="{}".format(i))

    # Создание словаря, в котором не измененные значения остануться нулями, а измененные поменяются
    param_Dict = {
        param_Name[i]: st.session_state[param_Name[i]] for i in range(len(param_Name))
    }
    paramDict_default.update(param_Dict)

    time_series = syntetic_time_series(date_range, **paramDict_default)

st.write("---") #Вставка разделителя

# Отображение ряда
st.subheader("Исходный временной ряд")
df_chart_display_loc(
    time_series,
    data_col_loc=time_series.select_dtypes(include=["int", "float"]).columns,
)
# dframe = st.dataframe(time_series)
# dframe_chart = st.line_chart(time_series.select_dtypes(include=['int', 'float']))
# st.dataframe(time_series)

# НАДО ПОНЯТЬ, КАК ИЗБЕГАТ ОШИБКУ ВЕЗДЕ, ПОКА Я ЕЩЕ НЕ ЗАРУЗИЛ ДАННЫЕ, А НЕ ПИСАТЬ ВЕЗДЕ if != None


st.sidebar.header("Предобработка ряда")
with st.sidebar.expander("Что это такое?"):
    txt.preprocessing()

numeric_columns = list(time_series.select_dtypes(include=["int", "float"]).columns)
other_columns = list(time_series.columns)

date_column_select = st.sidebar.selectbox(
    label="Выберите колонку отражающую время", options=([""] + other_columns)
)
value_select = st.sidebar.selectbox(
    label="Выберите колонку для анализа", options=([""] + numeric_columns)
)

# Блокер ошибок при отсутствии выбранного варианта
nothing_selected(date_column_select)
nothing_selected(value_select)

time_series["Время_формат"] = pd.to_datetime(time_series[date_column_select])
# date_column_formated = pd.to_datetime(time_series.loc[:, date_column_select])
time_series_selected = time_series.loc[:, ["Время_формат", value_select]]
# time_series_selected = time_series[date_column_formated, value_select]

# 'Отформатированный временной ряд'
# st.write(time_series_selected)
# st.line_chart(time_series_selected.iloc[:, 1])

# Слайдер для выбора и отображения нужного интервала ряда
start_point, end_point = st.sidebar.slider(
    "Выберите диапазон данных для дальнейшей работы",
    0,
    time_series_selected.shape[0] - 1,
    (0, time_series_selected.shape[0] - 1),
    key="time series borders",
)

time_series_selected_limited = time_series_selected.iloc[start_point:end_point]

st.write("---") #Вставка разделителя

st.subheader("Выбранный временной ряд")
df_chart_display_iloc(time_series_selected_limited, 1)
# st.write(time_series_selected.loc[start_point:end_point])
# st.line_chart(time_series_selected.iloc[start_point:end_point, 1])

T_s_len = end_point - start_point  # Размер выбранного диапазона
st.sidebar.write("Размер выбранного диапазона:", T_s_len)

MA_checkbox = st.sidebar.checkbox("Сгладить ряд", key="MA_checkbox")
with st.sidebar.expander("Что значит «сгладить»?"):
    txt.moving_average()
MA_step = 0  # объявление нулевого шага MA
if MA_checkbox == False:
    pass
else:
    st.write("---") #Вставка разделителя
    st.subheader("Сглаживание ряда")
    MA_step = st.sidebar.number_input(
        "Введите шаг скользящего среднего", min_value=1, max_value=T_s_len
    )
    
    st.write("Шаг скользящего среднего: ", MA_step, value=1)
    # time_series_avg = time_series_selected.loc[start_point:end_point].rolling(window=MA_step, min_periods=1).mean()
    time_series_selected_limited["Сглаженный"] = time_series_selected_limited.rolling(
        window=MA_step, min_periods=1
    ).mean()

    compare_dframe = st.dataframe(time_series_selected_limited)

    col1, col2 = st.columns(2)
    with col1:
        "Сглаженный ряд"
        dframe_chart_MA = st.line_chart(
            time_series_selected_limited.loc[MA_step:, ["Сглаженный"]]
        )
    with col2:
        "Исходный ряд"
        st.line_chart(time_series_selected_limited.iloc[:, 1])

    time_series_selected_limited[value_select] = time_series_selected_limited[
        "Сглаженный"
    ]
    del time_series_selected_limited["Сглаженный"]


stationar_test_checkbox = st.sidebar.checkbox(
    "Тест на стационарность", key="stat_test_checkbox"
)
with st.sidebar.expander("Что это за тест?"):
    txt.stationar_test()
if stationar_test_checkbox:
    stat_test_res = adfuller(time_series_selected_limited.iloc[MA_step:, -1])[1]
    st.sidebar.write("Результаты теста на стационарность (p-value): ", stat_test_res)
    if stat_test_res < 0.05:
        st.sidebar.write("Ряд стационарен")
    else:
        st.sidebar.write("Ряд НЕ стационарен")


st.session_state.final_dataframe = time_series_selected_limited
# st.session_state
