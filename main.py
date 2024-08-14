import streamlit as st
import pandas as pd
import datetime
from modules import (
    visualization,
    data_processing,
    utils,
    synthetic_data,
    time_series_analysis,
    visualization,
)
import App_descriptions_streamlit as txt
from config import DATA_TYPES, PARAM_NAMES


# Инициализация состояния сессии
# if "final_dataframe" not in st.session_state:
# st.session_state.final_dataframe = pd.DataFrame()


def args_init():
    utils.initialize_session_state(
        start_point=None,
        end_point=None,
        MA_checkbox=False,
        MA_step=1,
        date_column_select=None,
        value_select=None,
    )


def load_data():
    st.sidebar.info(
        "Важно, чтобы временной индекс был в одной колонке в файле!", icon="ℹ️"
    )
    upload_file = st.sidebar.file_uploader(label="Загрузите файл CSV", type="CSV")
    if upload_file is None:
        return None
    try:
        return pd.read_csv(upload_file, parse_dates=True)
    except Exception as e:
        st.sidebar.error(f"Ошибка при загрузке файла: {str(e)}")
        return None


def create_synthetic_data():
    utils.initialize_session_state(
        start_date=None, end_date=None, param_dict={}, selected_params=[]
    )
    start_date = st.sidebar.date_input(
        "Дата начала",
        value=st.session_state.start_date if st.session_state.start_date else "today",
    )
    end_date = st.sidebar.date_input(
        "Дата окончания",
        value=st.session_state.end_date if st.session_state.end_date else "today",
    )
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    if not isinstance(start_date, datetime.date) or not isinstance(
        end_date, datetime.date
    ):
        st.sidebar.error("Пожалуйста, выберите корректные даты.")
        return None

    if start_date >= end_date:
        st.sidebar.error("Дата начала должна быть раньше даты окончания.")
        st.stop()

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    st.session_state.param_dict = {name: 0.0 for name in PARAM_NAMES}

    selected_params = st.sidebar.multiselect(
        "Выберите параметры для генерируемого ряда",
        PARAM_NAMES,
        default=st.session_state.selected_params,
    )
    # Обновление session_state с выбранными параметрами
    st.session_state.selected_params = selected_params
    for name in selected_params:
        st.session_state.param_dict[name] = st.sidebar.number_input(
            name, value=st.session_state.param_dict.get(name, 0.0), key=name
        )

    try:
        return synthetic_data.generate_synthetic_time_series(
            date_range, **st.session_state.param_dict
        )
    except Exception as e:
        st.sidebar.error(f"Ошибка при генерации ряда: {str(e)}")
        return None


def process_time_series(time_series):

    st.subheader("Исходный временной ряд")
    visualization.df_chart_display_loc(
        time_series,
        data_col_loc=time_series.select_dtypes(include=["int", "float"]).columns,
    )

    st.sidebar.header("Предобработка ряда")
    with st.sidebar.expander("Что это такое?"):
        txt.preprocessing()

    if isinstance(time_series.index, pd.DatetimeIndex):
        time_series = time_series.reset_index()

    numeric_columns = list(time_series.select_dtypes(include=["int", "float"]).columns)
    other_columns = list(time_series.columns)

    st.session_state.date_column_select = st.sidebar.selectbox(
        label="Выберите колонку отражающую время", options=([""] + other_columns)
    )
    st.session_state.value_select = st.sidebar.selectbox(
        label="Выберите колонку для анализа", options=([""] + numeric_columns)
    )

    date_column_select = st.session_state.date_column_select
    value_select = st.session_state.value_select
    utils.nothing_selected_sidebar(date_column_select)
    utils.nothing_selected_sidebar(value_select)

    time_series_selected = time_series.loc[:, [date_column_select, value_select]]
    time_series_selected = time_series_selected.rename(
        columns={date_column_select: "date", value_select: "value"}
    )
    time_series_selected["date"] = pd.to_datetime(time_series_selected["date"])
    time_series_selected.set_index("date", inplace=True)

    st.session_state.start_point, st.session_state.end_point = st.sidebar.slider(
        "Выберите диапазон данных для дальнейшей работы",
        0,
        time_series_selected.shape[0] - 1,
        (0, time_series_selected.shape[0] - 1),
        key="time series borders",
    )

    time_series_selected_limited = time_series_selected.iloc[
        st.session_state.start_point : st.session_state.end_point
    ]

    st.subheader("Выбранный временной ряд")
    visualization.df_chart_display_iloc(time_series_selected_limited, 0)

    T_s_len = st.session_state.end_point - st.session_state.start_point
    st.sidebar.write("Размер выбранного диапазона:", T_s_len)
    st.dataframe(time_series_selected_limited)

    MA_checkbox = st.sidebar.checkbox("Сгладить ряд", key="MA_checkbox")
    with st.sidebar.expander("Что значит «сгладить»?"):
        txt.moving_average()

    if MA_checkbox:
        st.session_state.MA_step = int(
            st.sidebar.number_input(
                "Введите шаг скользящего среднего", min_value=1, max_value=T_s_len
            )
        )
        MA_step = st.session_state.MA_step
        time_series_selected_limited = data_processing.smooth_time_series(
            time_series_selected_limited, MA_step
        )
        st.subheader("Сглаженный ряд")
        st.line_chart(time_series_selected_limited)

    st.session_state.time_series = time_series_selected_limited

    stationar_test_checkbox = st.sidebar.checkbox(
        "Тест на стационарность", key="stat_test_checkbox"
    )
    with st.sidebar.expander("Что это за тест?"):
        txt.stationar_test()

    if stationar_test_checkbox:
        if MA_checkbox:
            data_processing.check_stationarity(
                time_series_selected_limited.iloc[MA_step:, 0]
            )
        else:
            data_processing.check_stationarity(time_series_selected_limited.iloc[:, 0])

    return time_series_selected_limited


def main():
    st.title("Комплекс для работы с временными рядами")
    st.sidebar.header("Выбор временного ряда для обработки")
    txt.intro_text()

    # Инициализация состояния сессии
    utils.initialize_session_state(
        time_series=None, analysis_results=None, data_radio=None
    )
    # Выбор данных для обработки с сохранением в session_state
    data_radio = st.sidebar.selectbox(
        "Выберите данные для обработки",
        DATA_TYPES,
        index=(
            DATA_TYPES.index(st.session_state.data_radio)
            if st.session_state.data_radio
            else 0
        ),
    )
    st.session_state.data_radio = data_radio

    # Кнопка для сброса временного ряда
    if st.sidebar.button("Сбросить временной ряд"):
        st.session_state.time_series = None
        st.session_state.analysis_results = None
        st.session_state.data_radio = None
        st.session_state.start_date = None
        st.session_state.end_date = None
        st.session_state.selected_params = None
        st.experimental_rerun()

    # Загрузка или создание временного ряда
    utils.nothing_selected_sidebar(data_radio)
    if st.session_state.data_radio == "Загруженный ряд":
        st.session_state.time_series = load_data()
    elif st.session_state.data_radio == "Искусственный ряд":
        st.session_state.time_series = create_synthetic_data()

    # Обработка временного ряда
    st.success("Временной ряд успешно загружен!")
    processed_time_series = process_time_series(st.session_state.time_series)
    st.session_state.time_series = processed_time_series

    # Кнопка для выполнения статистического анализа
    if st.button("Выполнить статистический анализ"):
        with st.spinner("Выполняется статистический анализ..."):
            st.session_state.analysis_results = (
                time_series_analysis.perform_statistical_analysis(
                    processed_time_series["value"]
                )
            )

    # Отображение результатов анализа
    if st.session_state.analysis_results is not None:
        analysis_results = st.session_state.analysis_results
        st.subheader("Статистический анализ временного ряда")
        st.write("Базовая статистика:")
        st.write(analysis_results["basic_stats"])
        st.write("Результаты теста на стационарность (тест Дики-Фуллера):")
        st.write(analysis_results["stationarity"])
        st.subheader("Автокорреляция и частичная автокорреляция")
        fig_acf_pacf = visualization.plot_statistical_analysis(analysis_results)
        st.pyplot(fig_acf_pacf)
        period = st.number_input(
            "Введите период сезонности для декомпозиции:", min_value=1, value=1
        )
        if st.checkbox("Выполнить декомпозицию временного ряда"):
            st.subheader("Декомпозиция временного ряда")
            decomposition = time_series_analysis.decompose_time_series(
                st.session_state.time_series["value"], period
            )
            fig_decomposition = visualization.plot_decomposition(decomposition)
            st.pyplot(fig_decomposition)
    else:
        st.warning("Статистический анализ ряда пока не проводился.")


if __name__ == "__main__":
    main()
