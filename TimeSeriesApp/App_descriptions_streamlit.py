import streamlit as st
import pandas as pd


def sample_csv_download_button():
    with st.expander("Что делать, если нет своего ряда?"):

        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")

        convert_df(pd.read_csv("Vietnam_CO2_Temp.csv").iloc[:, 2:])
        st.write("Настоящий ряд для тестирования можно скачать тут")
        st.download_button(
            label="Download data as CSV",
            data=convert_df(pd.read_csv("Vietnam_CO2_Temp.csv")),
            file_name="Vietnam.csv",
            # mime='text/csv',
        )


def intro_text():
    with st.expander("О комплексе?"):
        st.write("Комплекс разработан для работы с временными рядами метеорологических данных. Он позволяет совершить предобаботку ряда, а также его анализ и прогнозирование используя один из представленных в боковом меню методов. Комплекс доступен для запуска с любого устройства онлайн по ссылке https://time-series-msu-ff.streamlit.app/")
